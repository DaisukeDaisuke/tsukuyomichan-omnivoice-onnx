#!/usr/bin/env python3
"""Export the pinned Tsukuyomichan OmniVoice checkpoint without quantization.

The source checkpoint is only a build input. The release directory receives
split ONNX runtime models and metadata, never the original safetensors file.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import TensorProto
from onnx.external_data_helper import set_external_data

HIDDEN_SIZE = 1024
NUM_CODEBOOKS = 8
AUDIO_VOCAB = 1025
HIGGS_CODEBOOK_SIZE = 1024
MAX_RELEASE_ASSET_BYTES = 1_800_000_000

FORBIDDEN_QUANT_OPS = {
    "MatMulNBits",
    "QLinearConv",
    "QLinearMatMul",
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
}
FORBIDDEN_WEIGHT_TYPES = {
    TensorProto.FLOAT16,
    TensorProto.BFLOAT16,
    TensorProto.INT8,
    TensorProto.UINT8,
}
for enum_name in ("INT4", "UINT4", "FLOAT4E2M1"):
    if hasattr(TensorProto, enum_name):
        FORBIDDEN_WEIGHT_TYPES.add(getattr(TensorProto, enum_name))


class AudioEmbeddingsEncoder(nn.Module):
    def __init__(self, text_embed: nn.Embedding, audio_embed: nn.Embedding, layer_offsets: torch.Tensor):
        super().__init__()
        self.text_embed = text_embed
        self.audio_embed = audio_embed
        self.register_buffer("layer_offsets", layer_offsets.detach().clone())

    def forward(self, input_ids: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        text_embeddings = self.text_embed(input_ids[:, 0, :])
        shifted = (input_ids * audio_mask.unsqueeze(1)) + self.layer_offsets.view(1, -1, 1)
        audio_embeddings = self.audio_embed(shifted).sum(dim=1)
        return torch.where(audio_mask.unsqueeze(-1), audio_embeddings, text_embeddings)


class AudioHeadsDecoder(nn.Module):
    def __init__(self, heads: nn.Linear):
        super().__init__()
        self.heads = heads

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        logits = self.heads(hidden_states)
        return logits.view(batch, sequence, NUM_CODEBOOKS, AUDIO_VOCAB).permute(0, 2, 1, 3)


class HiggsDecoder(nn.Module):
    def __init__(self, quantizer: nn.Module, fc2: nn.Module, decoder: nn.Module):
        super().__init__()
        self.quantizer = quantizer
        self.fc2 = fc2
        self.decoder = decoder

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        decoded = self.quantizer.decode(codes)
        projected = self.fc2(decoded.transpose(1, 2)).transpose(1, 2)
        return self.decoder(projected)


def assert_fp32_module(module: nn.Module, label: str) -> None:
    wrong = [(name, str(param.dtype)) for name, param in module.named_parameters() if param.is_floating_point() and param.dtype != torch.float32]
    if wrong:
        preview = ", ".join(f"{name}:{dtype}" for name, dtype in wrong[:8])
        raise RuntimeError(f"{label} contains non-FP32 parameters: {preview}")


def strip_weight_norm(module: nn.Module) -> None:
    for submodule in module.modules():
        stripped = False
        try:
            from torch.nn.utils.parametrize import remove_parametrizations
            if hasattr(submodule, "parametrizations") and "weight" in getattr(submodule, "parametrizations", {}):
                remove_parametrizations(submodule, "weight", leave_parametrized=True)
                stripped = True
        except Exception:
            pass
        if not stripped:
            try:
                from torch.nn.utils import remove_weight_norm
                remove_weight_norm(submodule)
                stripped = True
            except (ValueError, AttributeError):
                pass
        if stripped and hasattr(submodule, "weight") and isinstance(submodule.weight, torch.Tensor):
            submodule.weight = nn.Parameter(submodule.weight.detach())
    module.requires_grad_(False)
    for submodule in module.modules():
        for attr_name in list(vars(submodule)):
            value = getattr(submodule, attr_name, None)
            if isinstance(value, torch.Tensor) and not isinstance(value, nn.Parameter) and value.requires_grad:
                setattr(submodule, attr_name, value.detach())


def torch_export(module: nn.Module, inputs: tuple[torch.Tensor, ...], path: Path, input_names: list[str], output_names: list[str], dynamic_axes: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )


def externalize(raw_path: Path, final_path: Path, data_name: str) -> None:
    model = onnx.load(str(raw_path), load_external_data=True)
    data_path = final_path.parent / data_name
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(final_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    raw_path.unlink()


def external_metadata(tensor: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def split_oversized_external_data(model_path: Path, limit: int = MAX_RELEASE_ASSET_BYTES) -> None:
    model = onnx.load(str(model_path), load_external_data=False)
    external_tensors = []
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        meta = external_metadata(tensor)
        location = meta.get("location")
        if not location:
            raise RuntimeError(f"External initializer {tensor.name} has no location")
        source = model_path.parent / location
        offset = int(meta.get("offset", "0"))
        length = int(meta.get("length", "0"))
        if length <= 0:
            # ONNX permits omitted length only when a tensor owns the rest of a file.
            length = source.stat().st_size - offset
        external_tensors.append((tensor, source, offset, length))

    oversized_sources = {source for _, source, _, _ in external_tensors if source.stat().st_size > limit}
    if not oversized_sources:
        return

    chunk_index = 0
    chunk_path: Path | None = None
    chunk_stream = None
    chunk_size = 0
    created: list[Path] = []
    try:
        for tensor, source, offset, length in external_tensors:
            if source not in oversized_sources:
                continue
            if length > limit:
                raise RuntimeError(f"Single external tensor {tensor.name} is too large for a GitHub Release asset: {length}")
            if chunk_stream is None or chunk_size + length > limit:
                if chunk_stream is not None:
                    chunk_stream.close()
                chunk_path = model_path.parent / f"{model_path.name}.data.{chunk_index:03d}"
                chunk_index += 1
                chunk_stream = chunk_path.open("wb")
                created.append(chunk_path)
                chunk_size = 0
            assert chunk_path is not None and chunk_stream is not None
            with source.open("rb") as src:
                src.seek(offset)
                remaining = length
                while remaining:
                    block = src.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise RuntimeError(f"Unexpected EOF while splitting {source}")
                    chunk_stream.write(block)
                    remaining -= len(block)
            tensor.ClearField("external_data")
            set_external_data(tensor, location=chunk_path.name, offset=chunk_size, length=length)
            chunk_size += length
    finally:
        if chunk_stream is not None:
            chunk_stream.close()

    temp_model = model_path.with_suffix(model_path.suffix + ".rewrite")
    onnx.save_model(model, str(temp_model))
    os.replace(temp_model, model_path)
    for source in oversized_sources:
        source.unlink()
    for path in created:
        if path.stat().st_size > limit:
            raise RuntimeError(f"Release chunk still exceeds size limit: {path} ({path.stat().st_size})")


def validate_unquantized_graph(model_path: Path) -> None:
    model = onnx.load(str(model_path), load_external_data=False)
    quant_ops = sorted({node.op_type for node in model.graph.node if node.op_type in FORBIDDEN_QUANT_OPS or "NBits" in node.op_type})
    if quant_ops:
        raise RuntimeError(f"Quantization operators found in {model_path.name}: {quant_ops}")
    bad_initializers = []
    for tensor in model.graph.initializer:
        if tensor.data_type in FORBIDDEN_WEIGHT_TYPES:
            bad_initializers.append((tensor.name, onnx.helper.tensor_dtype_to_string(tensor.data_type)))
    if bad_initializers:
        raise RuntimeError(f"Reduced/quantized weight tensors found in {model_path.name}: {bad_initializers[:10]}")


def ort_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def assert_close(label: str, expected: np.ndarray, actual: np.ndarray, rtol: float = 5e-4, atol: float = 5e-4) -> None:
    if expected.shape != actual.shape:
        raise RuntimeError(f"{label} shape mismatch: {expected.shape} != {actual.shape}")
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as error:
        max_abs = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        raise RuntimeError(f"{label} numerical equivalence failed (max_abs={max_abs})") from error


def save_backbone(source_dir: Path, work: Path, release: Path) -> None:
    from transformers import OmniVoice

    torch.manual_seed(0)
    np.random.seed(0)
    model = OmniVoice.from_pretrained(
        str(source_dir),
        train=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    assert_fp32_module(model, "Tsukuyomichan full-finetune source")

    qwen_dir = work / "qwen3-standalone"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(str(qwen_dir), safe_serialization=True)
    config_path = qwen_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architectures"] = ["Qwen3ForCausalLM"]
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Preserve runtime tokenizer/config from the exact voice checkpoint.
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, release / name)
    shutil.copy2(source_dir / "config.json", release / "omnivoice_config.json")

    sequence = 16
    input_ids = torch.randint(0, AUDIO_VOCAB, (1, NUM_CODEBOOKS, sequence), dtype=torch.int64)
    audio_mask = torch.zeros(1, sequence, dtype=torch.bool)
    audio_mask[:, sequence // 2 :] = True
    embeddings = AudioEmbeddingsEncoder(model.get_input_embeddings(), model.audio_embeddings, model.codebook_layer_offsets).eval()
    with torch.no_grad():
        expected_embeddings = embeddings(input_ids, audio_mask).cpu().numpy()
    raw_embeddings = work / "audio_embeddings_encoder.raw.onnx"
    torch_export(
        embeddings,
        (input_ids, audio_mask),
        raw_embeddings,
        ["input_ids", "audio_mask"],
        ["inputs_embeds"],
        {
            "input_ids": {0: "batch", 2: "seq"},
            "audio_mask": {0: "batch", 1: "seq"},
            "inputs_embeds": {0: "batch", 1: "seq"},
        },
    )
    np.savez(work / "audio_embeddings_equiv.npz", input_ids=input_ids.numpy(), audio_mask=audio_mask.numpy(), expected=expected_embeddings)

    hidden_states = torch.randn(1, sequence, HIDDEN_SIZE, dtype=torch.float32)
    heads = AudioHeadsDecoder(model.audio_heads).eval()
    with torch.no_grad():
        expected_heads = heads(hidden_states).cpu().numpy()
    raw_heads = work / "audio_heads_decoder.raw.onnx"
    torch_export(
        heads,
        (hidden_states,),
        raw_heads,
        ["hidden_states"],
        ["logits"],
        {
            "hidden_states": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 2: "seq"},
        },
    )
    np.savez(work / "audio_heads_equiv.npz", hidden_states=hidden_states.numpy(), expected=expected_heads)

    llm_inputs = torch.randn(1, sequence, HIDDEN_SIZE, dtype=torch.float32)
    attention_mask = torch.ones(1, sequence, dtype=torch.int64)
    with torch.no_grad():
        llm_expected = model.llm(inputs_embeds=llm_inputs, attention_mask=attention_mask, use_cache=False, return_dict=True).last_hidden_state.cpu().numpy()
    np.savez(work / "llm_equiv.npz", inputs_embeds=llm_inputs.numpy(), attention_mask=attention_mask.numpy(), expected=llm_expected)

    # Release the 2.45 GB source checkpoint and PyTorch model before building the
    # FP32 LLM ONNX. The qwen3-standalone directory is a temporary build input.
    del embeddings, heads, model
    gc.collect()

    externalize(raw_embeddings, release / "audio_embeddings_encoder.onnx", "audio_embeddings_encoder.onnx.data")
    externalize(raw_heads, release / "audio_heads_decoder.onnx", "audio_heads_decoder.onnx.data")
    split_oversized_external_data(release / "audio_embeddings_encoder.onnx")
    split_oversized_external_data(release / "audio_heads_decoder.onnx")

    emb_case = np.load(work / "audio_embeddings_equiv.npz")
    emb_actual = ort_session(release / "audio_embeddings_encoder.onnx").run(
        ["inputs_embeds"],
        {"input_ids": emb_case["input_ids"], "audio_mask": emb_case["audio_mask"]},
    )[0]
    assert_close("audio_embeddings_encoder", emb_case["expected"], emb_actual)

    head_case = np.load(work / "audio_heads_equiv.npz")
    head_actual = ort_session(release / "audio_heads_decoder.onnx").run(
        ["logits"], {"hidden_states": head_case["hidden_states"]}
    )[0]
    assert_close("audio_heads_decoder", head_case["expected"], head_actual)

    source_model = source_dir / "model.safetensors"
    if source_model.exists():
        source_model.unlink()
    shutil.rmtree(source_dir, ignore_errors=True)
    gc.collect()

    builder_cache = work / "builder-cache"
    command = [
        sys.executable,
        "-m",
        "onnxruntime_genai.models.builder",
        "-i",
        str(qwen_dir),
        "-o",
        str(release),
        "-p",
        "fp32",
        "-e",
        "cpu",
        "-c",
        str(builder_cache),
        "--extra_options",
        "filename=llm_decoder.onnx",
        "exclude_embeds=true",
        "exclude_lm_head=true",
        "include_hidden_states=true",
        "shared_embeddings=false",
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    split_oversized_external_data(release / "llm_decoder.onnx")

    for model_path in (
        release / "audio_embeddings_encoder.onnx",
        release / "audio_heads_decoder.onnx",
        release / "llm_decoder.onnx",
    ):
        validate_unquantized_graph(model_path)

    verify_llm_equivalence(release / "llm_decoder.onnx", work / "llm_equiv.npz")
    shutil.rmtree(qwen_dir, ignore_errors=True)
    shutil.rmtree(builder_cache, ignore_errors=True)
    gc.collect()


def infer_dynamic_shape(shape: list, name: str) -> tuple[int, ...]:
    result: list[int] = []
    for index, dim in enumerate(shape):
        if isinstance(dim, int) and dim >= 0:
            result.append(dim)
            continue
        text = str(dim).lower()
        if "batch" in text or index == 0:
            result.append(1)
        elif "head" in text and "size" not in text:
            result.append(8)
        elif "head" in text and "size" in text:
            result.append(128)
        elif "past" in text or "cache" in text:
            result.append(0)
        else:
            # Qwen3 cache tensors are [B, kv_heads, past_seq, head_dim].
            if len(shape) == 4 and index == 1:
                result.append(8)
            elif len(shape) == 4 and index == 2:
                result.append(0)
            elif len(shape) == 4 and index == 3:
                result.append(128)
            else:
                raise RuntimeError(f"Cannot infer dynamic input dimension for {name}: {shape}")
    return tuple(result)


def verify_llm_equivalence(model_path: Path, case_path: Path) -> None:
    case = np.load(case_path)
    session = ort_session(model_path)
    embeds = case["inputs_embeds"].astype(np.float32)
    attention = case["attention_mask"].astype(np.int64)
    sequence = embeds.shape[1]
    feed = {}
    for input_meta in session.get_inputs():
        name = input_meta.name
        lower = name.lower()
        if name == "inputs_embeds":
            feed[name] = embeds
        elif name == "attention_mask":
            feed[name] = attention
        elif name == "position_ids":
            feed[name] = np.arange(sequence, dtype=np.int64)[None, :]
        elif "past_key_values" in lower or "past_key" in lower or "past_value" in lower:
            shape = infer_dynamic_shape(list(input_meta.shape), name)
            feed[name] = np.zeros(shape, dtype=np.float32)
        elif "past_sequence" in lower:
            feed[name] = np.array([0], dtype=np.int64)
        else:
            raise RuntimeError(f"Unexpected LLM input generated by ModelBuilder: {name} {input_meta.shape} {input_meta.type}")
    output_names = [output.name for output in session.get_outputs()]
    hidden_name = next((name for name in output_names if "hidden" in name.lower()), output_names[0])
    actual = session.run([hidden_name], feed)[0]
    assert_close("llm_decoder", case["expected"], actual, rtol=1e-3, atol=1e-3)


def save_higgs(higgs_dir: Path, work: Path, release: Path) -> None:
    from transformers import HiggsAudioV2TokenizerModel

    torch.manual_seed(1)
    tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(str(higgs_dir), dtype=torch.float32)
    tokenizer.eval()
    strip_weight_norm(tokenizer)
    assert_fp32_module(tokenizer, "Higgs Audio 2 source codec")
    wrapper = HiggsDecoder(tokenizer.quantizer, tokenizer.fc2, tokenizer.acoustic_decoder).eval()
    assert_fp32_module(wrapper, "Higgs Audio 2 decoder")

    codes = torch.randint(0, HIGGS_CODEBOOK_SIZE, (NUM_CODEBOOKS, 1, 8), dtype=torch.int64)
    with torch.no_grad():
        expected = wrapper(codes).cpu().numpy()
    raw = work / "higgs_decoder.raw.onnx"
    torch_export(
        wrapper,
        (codes,),
        raw,
        ["codes"],
        ["waveform_24k"],
        {
            "codes": {1: "batch", 2: "frames"},
            "waveform_24k": {0: "batch", 2: "samples"},
        },
    )
    np.savez(work / "higgs_equiv.npz", codes=codes.numpy(), expected=expected)
    del wrapper, tokenizer
    gc.collect()

    externalize(raw, release / "higgs_decoder.onnx", "higgs_decoder.onnx.data")
    split_oversized_external_data(release / "higgs_decoder.onnx")
    validate_unquantized_graph(release / "higgs_decoder.onnx")
    case = np.load(work / "higgs_equiv.npz")
    actual = ort_session(release / "higgs_decoder.onnx").run(["waveform_24k"], {"codes": case["codes"]})[0]
    assert_close("higgs_decoder", case["expected"], actual, rtol=1e-4, atol=1e-4)
    shutil.copy2(higgs_dir / "config.json", release / "higgs_config.json")
    if (higgs_dir / "preprocessor_config.json").exists():
        shutil.copy2(higgs_dir / "preprocessor_config.json", release / "higgs_preprocessor_config.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    backbone = subparsers.add_parser("backbone")
    backbone.add_argument("--source-dir", required=True)
    backbone.add_argument("--work-dir", required=True)
    backbone.add_argument("--release-dir", required=True)
    higgs = subparsers.add_parser("higgs")
    higgs.add_argument("--source-dir", required=True)
    higgs.add_argument("--work-dir", required=True)
    higgs.add_argument("--release-dir", required=True)
    args = parser.parse_args()

    work = Path(args.work_dir).resolve()
    release = Path(args.release_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    release.mkdir(parents=True, exist_ok=True)
    if args.command == "backbone":
        save_backbone(Path(args.source_dir).resolve(), work, release)
    else:
        save_higgs(Path(args.source_dir).resolve(), work, release)


if __name__ == "__main__":
    main()
