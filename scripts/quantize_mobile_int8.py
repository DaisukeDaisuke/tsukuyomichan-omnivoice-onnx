#!/usr/bin/env python3
"""Convert only the already-correct OmniVoice LLM ONNX weights to MatMulNBits.

The FP32 exporter remains the single source of truth for graph structure and
OmniVoice's non-causal rank-4 Boolean attention contract. This script runs
after that export and replaces constant MatMul weights in llm_decoder.onnx
with block-wise 4-bit or 8-bit MatMulNBits weights. Audio embeddings, audio heads and
the Higgs decoder are deliberately left unchanged.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from onnxruntime.quantization.quant_utils import QuantFormat


BITS = 8
SUPPORTED_BITS = (4, 8)
BLOCK_SIZE = 128
# Keep activations in FP32. The mobile experiment is weight-only.
ACCURACY_LEVEL = 1
MAX_RELATIVE_RMSE = 0.05
MIN_COSINE_SIMILARITY = 0.995
MAX_RELEASE_ASSET_BYTES = 1_800_000_000


def external_metadata(tensor: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def retarget_external_data(tensor: onnx.TensorProto, location: str, offset: int, length: int) -> None:
    tensor.ClearField("external_data")
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value


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
                raise RuntimeError(f"Single external tensor {tensor.name} is too large: {length}")
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
            retarget_external_data(tensor, chunk_path.name, chunk_size, length)
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


def validate_llm_attention_contract(model_path: Path) -> None:
    model = onnx.load(str(model_path), load_external_data=False)
    inputs = {value.name: value for value in model.graph.input}
    if set(inputs) != {"inputs_embeds", "attention_mask"}:
        raise RuntimeError(f"LLM runtime inputs must be inputs_embeds + attention_mask only, got {sorted(inputs)}")
    embeds = inputs["inputs_embeds"].type.tensor_type
    attention = inputs["attention_mask"].type.tensor_type
    if embeds.elem_type != TensorProto.FLOAT or len(embeds.shape.dim) != 3:
        raise RuntimeError("LLM inputs_embeds must be rank-3 FP32")
    if attention.elem_type != TensorProto.BOOL or len(attention.shape.dim) != 4:
        raise RuntimeError("LLM attention_mask must be rank-4 BOOL to preserve OmniVoice non-causal attention")
    head_axis = attention.shape.dim[1]
    if not head_axis.HasField("dim_value") or head_axis.dim_value != 1:
        raise RuntimeError("LLM attention_mask must have shape [batch, 1, sequence, sequence]")


def node_int_attribute(node: onnx.NodeProto, name: str) -> int | None:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(onnx.helper.get_attribute_value(attribute))
    return None


def external_locations(model_path: Path) -> set[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    locations: set[str] = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        location = external_metadata(tensor).get("location")
        if not location:
            raise RuntimeError(f"External initializer {tensor.name} has no location")
        locations.add(location)
    return locations


def validate_mobile_graph(model_path: Path, bits: int = BITS) -> int:
    validate_llm_attention_contract(model_path)
    model = onnx.load(str(model_path), load_external_data=False)
    quantized = [node for node in model.graph.node if node.domain == "com.microsoft" and node.op_type == "MatMulNBits"]
    if not quantized:
        raise RuntimeError("Mobile LLM contains no MatMulNBits nodes")
    for node in quantized:
        node_bits = node_int_attribute(node, "bits")
        block_size = node_int_attribute(node, "block_size")
        accuracy_level = node_int_attribute(node, "accuracy_level")
        if node_bits != bits:
            raise RuntimeError(f"Unexpected MatMulNBits bits={node_bits} in {node.name!r}; expected {bits}")
        if block_size != BLOCK_SIZE:
            raise RuntimeError(f"Unexpected MatMulNBits block_size={block_size} in {node.name!r}")
        if accuracy_level != ACCURACY_LEVEL:
            raise RuntimeError(f"Unexpected MatMulNBits accuracy_level={accuracy_level} in {node.name!r}")
    return len(quantized)


def make_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def compare_case(label: str, expected: np.ndarray, actual: np.ndarray, bits: int = BITS) -> dict[str, float]:
    expected64 = expected.astype(np.float64, copy=False)
    actual64 = actual.astype(np.float64, copy=False)
    if expected64.shape != actual64.shape:
        raise RuntimeError(f"{label} shape mismatch: {expected64.shape} != {actual64.shape}")
    if not np.isfinite(actual64).all():
        raise RuntimeError(f"{label} produced non-finite values")
    delta = actual64 - expected64
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    reference_rms = float(np.sqrt(np.mean(np.square(expected64))))
    relative_rmse = rmse / max(reference_rms, 1e-12)
    expected_flat = expected64.reshape(-1)
    actual_flat = actual64.reshape(-1)
    denominator = float(np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat))
    cosine = float(np.dot(expected_flat, actual_flat) / denominator) if denominator else 1.0
    max_abs = float(np.max(np.abs(delta)))
    if relative_rmse > MAX_RELATIVE_RMSE or cosine < MIN_COSINE_SIMILARITY:
        raise RuntimeError(
            f"{label} {bits}-bit equivalence is outside the PoC safety envelope: "
            f"relative_rmse={relative_rmse:.6f}, cosine={cosine:.6f}, max_abs={max_abs:.6f}"
        )
    return {"relative_rmse": relative_rmse, "cosine": cosine, "max_abs": max_abs}


def verify_equivalence(model_path: Path, case_path: Path, bits: int = BITS) -> dict[str, dict[str, float]]:
    case = np.load(case_path)
    session = make_session(model_path)
    if [item.name for item in session.get_inputs()] != ["inputs_embeds", "attention_mask"]:
        raise RuntimeError(f"{bits}-bit LLM input contract changed unexpectedly")
    metrics: dict[str, dict[str, float]] = {}
    for prefix, label in (("", "primary"), ("alt_", "alternate")):
        actual = session.run(
            ["hidden_states"],
            {
                "inputs_embeds": case[f"{prefix}inputs_embeds"].astype(np.float32),
                "attention_mask": case[f"{prefix}attention_mask"].astype(np.bool_),
            },
        )[0]
        metrics[label] = compare_case(label, case[f"{prefix}expected"], actual, bits)
    return metrics


def quantize_llm(source_model: Path, output_model: Path, bits: int = BITS) -> None:
    if bits not in SUPPORTED_BITS:
        raise RuntimeError(f"Unsupported MatMulNBits width: {bits}")
    config = DefaultWeightOnlyQuantConfig(
        block_size=BLOCK_SIZE,
        is_symmetric=True,
        accuracy_level=ACCURACY_LEVEL,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        bits=bits,
    )
    quantizer = MatMulNBitsQuantizer(str(source_model), algo_config=config)
    quantizer.process()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    quantizer.model.save_model_to_file(str(output_model), True)
    del quantizer
    gc.collect()
    split_oversized_external_data(output_model)


def replace_release_llm(release: Path, quantized_model: Path) -> None:
    original_model = release / "llm_decoder.onnx"
    old_files = {original_model, *(release / location for location in external_locations(original_model))}
    new_files = {quantized_model, *(quantized_model.parent / location for location in external_locations(quantized_model))}
    for path in old_files:
        path.unlink(missing_ok=True)
    for source in new_files:
        destination = release / source.name
        if destination.exists():
            destination.unlink()
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--bits", type=int, choices=SUPPORTED_BITS, default=BITS)
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    release = Path(args.release_dir).resolve()
    source_model = release / "llm_decoder.onnx"
    case_path = work / "llm_equiv.npz"
    if not source_model.is_file() or not case_path.is_file():
        raise RuntimeError("FP32 LLM export and llm_equiv.npz must exist before mobile quantization")

    bits = int(args.bits)
    profile = f"mobile-int{bits}"
    output_dir = work / f"{profile}-llm"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_model = output_dir / "llm_decoder.onnx"
    quantize_llm(source_model, output_model, bits)
    node_count = validate_mobile_graph(output_model, bits)
    metrics = verify_equivalence(output_model, case_path, bits)
    (work / f"{profile}-quantization.json").write_text(
        json.dumps(
            {
                "bits": bits,
                "blockSize": BLOCK_SIZE,
                "accuracyLevel": ACCURACY_LEVEL,
                "matMulNBitsNodes": node_count,
                "equivalence": metrics,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    replace_release_llm(release, output_model)
    validate_mobile_graph(release / "llm_decoder.onnx", bits)
    print(
        f"Mobile LLM quantization PASS: MatMulNBits={node_count}, bits={bits}, "
        f"block={BLOCK_SIZE}, accuracy_level={ACCURACY_LEVEL}, metrics={metrics}"
    )


if __name__ == "__main__":
    main()
