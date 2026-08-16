#!/usr/bin/env python3
"""Behavioral safety tests that run before downloading multi-GB checkpoints."""
from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from export_fp32 import (
    ort_session,
    split_oversized_external_data,
    torch_export,
    validate_llm_attention_contract,
    validate_unquantized_graph,
)
from finalize_release import (
    REQUIRED_FILES,
    REQUIRED_MODELS,
    build_distribution,
    check_release_files,
    release_asset,
    resolve_llm_runtime_dimensions,
    verify_source_runtime_files,
)
from upload_huggingface import BASE_MODEL, DEFAULT_HF_REPO_ID, SAMPLE_SPECS, build_model_card


def test_required_model_apis_import() -> None:
    # Run before any multi-GB checkpoint download.  The converter depends on
    # OmniVoice's own package for the model class; it is not a Transformers
    # top-level model export.
    from omnivoice import OmniVoice
    from huggingface_hub import HfApi
    import onnx_ir
    import onnxscript
    from transformers import HiggsAudioV2TokenizerModel

    assert callable(getattr(OmniVoice, "from_pretrained", None))
    assert callable(getattr(onnx_ir, "load", None))
    assert hasattr(onnxscript, "__version__")
    assert callable(getattr(HiggsAudioV2TokenizerModel, "from_pretrained", None))
    assert callable(getattr(HfApi, "whoami", None))
    assert callable(getattr(HfApi, "create_repo", None))
    upload_parameters = inspect.signature(HfApi.upload_folder).parameters
    tag_parameters = inspect.signature(HfApi.create_tag).parameters
    assert "delete_patterns" in upload_parameters
    assert "path_in_repo" in upload_parameters
    assert "exist_ok" in tag_parameters

def test_modern_torch_onnx_export(root: Path) -> None:
    class TinyAdd(torch.nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return left + right

    path = root / "modern-export.onnx"
    left = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    right = torch.ones_like(left)
    torch_export(
        TinyAdd().eval(),
        (left, right),
        path,
        ["left", "right"],
        ["result"],
        {
            "left": {0: "batch", 1: "seq"},
            "right": {0: "batch", 1: "seq"},
            "result": {0: "batch", 1: "seq"},
        },
    )
    onnx.checker.check_model(onnx.load(str(path), load_external_data=False))
    session = ort_session(path)
    actual = session.run(["result"], {"left": left.numpy(), "right": right.numpy()})[0]
    np.testing.assert_allclose(actual, left.numpy() + right.numpy(), rtol=0.0, atol=0.0)

    # Prove the new exporter kept the shared batch/sequence dimensions dynamic,
    # rather than only accepting the example shape used during export.
    alt_left = np.arange(18, dtype=np.float32).reshape(2, 9)
    alt_right = np.full((2, 9), 2.0, dtype=np.float32)
    alt_actual = session.run(["result"], {"left": alt_left, "right": alt_right})[0]
    np.testing.assert_allclose(alt_actual, alt_left + alt_right, rtol=0.0, atol=0.0)


def save_llm_contract_model(path: Path, valid_attention: bool) -> None:
    attention_type = TensorProto.BOOL if valid_attention else TensorProto.INT64
    attention_shape = ["batch", 1, "seq", "seq"] if valid_attention else ["batch", "seq"]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["inputs_embeds"], ["hidden_states"])],
        "llm-attention-contract-test",
        [
            helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, ["batch", "seq", 1024]),
            helper.make_tensor_value_info("attention_mask", attention_type, attention_shape),
        ],
        [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, ["batch", "seq", 1024])],
    )
    onnx.save_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), str(path))


def test_llm_attention_contract_rejects_causal_builder_shape(root: Path) -> None:
    valid = root / "llm-full-attention.onnx"
    save_llm_contract_model(valid, True)
    validate_llm_attention_contract(valid)

    causal = root / "llm-causal-attention.onnx"
    save_llm_contract_model(causal, False)
    try:
        validate_llm_attention_contract(causal)
    except RuntimeError as error:
        assert "rank-4 BOOL" in str(error)
    else:
        raise AssertionError("2-D causal/padding attention contract was accepted")


def test_manifest_uses_explicit_qwen_head_dim() -> None:
    hidden_size, kv_heads, head_dim = resolve_llm_runtime_dimensions(
        {
            "llm_config": {
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
            }
        }
    )
    assert hidden_size == 1024
    assert kv_heads == 8
    assert head_dim == 128


def test_huggingface_distribution_and_model_card(root: Path) -> None:
    revision = "gh-0123456789abcdef-12345"
    distribution = build_distribution(revision)
    assert distribution["provider"] == "github-release"
    mirror = distribution["mirrors"][0]
    assert mirror["repo"] == DEFAULT_HF_REPO_ID
    assert mirror["revision"] == revision
    assert mirror["assetBaseUrl"].endswith(f"/{revision}/")

    release = root / "hf-card"
    release.mkdir()
    (release / "runtime-manifest.json").write_text(
        json.dumps(
            {
                "qualityProfile": "fp32-unquantized",
                "source": {
                    "voiceCheckpoint": {"revision": "voice-revision", "modelSha256": "a" * 64},
                    "audioCodec": {"revision": "codec-revision"},
                },
            }
        ),
        encoding="utf-8",
    )
    card = build_model_card(release, revision, "b" * 40, "10")
    assert f"base_model: {BASE_MODEL}" in card
    assert "license: other" in card
    assert revision in card
    assert "### 出力音声の利用制限" in card
    assert "人を批判・攻撃すること。" in card
    assert "特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。" in card
    assert "刺激の強い表現をゾーニングなしで公開すること。" in card
    assert "他者に対して二次利用（素材としての利用）を許可する形で公開すること。" in card
    assert "### 改変・再配布について" in card
    assert "派生ソフトや再配布されたデータにもコピーレフトされます。" in card
    assert "## Audio Samples" in card
    assert "native Python + ONNX Runtime" in card
    assert "### Japanese" in card
    assert "### English" in card
    assert "ChatGPT GPT-5.6 Sol high" in card
    assert "https://github.com/DaisukeDaisuke/tsukuyomichan-omnivoice-onnx" in card
    assert "reproducible numerical checks" in card
    assert "samples/SAMPLES_SHA256SUMS" in card
    assert "XXH3-128" in card
    assert "first-download and reload validation" in card
    for filename, text in SAMPLE_SPECS:
        assert text in card
        assert f"/resolve/main/samples/{filename}" in card
        assert f"[samples/{filename}](./samples/{filename})" in card


def test_release_asset_has_sha256_and_xxh3_128(root: Path) -> None:
    path = root / "hash-contract.bin"
    path.write_bytes(b"typed-voice-fast-cache-check")
    asset = release_asset(path, "runtime")
    assert asset["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(asset["xxh3_128"]) == 32
    assert all(char in "0123456789abcdef" for char in asset["xxh3_128"])


def test_source_runtime_files_cannot_be_clobbered(root: Path) -> None:
    release = root / "source-runtime-release"
    work = root / "source-runtime-work"
    release.mkdir()
    work.mkdir()
    mapping = {
        "config.json": "omnivoice_config.json",
        "tokenizer.json": "tokenizer.json",
        "tokenizer_config.json": "tokenizer_config.json",
        "chat_template.jinja": "chat_template.jinja",
    }
    runtime_files = {}
    for index, (source_name, release_name) in enumerate(mapping.items()):
        data = f"pinned-runtime-file-{index}\n".encode("utf-8")
        (release / release_name).write_bytes(data)
        runtime_files[source_name] = {
            "byte_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (work / "full-source.json").write_text(
        json.dumps({"runtime_files": runtime_files}) + "\n",
        encoding="utf-8",
    )
    verify_source_runtime_files(release, work)
    (release / "tokenizer.json").write_text("builder-overwrite\n", encoding="utf-8")
    try:
        verify_source_runtime_files(release, work)
    except RuntimeError as error:
        assert "modified after download" in str(error)
    else:
        raise AssertionError("Source tokenizer clobber was not rejected")


def test_pinned_license_fetches(root: Path) -> None:
    scripts_dir = Path(__file__).resolve().parent
    for script_name, output_name in (
        ("fetch_omnivoice_code_license.py", "OMNIVOICE_CODE_LICENSE.txt"),
        ("fetch_meta_license.py", "META-LLAMA-3-LICENSE.txt"),
    ):
        output = root / output_name
        subprocess.run(
            [sys.executable, str(scripts_dir / script_name), "--output", str(output)],
            check=True,
        )
        assert output.is_file() and output.stat().st_size > 0


def save_two_weight_model(path: Path) -> tuple[np.ndarray, np.ndarray]:
    left = np.arange(64, dtype=np.float32).reshape(8, 8)
    right = np.arange(64, 128, dtype=np.float32).reshape(8, 8)
    graph = helper.make_graph(
        [helper.make_node("Add", ["left", "right"], ["sum"])],
        "external-data-test",
        [],
        [helper.make_tensor_value_info("sum", TensorProto.FLOAT, [8, 8])],
        [numpy_helper.from_array(left, name="left"), numpy_helper.from_array(right, name="right")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{path.name}.data",
        size_threshold=0,
        convert_attribute=False,
    )
    return left, right


def test_external_data_split(root: Path) -> None:
    model_path = root / "split.onnx"
    expected_left, expected_right = save_two_weight_model(model_path)
    original_data = root / "split.onnx.data"
    assert original_data.stat().st_size > 300
    split_oversized_external_data(model_path, limit=300)
    assert not original_data.exists()
    chunks = sorted(root.glob("split.onnx.data.*"))
    assert len(chunks) == 2
    assert all(chunk.stat().st_size <= 300 for chunk in chunks)
    loaded = onnx.load(str(model_path), load_external_data=True)
    actual = {item.name: numpy_helper.to_array(item) for item in loaded.graph.initializer}
    np.testing.assert_array_equal(actual["left"], expected_left)
    np.testing.assert_array_equal(actual["right"], expected_right)


def test_quantized_graph_rejected(root: Path) -> None:
    path = root / "quantized.onnx"
    scale = numpy_helper.from_array(np.array([0.1], dtype=np.float32), name="scale")
    zero = numpy_helper.from_array(np.array([0], dtype=np.uint8), name="zero")
    graph = helper.make_graph(
        [helper.make_node("QuantizeLinear", ["input", "scale", "zero"], ["output"])],
        "quantization-rejection-test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1])],
        [scale, zero],
    )
    onnx.save_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), str(path))
    try:
        validate_unquantized_graph(path)
    except RuntimeError as error:
        assert "Quantization operators found" in str(error)
    else:
        raise AssertionError("Quantized ONNX graph was not rejected")


def save_empty_model(path: Path) -> None:
    graph = helper.make_graph([], path.stem, [], [])
    onnx.save_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), str(path))


def test_release_rejects_safetensors(root: Path) -> None:
    release = root / "release"
    release.mkdir()
    for name in REQUIRED_MODELS:
        save_empty_model(release / name)
    for name in REQUIRED_FILES:
        (release / name).write_text("{}\n" if name.endswith(".json") else "test\n", encoding="utf-8")
    (release / "do-not-release.safetensors").write_bytes(b"not a model")
    try:
        check_release_files(release)
    except RuntimeError as error:
        assert "Safetensors build inputs must never be released" in str(error)
    else:
        raise AssertionError("Release gate accepted a .safetensors build input")


def main() -> None:
    test_required_model_apis_import()
    test_manifest_uses_explicit_qwen_head_dim()
    with tempfile.TemporaryDirectory(prefix="tsukuyomichan-onnx-self-test-") as temp:
        root = Path(temp)
        test_modern_torch_onnx_export(root)
        test_llm_attention_contract_rejects_causal_builder_shape(root)
        test_huggingface_distribution_and_model_card(root)
        test_source_runtime_files_cannot_be_clobbered(root)
        test_pinned_license_fetches(root)
        test_external_data_split(root)
        test_quantized_graph_rejected(root)
        test_release_rejects_safetensors(root)
    print("converter behavioral safety tests: PASS")


if __name__ == "__main__":
    main()