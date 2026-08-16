#!/usr/bin/env python3
"""Behavioral safety tests that run before downloading multi-GB checkpoints."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from export_fp32 import split_oversized_external_data, validate_unquantized_graph
from finalize_release import REQUIRED_FILES, REQUIRED_MODELS, check_release_files


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
    with tempfile.TemporaryDirectory(prefix="tsukuyomichan-onnx-self-test-") as temp:
        root = Path(temp)
        test_external_data_split(root)
        test_quantized_graph_rejected(root)
        test_release_rejects_safetensors(root)
    print("converter behavioral safety tests: PASS")


if __name__ == "__main__":
    main()