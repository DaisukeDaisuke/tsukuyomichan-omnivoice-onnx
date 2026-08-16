#!/usr/bin/env python3
"""Convert only the expensive constant-weight OmniVoice LLM MatMuls to FP16.

The surrounding runtime stays FP32.  The LLM keeps FP32 graph inputs/outputs
so typed-voice can use exactly the same session contract as the FP32 profile.
For each constant-weight MatMul, the activation is cast to FP16 immediately
before the MatMul and the result is cast back to FP32 immediately afterwards.
RMSNorm, attention/softmax, residual math, nonlinearities, and dynamic MatMuls
therefore remain FP32 while the dominant LLM weight storage and GEMM work use
the WebGPU FP16 path.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto
from onnxruntime.transformers.float16 import convert_tensor_float_to_float16

from export_fp32 import split_oversized_external_data, validate_llm_attention_contract


MAX_RELATIVE_RMSE = 0.02
MIN_COSINE_SIMILARITY = 0.999
MATMUL_FP16_PREFIX = "matmul_fp16_"


def external_locations(model_path: Path) -> set[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    locations: set[str] = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        metadata = {entry.key: entry.value for entry in tensor.external_data}
        location = metadata.get("location")
        if not location:
            raise RuntimeError(f"External initializer {tensor.name} has no location")
        locations.add(location)
    return locations


def runtime_bytes(model_path: Path) -> int:
    total = model_path.stat().st_size
    total += sum((model_path.parent / location).stat().st_size for location in external_locations(model_path))
    return total


def validate_fp16_llm_graph(model_path: Path) -> int:
    validate_llm_attention_contract(model_path)
    model = onnx.load(str(model_path), load_external_data=False)
    outputs = {value.name: value for value in model.graph.output}
    hidden = outputs.get("hidden_states")
    if hidden is None or hidden.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError("FP16 LLM must keep hidden_states output as FP32")

    quant_ops = sorted({
        node.op_type
        for node in model.graph.node
        if node.op_type in {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "MatMulNBits"}
        or "NBits" in node.op_type
    })
    if quant_ops:
        raise RuntimeError(f"FP16 LLM unexpectedly contains quantization operators: {quant_ops}")

    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    matmul_weights = {
        node.input[1]
        for node in model.graph.node
        if node.op_type == "MatMul" and len(node.input) > 1 and node.input[1] in initializers
    }
    if not matmul_weights:
        raise RuntimeError("FP16 LLM contains no constant MatMul weights")
    wrong = [name for name in sorted(matmul_weights) if initializers[name].data_type != TensorProto.FLOAT16]
    if wrong:
        raise RuntimeError(f"FP16 LLM still has non-FP16 MatMul weights: {wrong[:10]}")
    cast_in = {
        node.name
        for node in model.graph.node
        if node.op_type == "Cast"
        and node.name.startswith(MATMUL_FP16_PREFIX)
        and node.name.endswith("_cast_in")
        and {attribute.name: attribute.i for attribute in node.attribute}.get("to") == TensorProto.FLOAT16
    }
    cast_out = {
        node.name
        for node in model.graph.node
        if node.op_type == "Cast"
        and node.name.startswith(MATMUL_FP16_PREFIX)
        and node.name.endswith("_cast_out")
        and {attribute.name: attribute.i for attribute in node.attribute}.get("to") == TensorProto.FLOAT
    }
    if len(cast_in) != len(matmul_weights) or len(cast_out) != len(matmul_weights):
        raise RuntimeError(
            "FP16 LLM must wrap every constant MatMul with FP32->FP16 and FP16->FP32 casts: "
            f"matmul={len(matmul_weights)} input_casts={len(cast_in)} output_casts={len(cast_out)}"
        )
    return len(matmul_weights)


def convert_llm(source_model: Path, output_model: Path) -> int:
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_data = output_model.parent / f"{output_model.name}.data"
    if output_model.exists():
        output_model.unlink()
    if output_data.exists():
        output_data.unlink()

    converted = onnx.load(str(source_model), load_external_data=True)
    initializers = {tensor.name: tensor for tensor in converted.graph.initializer}
    users: dict[str, list[onnx.NodeProto]] = {}
    for node in converted.graph.node:
        for name in node.input:
            users.setdefault(name, []).append(node)

    targets = [
        node
        for node in converted.graph.node
        if node.op_type == "MatMul"
        and len(node.input) > 1
        and node.input[1] in initializers
        and initializers[node.input[1]].data_type == TensorProto.FLOAT
    ]
    if not targets:
        raise RuntimeError("FP16 LLM conversion found no constant FP32 MatMul weights")
    target_ids = {id(node) for node in targets}
    weights = {node.input[1] for node in targets}
    for weight_name in sorted(weights):
        unexpected = [node.name for node in users.get(weight_name, []) if id(node) not in target_ids]
        if unexpected:
            raise RuntimeError(
                f"MatMul weight {weight_name} is shared with non-target nodes and cannot be converted safely: {unexpected[:10]}"
            )
        convert_tensor_float_to_float16(initializers[weight_name])

    rewritten_nodes = []
    converted_count = 0
    for node in converted.graph.node:
        if id(node) not in target_ids:
            rewritten_nodes.append(node)
            continue
        tag = f"{MATMUL_FP16_PREFIX}{converted_count:03d}"
        original_input = node.input[0]
        original_output = node.output[0]
        fp16_input = f"{tag}_input"
        fp16_output = f"{tag}_output"
        rewritten_nodes.append(
            onnx.helper.make_node(
                "Cast",
                [original_input],
                [fp16_input],
                to=TensorProto.FLOAT16,
                name=f"{tag}_cast_in",
            )
        )
        node.input[0] = fp16_input
        node.output[0] = fp16_output
        rewritten_nodes.append(node)
        rewritten_nodes.append(
            onnx.helper.make_node(
                "Cast",
                [fp16_output],
                [original_output],
                to=TensorProto.FLOAT,
                name=f"{tag}_cast_out",
            )
        )
        converted_count += 1
    del converted.graph.node[:]
    converted.graph.node.extend(rewritten_nodes)
    onnx.save_model(
        converted,
        str(output_model),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_data.name,
        size_threshold=1024,
        convert_attribute=False,
    )
    del converted
    gc.collect()
    matmul_weight_count = validate_fp16_llm_graph(output_model)
    if matmul_weight_count != converted_count:
        raise RuntimeError(f"FP16 MatMul count changed after serialization: {matmul_weight_count} != {converted_count}")
    return matmul_weight_count


def compare_case(label: str, expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
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
            f"{label} FP16 equivalence is outside the safety envelope: "
            f"relative_rmse={relative_rmse:.6f}, cosine={cosine:.6f}, max_abs={max_abs:.6f}"
        )
    return {"relative_rmse": relative_rmse, "cosine": cosine, "max_abs": max_abs}


def verify_equivalence(model_path: Path, case_path: Path) -> dict[str, dict[str, float]]:
    case = np.load(case_path)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    if [item.name for item in session.get_inputs()] != ["inputs_embeds", "attention_mask"]:
        raise RuntimeError("FP16 LLM input contract changed unexpectedly")
    metrics: dict[str, dict[str, float]] = {}
    for prefix, label in (("", "primary"), ("alt_", "alternate")):
        actual = session.run(
            ["hidden_states"],
            {
                "inputs_embeds": case[f"{prefix}inputs_embeds"].astype(np.float32),
                "attention_mask": case[f"{prefix}attention_mask"].astype(np.bool_),
            },
        )[0]
        metrics[label] = compare_case(label, case[f"{prefix}expected"], actual)
    return metrics


def replace_release_llm(release_model: Path, converted_model: Path) -> None:
    original_locations = external_locations(release_model)
    converted_locations = external_locations(converted_model)
    if converted_locations != {f"{converted_model.name}.data"}:
        raise RuntimeError(f"Unexpected converted external-data layout: {sorted(converted_locations)}")

    for location in original_locations:
        path = release_model.parent / location
        if path.exists():
            path.unlink()
    release_model.unlink()

    converted_data = converted_model.parent / f"{converted_model.name}.data"
    final_data = release_model.parent / f"{release_model.name}.data"
    shutil.move(str(converted_model), str(release_model))
    shutil.move(str(converted_data), str(final_data))

    # The temporary and final model names are intentionally identical, so the
    # external-data location embedded in the graph remains valid after moving.
    validate_fp16_llm_graph(release_model)
    split_oversized_external_data(release_model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--release-dir", required=True)
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    release = Path(args.release_dir).resolve()
    source_model = release / "llm_decoder.onnx"
    case_path = work / "llm_equiv.npz"
    if not source_model.is_file() or not case_path.is_file():
        raise RuntimeError("FP16 conversion requires the complete validated FP32 backbone export")

    before_bytes = runtime_bytes(source_model)
    temp_dir = work / "fp16-llm"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True)
    converted_model = temp_dir / "llm_decoder.onnx"
    matmul_weight_count = convert_llm(source_model, converted_model)
    metrics = verify_equivalence(converted_model, case_path)
    after_bytes = runtime_bytes(converted_model)
    if after_bytes >= before_bytes:
        raise RuntimeError(f"FP16 LLM did not reduce runtime size: {after_bytes} >= {before_bytes}")

    replace_release_llm(source_model, converted_model)
    metadata = {
        "scope": "llm-only",
        "weightDtype": "float16",
        "computeDtype": "fp16-constant-matmul-fp32-otherwise",
        "ioDtype": "float32",
        "fp16ComputeScope": "constant-matmul-only",
        "matMulWeightCount": matmul_weight_count,
        "sourceBytes": before_bytes,
        "convertedBytes": after_bytes,
        "equivalence": metrics,
    }
    (work / "fp16-conversion.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
