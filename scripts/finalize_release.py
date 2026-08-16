#!/usr/bin/env python3
"""Generate the release manifest, notices, hashes, and hard release gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import xxhash
from onnx import TensorProto

REQUIRED_MODELS = (
    "audio_embeddings_encoder.onnx",
    "llm_decoder.onnx",
    "audio_heads_decoder.onnx",
    "higgs_decoder.onnx",
)
REQUIRED_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "omnivoice_config.json",
    "higgs_config.json",
    "BOSON-HIGGS-AUDIO-2-LICENSE.txt",
    "META-LLAMA-3-LICENSE.txt",
    "TSUKUYOMICHAN_MODEL_CARD.md",
    "TSUKUYOMICHAN_MODEL_CARD_JA.md",
    "OMNIVOICE_MODEL_CARD.md",
    "OMNIVOICE_CODE_LICENSE.txt",
)
MAX_RELEASE_ASSET_BYTES = 1_800_000_000
SOURCE_MODEL_SIZE = 2_450_344_144
SOURCE_MODEL_SHA256 = "9ebaa8dd3bf35ceb6217cd19142bdabe6d6c044cca40672d2ae163d1a90ab47e"
HIGGS_MODEL_SIZE = 805_665_628
HIGGS_MODEL_SHA256 = "fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d"
HF_MIRROR_REPO = "RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx"
PROFILE_CONFIGS = {
    "fp32": {
        "id": "higgs-audio-2-tsukuyomichan-omnivoice-full-finetune-fp32",
        "displayName": "Higgs Audio 2 Tsukuyomichan OmniVoice Full Finetune ONNX FP32",
        "qualityProfile": "fp32-unquantized",
        "releaseTag": "full-finetune-latest",
        "bits": None,
    },
    "fp16": {
        "id": "higgs-audio-2-tsukuyomichan-omnivoice-full-finetune-llm-fp16",
        "displayName": "Higgs Audio 2 Tsukuyomichan OmniVoice LLM FP16",
        "qualityProfile": "fp16-llm",
        "releaseTag": "fp16-latest",
        "bits": None,
    },
    "mobile-int8": {
        "id": "higgs-audio-2-tsukuyomichan-omnivoice-full-finetune-mobile-int8",
        "displayName": "Higgs Audio 2 Tsukuyomichan OmniVoice Mobile INT8",
        "qualityProfile": "mobile-int8-weight-only",
        "releaseTag": "mobile-int8-latest",
        "bits": 8,
    },
    "mobile-int4": {
        "id": "higgs-audio-2-tsukuyomichan-omnivoice-full-finetune-mobile-int4",
        "displayName": "Higgs Audio 2 Tsukuyomichan OmniVoice Mobile INT4",
        "qualityProfile": "mobile-int4-weight-only",
        "releaseTag": "mobile-int4-latest",
        "bits": 4,
    },
}
PROFILES = tuple(PROFILE_CONFIGS)
SOURCE_RUNTIME_RELEASE_FILES = {
    "config.json": "omnivoice_config.json",
    "tokenizer.json": "tokenizer.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "chat_template.jinja": "chat_template.jinja",
}


def build_distribution(hf_revision: str, profile: str = "fp32") -> dict:
    if not hf_revision or hf_revision in {"main", "master"}:
        raise RuntimeError("Hugging Face mirror revision must be an immutable build-specific revision")
    if profile not in PROFILES:
        raise RuntimeError(f"Unsupported release profile: {profile}")
    github_tag = PROFILE_CONFIGS[profile]["releaseTag"]
    return {
        "provider": "github-release",
        "repo": "DaisukeDaisuke/tsukuyomichan-omnivoice-onnx",
        "tag": github_tag,
        "assetBaseUrl": f"https://github.com/DaisukeDaisuke/tsukuyomichan-omnivoice-onnx/releases/download/{github_tag}/",
        "mirrors": [
            {
                "provider": "huggingface",
                "repo": HF_MIRROR_REPO,
                "revision": hf_revision,
                "assetBaseUrl": f"https://huggingface.co/{HF_MIRROR_REPO}/resolve/{hf_revision}/",
            }
        ],
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def xxh3_128_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = xxhash.xxh3_128()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def external_locations(model_path: Path) -> list[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    locations: set[str] = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        metadata = {entry.key: entry.value for entry in tensor.external_data}
        location = metadata.get("location")
        if not location:
            raise RuntimeError(f"{model_path.name}: external tensor {tensor.name} has no location")
        if Path(location).is_absolute() or ".." in Path(location).parts:
            raise RuntimeError(f"{model_path.name}: unsafe external-data location: {location}")
        locations.add(location)
    return sorted(locations)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def release_asset(path: Path, role: str) -> dict:
    return {
        "id": path.name,
        "role": role,
        "localPath": path.name,
        "byteSize": path.stat().st_size,
        "sha256": sha256_file(path),
        "xxh3_128": xxh3_128_file(path),
    }


def check_release_files(release: Path) -> None:
    for required in (*REQUIRED_MODELS, *REQUIRED_FILES):
        if not (release / required).is_file():
            raise RuntimeError(f"Required release file is missing: {required}")

    safetensors = list(release.rglob("*.safetensors"))
    if safetensors:
        raise RuntimeError(f"Safetensors build inputs must never be released: {safetensors}")

    for path in release.iterdir():
        if not path.is_file():
            raise RuntimeError(f"Release directory must be flat: {path}")
        size = path.stat().st_size
        if size > MAX_RELEASE_ASSET_BYTES:
            raise RuntimeError(f"Release asset exceeds the conservative 1.8 GB limit: {path.name} ({size})")
        if size == SOURCE_MODEL_SIZE and sha256_file(path) == SOURCE_MODEL_SHA256:
            raise RuntimeError(f"Original 2.45 GB source checkpoint leaked into release as {path.name}")
        if size == HIGGS_MODEL_SIZE and sha256_file(path) == HIGGS_MODEL_SHA256:
            raise RuntimeError(f"Original Higgs source checkpoint leaked into release as {path.name}")

    for model_name in REQUIRED_MODELS:
        for location in external_locations(release / model_name):
            target = release / location
            if not target.is_file():
                raise RuntimeError(f"{model_name} references missing external data: {location}")


def verify_source_runtime_files(release: Path, work: Path) -> None:
    full_source = load_json(work / "full-source.json")
    expected_files = full_source.get("runtime_files")
    if not isinstance(expected_files, dict):
        raise RuntimeError("full-source.json is missing pinned runtime_files integrity metadata")
    for source_name, release_name in SOURCE_RUNTIME_RELEASE_FILES.items():
        metadata = expected_files.get(source_name)
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Missing integrity metadata for pinned runtime file: {source_name}")
        path = release / release_name
        if not path.is_file():
            raise RuntimeError(f"Pinned runtime file is missing from release: {release_name}")
        expected_size = int(metadata["byte_size"])
        expected_sha = str(metadata["sha256"])
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != expected_size or actual_sha != expected_sha:
            raise RuntimeError(
                f"Pinned runtime file was modified after download: {release_name} "
                f"(size {actual_size} != {expected_size}, sha256 {actual_sha} != {expected_sha})"
            )


def resolve_llm_runtime_dimensions(config: dict) -> tuple[int, int, int]:
    llm_config = config.get("llm_config", {})
    hidden_size = int(llm_config.get("hidden_size", 1024))
    attention_heads = int(llm_config.get("num_attention_heads", 16))
    kv_heads = int(llm_config.get("num_key_value_heads", 8))
    head_dim = int(llm_config.get("head_dim", hidden_size // attention_heads))
    if hidden_size <= 0 or attention_heads <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise RuntimeError("Invalid LLM runtime dimensions in OmniVoice config")
    return hidden_size, kv_heads, head_dim


def build_manifest(release: Path, work: Path, hf_revision: str, profile: str = "fp32") -> dict:
    if profile not in PROFILES:
        raise RuntimeError(f"Unsupported release profile: {profile}")
    full_source = load_json(work / "full-source.json")
    higgs_source = load_json(work / "higgs-source.json")
    config = load_json(release / "omnivoice_config.json")
    hidden_size, kv_heads, head_dim = resolve_llm_runtime_dimensions(config)
    profile_config = PROFILE_CONFIGS[profile]
    mobile_quantization = None
    fp16_conversion = None
    expected_bits = profile_config["bits"]
    if expected_bits is not None:
        quantization_path = work / f"{profile}-quantization.json"
        if not quantization_path.is_file():
            raise RuntimeError(f"{profile} profile is missing {profile}-quantization.json")
        mobile_quantization = load_json(quantization_path)
        if int(mobile_quantization.get("bits", 0)) != expected_bits:
            raise RuntimeError(f"{profile} profile did not record a {expected_bits}-bit LLM")
    if profile == "fp16":
        conversion_path = work / "fp16-conversion.json"
        if not conversion_path.is_file():
            raise RuntimeError("fp16 profile is missing fp16-conversion.json")
        fp16_conversion = load_json(conversion_path)
        if fp16_conversion.get("scope") != "llm-only":
            raise RuntimeError("fp16 profile must convert only the LLM")
        if fp16_conversion.get("weightDtype") != "float16" or fp16_conversion.get("ioDtype") != "float32":
            raise RuntimeError("fp16 profile must keep FP32 LLM I/O around FP16 MatMul weights")
        if fp16_conversion.get("computeDtype") != "fp16-constant-matmul-fp32-otherwise":
            raise RuntimeError("fp16 profile must record constant-MatMul-only FP16 compute")
        if fp16_conversion.get("fp16ComputeScope") != "constant-matmul-only":
            raise RuntimeError("fp16 profile must limit FP16 compute to constant MatMul nodes")

    sessions = {}
    session_specs = {
        "audioEmbeddings": "audio_embeddings_encoder.onnx",
        "llm": "llm_decoder.onnx",
        "audioHeads": "audio_heads_decoder.onnx",
        "higgsDecoder": "higgs_decoder.onnx",
    }
    runtime_paths: set[str] = set()
    for session_name, model_name in session_specs.items():
        runtime_paths.add(model_name)
        externals = []
        for location in external_locations(release / model_name):
            runtime_paths.add(location)
            externals.append({"path": location, "localPath": location})
        sessions[session_name] = {"model": model_name, "externalData": externals}

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "omnivoice_config.json",
        "higgs_config.json",
        "higgs_preprocessor_config.json",
    ):
        if (release / name).is_file():
            runtime_paths.add(name)

    assets = [release_asset(release / path, "runtime") for path in sorted(runtime_paths)]
    manifest = {
        "schemaVersion": 1,
        "id": profile_config["id"],
        "displayName": profile_config["displayName"],
        "qualityProfile": profile_config["qualityProfile"],
        "quantized": expected_bits is not None,
        "distribution": build_distribution(hf_revision, profile),
        "source": {
            "omnivoiceCode": {
                "repo": "k2-fsa/OmniVoice",
                "release": "0.2.1",
                "revision": "5ba967c4d5b0f08244ae856b033eea583d1e4517",
                "license": "Apache-2.0",
            },
            "voiceCheckpoint": {
                "repo": full_source["repo"],
                "revision": full_source["revision"],
                "modelSha256": full_source["model_sha256"],
                "modelByteSize": full_source["model_byte_size"],
                "redistributedOriginalSafetensors": False,
            },
            "audioCodec": {
                "repo": higgs_source["repo"],
                "revision": higgs_source["revision"],
                "modelSha256": higgs_source["model_sha256"],
                "modelByteSize": higgs_source["model_byte_size"],
                "redistributedOriginalSafetensors": False,
            },
        },
        "runtime": {
            "sampleRate": 24000,
            "tokenizerDirectory": ".",
            "hiddenSize": hidden_size,
            "numKvHeads": kv_heads,
            "headDim": head_dim,
            "llmAttention": {
                "mode": "omnivoice-noncausal",
                "maskDtype": "bool",
                "maskRank": 4,
                "useCache": False,
            },
            "decoderInputName": "codes",
            "decoderOutputName": "waveform_24k",
            "generation": {
                "num_audio_codebook": int(config.get("num_audio_codebook", 8)),
                "audio_mask_id": int(config.get("audio_mask_id", 1024)),
                "audio_vocab_size": int(config.get("audio_vocab_size", 1025)),
                "numStep": 16,
                "guidanceScale": 2.0,
                "tShift": 0.1,
                "layerPenalty": 5.0,
                "positionTemperature": 5.0,
                "classTemperature": 0.0,
                "denoise": True,
            },
            "sessions": sessions,
        },
        "licenses": {
            "converterCode": "MIT",
            "voiceCheckpoint": "See TSUKUYOMICHAN_MODEL_CARD.md and TSUKUYOMICHAN_MODEL_CARD_JA.md",
            "upstreamOmniVoiceModel": "See OMNIVOICE_MODEL_CARD.md; the pinned upstream model card states CC-BY-NC for the pre-trained model",
            "upstreamOmniVoiceCode": "Apache-2.0; see OMNIVOICE_CODE_LICENSE.txt",
            "audioCodec": "Boson Higgs Audio 2 Community License Agreement",
            "metaDependency": "Meta Llama 3 Community License Agreement",
            "notice": "NOTICE.txt",
        },
        "assets": assets,
    }
    if mobile_quantization is not None:
        manifest["quantization"] = {
            "scope": "llm-only",
            "operator": "com.microsoft::MatMulNBits",
            "weightBits": int(mobile_quantization["bits"]),
            "blockSize": int(mobile_quantization["blockSize"]),
            "accuracyLevel": int(mobile_quantization["accuracyLevel"]),
            "activationDtype": "float32",
            "audioEmbeddingsQuantized": False,
            "audioHeadsQuantized": False,
            "higgsDecoderQuantized": False,
            "equivalence": mobile_quantization["equivalence"],
        }
    if fp16_conversion is not None:
        manifest["precision"] = {
            "scope": "llm-only",
            "weightDtype": "float16",
            "computeDtype": "fp16-constant-matmul-fp32-otherwise",
            "ioDtype": "float32",
            "fp16ComputeScope": "constant-matmul-only",
            "audioEmbeddingsDtype": "float32",
            "audioHeadsDtype": "float32",
            "higgsDecoderDtype": "float32",
            "equivalence": fp16_conversion["equivalence"],
            "sourceBytes": int(fp16_conversion["sourceBytes"]),
            "convertedBytes": int(fp16_conversion["convertedBytes"]),
        }
    return manifest


def build_typed_voice_manifest(runtime_manifest: dict, hf_revision: str) -> dict:
    source = runtime_manifest["source"]["voiceCheckpoint"]
    assets = []
    for asset in runtime_manifest["assets"]:
        item = dict(asset)
        item["source"] = {
            "provider": "huggingface",
            "repo": HF_MIRROR_REPO,
            "revision": hf_revision,
            "path": asset["localPath"],
        }
        assets.append(item)
    return {
        "schemaVersion": 2,
        "id": runtime_manifest["id"],
        "displayName": runtime_manifest["displayName"],
        "preparable": True,
        "installable": True,
        "voice": {
            "engine": "omnivoice",
            "source": {
                "provider": "huggingface",
                "repo": source["repo"],
                "revision": source["revision"],
                "modelSha256": source["modelSha256"],
                "modelByteSize": source["modelByteSize"],
            },
        },
        "runtimeSource": {
            "provider": "huggingface",
            "repo": HF_MIRROR_REPO,
            "revision": hf_revision,
            "qualityProfile": runtime_manifest["qualityProfile"],
        },
        "conversion": {
            "sourceFormat": "safetensors",
            "targetProfile": runtime_manifest["qualityProfile"],
            "quantized": bool(runtime_manifest["quantized"]),
            "sourceRedistributed": False,
        },
        "licenses": {
            "projectCode": "Apache-2.0",
            "voiceModel": "other",
            "voiceModelNotice": "TSUKUYOMICHAN_MODEL_CARD.md",
            "upstreamOmniVoiceModel": "CC-BY-NC as stated by the pinned upstream model card",
            "audioCodec": "Boson Higgs Audio 2 Community License Agreement",
            "metaDependency": "Meta Llama 3 Community License Agreement",
        },
        "runtime": runtime_manifest["runtime"],
        "assets": assets,
    }


def write_notices(release: Path, manifest: dict) -> None:
    voice = manifest["source"]["voiceCheckpoint"]
    codec = manifest["source"]["audioCodec"]
    is_mobile = bool(manifest.get("quantized"))
    is_fp16 = manifest.get("qualityProfile") == "fp16-llm"
    mobile_bits = int(manifest.get("quantization", {}).get("weightBits", 0)) if is_mobile else None
    title = manifest["displayName"]
    notice = f"""{title}

This release contains converted runtime artifacts. The repository's MIT license applies to converter code only; it does not relicense model weights, converted model artifacts, datasets, voice data, or codec materials.

Tsukuyomichan / つくよみちゃん
本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データを使用しています。
■つくよみちゃんコーパス（CV.夢前黎）
https://tyc.rei-yumesaki.net/material/corpus/

Voice checkpoint: {voice['repo']} @ {voice['revision']}
Original checkpoint SHA-256: {voice['modelSha256']}
The original model.safetensors is a temporary build input and is not redistributed in this release.
See TSUKUYOMICHAN_MODEL_CARD.md / TSUKUYOMICHAN_MODEL_CARD_JA.md for the model's usage, output-audio, modification, and redistribution conditions.

Upstream OmniVoice
The pinned k2-fsa/OmniVoice model card states that OmniVoice code is Apache-2.0 while its pre-trained model is CC-BY-NC because of training-data constraints. This conversion does not claim that those upstream model terms are superseded or relaxed. See OMNIVOICE_MODEL_CARD.md and OMNIVOICE_CODE_LICENSE.txt.

Higgs Audio 2 codec source: {codec['repo']} @ {codec['revision']}
The original Higgs safetensors checkpoint is a temporary build input and is not redistributed in this release.
Boson Higgs Audio 2 is licensed under the Boson Community License, Copyright © Boson AI USA, Inc. All Rights Reserved.
Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
Built with Higgs Materials licensed from Boson AI USA, Inc.
Built with Meta Llama 3.

Copies of the Boson Higgs Audio 2 and Meta Llama 3 license agreements are included in this release.
"""
    (release / "NOTICE.txt").write_text(notice, encoding="utf-8")

    if is_mobile:
        profile_notes = f"""This is the **mobile {mobile_bits}-bit weight-only PoC conversion** of the same pinned Tsukuyomichan OmniVoice full-finetune checkpoint used by the FP32 quality baseline.

- The graph structure still comes from the validated FP32 exporter.
- Only constant `MatMul` weights in `llm_decoder.onnx` are converted to `com.microsoft::MatMulNBits` with {mobile_bits}-bit weights.
- LLM activations stay FP32 (`accuracy_level=1`).
- Audio embeddings, audio heads, and the Higgs decoder remain unquantized FP32.
- `llm_decoder.onnx` still preserves OmniVoice's rank-4 Boolean non-causal attention mask and does not use KV cache.
- The quantized LLM is compared against the FP32 PyTorch golden cases before publication; the measured error metrics are recorded in `runtime-manifest.json`.
"""
    elif is_fp16:
        profile_notes = """This is the **LLM-only FP16 MatMul runtime** of the same pinned Tsukuyomichan OmniVoice full-finetune checkpoint used by the FP32 quality baseline.

- The complete FP32 runtime is exported and verified first; no model-export stage is skipped for this profile.
- Only constant-weight `MatMul` nodes in `llm_decoder.onnx` are converted as a post-export step: FP32 activation -> FP16 MatMul/weight -> FP32 result.
- RMSNorm, attention/softmax, residual math, nonlinearities, and dynamic MatMul nodes remain FP32.
- The LLM keeps FP32 `inputs_embeds` / `hidden_states` I/O and the same rank-4 Boolean non-causal attention contract.
- Audio embeddings, audio heads, and the Higgs waveform decoder remain FP32.
- The FP16 LLM is compared against the FP32 PyTorch golden cases before publication; the measured error metrics and size reduction are recorded in `runtime-manifest.json`.
"""
    else:
        profile_notes = """This is an **unquantized FP32 quality-baseline conversion** of the pinned Tsukuyomichan OmniVoice full-finetune checkpoint for browser/runtime evaluation.

- No INT4, INT8, GPTQ, FP16, or BF16 conversion is permitted by this profile.
- Every ONNX graph is checked for quantization operators and reduced-precision weight initializers before release.
- PyTorch and ONNX Runtime outputs are numerically compared for the exported components before release.
- `llm_decoder.onnx` preserves OmniVoice's rank-4 Boolean non-causal attention mask and does not use KV cache; a 2-D causal/padding-mask contract is rejected.
"""
    notes = f"""# {title}

{profile_notes}

- Voice checkpoint: `{voice['repo']}` @ `{voice['revision']}`
- Higgs Audio 2 codec source: `{codec['repo']}` @ `{codec['revision']}`
- The original 2.45 GB `model.safetensors` and the original Higgs `model.safetensors` are runner-local build inputs only. They are not Actions artifacts, cache entries, or release assets.

## Tsukuyomichan credit

本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データを使用しています。

■つくよみちゃんコーパス（CV.夢前黎）
https://tyc.rei-yumesaki.net/material/corpus/

The converted voice artifacts remain subject to the source model / Tsukuyomichan Corpus conditions. See the included model cards and `NOTICE.txt`.

## Upstream OmniVoice terms

The pinned `k2-fsa/OmniVoice` model card states that its code is Apache-2.0 and its pre-trained model is CC-BY-NC due to training-data constraints. This conversion does not assert that the Tsukuyomichan full-finetune or this ONNX representation removes those upstream model conditions. The pinned upstream model card and code license are included in the release.

## Higgs Audio 2 / Meta Llama 3

Built with Higgs Materials licensed from Boson AI USA, Inc.

Built with Meta Llama 3.

The exact Boson and Meta license agreements and required notices are included with the release. The repository MIT license does not replace those terms.

## Integrity

Use `runtime-manifest.json` for per-runtime-asset SHA-256 values and `SHA256SUMS` for the complete release-file checksum list.
"""
    (release / "RELEASE_NOTES.md").write_text(notes, encoding="utf-8")


def write_hashes(release: Path) -> None:
    target = release / "SHA256SUMS"
    if target.exists():
        target.unlink()
    lines = []
    for path in sorted(release.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != target.name:
            lines.append(f"{sha256_file(path)}  {path.name}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--profile", choices=PROFILES, default="fp32")
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    release = Path(args.release_dir).resolve()

    check_release_files(release)
    verify_source_runtime_files(release, work)
    manifest = build_manifest(release, work, args.hf_revision, args.profile)
    (release / "runtime-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    typed_voice_manifest = build_typed_voice_manifest(manifest, args.hf_revision)
    (release / "typed-voice-manifest.json").write_text(
        json.dumps(typed_voice_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_notices(release, manifest)
    check_release_files(release)
    write_hashes(release)
    check_release_files(release)
    print(f"Release finalized with {len(list(release.iterdir()))} files; original safetensors checkpoints are absent.")


if __name__ == "__main__":
    main()