#!/usr/bin/env python3
"""Generate the release manifest, notices, hashes, and hard release gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
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
SOURCE_RUNTIME_RELEASE_FILES = {
    "config.json": "omnivoice_config.json",
    "tokenizer.json": "tokenizer.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "chat_template.jinja": "chat_template.jinja",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
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


def build_manifest(release: Path, work: Path) -> dict:
    full_source = load_json(work / "full-source.json")
    higgs_source = load_json(work / "higgs-source.json")
    config = load_json(release / "omnivoice_config.json")
    hidden_size, kv_heads, head_dim = resolve_llm_runtime_dimensions(config)

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
    return {
        "schemaVersion": 1,
        "id": "higgs-audio-2-tsukuyomichan-omnivoice-full-finetune-fp32",
        "displayName": "Higgs Audio 2 Tsukuyomichan OmniVoice Full Finetune ONNX FP32",
        "qualityProfile": "fp32-unquantized",
        "quantized": False,
        "distribution": {
            "provider": "github-release",
            "repo": "DaisukeDaisuke/tsukuyomichan-omnivoice-onnx",
            "tag": "full-finetune-latest",
            "assetBaseUrl": "https://github.com/DaisukeDaisuke/tsukuyomichan-omnivoice-onnx/releases/download/full-finetune-latest/",
        },
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


def write_notices(release: Path, manifest: dict) -> None:
    voice = manifest["source"]["voiceCheckpoint"]
    codec = manifest["source"]["audioCodec"]
    notice = f"""Higgs Audio 2 Tsukuyomichan OmniVoice Full Finetune ONNX FP32

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

    notes = f"""# Higgs Audio 2 Tsukuyomichan OmniVoice ONNX FP32

This is an **unquantized FP32 quality-baseline conversion** of the pinned Tsukuyomichan OmniVoice full-finetune checkpoint for browser/runtime evaluation.

- Voice checkpoint: `{voice['repo']}` @ `{voice['revision']}`
- Higgs Audio 2 codec source: `{codec['repo']}` @ `{codec['revision']}`
- No INT4, INT8, GPTQ, FP16, or BF16 conversion is permitted by this workflow.
- Every ONNX graph is checked for quantization operators and reduced-precision weight initializers before release.
- PyTorch and ONNX Runtime outputs are numerically compared for the exported components before release.
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
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    release = Path(args.release_dir).resolve()

    check_release_files(release)
    verify_source_runtime_files(release, work)
    manifest = build_manifest(release, work)
    (release / "runtime-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_notices(release, manifest)
    check_release_files(release)
    write_hashes(release)
    check_release_files(release)
    print(f"Release finalized with {len(list(release.iterdir()))} files; original safetensors checkpoints are absent.")


if __name__ == "__main__":
    main()