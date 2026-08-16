#!/usr/bin/env python3
"""Mirror the already-verified Release directory to a Hugging Face model repo."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_HF_REPO_ID = "RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx"
BASE_MODEL = "kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune"
JAPANESE_SAMPLE_SPECS = (
    ("01_customs_tariff_rejection.wav", "税関関税許可局、関税許可を急遽却下"),
    ("02_webassembly_vibe_coding.wav", "WebAssemblyをLLMでVibe Coding中"),
    ("03_found_me_waiting.wav", "えへへ、見つけてくれたんだ！ずっとここで待ってたんだよ？"),
)
ENGLISH_SAMPLE_SPECS = (
    ("04_found_me_waiting_English.wav", "Hey, you finally made it! How does it feel, looking back at everything we've been through?"),
)
SAMPLE_SPECS = (*JAPANESE_SAMPLE_SPECS, *ENGLISH_SAMPLE_SPECS)


def api_from_environment(repo_id: str) -> HfApi:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required for the Hugging Face mirror upload")
    api = HfApi(token=token)
    identity = api.whoami()
    owner = repo_id.split("/", 1)[0]
    if identity.get("name") != owner:
        raise RuntimeError(f"HF_TOKEN belongs to {identity.get('name')!r}, expected {owner!r}")
    return api


def prepare_repo(repo_id: str) -> None:
    api = api_from_environment(repo_id)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    print(f"Hugging Face mirror repository is writable: {repo_id}")


def build_model_card(release: Path, revision: str, github_sha: str, github_run_number: str) -> str:
    manifest = json.loads((release / "runtime-manifest.json").read_text(encoding="utf-8"))
    voice = manifest["source"]["voiceCheckpoint"]
    codec = manifest["source"]["audioCodec"]
    def render_samples(specs) -> str:
        sample_blocks = []
        for filename, text in specs:
            sample_blocks.append(
            f"""Text: `{text}`

<audio controls src="https://huggingface.co/{DEFAULT_HF_REPO_ID}/resolve/main/samples/{filename}"></audio>

Direct file: [samples/{filename}](./samples/{filename})"""
            )
        return "\n\n".join(sample_blocks)

    japanese_samples = render_samples(JAPANESE_SAMPLE_SPECS)
    english_samples = render_samples(ENGLISH_SAMPLE_SPECS)
    return f"""---
base_model: {BASE_MODEL}
library_name: onnxruntime
pipeline_tag: text-to-speech
license: other
tags:
- onnx
- onnxruntime
- text-to-speech
- japanese
- tsukuyomichan
---

# Tsukuyomichan OmniVoice Full Finetune ONNX FP32

This repository is the browser/runtime mirror of the **unquantized FP32** ONNX conversion produced by
`DaisukeDaisuke/tsukuyomichan-omnivoice-onnx`.

The converter is open source and its source code, release workflow, verification gates, and conversion
scripts are available at:
https://github.com/DaisukeDaisuke/tsukuyomichan-omnivoice-onnx

The corrected OmniVoice non-causal attention conversion and the resulting native-Python PoC path were
also reviewed and validated with **ChatGPT GPT-5.6 Sol high**. This is an additional engineering review
signal, not a substitute for the repository's reproducible numerical checks, SHA-256 verification, or
runtime tests.

- Parent model: `{BASE_MODEL}`
- Voice revision: `{voice['revision']}`
- Original voice checkpoint SHA-256: `{voice['modelSha256']}`
- Original voice checkpoint redistributed here: **no**
- Higgs Audio 2 source revision: `{codec['revision']}`
- Original Higgs checkpoint redistributed here: **no**
- Mirror revision/tag: `{revision}`
- GitHub source commit: `{github_sha}`
- GitHub Actions build: `{github_run_number}`
- Quality profile: `{manifest['qualityProfile']}`

The runtime is split into `audio_embeddings_encoder`, `llm_decoder`, `audio_heads_decoder`, and
`higgs_decoder` ONNX graphs with external data. The conversion workflow rejects INT4, INT8, GPTQ,
FP16 and BF16 weights/operators and numerically compares exported components against their PyTorch
outputs before publication. `llm_decoder` preserves OmniVoice's rank-4 Boolean non-causal attention
mask and runs without KV cache; a 2-D causal/padding-mask LLM contract is rejected by the release gate.

## Audio Samples

These WAV files are generated on the GitHub Actions CPU runner with native Python + ONNX Runtime from
the finalized FP32 ONNX release artifacts. They are static sample files; the player below does not run
the model in the browser.

### Japanese

{japanese_samples}

### English

{english_samples}

## Integrity

Use `runtime-manifest.json` for runtime asset sizes, SHA-256 audit values, and XXH3-128 browser integrity
values, and `SHA256SUMS` for the runtime Release file set. Browser clients intentionally use XXH3-128 for
both first-download and reload validation to avoid multi-GiB client-side SHA-256 work, while SHA-256 is
retained for CI and offline release auditing. Generated audio samples have their own `samples/SAMPLES_SHA256SUMS`. The GitHub Release
remains the audit/release archive; this Hugging Face repository is the CORS/Range-friendly browser
distribution mirror of the same verified files.

## Tsukuyomichan credit

本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データを使用しています。

■つくよみちゃんコーパス（CV.夢前黎）
https://tyc.rei-yumesaki.net/material/corpus/

### 出力音声の利用制限

本モデルから出力した音声は、次の目的ではご利用いただけません。

- 人を批判・攻撃すること。（「批判・攻撃」の定義は、[**つくよみちゃんキャラクターライセンス**](https://tyc.rei-yumesaki.net/about/terms/#condition3) に準じます）
- 特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。
- 刺激の強い表現をゾーニングなしで公開すること。
- 他者に対して二次利用（素材としての利用）を許可する形で公開すること。

※鑑賞用の作品として配布・販売していただくことは問題ございません。

### 改変・再配布について

つくよみちゃんのモデルそのものを素材として使用する場合（改変、ファインチューニング、他モデルとのマージ、再配布などを行う場合）、つくよみちゃんコーパスに由来する部分の取り扱いについては、[**つくよみちゃんコーパスの利用規約**](https://tyc.rei-yumesaki.net/material/corpus/) に従ってください。この規定は、派生ソフトや再配布されたデータにもコピーレフトされます。

## License / upstream terms

The `license: other` metadata is intentional. This repository does not claim that conversion to ONNX
relicenses the voice checkpoint, Tsukuyomichan corpus-derived materials, upstream OmniVoice model,
Higgs Audio 2 materials, or Meta Llama 3 materials. Read `TSUKUYOMICHAN_MODEL_CARD.md`,
`TSUKUYOMICHAN_MODEL_CARD_JA.md`, `OMNIVOICE_MODEL_CARD.md`, `OMNIVOICE_CODE_LICENSE.txt`,
`BOSON-HIGGS-AUDIO-2-LICENSE.txt`, `META-LLAMA-3-LICENSE.txt`, and `NOTICE.txt` before use or
redistribution.
"""


def assert_upload_surface(release: Path) -> None:
    if not (release / "runtime-manifest.json").is_file() or not (release / "SHA256SUMS").is_file():
        raise RuntimeError("Release directory is not finalized")
    safetensors = list(release.rglob("*.safetensors"))
    if safetensors:
        raise RuntimeError(f"Refusing to upload safetensors build inputs to Hugging Face: {safetensors}")


def validate_samples(samples: Path) -> None:
    if not samples.is_dir():
        raise RuntimeError(f"Audio sample directory is missing: {samples}")
    expected = {name for name, _ in SAMPLE_SPECS}
    actual = {path.name for path in samples.iterdir() if path.is_file() and path.suffix.lower() == ".wav"}
    if actual != expected:
        raise RuntimeError(f"Audio sample set mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    for name in expected:
        path = samples / name
        if path.suffix.lower() != ".wav" or path.stat().st_size <= 44:
            raise RuntimeError(f"Invalid WAV sample: {path}")

    checksums = samples / "SAMPLES_SHA256SUMS"
    if not checksums.is_file():
        raise RuntimeError(f"Audio sample checksums are missing: {checksums}")
    entries = {}
    for line in checksums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            entries[name.lstrip("*")] = digest.lower()
    if set(entries) != expected:
        raise RuntimeError(f"Audio sample checksum set mismatch: expected {sorted(expected)}, got {sorted(entries)}")

    import hashlib

    for name in expected:
        digest = hashlib.sha256((samples / name).read_bytes()).hexdigest()
        if entries[name] != digest:
            raise RuntimeError(f"Audio sample SHA-256 mismatch: {name}")


def publish(
    release: Path,
    samples: Path,
    repo_id: str,
    revision: str,
    github_sha: str,
    github_run_number: str,
) -> str:
    if not revision or revision in {"main", "master"}:
        raise RuntimeError("Hugging Face mirror revision must be build-specific")

    assert_upload_surface(release)
    validate_samples(samples)
    readme = release / "README.md"
    if readme.exists():
        raise RuntimeError("README.md unexpectedly exists in the GitHub Release directory")
    readme.write_text(build_model_card(release, revision, github_sha, github_run_number), encoding="utf-8")

    api = api_from_environment(repo_id)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(release),
            commit_message=f"Mirror verified FP32 runtime from GitHub Actions build {github_run_number}",
            delete_patterns=["*"],
        )
        oid = getattr(commit, "oid", None)
        if not oid:
            raise RuntimeError("Hugging Face upload did not return a commit oid")
        sample_commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(samples),
            path_in_repo="samples",
            commit_message=f"Add native CPU audio samples from GitHub Actions build {github_run_number}",
        )
        oid = getattr(sample_commit, "oid", None)
        if not oid:
            raise RuntimeError("Hugging Face sample upload did not return a commit oid")
        api.create_tag(
            repo_id=repo_id,
            repo_type="model",
            tag=revision,
            revision=oid,
            exist_ok=True,
        )
        print(f"Hugging Face mirror published: {repo_id} @ {revision} ({oid})")
        return oid
    finally:
        readme.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--release-dir", required=True)
    publish_parser.add_argument("--samples-dir", required=True)
    publish_parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    publish_parser.add_argument("--revision", required=True)
    publish_parser.add_argument("--github-sha", required=True)
    publish_parser.add_argument("--github-run-number", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_repo(args.repo_id)
        return
    publish(
        Path(args.release_dir).resolve(),
        Path(args.samples_dir).resolve(),
        args.repo_id,
        args.revision,
        args.github_sha,
        args.github_run_number,
    )


if __name__ == "__main__":
    main()
