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


def prepare_repo(repo_id: str, branch: str = "main") -> None:
    api = api_from_environment(repo_id)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    if branch != "main":
        api.create_branch(repo_id=repo_id, repo_type="model", branch=branch, exist_ok=True)
    print(f"Hugging Face mirror repository is writable: {repo_id} @ {branch}")


def build_model_card(
    release: Path,
    revision: str,
    github_sha: str,
    github_run_number: str,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    branch: str = "main",
) -> str:
    manifest = json.loads((release / "runtime-manifest.json").read_text(encoding="utf-8"))
    voice = manifest["source"]["voiceCheckpoint"]
    codec = manifest["source"]["audioCodec"]
    is_mobile = bool(manifest.get("quantized"))
    is_fp16 = manifest.get("qualityProfile") == "fp16-llm"
    mobile_bits = int(manifest.get("quantization", {}).get("weightBits", 0)) if is_mobile else None
    if is_mobile:
        title = f"Tsukuyomichan OmniVoice Mobile INT{mobile_bits}"
        profile_summary = f"This branch contains the **mobile {mobile_bits}-bit weight-only** runtime. Only the LLM's constant MatMul weights are quantized to `com.microsoft::MatMulNBits`; LLM activations, audio embeddings, audio heads, and the Higgs decoder remain FP32."
        conversion_summary = f"The LLM keeps the validated rank-4 Boolean non-causal attention contract and no KV cache, then applies {mobile_bits}-bit weight-only `MatMulNBits` quantization as a post-export step. The release records FP32-golden equivalence metrics in `runtime-manifest.json`."
        storage_summary = "The mobile profile is intended to reduce the LLM weight footprint substantially versus the FP32 baseline; use `runtime-manifest.json` for the exact build size."
    elif is_fp16:
        title = "Tsukuyomichan OmniVoice LLM FP16"
        profile_summary = "This branch contains the **LLM-only FP16 MatMul** runtime. Constant-weight LLM MatMul nodes use FP16 weights and FP16 activation/result boundaries, while RMSNorm, attention/softmax, residual math, nonlinearities, dynamic MatMuls, graph I/O, audio embeddings, audio heads, and the Higgs decoder remain FP32."
        conversion_summary = "The complete FP32 runtime is exported and verified first. Each constant-weight LLM MatMul is rewritten as FP32 activation -> FP16 MatMul/weight -> FP32 result, preserving the rank-4 Boolean non-causal attention contract and no KV cache. The resulting LLM is numerically compared against the FP32 PyTorch golden cases before publication."
        storage_summary = "The FP16 profile reduces the dominant LLM weight footprint without converting the lower-cost audio embeddings, audio heads, or Higgs waveform decoder. Use `runtime-manifest.json` for exact build sizes."
    else:
        title = "Tsukuyomichan OmniVoice Full Finetune ONNX FP32"
        profile_summary = "This branch contains the **unquantized FP32** ONNX quality-baseline runtime."
        conversion_summary = "The conversion workflow rejects INT4, INT8, GPTQ, FP16 and BF16 weights/operators and numerically compares exported components against their PyTorch outputs before publication. `llm_decoder` preserves OmniVoice's rank-4 Boolean non-causal attention mask and runs without KV cache; a 2-D causal/padding-mask LLM contract is rejected by the release gate."
        storage_summary = "The current FP32 runtime is about **2.4 GiB**, so the first preparation consumes more than 2 GB of network transfer and persistent local storage."
    def render_samples(specs) -> str:
        sample_blocks = []
        for filename, text in specs:
            sample_blocks.append(
            f"""Text: `{text}`

FP32:
<audio controls src="https://huggingface.co/{repo_id}/resolve/main/samples/{filename}"></audio>

LLM FP16:
<audio controls src="https://huggingface.co/{repo_id}/resolve/fp16/samples/{filename}"></audio>

Mobile INT8:
<audio controls src="https://huggingface.co/{repo_id}/resolve/mobile-int8/samples/{filename}"></audio>

Mobile INT4:
<audio controls src="https://huggingface.co/{repo_id}/resolve/mobile-int4/samples/{filename}"></audio>

Direct files: [FP32](https://huggingface.co/{repo_id}/resolve/main/samples/{filename}) / [LLM FP16](https://huggingface.co/{repo_id}/resolve/fp16/samples/{filename}) / [Mobile INT8](https://huggingface.co/{repo_id}/resolve/mobile-int8/samples/{filename}) / [Mobile INT4](https://huggingface.co/{repo_id}/resolve/mobile-int4/samples/{filename})"""
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

# {title}

| Distribution profile | Hugging Face | Quantization |
| --- | --- | --- |
| Mobile INT4 | [mobile-int4](https://huggingface.co/{repo_id}/tree/mobile-int4) | LLM constant MatMul weights: 4-bit; activations/audio/Higgs remain FP32 |
| Mobile INT8 | [mobile-int8](https://huggingface.co/{repo_id}/tree/mobile-int8) | LLM constant MatMul weights: 8-bit; activations/audio/Higgs remain FP32 |
| LLM FP16 | [fp16](https://huggingface.co/{repo_id}/tree/fp16) | Constant-weight LLM MatMul: FP16; all other LLM math + graph I/O/audio/Higgs: FP32 |
| FP32 baseline | [main](https://huggingface.co/{repo_id}/tree/main) | None |

{profile_summary}

This repository/branch is a browser/runtime mirror produced by `DaisukeDaisuke/tsukuyomichan-omnivoice-onnx`.

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
`higgs_decoder` ONNX graphs with external data. {conversion_summary}

# Download and try your computer!

You can run the browser PoC directly here:

https://daisukedaisuke.github.io/typed-voice/poc.html

The PoC downloads the converted runtime into persistent browser storage so it can be reused across
reloads. {storage_summary} Make sure your connection and device have
enough capacity before starting the download. Clearing this site's stored data removes the persistent
model cache when you want to reclaim the space.

The browser app uses a **Service Worker** for the offline model-asset path, persistent Cache Storage,
and cross-origin isolation. The FP32 validated fast path uses a **WebGPU + WebAssembly hybrid**:
audio embeddings, the LLM, and audio heads run on WebGPU, while the Higgs waveform decoder runs on
WebAssembly because this preserves the clean reference audio quality. The LLM FP16 profile keeps the
same FP32 graph I/O contract as the FP32 baseline, uses FP16 MatMul weights/compute on the ordinary
WebGPU path, and returns each constant MatMul result to FP32 immediately so the remaining LLM math
stays FP32. It does not use `MatMulNBits`. ONNX Runtime Web's JSEP `MatMulNBits` path does not support the Mobile INT8 LLM on
WebGPU, so Mobile INT8 keeps the LLM on WASM while still allowing the unquantized audio embeddings and
audio heads to use WebGPU. Browser testing also found reproducible token divergence when the Mobile
INT4 LLM alone runs on WebGPU; Mobile INT4 therefore uses the same LLM-on-WASM hybrid while its
embeddings and heads remain on WebGPU. Browsers without the required WebGPU path fall back to
WebAssembly where supported.

## Audio Samples

These WAV files are generated on the GitHub Actions CPU runner with native Python + ONNX Runtime from
the finalized `{manifest['qualityProfile']}` release artifacts. They are static sample files; the player below does not run
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

このモデルは、**つくよみちゃんというキャラクターの声をもとにした音声合成モデルです。**
モデルの変換形式や配布形態が変わっても、誰の声をもとにしているのか分からなくならないよう、つくよみちゃん由来であることを明記しています。

■つくよみちゃんコーパス（CV.夢前黎）
https://tyc.rei-yumesaki.net/material/corpus/

### 出力音声の利用制限

本モデルから出力した音声は、次の目的ではご利用いただけません。

- 人を批判・攻撃すること。（「批判・攻撃」の定義は、[**つくよみちゃんキャラクターライセンス**](https://tyc.rei-yumesaki.net/about/terms/#condition3) に準じます）
- 特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。
- 刺激の強い表現をゾーニングなしで公開すること。
- 他者に対して二次利用（素材としての利用）を許可する形で公開すること。

※鑑賞用の作品として配布・販売していただくことは問題ございません。

出力音声の利用条件の詳細は、つくよみちゃんコーパスの利用規約をご確認ください。
https://tyc.rei-yumesaki.net/material/corpus/

### 改変・再配布について

つくよみちゃんのモデルそのものを素材として使用する場合（改変、ファインチューニング、他モデルとのマージ、再配布などを行う場合）、つくよみちゃんコーパスに由来する部分の取り扱いについては、つくよみちゃんコーパスの利用規約に従ってください。この規定は、派生ソフトや再配布されたデータにもコピーレフトされます。

https://tyc.rei-yumesaki.net/material/corpus/

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
    branch: str = "main",
) -> str:
    if not revision or revision in {"main", "master"}:
        raise RuntimeError("Hugging Face mirror revision must be build-specific")

    assert_upload_surface(release)
    validate_samples(samples)
    readme = release / "README.md"
    if readme.exists():
        raise RuntimeError("README.md unexpectedly exists in the GitHub Release directory")
    readme.write_text(
        build_model_card(
            release,
            revision,
            github_sha,
            github_run_number,
            repo_id=repo_id,
            branch=branch,
        ),
        encoding="utf-8",
    )

    api = api_from_environment(repo_id)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        if branch != "main":
            api.create_branch(repo_id=repo_id, repo_type="model", branch=branch, exist_ok=True)
        profile = manifest_profile(release)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(release),
            revision=branch,
            commit_message=f"Mirror verified {profile} runtime from GitHub Actions build {github_run_number}",
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
            revision=branch,
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
        print(f"Hugging Face mirror published: {repo_id} branch={branch} tag={revision} ({oid})")
        return oid
    finally:
        readme.unlink(missing_ok=True)


def manifest_profile(release: Path) -> str:
    manifest = json.loads((release / "runtime-manifest.json").read_text(encoding="utf-8"))
    return str(manifest.get("qualityProfile", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    prepare_parser.add_argument("--branch", default="main")
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--release-dir", required=True)
    publish_parser.add_argument("--samples-dir", required=True)
    publish_parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    publish_parser.add_argument("--revision", required=True)
    publish_parser.add_argument("--github-sha", required=True)
    publish_parser.add_argument("--github-run-number", required=True)
    publish_parser.add_argument("--branch", default="main")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_repo(args.repo_id, args.branch)
        return
    publish(
        Path(args.release_dir).resolve(),
        Path(args.samples_dir).resolve(),
        args.repo_id,
        args.revision,
        args.github_sha,
        args.github_run_number,
        args.branch,
    )


if __name__ == "__main__":
    main()
