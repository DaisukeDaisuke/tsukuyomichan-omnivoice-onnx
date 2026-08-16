#!/usr/bin/env python3
"""Download pinned build inputs into runner-local temporary storage only."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

FULL_REPO = "kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune"
FULL_REVISION = "c1d7ff9477d0b21f220c58070da63355f69607e9"
FULL_MODEL_SHA256 = "9ebaa8dd3bf35ceb6217cd19142bdabe6d6c044cca40672d2ae163d1a90ab47e"
FULL_MODEL_SIZE = 2_450_344_144

BASE_REPO = "k2-fsa/OmniVoice"
BASE_REVISION = "5337ba6bfe0ab30725fed141678a054fbedbf7da"
HIGGS_MODEL_SHA256 = "fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d"
HIGGS_MODEL_SIZE = 805_665_628

FULL_PATTERNS = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
    "README_ja.md",
    "train_config.json",
]
HIGGS_PATTERNS = [
    "README.md",
    "audio_tokenizer/model.safetensors",
    "audio_tokenizer/config.json",
    "audio_tokenizer/preprocessor_config.json",
    "audio_tokenizer/LICENSE",
    "audio_tokenizer/README.md",
]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_size: int, expected_sha256: str) -> None:
    size = path.stat().st_size
    if size != expected_size:
        raise RuntimeError(f"Unexpected size for {path}: {size} != {expected_size}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")


def download_full(work: Path, release: Path) -> None:
    full_dir = work / "full-finetune"
    snapshot_download(
        repo_id=FULL_REPO,
        revision=FULL_REVISION,
        allow_patterns=FULL_PATTERNS,
        local_dir=full_dir,
    )
    verify(full_dir / "model.safetensors", FULL_MODEL_SIZE, FULL_MODEL_SHA256)
    for source_name, release_name in (
        ("README.md", "TSUKUYOMICHAN_MODEL_CARD.md"),
        ("README_ja.md", "TSUKUYOMICHAN_MODEL_CARD_JA.md"),
    ):
        source = full_dir / source_name
        if source.exists():
            shutil.copy2(source, release / release_name)
    (work / "full-source.json").write_text(
        json.dumps(
            {
                "repo": FULL_REPO,
                "revision": FULL_REVISION,
                "model_sha256": FULL_MODEL_SHA256,
                "model_byte_size": FULL_MODEL_SIZE,
                "path": str(full_dir),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def download_higgs(work: Path, release: Path) -> None:
    base_dir = work / "base-omnivoice"
    snapshot_download(
        repo_id=BASE_REPO,
        revision=BASE_REVISION,
        allow_patterns=HIGGS_PATTERNS,
        local_dir=base_dir,
    )
    higgs_dir = base_dir / "audio_tokenizer"
    verify(higgs_dir / "model.safetensors", HIGGS_MODEL_SIZE, HIGGS_MODEL_SHA256)
    shutil.copy2(higgs_dir / "LICENSE", release / "BOSON-HIGGS-AUDIO-2-LICENSE.txt")
    shutil.copy2(base_dir / "README.md", release / "OMNIVOICE_MODEL_CARD.md")
    (work / "higgs-source.json").write_text(
        json.dumps(
            {
                "repo": BASE_REPO,
                "revision": BASE_REVISION,
                "model_sha256": HIGGS_MODEL_SHA256,
                "model_byte_size": HIGGS_MODEL_SIZE,
                "path": str(higgs_dir),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=["full", "higgs"])
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--release-dir", required=True)
    args = parser.parse_args()

    work = Path(args.work_dir).resolve()
    release = Path(args.release_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    release.mkdir(parents=True, exist_ok=True)
    if args.component == "full":
        download_full(work, release)
    else:
        download_higgs(work, release)


if __name__ == "__main__":
    main()
