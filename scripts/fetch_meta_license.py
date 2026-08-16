#!/usr/bin/env python3
"""Fetch the exact Meta Llama 3 license required by the Higgs Audio 2 license."""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/meta-llama/llama3/main/LICENSE"
EXPECTED_SHA256 = "4fa551d4f938f68b8c1e6afa9d28befb70e3f33f75d0753248d530364aeea40f"
MAX_BYTES = 256 * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request = urllib.request.Request(URL, headers={"User-Agent": "tsukuyomichan-omnivoice-onnx-builder/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise RuntimeError("Meta Llama 3 license unexpectedly exceeds the bounded download size")
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Meta Llama 3 license SHA-256 mismatch: {digest} != {EXPECTED_SHA256}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f"Verified Meta Llama 3 license: {digest}")


if __name__ == "__main__":
    main()