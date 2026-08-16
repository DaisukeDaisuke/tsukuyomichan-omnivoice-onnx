#!/usr/bin/env python3
"""Fetch the Apache-2.0 license from the exact OmniVoice 0.2.1 code release."""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

OMNIVOICE_CODE_COMMIT = "5ba967c4d5b0f08244ae856b033eea583d1e4517"
URL = f"https://raw.githubusercontent.com/k2-fsa/OmniVoice/{OMNIVOICE_CODE_COMMIT}/LICENSE"
MAX_BYTES = 64 * 1024
REQUIRED_MARKERS = (
    b"Apache License\n",
    b"Version 2.0, January 2004",
    b"END OF TERMS AND CONDITIONS",
    b"Copyright 2026 Xiaomi Corp.",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request = urllib.request.Request(URL, headers={"User-Agent": "tsukuyomichan-omnivoice-onnx-builder/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise RuntimeError("OmniVoice code license unexpectedly exceeds the bounded download size")
    missing = [marker.decode("ascii", errors="replace") for marker in REQUIRED_MARKERS if marker not in data]
    if missing:
        raise RuntimeError(f"Pinned OmniVoice code license is missing expected markers: {missing}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f"Fetched pinned OmniVoice code license @ {OMNIVOICE_CODE_COMMIT}: sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
