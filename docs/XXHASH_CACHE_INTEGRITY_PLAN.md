# Browser cache integrity plan

## Goal

Reduce the cost of validating ~2.4 GiB of OmniVoice runtime assets on iPad/browser startup. Client-side SHA-256 is too expensive even on the first download, so browser validation uses XXH3-128 for both first download and reload.

## Integrity policy

- SHA-256 remains in CI/release generation and release auditing, but the browser does not calculate SHA-256 over model assets.
- XXH3-128 is the browser integrity hash for both first download and reload.
- The runtime manifest should carry both `sha256` and a separately named fast-cache digest such as `xxh3_128` for each runtime asset.
- Cache metadata records the immutable source revision, byte size, and verified XXH3-128 digest.
- Any XXH3-128 mismatch deletes both the cached object and its metadata and requires a fresh download + XXH3-128 verification.
- If the XXH3 implementation is unavailable or fails to initialize, fail closed rather than silently falling back to metadata-only trust or client-side SHA-256.

This deliberately weakens the browser-side cryptographic integrity boundary compared with SHA-256. The PoC accepts that trade-off to make iPad startup practical. Transport security and immutable Hugging Face revisions still prevent ordinary mutable-source drift, while CI/release artifacts continue to retain SHA-256 for offline auditing.

## Implementation constraints

- Do not write a new XXH3 implementation unless existing maintained implementations are unsuitable after source review.
- Prefer a small browser-compatible implementation that supports incremental/streaming hashing so multi-GiB assets are never materialized as one ArrayBuffer solely for hashing.
- Review implementation source, WASM loading behavior, CSP/cross-origin requirements, worker compatibility, Safari/iPad compatibility, package size, maintenance status, and license before adopting it.
- Avoid introducing a network fetch for the hashing runtime at application startup; any required JS/WASM must be bundled with the application and served same-origin.
- Do not run the existing incremental SHA-256 implementation over multi-GiB browser assets during prepare or initialization.

## PoC acceptance

- First download: XXH3-128 must be checked against the immutable manifest.
- Reload: XXH3-128 must detect deliberate byte corruption and trigger cache deletion.
- Reload validation must be materially faster than the existing JavaScript SHA-256 pass over the same asset set.
- The change must not make iPad/Safari initialization less reliable or materially increase peak memory.

## Source review result

Reviewed on 2026-08-16 without using the `Internet__` MCP:

- Upstream xxHash reference: `Cyan4973/xxHash` at `c0b5ea995d66691734b1a79ad89e73a0d2fd5a53`.
- Browser wrapper candidate: `Daninet/hash-wasm` at `373b796205ab55fb4a657374dad6ea589bf75815` / npm `hash-wasm@4.12.0`.
- A new WebAssembly port is **not** required. `hash-wasm` already provides `createXXHash128()` with incremental `update()` and `digest()` APIs backed by an XXH3-128 C implementation derived from Yann Collet's xxHash.
- The wrapper is MIT licensed. The embedded xxHash C implementation retains the upstream BSD-2-Clause notice.
- WASM is compiled with a 128 KiB maximum memory. The JavaScript bridge copies input through a 16 KiB window, so multi-GiB inputs are streamed rather than materialized in WASM memory.
- The published package embeds the WASM payload in JavaScript; no extra runtime network request is required. A Vite production bundle importing only `createXXHash128` measured 19,756 bytes uncompressed / 8,686 bytes gzip.
- Digests from `hash-wasm@4.12.0` `xxhash128` exactly matched Python `xxhash.xxh3_128` for single-shot and multi-chunk streaming vectors.
- On the Codespace benchmark, hashing 512 MiB in chunks measured about 5,139 MiB/s for `hash-wasm` XXH128 versus about 45.25 MiB/s for typed-voice's current pure-JavaScript incremental SHA-256, roughly 113x faster in that environment. This benchmark is not an iPad performance guarantee, but it confirms that the current SHA-256 implementation is the dominant reload-verification CPU cost.

Decision: use Python `xxhash.xxh3_128` in the converter to publish `xxh3_128`, and use `hash-wasm@4.12.0` `createXXHash128()` in typed-voice for both first-download and reload verification. SHA-256 remains a CI/release-audit field only and is not computed by the browser over runtime assets.
