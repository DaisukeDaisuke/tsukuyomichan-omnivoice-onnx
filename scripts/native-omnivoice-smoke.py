#!/usr/bin/env python3
"""Generate a native CPU OmniVoice sample from a finalized ONNX release directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_session(model: Path, threads: int):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])


def log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def gumbel_score(score: float, temperature: float, rng: random.Random) -> float:
    if not temperature > 0:
        return float(score)
    uniform = min(1 - 1e-10, max(0.0, rng.random()))
    noise = -math.log(-math.log(uniform + 1e-10) + 1e-10)
    return float(score) / temperature + noise


def text_weight(text: str) -> float:
    total = 0.0
    for char in text:
        code = ord(char)
        if code == 0x20 or code == 0x3000 or 0x2000 <= code <= 0x200A:
            total += 0.2
        elif 0x30 <= code <= 0x39:
            total += 3.5
        elif 0x21 <= code <= 0x2F or 0x3A <= code <= 0x40 or 0x3000 <= code <= 0x303F:
            total += 0.5
        elif 0x300 <= code <= 0x36F or 0xFE20 <= code <= 0xFE2F:
            total += 0.0
        elif 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
            total += 2.2
        elif 0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or code >= 0x20000:
            total += 3.0
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            total += 2.5
        elif 0x590 <= code <= 0x5FF:
            total += 1.5
        elif 0x600 <= code <= 0x8FF or 0xFB50 <= code <= 0xFEFF:
            total += 1.5
        elif 0x900 <= code <= 0xDFF:
            total += 1.8
        elif 0xE00 <= code <= 0xEFF:
            total += 1.5
        elif 0x1000 <= code <= 0x109F or 0x1780 <= code <= 0x17FF:
            total += 1.8
        elif 0x1200 <= code <= 0x139F:
            total += 3.0
        elif 0xA000 <= code <= 0xA4CF:
            total += 3.0
        elif 0x41 <= code <= 0x7A or 0xC0 <= code <= 0x2AF:
            total += 1.0
        elif 0x370 <= code <= 0x3FF or 0x400 <= code <= 0x52F:
            total += 1.0
        elif 0x530 <= code <= 0x58F or 0x10A0 <= code <= 0x10FF:
            total += 1.0
        else:
            total += 1.0
    return total


def estimate_target_tokens(text: str, speed: float) -> int:
    if not speed > 0:
        raise ValueError("speed must be greater than zero")
    reference_weight = text_weight("Nice to meet you.")
    estimate = text_weight(text) / (reference_weight / 25.0)
    if estimate < 50:
        estimate = 50 * math.pow(estimate / 50, 1 / 3)
    return max(1, round(estimate / speed))


def write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def verify_release_assets(model: Path, manifest: dict) -> None:
    sums_path = model / "SHA256SUMS"
    if not sums_path.is_file():
        raise RuntimeError(f"Finalized release is missing {sums_path}")
    sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            sums[name.lstrip("*")] = digest.lower()

    print("verifying finalized release runtime assets...", flush=True)
    for asset in manifest.get("assets", []):
        name = str(asset["localPath"])
        path = model / name
        expected = str(asset["sha256"]).lower()
        expected_size = int(asset["byteSize"])
        if not path.is_file():
            raise RuntimeError(f"runtime asset is missing: {name}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"byte size mismatch: {name}: {actual_size} != {expected_size}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"manifest SHA256 mismatch: {name}: {actual} != {expected}")
        if sums.get(name) != actual:
            raise RuntimeError(f"SHA256SUMS mismatch: {name}: {sums.get(name)!r} != {actual}")
        print("  ok", name, actual, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True, help="Finalized release directory containing runtime-manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="ja", help="OmniVoice language tag, for example ja or en")
    parser.add_argument("--target-tokens", type=int, default=0, help="0 estimates duration from text and --speed")
    parser.add_argument("--speed", type=float, default=1.0, help="Relative speaking speed used by the duration estimator")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.threads < 1:
        raise ValueError("threads must be at least 1")
    if not args.speed > 0:
        raise ValueError("speed must be greater than zero")
    if not args.language or any(char in args.language for char in "<>|"):
        raise ValueError("language must be a plain non-empty OmniVoice language tag")

    model = args.model_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((model / "runtime-manifest.json").read_text(encoding="utf-8"))
    runtime = manifest["runtime"]
    attention_contract = runtime.get("llmAttention", {})
    if attention_contract != {
        "mode": "omnivoice-noncausal",
        "maskDtype": "bool",
        "maskRank": 4,
        "useCache": False,
    }:
        raise RuntimeError(f"unexpected LLM attention contract: {attention_contract}")
    verify_release_assets(model, manifest)

    config = runtime["generation"]
    codebooks = int(config["num_audio_codebook"])
    mask_id = int(config["audio_mask_id"])
    vocab = int(config["audio_vocab_size"])
    num_step = int(config["numStep"])
    guidance = float(config["guidanceScale"])
    t_shift = float(config["tShift"])
    layer_penalty = float(config["layerPenalty"])
    position_temperature = float(config["positionTemperature"])
    class_temperature = float(config["classTemperature"])
    if class_temperature != 0:
        raise RuntimeError(f"classTemperature={class_temperature} is unsupported by the reference greedy sampler")

    sample_rate = int(runtime["sampleRate"])
    target = args.target_tokens if args.target_tokens > 0 else estimate_target_tokens(args.text, args.speed)
    rng = random.Random(args.seed)

    print("Native Python / ONNX Runtime CPU sample", flush=True)
    print("onnxruntime", ort.__version__, "providers", ort.get_available_providers(), flush=True)
    print("text:", args.text, flush=True)
    print(
        "generation:",
        json.dumps(
            {
                "targetTokens": target,
                "speed": args.speed,
                "numStep": num_step,
                "guidanceScale": guidance,
                "tShift": t_shift,
                "layerPenalty": layer_penalty,
                "positionTemperature": position_temperature,
                "classTemperature": class_temperature,
                "seed": args.seed,
                "threads": args.threads,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    style = f"<|lang_start|>{args.language}<|lang_end|><|instruct_start|>None<|instruct_end|>"
    wrapped = f"<|text_start|>{args.text.strip()}<|text_end|>"
    style_ids = tokenizer.encode(style).ids
    text_ids = tokenizer.encode(wrapped).ids
    target_offset = len(style_ids) + len(text_ids)
    sequence = target_offset + target

    base = np.zeros((codebooks, sequence), dtype=np.int64)
    for codebook in range(codebooks):
        base[codebook, : len(style_ids)] = style_ids
        base[codebook, len(style_ids) : target_offset] = text_ids
        base[codebook, target_offset:] = mask_id

    batch_ids = np.full((2, codebooks, sequence), mask_id, dtype=np.int64)
    batch_ids[0] = base
    batch_ids[1, :, :target] = mask_id

    audio_mask = np.zeros((2, sequence), dtype=bool)
    audio_mask[0, target_offset:] = True
    audio_mask[1, :target] = True

    attention = np.zeros((2, 1, sequence, sequence), dtype=bool)
    attention[0, 0, :, :] = True
    attention[1, 0, :target, :target] = True
    diagonal = np.arange(target, sequence)
    attention[1, 0, diagonal, diagonal] = True

    sessions = runtime["sessions"]
    print("loading native ONNX Runtime CPU sessions...", flush=True)
    embeddings_session = make_session(model / sessions["audioEmbeddings"]["model"], args.threads)
    llm_session = make_session(model / sessions["llm"]["model"], args.threads)
    heads_session = make_session(model / sessions["audioHeads"]["model"], args.threads)
    decoder_session = make_session(model / sessions["higgsDecoder"]["model"], args.threads)

    llm_inputs = [(item.name, item.type, item.shape) for item in llm_session.get_inputs()]
    print("LLM inputs:", llm_inputs, flush=True)
    if (
        len(llm_inputs) != 2
        or llm_inputs[0][0] != "inputs_embeds"
        or llm_inputs[1][0] != "attention_mask"
        or "bool" not in llm_inputs[1][1]
        or len(llm_inputs[1][2]) != 4
    ):
        raise RuntimeError(f"unexpected corrected LLM contract: {llm_inputs}")

    def backbone(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        embeddings = embeddings_session.run(["inputs_embeds"], {"input_ids": ids, "audio_mask": mask})[0]
        hidden = llm_session.run(
            None,
            {
                "inputs_embeds": embeddings.astype(np.float32, copy=False),
                "attention_mask": attention,
            },
        )[0]
        logits = heads_session.run(
            ["logits"],
            {"hidden_states": hidden.astype(np.float32, copy=False)},
        )[0]
        return logits.astype(np.float32, copy=False)

    tokens = np.full((codebooks, target), mask_id, dtype=np.int64)
    schedule = []
    for index in range(num_step + 1):
        linear = index / num_step
        schedule.append((t_shift * linear) / (1 + (t_shift - 1) * linear))

    total = tokens.size
    remaining = total
    for step in range(num_step):
        scheduled = remaining if step == num_step - 1 else min(
            remaining,
            math.ceil(total * (schedule[step + 1] - schedule[step])),
        )
        logits = backbone(batch_ids, audio_mask)
        conditional = log_softmax(logits[0, :, target_offset : target_offset + target, :])
        unconditional = log_softmax(logits[1, :, :target, :])
        guided = log_softmax((1 + guidance) * conditional - guidance * unconditional)
        guided[:, :, mask_id] = -np.inf
        predictions = np.argmax(guided, axis=-1)
        scores = np.max(guided, axis=-1)
        scores = scores - np.arange(codebooks, dtype=np.float32)[:, None] * layer_penalty

        candidates = []
        for codebook in range(codebooks):
            for position in range(target):
                if tokens[codebook, position] == mask_id:
                    candidates.append(
                        (gumbel_score(scores[codebook, position], position_temperature, rng), codebook, position)
                    )
        candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = candidates[:scheduled]
        for _, codebook, position in chosen:
            tokens[codebook, position] = predictions[codebook, position]
        remaining -= len(chosen)
        batch_ids[0, :, target_offset : target_offset + target] = tokens
        batch_ids[1, :, :target] = tokens
        print(f"step {step + 1}/{num_step}: scheduled={scheduled} remaining={remaining}", flush=True)

    if np.any(tokens == mask_id):
        raise RuntimeError(f"masked tokens remain: {int(np.sum(tokens == mask_id))}")

    np.save(output / "codes.npy", tokens)
    waveform = decoder_session.run(
        [runtime.get("decoderOutputName", "waveform_24k")],
        {runtime.get("decoderInputName", "codes"): tokens[:, None, :]},
    )[0].astype(np.float32).reshape(-1)
    write_pcm16_wav(output / "python_native_raw.wav", waveform, sample_rate)

    start = 0
    end = len(waveform)
    threshold = 0.0005
    while start < end and abs(float(waveform[start])) < threshold:
        start += 1
    while end > start and abs(float(waveform[end - 1])) < threshold:
        end -= 1
    normalized = waveform[start:end].copy()
    peak = float(np.max(np.abs(normalized))) if len(normalized) else 0.0
    scale = min(0.95 / peak, 3.0) if peak > 0 else 1.0
    normalized *= scale
    listen_wav = output / "python_native_listen.wav"
    write_pcm16_wav(listen_wav, normalized, sample_rate)

    metadata = {
        "text": args.text,
        "language": args.language,
        "speed": args.speed,
        "modelDir": str(model),
        "attention": {
            "dtype": "bool",
            "shape": list(attention.shape),
            "mode": "omnivoice-noncausal",
            "useCache": False,
        },
        "generation": {
            "targetTokens": target,
            "numStep": num_step,
            "guidanceScale": guidance,
            "tShift": t_shift,
            "layerPenalty": layer_penalty,
            "positionTemperature": position_temperature,
            "classTemperature": class_temperature,
            "seed": args.seed,
        },
        "codes": {
            "min": int(tokens.min()),
            "max": int(tokens.max()),
            "unique": int(len(np.unique(tokens))),
        },
        "audio": {
            "sampleRate": sample_rate,
            "rawSamples": int(len(waveform)),
            "normalizedSamples": int(len(normalized)),
            "rawMin": float(waveform.min()),
            "rawMax": float(waveform.max()),
            "rawRms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
            "peakBeforeNormalize": peak,
            "normalizeScale": scale,
            "listenSha256": sha256(listen_wav),
        },
    }
    (output / "diagnostic.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print("WAV:", listen_wav, flush=True)


if __name__ == "__main__":
    main()
