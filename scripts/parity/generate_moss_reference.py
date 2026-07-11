#!/usr/bin/env python3
"""Generate a local MOSS parity fixture with the pinned mlx-audio reference.

This is deliberately a developer tool: it never downloads a model and is not
part of the application dependency graph.  The pinned mlx-audio MOSS adapter
must expose a `parity_trace` (or `generate_with_trace`) method.  Requiring that
explicit trace API is intentional: silently replacing intermediates with a
normal `generate` result would make a parity fixture unverifiable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any, Mapping


MODEL_REVISION = "d7231bbae2587a4af278735eb765b318c4f64edd"
MLX_AUDIO_REVISION = "64e8416c303fb3b3463dab8eb4ebd78c55a87c1a"
MLX_RS_REVISION = "f4aa309c79b6be35255ca7d34157dfc10d9ed4c9"
SCHEMA_VERSION = 1
SAMPLE_RATE = 16_000
MIN_DURATION_SECONDS = 30.0
TRACE_KEYS = (
    "log_mel",
    "whisper_encoder",
    "vq_adaptor",
    "expanded_input_ids",
    "fused_embeddings",
    "first_token_logits",
    "greedy_token_ids",
    "final_transcript",
)
FLOAT_TOLERANCES = {
    "log_mel": {"atol": 1e-4, "rtol": 1e-4},
    "whisper_encoder": {"atol": 2e-3, "rtol": 2e-3},
    "vq_adaptor": {"atol": 2e-3, "rtol": 2e-3},
    "fused_embeddings": {"atol": 2e-3, "rtol": 2e-3},
    "first_token_logits": {"atol": 2e-2, "rtol": 2e-3},
}


def fail(message: str) -> None:
    raise RuntimeError(f"MOSS parity fixture was not generated: {message}")


def np_module() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        fail(f"numpy is not installed; install scripts/parity/requirements.txt: {error}")
    return np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="already-downloaded model snapshot at the pinned HF revision")
    parser.add_argument("--audio", type=Path, required=True,
                        help="local source audio; must be longer than 30 seconds")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="empty/new directory for metadata.json and tensors.npz")
    parser.add_argument("--prompt", default=None,
                        help="optional prompt; omit to use the model's default prompt")
    return parser.parse_args()


def read_audio(path: Path) -> Any:
    np = np_module()
    try:
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as error:
        fail(f"audio dependencies are not installed; install scripts/parity/requirements.txt: {error}")
    if not path.is_file():
        fail(f"audio is missing: {path}")
    pcm, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    pcm = np.mean(pcm, axis=1, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        divisor = np.gcd(sample_rate, SAMPLE_RATE)
        pcm = resample_poly(pcm, SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)
    if pcm.size / SAMPLE_RATE <= MIN_DURATION_SECONDS:
        fail(f"audio must be > {MIN_DURATION_SECONDS:g}s after 16 kHz conversion; got {pcm.size / SAMPLE_RATE:.3f}s")
    return np.ascontiguousarray(pcm)


def installed_mlx_audio_revision() -> str | None:
    """Return the commit from PEP 610 metadata when pip/uv recorded it."""
    try:
        dist = importlib.metadata.distribution("mlx-audio")
        for file in dist.files or ():
            if file.name == "direct_url.json":
                direct_url = dist.locate_file(file)
                data = json.loads(direct_url.read_text(encoding="utf-8"))
                return data.get("vcs_info", {}).get("commit_id")
    except importlib.metadata.PackageNotFoundError:
        fail("mlx-audio is not installed; create the dev environment from scripts/parity/requirements.txt")
    return None


def load_trace(model_dir: Path, pcm: np.ndarray, prompt: str | None) -> Mapping[str, Any]:
    try:
        from mlx_audio.stt.utils import load  # type: ignore[import-not-found]
    except ImportError as error:
        fail(f"could not import mlx_audio.stt.utils.load: {error}")

    model = load(str(model_dir))
    trace_method = getattr(model, "parity_trace", None) or getattr(model, "generate_with_trace", None)
    if trace_method is None:
        fail(
            "the pinned mlx-audio MOSS adapter does not expose `parity_trace` or "
            "`generate_with_trace`. Add the narrow, local tracing adapter documented "
            "in scripts/parity/README.md; do not replace this with plain generate()."
        )
    trace = trace_method(audio=pcm, sample_rate=SAMPLE_RATE, prompt=prompt, max_new_tokens=4096)
    if not isinstance(trace, Mapping):
        fail("trace API returned a non-mapping result")
    missing = [key for key in TRACE_KEYS if key not in trace]
    if missing:
        fail(f"trace API omitted required stages: {', '.join(missing)}")
    return trace


def to_numpy(value: Any, key: str) -> Any:
    np = np_module()
    if key == "final_transcript":
        return np.asarray(value, dtype=np.str_)
    if hasattr(value, "tolist") and not isinstance(value, np.ndarray):
        value = value.tolist()
    array = np.asarray(value)
    if key in {"expanded_input_ids", "greedy_token_ids"}:
        array = array.astype(np.int64, copy=False)
    else:
        array = array.astype(np.float32, copy=False)
    return np.ascontiguousarray(array)


def tensor_summary(array: Any, key: str) -> dict[str, Any]:
    raw = array.tobytes(order="C")
    summary: dict[str, Any] = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if key in FLOAT_TOLERANCES:
        flat = array.reshape(-1)
        # Fixed, evenly distributed probes keep the JSON small but diagnostic.
        probes = np.unique(np.linspace(0, flat.size - 1, num=min(64, flat.size), dtype=np.int64))
        summary["sample_indices"] = probes.tolist()
        summary["sample_values"] = flat[probes].astype(np.float64).tolist()
        summary["min"] = float(np.min(flat))
        summary["max"] = float(np.max(flat))
        summary["mean"] = float(np.mean(flat, dtype=np.float64))
    return summary


def main() -> int:
    args = parse_args()
    if not args.model_dir.is_dir():
        fail(f"model directory is missing: {args.model_dir}")
    if not (args.model_dir / "config.json").is_file():
        fail("model directory does not contain config.json")
    revision = installed_mlx_audio_revision()
    if revision != MLX_AUDIO_REVISION:
        fail(
            "mlx-audio revision is not pinned correctly; expected "
            f"{MLX_AUDIO_REVISION}, got {revision or 'unknown (install from requirements.txt)'}"
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        fail(f"output directory must not already exist: {output_dir}")
    pcm = read_audio(args.audio)
    trace = load_trace(args.model_dir, pcm, args.prompt)
    arrays = {key: to_numpy(trace[key], key) for key in TRACE_KEYS}
    if arrays["greedy_token_ids"].size < 32:
        fail("greedy_token_ids must contain the complete decode and at least 32 tokens")
    transcript = str(trace["final_transcript"])

    output_dir.mkdir(parents=True)
    np_module().savez_compressed(output_dir / "tensors.npz", **arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "model_repo": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "model_revision": MODEL_REVISION,
            "mlx_audio_revision": MLX_AUDIO_REVISION,
            "mlx_rs_revision": MLX_RS_REVISION,
        },
        "input": {
            "sample_rate": SAMPLE_RATE,
            "duration_seconds": pcm.size / SAMPLE_RATE,
            "source_basename": args.audio.name,
            "sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
            "prompt": args.prompt,
        },
        "tensors_file": "tensors.npz",
        "tensors": {key: tensor_summary(array, key) for key, array in arrays.items()},
        "comparison": {
            "float_tolerances": FLOAT_TOLERANCES,
            "exact": ["expanded_input_ids", "greedy_token_ids", "final_transcript"],
        },
        "decode": {
            "expanded_input_ids": arrays["expanded_input_ids"].reshape(-1).tolist(),
            "first_token_id": int(arrays["greedy_token_ids"].reshape(-1)[0]),
            "greedy_token_ids": arrays["greedy_token_ids"].reshape(-1).tolist(),
            "first_32_greedy_tokens": arrays["greedy_token_ids"].reshape(-1)[:32].tolist(),
            "final_transcript": transcript,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote parity fixture to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2)
