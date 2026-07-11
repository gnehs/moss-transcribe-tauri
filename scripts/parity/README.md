# MOSS parity fixture harness

This directory is a development-only compatibility harness. It proves that the
native MLX-Rust implementation follows the pinned Python `mlx-audio` reference;
it is not packaged with the Tauri application and never downloads a model.

## Pinned inputs

| Component | Revision |
| --- | --- |
| `OpenMOSS-Team/MOSS-Transcribe-Diarize` | `d7231bbae2587a4af278735eb765b318c4f64edd` |
| `Blaizzy/mlx-audio` | `64e8416c303fb3b3463dab8eb4ebd78c55a87c1a` |
| `mlx-rs` | `f4aa309c79b6be35255ca7d34157dfc10d9ed4c9` |

Create a Python environment outside the app dependency graph, install
`requirements.txt`, and point it at an already-downloaded snapshot and a local
audio file longer than 30 seconds:

```sh
python -m venv /tmp/moss-parity-venv
/tmp/moss-parity-venv/bin/pip install -r scripts/parity/requirements.txt
/tmp/moss-parity-venv/bin/python scripts/parity/generate_moss_reference.py \
  --model-dir /path/to/MOSS-Transcribe-Diarize \
  --audio /path/to/long-audio.wav \
  --output-dir tests/fixtures/parity/generated/long-audio
```

The generator deliberately requires the pinned MOSS adapter to expose
`parity_trace(...)` or `generate_with_trace(...)`. That narrow local-only trace
API must return these keys: `log_mel`, `whisper_encoder`, `vq_adaptor`,
`expanded_input_ids`, `fused_embeddings`, `first_token_logits`,
`greedy_token_ids` (the complete greedy decode through EOS or the 4096-token
limit), and `final_transcript`. The first 32 IDs are recorded separately as
fixed decode probes. A normal transcription API is insufficient because it
cannot prove intermediate parity. If the API is absent, the command exits
non-zero without creating a partial fixture.

`--output-dir` must not exist yet. The output contains `tensors.npz` (full
tensors) and `metadata.json` (schema, pins, shapes, hashes, and deterministic
numeric probes). Generated fixtures and audio are gitignored; do not commit
model weights or large binary audio.

## Generate a native Rust trace

The native trace generator is feature-gated and never becomes a Tauri command.
It requires the already-downloaded model and the same canonical mono 16 kHz
PCM16 WAV used for the reference fixture:

```sh
cargo run --manifest-path src-tauri/Cargo.toml \
  --features parity-trace --bin moss-parity-trace -- \
  --model-dir /path/to/MOSS-Transcribe-Diarize \
  --wav /path/to/long-audio.wav \
  --output /tmp/moss-native-trace.json
```

The output contains only shapes, canonical dtypes, hashes, fixed numeric probes,
tokens, and transcript text. Full intermediate tensors remain inside the native
process and are never sent over Tauri IPC. Missing inputs, an existing output
path, an invalid model manifest, audio of 30 seconds or less, or a complete
decode that contains fewer than the required 32-token probe window all exit
non-zero without a successful trace.

## Running Rust parity

The integration test is intentionally `#[ignore]`; ordinary `cargo test` must
not claim an ML parity result without the local model/audio fixture. After the
native trace hook writes a JSON trace with the same summary layout, run:

```sh
MOSS_PARITY_FIXTURE_DIR="$PWD/tests/fixtures/parity/generated/long-audio" \
MOSS_PARITY_MODEL_DIR=/path/to/MOSS-Transcribe-Diarize \
MOSS_PARITY_WAV=/path/to/long-audio.wav \
cargo test --manifest-path src-tauri/Cargo.toml \
  --features parity-trace --test moss_parity -- --ignored
```

The integration test generates the native trace in-process; it no longer
depends on a manually prepared `MOSS_PARITY_RUST_TRACE`. Missing paths, an
invalid schema, mismatched pins, or native generation failure remain hard
failures.

Integer/token values and the final transcript compare exactly. Float stages use
`abs(actual-reference) <= atol + rtol * abs(reference)`: log-Mel uses `1e-4`;
encoder, VQ adaptor, and fused embeddings use `2e-3`; first-token logits use
`2e-2` absolute and `2e-3` relative tolerance. The NPZ data remains the
authoritative full-tensor evidence; JSON probes make failures small and readable.
