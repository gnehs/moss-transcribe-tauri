# MOSS Transcribe Studio

English | [繁體中文](README.zh.md)

MOSS Transcribe Studio is a native long-form audio transcription app for Apple
Silicon Macs. It runs `OpenMOSS-Team/MOSS-Transcribe-Diarize` with Rust,
`mlx-rs`, and Metal, supports timestamps and speaker diarization, and exports
TXT, JSON, and SRT files. Python and PyTorch are not required at runtime.

## Install from a release

1. Open the [latest GitHub release](https://github.com/gnehs/moss-transcribe-tauri/releases/latest).
2. Under **Assets**, download the `.dmg` file. Do not download the automatically
   generated source-code ZIP or tar.gz archives.
3. Open the DMG and drag MOSS Transcribe Studio into Applications.
4. Releases are currently unsigned. If macOS blocks the first launch, run:

```sh
xattr -cr "/Applications/MOSS Transcribe Studio.app"
```

## Download the model

The model is not bundled with the release DMG. Before your first transcription:

1. Open **Settings** in the app.
2. Find the **Model** section and select **Download Model**.
3. Wait for `MOSS-Transcribe-Diarize 0.9B` to finish downloading (approximately
   1.83 GB).

Creating your first transcription task also starts the download automatically if
the model is missing.

The app downloads a pinned revision of
[`OpenMOSS-Team/MOSS-Transcribe-Diarize`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
to the local Application Support directory. An internet connection is required
for this download. Afterward, transcription and export are processed locally.
On macOS, the default model location is
`~/Library/Application Support/MOSS Transcribe Studio/models/moss-transcribe-diarize`.

## Requirements

- An Apple Silicon Mac running macOS 14 or later
- FFmpeg (the app checks Homebrew, MacPorts, and `PATH` by default)
- An internet connection for the initial model download

The following are additionally required for development:

- Xcode Command Line Tools
- Rust toolchain
- pnpm

## Development

```sh
pnpm install
pnpm tauri:dev
```

Build the frontend for production:

```sh
pnpm build
```

Build the macOS app:

```sh
pnpm tauri build
```

Model weights, PCM data, embeddings, and logits are never sent to the frontend
over Tauri IPC.

## Inference and tests

The native inference implementation lives in `src-tauri/src/inference/`. It
includes Whisper log-Mel processing, a Whisper Medium encoder, 4x time merge,
VQAdaptor, audio embedding injection, a Qwen3 KV cache, 4,096-token chunked
prefill, and greedy decoding.

```sh
cd src-tauri
cargo test
```

End-to-end tests with the real model and Python `mlx-audio` parity tests are
opt-in development gates. They require the downloaded pinned model and the same
mono 16 kHz PCM16 WAV file longer than 30 seconds. See
[`scripts/parity/README.md`](scripts/parity/README.md) for instructions. Python
packages are only part of that development verification environment and are not
app dependencies.
