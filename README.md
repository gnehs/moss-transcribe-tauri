# MOSS Transcribe Studio

Apple Silicon macOS 專用的原生長音訊轉錄工具。App 透過 Rust、`mlx-rs`
與 Metal 執行 `OpenMOSS-Team/MOSS-Transcribe-Diarize`，支援時間戳、說話者
辨識，以及 TXT、JSON、SRT 匯出；正式執行不需要 Python 或 PyTorch。

## 系統需求

- Apple Silicon Mac，macOS 14 或更新版本
- Xcode Command Line Tools
- Rust toolchain
- pnpm
- FFmpeg（預設偵測 Homebrew、MacPorts、`PATH`）

## 開發

```sh
pnpm install
pnpm tauri:dev
```

前端 production build：

```sh
pnpm build
```

建立 `.app`：

```sh
pnpm tauri build
```

模型會由 App 固定從
`OpenMOSS-Team/MOSS-Transcribe-Diarize` 的 pinned revision 下載到 App
Support 目錄；權重、PCM、embedding 與 logits 不會透過 Tauri IPC 傳到前端。

## 推論與測試

原生推論位於 `src-tauri/src/inference/`，包含 Whisper log-Mel、Whisper
Medium encoder、4x time merge、VQAdaptor、audio embedding injection、Qwen3
KV cache、4096-token chunked prefill 與 greedy decoding。

```sh
cd src-tauri
cargo test
```

真模型的端對端與 Python `mlx-audio` parity 測試是開發用的 opt-in gate，
需要已下載的 pinned 模型與同一段超過 30 秒的 mono 16 kHz PCM16 WAV；詳細
步驟見 [`scripts/parity/README.md`](scripts/parity/README.md)。Python 套件只存在
於該開發驗證環境，不屬於 App 依賴。
