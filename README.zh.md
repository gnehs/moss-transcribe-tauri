# MOSS Transcribe Studio

[English](README.md) | 繁體中文

Apple Silicon macOS 專用的原生長音訊轉錄工具。App 透過 Rust、`mlx-rs`
與 Metal 執行 `OpenMOSS-Team/MOSS-Transcribe-Diarize`，支援時間戳、說話者
辨識，以及 TXT、JSON、SRT 匯出；正式執行不需要 Python 或 PyTorch。

## 從 Release 安裝

1. 前往 [最新版 GitHub Release](https://github.com/gnehs/moss-transcribe-tauri/releases/latest)。
2. 在 **Assets** 下載 `.dmg` 檔（不是原始碼的 ZIP 或 tar.gz）。
3. 開啟 DMG，將 MOSS Transcribe Studio 拖曳到「應用程式」。
4. 目前的 release 未簽章。若 macOS 阻擋首次開啟，請執行：

```sh
xattr -cr "/Applications/MOSS Transcribe Studio.app"
```

## 下載模型

模型不包含在 release DMG 中。首次轉錄前：

1. 開啟 App 的「設定」。
2. 在「模型」區塊選擇「下載模型」。
3. 等待 `MOSS-Transcribe-Diarize 0.9B` 下載完成（約 1.83 GB）。

若直接建立第一個轉錄任務，App 也會在偵測到缺少模型時開始下載。
App 會從
[`OpenMOSS-Team/MOSS-Transcribe-Diarize`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
的固定 revision 下載模型，並儲存在本機 App Support 目錄。下載過程需要
網路連線；完成後的轉錄與匯出都在本機處理。在 macOS 上的預設
模型位置為
`~/Library/Application Support/MOSS Transcribe Studio/models/moss-transcribe-diarize`。

## 系統需求

- Apple Silicon Mac，macOS 14 或更新版本
- FFmpeg（預設偵測 Homebrew、MacPorts、`PATH`）
- 網路連線（首次下載模型）

開發需求：

- Xcode Command Line Tools
- Rust toolchain
- pnpm

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

模型權重、PCM、embedding 與 logits 不會透過 Tauri IPC 傳到前端。

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
