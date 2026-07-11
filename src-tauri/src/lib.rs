mod audio;
mod conversion;
mod downloader;
mod error;
mod export;
pub mod inference;
pub mod models;
mod paths;
mod transcript;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("MOSS Transcribe Studio only supports Apple Silicon macOS");

use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
    time::Instant,
};

use error::{AppError, AppResult};
pub use inference::MossTranscriber;
#[cfg(feature = "parity-trace")]
pub use inference::{generate_native_parity_trace, NativeParityTrace};
use models::{
    FfmpegStatus, ModelStatus, RuntimeInfo, TaskStage, TranscribeFileRequest, TranscriptionResponse,
};
pub use models::{ProgressEvent, TranscribeOptions, TranscriptResult, TranscriptSegment};
use tauri::{AppHandle, Emitter, State, WebviewUrl, WebviewWindowBuilder};

#[cfg(target_os = "macos")]
use tauri::{LogicalPosition, TitleBarStyle};

#[derive(Default)]
struct InferenceState {
    transcriber: Arc<Mutex<Option<MossTranscriber>>>,
    model_operation: Arc<Mutex<()>>,
}

#[tauri::command]
fn get_model_status() -> AppResult<ModelStatus> {
    paths::model_status()
}

#[tauri::command]
fn get_ffmpeg_status() -> FfmpegStatus {
    audio::ffmpeg_status()
}

#[tauri::command]
fn get_runtime_info(app: AppHandle) -> RuntimeInfo {
    RuntimeInfo {
        platform: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        mlx_available: cfg!(all(
            target_os = "macos",
            target_arch = "aarch64",
            feature = "mlx"
        )),
        metal_device: apple_chip_name(),
        app_version: app.package_info().version.to_string(),
    }
}

#[tauri::command]
async fn download_model(
    app: AppHandle,
    state: State<'_, InferenceState>,
) -> AppResult<ModelStatus> {
    let operation = state.model_operation.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = operation
            .lock()
            .map_err(|_| AppError::Model("Model operation state is unavailable".into()))?;
        downloader::download_model(app, false)
    })
    .await
    .map_err(|error| AppError::Download(error.to_string()))?
}

#[tauri::command]
async fn redownload_model(
    app: AppHandle,
    state: State<'_, InferenceState>,
) -> AppResult<ModelStatus> {
    drop_loaded_model(&state)?;
    let operation = state.model_operation.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = operation
            .lock()
            .map_err(|_| AppError::Model("Model operation state is unavailable".into()))?;
        downloader::download_model(app, true)
    })
    .await
    .map_err(|error| AppError::Download(error.to_string()))?
}

#[tauri::command]
async fn delete_model(state: State<'_, InferenceState>) -> AppResult<ModelStatus> {
    drop_loaded_model(&state)?;
    let operation = state.model_operation.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = operation
            .lock()
            .map_err(|_| AppError::Model("Model operation state is unavailable".into()))?;
        paths::delete_model()
    })
    .await
    .map_err(|error| AppError::Model(error.to_string()))?
}

#[tauri::command]
async fn transcribe_file(
    app: AppHandle,
    state: State<'_, InferenceState>,
    request: TranscribeFileRequest,
) -> AppResult<TranscriptionResponse> {
    validate_request(&request)?;
    let state = state.inner().transcriber.clone();
    tauri::async_runtime::spawn_blocking(move || run_transcription(app, state, request))
        .await
        .map_err(|error| AppError::Transcription(error.to_string()))?
}

fn run_transcription(
    app: AppHandle,
    state: Arc<Mutex<Option<MossTranscriber>>>,
    request: TranscribeFileRequest,
) -> AppResult<TranscriptionResponse> {
    let started = Instant::now();
    let emit = |stage, percent, message: String, duration, prompt_tokens, generated_tokens| {
        let _ = app.emit(
            "transcription-progress",
            ProgressEvent {
                task_id: request.task_id.clone(),
                stage,
                percent,
                message,
                elapsed_ms: started.elapsed().as_millis() as u64,
                audio_duration_ms: duration,
                prompt_tokens,
                generated_tokens,
                estimated_generated_tokens: 0,
            },
        );
    };

    let result: AppResult<TranscriptionResponse> = (|| {
        emit(
            TaskStage::Preparing,
            0.5,
            "Decoding media".into(),
            None,
            0,
            0,
        );
        let audio_path = PathBuf::from(&request.audio_path);
        let decoded = audio::decode_to_pcm(&audio_path)?;
        let duration = Some(decoded.duration_ms);
        emit(
            TaskStage::Encoding,
            1.0,
            "Loading MOSS model".into(),
            duration,
            0,
            0,
        );

        let mut guard = state
            .lock()
            .map_err(|_| AppError::Transcription("Inference worker state is unavailable".into()))?;
        if guard.is_none() {
            *guard = Some(MossTranscriber::load(&paths::model_dir()?)?);
        }
        let transcriber = guard
            .as_mut()
            .ok_or_else(|| AppError::Transcription("MOSS model could not be loaded".into()))?;
        let task_id = request.task_id.clone();
        let mut transcript =
            transcriber.transcribe(&decoded.pcm, &request.options, |mut progress| {
                progress.task_id.clone_from(&task_id);
                progress.elapsed_ms = started.elapsed().as_millis() as u64;
                progress.audio_duration_ms = duration;
                let _ = app.emit("transcription-progress", progress);
            })?;

        if request.options.convert_to_traditional {
            conversion::simplified_to_traditional(&mut transcript)?;
        }

        let outputs = export::export_result(&audio_path, &request.export, &transcript)?;
        emit(
            TaskStage::Completed,
            100.0,
            "Transcription complete".into(),
            duration,
            transcript.prompt_tokens,
            transcript.generated_tokens,
        );
        Ok(TranscriptionResponse {
            audio_path: request.audio_path.clone(),
            audio_duration_ms: decoded.duration_ms,
            text: transcript.text,
            segments: transcript.segments,
            prompt_tokens: transcript.prompt_tokens,
            generated_tokens: transcript.generated_tokens,
            truncated: transcript.truncated,
            outputs,
        })
    })();

    if let Err(error) = &result {
        emit(TaskStage::Failed, 0.0, error.to_string(), None, 0, 0);
    }
    result
}

fn validate_request(request: &TranscribeFileRequest) -> AppResult<()> {
    if request.task_id.trim().is_empty() {
        return Err(AppError::Transcription("Task ID is required".into()));
    }
    if request
        .options
        .max_new_tokens
        .is_some_and(|limit| limit == 0 || limit > 131_072)
    {
        return Err(AppError::Transcription(
            "maxNewTokens must be between 1 and 131072 when provided".into(),
        ));
    }
    Ok(())
}

fn drop_loaded_model(state: &State<'_, InferenceState>) -> AppResult<()> {
    let mut transcriber = state
        .transcriber
        .try_lock()
        .map_err(|_| AppError::Model("The model is busy with a transcription task".into()))?;
    *transcriber = None;
    Ok(())
}

fn apple_chip_name() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()?;
        if output.status.success() {
            let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

pub fn run() -> Result<(), tauri::Error> {
    tauri::Builder::default()
        .manage(InferenceState::default())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let window_builder = WebviewWindowBuilder::new(app, "main", WebviewUrl::default())
                .title("MOSS Transcribe Studio")
                .inner_size(1180.0, 760.0)
                .min_inner_size(980.0, 680.0);

            #[cfg(target_os = "macos")]
            let window_builder = window_builder
                .title_bar_style(TitleBarStyle::Overlay)
                .hidden_title(true)
                .traffic_light_position(LogicalPosition::new(14.0, 25.0));

            window_builder.build()?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_model_status,
            get_ffmpeg_status,
            get_runtime_info,
            download_model,
            redownload_model,
            delete_model,
            transcribe_file,
        ])
        .run(tauri::generate_context!())
}
