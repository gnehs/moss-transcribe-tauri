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
    sync::{Arc, Mutex, MutexGuard, OnceLock, TryLockError},
    time::Instant,
};

use error::{AppError, AppResult};
pub use inference::MossTranscriber;
#[cfg(feature = "parity-trace")]
pub use inference::{generate_native_parity_trace, NativeParityTrace};
use models::{
    FfmpegStatus, ModelStatus, RuntimeInfo, TaskStage, TranscribeFileRequest,
    TranscriptStreamEvent, TranscriptionResponse,
};
pub use models::{ProgressEvent, TranscribeOptions, TranscriptResult, TranscriptSegment};
use serde::Deserialize;
use tauri::{
    ipc::Channel, AppHandle, Emitter, State, WebviewUrl, WebviewWindow, WebviewWindowBuilder,
};

#[cfg(target_os = "macos")]
use tauri::{
    menu::{MenuBuilder, MenuItemBuilder, SubmenuBuilder},
    LogicalPosition, Manager, PhysicalPosition, TitleBarStyle, WindowEvent,
};

#[cfg(target_os = "macos")]
const ABOUT_WINDOW_LABEL: &str = "about";
#[cfg(target_os = "macos")]
const ABOUT_MENU_ID: &str = "about";
#[cfg(target_os = "macos")]
const SETTINGS_MENU_ID: &str = "settings";
#[cfg(target_os = "macos")]
const NEW_TASK_MENU_ID: &str = "new-task";
#[cfg(target_os = "macos")]
const GITHUB_MENU_ID: &str = "github";
const OPEN_SETTINGS_EVENT: &str = "open-settings";
const NEW_TASK_EVENT: &str = "new-task";
const OPEN_GITHUB_EVENT: &str = "open-github";

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NativeMenuText {
    about: String,
    settings: String,
    file: String,
    new_task: String,
    edit: String,
    window: String,
    help: String,
    github: String,
}

impl Default for NativeMenuText {
    fn default() -> Self {
        Self {
            about: "關於 MOSS Transcribe Studio".into(),
            settings: "設定".into(),
            file: "檔案".into(),
            new_task: "新增任務".into(),
            edit: "編輯".into(),
            window: "視窗".into(),
            help: "說明".into(),
            github: "GitHub".into(),
        }
    }
}

#[derive(Default)]
struct InferenceState {
    transcriber: Arc<Mutex<Option<MossTranscriber>>>,
    model_operation: Arc<Mutex<()>>,
}

struct NativeMenuState(Mutex<NativeMenuText>);

impl Default for NativeMenuState {
    fn default() -> Self {
        Self(Mutex::new(NativeMenuText::default()))
    }
}

struct InferenceMemorySession<'a> {
    transcriber: MutexGuard<'a, Option<MossTranscriber>>,
    cleaned: bool,
}

impl<'a> InferenceMemorySession<'a> {
    fn begin(state: &'a Mutex<Option<MossTranscriber>>) -> AppResult<Self> {
        let transcriber = state
            .lock()
            .map_err(|_| AppError::Transcription("Inference worker state is unavailable".into()))?;
        inference::begin_mlx_memory_session()?;
        Ok(Self {
            transcriber,
            cleaned: false,
        })
    }

    fn cleanup(&mut self, unload_model: bool) -> AppResult<()> {
        if unload_model {
            let model = self.transcriber.take();
            drop(model);
        }
        inference::cleanup_mlx_memory()?;
        self.cleaned = true;
        Ok(())
    }
}

impl Drop for InferenceMemorySession<'_> {
    fn drop(&mut self) {
        if !self.cleaned {
            let model = self.transcriber.take();
            drop(model);
            if let Err(error) = inference::cleanup_mlx_memory() {
                eprintln!("MLX fallback cleanup failed: {error}");
            }
        }
    }
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
fn set_native_menu_text(app: AppHandle, menu: NativeMenuText) -> Result<(), String> {
    if [
        &menu.about,
        &menu.settings,
        &menu.file,
        &menu.new_task,
        &menu.edit,
        &menu.window,
        &menu.help,
        &menu.github,
    ]
    .iter()
    .any(|text| text.trim().is_empty())
    {
        return Err("Native menu text cannot be empty".into());
    }

    #[cfg(target_os = "macos")]
    {
        set_application_menu(&app, &menu).map_err(|error| error.to_string())?;

        if let Some(about) = app.get_webview_window(ABOUT_WINDOW_LABEL) {
            about
                .set_title(&menu.about)
                .map_err(|error| error.to_string())?;
        }
    }

    let state = app.state::<NativeMenuState>();
    let mut current = state
        .0
        .lock()
        .map_err(|_| "Native menu text state is unavailable".to_string())?;
    *current = menu;
    Ok(())
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
    let operation = state.model_operation.clone();
    let transcriber = state.transcriber.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = operation
            .lock()
            .map_err(|_| AppError::Model("Model operation state is unavailable".into()))?;
        drop_loaded_model(&transcriber)?;
        downloader::download_model(app, true)
    })
    .await
    .map_err(|error| AppError::Download(error.to_string()))?
}

#[tauri::command]
async fn delete_model(state: State<'_, InferenceState>) -> AppResult<ModelStatus> {
    let operation = state.model_operation.clone();
    let transcriber = state.transcriber.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = operation
            .lock()
            .map_err(|_| AppError::Model("Model operation state is unavailable".into()))?;
        drop_loaded_model(&transcriber)?;
        paths::delete_model()
    })
    .await
    .map_err(|error| AppError::Model(error.to_string()))?
}

#[tauri::command]
async fn unload_model(state: State<'_, InferenceState>) -> AppResult<()> {
    let operation = state.model_operation.clone();
    let transcriber = state.transcriber.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = match operation.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(()),
            Err(TryLockError::Poisoned(_)) => {
                return Err(AppError::Model(
                    "Model operation state is unavailable".into(),
                ))
            }
        };
        drop_loaded_model(&transcriber)
    })
    .await
    .map_err(|error| AppError::Model(error.to_string()))?
}

#[tauri::command]
async fn transcribe_file(
    app: AppHandle,
    state: State<'_, InferenceState>,
    request: TranscribeFileRequest,
    on_stream: Channel<TranscriptStreamEvent>,
) -> AppResult<TranscriptionResponse> {
    validate_request(&request)?;
    let transcriber = state.inner().transcriber.clone();
    let operation = state.inner().model_operation.clone();
    tauri::async_runtime::spawn_blocking(move || {
        run_transcription(app, transcriber, operation, request, on_stream)
    })
    .await
    .map_err(|error| AppError::Transcription(error.to_string()))?
}

fn run_transcription(
    app: AppHandle,
    state: Arc<Mutex<Option<MossTranscriber>>>,
    operation: Arc<Mutex<()>>,
    request: TranscribeFileRequest,
    on_stream: Channel<TranscriptStreamEvent>,
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

    let operation_guard = match operation.lock() {
        Ok(guard) => guard,
        Err(_) => {
            let error = AppError::Transcription("Model operation state is unavailable".into());
            emit(TaskStage::Failed, 0.0, error.to_string(), None, 0, 0);
            return Err(error);
        }
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

        let mut session = InferenceMemorySession::begin(&state)?;
        let inference_result = (|| {
            if session.transcriber.is_none() {
                *session.transcriber = Some(MossTranscriber::load(&paths::model_dir()?)?);
            }
            if let Err(error) = inference::log_mlx_memory("after model load") {
                eprintln!("{error}");
            }
            let transcriber = session
                .transcriber
                .as_mut()
                .ok_or_else(|| AppError::Transcription("MOSS model could not be loaded".into()))?;
            let task_id = request.task_id.clone();
            let stream_task_id = request.task_id.clone();
            let traditional_converter = request
                .options
                .convert_to_traditional
                .then(conversion::TraditionalConverter::new)
                .transpose()?;
            let mut transcript = transcriber.transcribe_owned(
                decoded.pcm,
                &request.options,
                |mut progress| {
                    progress.task_id.clone_from(&task_id);
                    progress.elapsed_ms = started.elapsed().as_millis() as u64;
                    progress.audio_duration_ms = duration;
                    let _ = app.emit("transcription-progress", progress);
                },
                |mut partial| {
                    partial.task_id.clone_from(&stream_task_id);
                    if let Some(converter) = &traditional_converter {
                        converter.convert_stream(&mut partial);
                    }
                    let _ = on_stream.send(partial);
                },
            )?;
            if let Err(error) = inference::log_mlx_memory("after inference") {
                eprintln!("{error}");
            }

            if let Some(converter) = traditional_converter {
                converter.convert_result(&mut transcript);
            }

            let outputs = export::export_result(&audio_path, &request.export, &transcript)?;
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

        let unload_model = inference_result.is_err() || !request.keep_model_loaded;
        let cleanup_result = session.cleanup(unload_model);
        match (inference_result, cleanup_result) {
            (Ok(response), Ok(())) => Ok(response),
            (Ok(_), Err(cleanup_error)) => Err(cleanup_error),
            (Err(error), Ok(())) => Err(error),
            (Err(error), Err(cleanup_error)) => {
                eprintln!("MLX cleanup after task failure also failed: {cleanup_error}");
                Err(error)
            }
        }
    })();

    match &result {
        Ok(response) => emit(
            TaskStage::Completed,
            100.0,
            "Transcription complete".into(),
            Some(response.audio_duration_ms),
            response.prompt_tokens,
            response.generated_tokens,
        ),
        Err(error) => emit(TaskStage::Failed, 0.0, error.to_string(), None, 0, 0),
    }
    drop(operation_guard);
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

fn drop_loaded_model(state: &Mutex<Option<MossTranscriber>>) -> AppResult<()> {
    let mut transcriber = state
        .try_lock()
        .map_err(|_| AppError::Model("The model is busy with a transcription task".into()))?;
    let model = transcriber.take();
    drop(model);
    inference::cleanup_mlx_memory()
}

fn apple_chip_name() -> Option<String> {
    static CHIP_NAME: OnceLock<Option<String>> = OnceLock::new();
    CHIP_NAME.get_or_init(query_apple_chip_name).clone()
}

fn query_apple_chip_name() -> Option<String> {
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

#[cfg(target_os = "macos")]
fn centered_position(
    main: &WebviewWindow,
    about: &WebviewWindow,
) -> tauri::Result<PhysicalPosition<i32>> {
    let main_position = main.outer_position()?;
    let main_size = main.outer_size()?;
    let about_size = about.outer_size()?;

    let center_axis = |position: i32, main_length: u32, about_length: u32| {
        let centered = i64::from(position) + (i64::from(main_length) - i64::from(about_length)) / 2;
        centered.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    };

    Ok(PhysicalPosition::new(
        center_axis(main_position.x, main_size.width, about_size.width),
        center_axis(main_position.y, main_size.height, about_size.height),
    ))
}

#[cfg(target_os = "macos")]
fn show_about_window(app: &AppHandle) -> tauri::Result<()> {
    let Some(main) = app.get_webview_window("main") else {
        return Ok(());
    };

    let about = if let Some(about) = app.get_webview_window(ABOUT_WINDOW_LABEL) {
        about
    } else {
        let about_title = app
            .state::<NativeMenuState>()
            .0
            .lock()
            .map(|menu| menu.about.clone())
            .unwrap_or_else(|_| NativeMenuText::default().about);
        let about = WebviewWindowBuilder::new(
            app,
            ABOUT_WINDOW_LABEL,
            WebviewUrl::App("index.html?window=about".into()),
        )
        .title(about_title)
        .inner_size(540.0, 400.0)
        .resizable(false)
        .visible(false)
        .build()?;

        let about_for_close = about.clone();
        about.on_window_event(move |event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = about_for_close.hide();
            }
        });
        about
    };

    about.set_position(centered_position(&main, &about)?)?;
    about.show()?;
    about.set_focus()?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn set_application_menu(app: &AppHandle, menu_text: &NativeMenuText) -> tauri::Result<()> {
    let about_item = MenuItemBuilder::with_id(ABOUT_MENU_ID, &menu_text.about).build(app)?;
    let settings_item = MenuItemBuilder::with_id(SETTINGS_MENU_ID, &menu_text.settings)
        .accelerator("CmdOrCtrl+,")
        .build(app)?;
    let application_menu = SubmenuBuilder::new(app, "MOSS Transcribe Studio")
        .item(&about_item)
        .separator()
        .item(&settings_item)
        .separator()
        .services()
        .separator()
        .hide()
        .hide_others()
        .show_all()
        .separator()
        .quit()
        .build()?;
    let new_task_item = MenuItemBuilder::with_id(NEW_TASK_MENU_ID, &menu_text.new_task)
        .accelerator("CmdOrCtrl+N")
        .build(app)?;
    let file_menu = SubmenuBuilder::new(app, &menu_text.file)
        .item(&new_task_item)
        .separator()
        .close_window()
        .build()?;
    let edit_menu = SubmenuBuilder::new(app, &menu_text.edit)
        .undo()
        .redo()
        .separator()
        .cut()
        .copy()
        .paste()
        .select_all()
        .build()?;
    let window_menu = SubmenuBuilder::new(app, &menu_text.window)
        .minimize()
        .maximize()
        .fullscreen()
        .separator()
        .bring_all_to_front()
        .build()?;
    let help_menu = SubmenuBuilder::new(app, &menu_text.help)
        .text(GITHUB_MENU_ID, &menu_text.github)
        .build()?;
    app.set_menu(
        MenuBuilder::new(app)
            .item(&application_menu)
            .item(&file_menu)
            .item(&edit_menu)
            .item(&window_menu)
            .item(&help_menu)
            .build()?,
    )?;
    Ok(())
}

pub fn run() -> Result<(), tauri::Error> {
    tauri::Builder::default()
        .manage(InferenceState::default())
        .manage(NativeMenuState::default())
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

            #[cfg(target_os = "macos")]
            {
                set_application_menu(app.handle(), &NativeMenuText::default())?;

                app.on_menu_event(|app, event| {
                    let menu_id = event.id().0.as_str();
                    if menu_id == ABOUT_MENU_ID {
                        if let Err(error) = show_about_window(app) {
                            eprintln!("Failed to show About window: {error}");
                        }
                    } else if menu_id == SETTINGS_MENU_ID {
                        let _ = app.emit(OPEN_SETTINGS_EVENT, ());
                    } else if menu_id == NEW_TASK_MENU_ID {
                        let _ = app.emit(NEW_TASK_EVENT, ());
                    } else if menu_id == GITHUB_MENU_ID {
                        let _ = app.emit(OPEN_GITHUB_EVENT, ());
                    }
                });
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_model_status,
            get_ffmpeg_status,
            get_runtime_info,
            download_model,
            redownload_model,
            delete_model,
            unload_model,
            transcribe_file,
            set_native_menu_text,
        ])
        .run(tauri::generate_context!())
}
