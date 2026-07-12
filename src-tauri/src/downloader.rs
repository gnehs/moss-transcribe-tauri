use std::{fs, sync::Mutex, time::Instant};

use hf_hub::{
    progress::{DownloadEvent, FileStatus, ProgressEvent as HfProgressEvent, ProgressHandler},
    HFClient,
};
use tauri::{AppHandle, Emitter};

use crate::{
    error::{AppError, AppResult},
    models::{DownloadProgress, ModelStatus, MODEL_FILES, MODEL_ID, MODEL_REPO},
    paths,
};

pub const MODEL_REVISION: &str = "d7231bbae2587a4af278735eb765b318c4f64edd";

pub fn download_model(app: AppHandle, force: bool) -> AppResult<ModelStatus> {
    let model_dir = paths::model_dir()?;
    fs::create_dir_all(&model_dir)?;
    let cache_dir = paths::hf_cache_dir()?;
    fs::create_dir_all(&cache_dir)?;

    let client = HFClient::builder()
        .cache_dir(cache_dir)
        .user_agent(format!(
            "moss-transcribe-studio/{}",
            env!("CARGO_PKG_VERSION")
        ))
        .build_sync()
        .map_err(|error| AppError::Download(error.to_string()))?;
    let (owner, name) = MODEL_REPO
        .split_once('/')
        .ok_or_else(|| AppError::Model("Invalid Hugging Face repository".into()))?;
    let repo = client.model(owner, name);
    let total_files = MODEL_FILES.len();
    let file_sizes = MODEL_FILES
        .iter()
        .map(|file| {
            let path = model_dir.join(file);
            if path.is_file() && !force {
                fs::metadata(path)
                    .map(|metadata| metadata.len())
                    .map_err(AppError::from)
            } else {
                repo.get_file_metadata()
                    .filepath((*file).to_string())
                    .revision(MODEL_REVISION.to_string())
                    .send()
                    .map(|metadata| metadata.file_size)
                    .map_err(|error| AppError::Download(error.to_string()))
            }
        })
        .collect::<AppResult<Vec<_>>>()?;
    let total_bytes = file_sizes.iter().sum::<u64>();

    emit(
        &app,
        DownloadProgress {
            model_id: MODEL_ID.into(),
            state: "starting".into(),
            current_file: None,
            file_index: 0,
            total_files,
            file_bytes_completed: 0,
            file_total_bytes: total_bytes,
            speed_bytes_per_sec: 0.0,
            percent: 0.0,
            message: "Preparing model download".into(),
        },
    );

    for (index, file) in MODEL_FILES.iter().enumerate() {
        if model_dir.join(file).is_file() && !force {
            let file_size = file_sizes[index];
            let completed_bytes = file_sizes[..=index].iter().sum::<u64>();
            emit(
                &app,
                DownloadProgress {
                    model_id: MODEL_ID.into(),
                    state: "cached".into(),
                    current_file: Some((*file).into()),
                    file_index: index + 1,
                    total_files,
                    file_bytes_completed: file_size,
                    file_total_bytes: file_size,
                    speed_bytes_per_sec: 0.0,
                    percent: download_percent(completed_bytes, total_bytes),
                    message: format!("{file} is already available"),
                },
            );
            continue;
        }

        let handler = DownloadProgressHandler {
            app: app.clone(),
            file: (*file).into(),
            index,
            total_files,
            file_total_bytes: file_sizes[index],
            completed_bytes_before_file: file_sizes[..index].iter().sum(),
            total_bytes,
            last_sample: Mutex::new((Instant::now(), 0)),
        };
        repo.download_file()
            .filename((*file).to_string())
            .local_dir(model_dir.clone())
            .revision(MODEL_REVISION.to_string())
            .force_download(force)
            .progress(hf_hub::progress::Progress::new(handler))
            .send()
            .map_err(|error| AppError::Download(error.to_string()))?;
    }

    validate_download()?;
    emit(
        &app,
        DownloadProgress {
            model_id: MODEL_ID.into(),
            state: "complete".into(),
            current_file: None,
            file_index: total_files,
            total_files,
            file_bytes_completed: total_bytes,
            file_total_bytes: total_bytes,
            speed_bytes_per_sec: 0.0,
            percent: 100.0,
            message: "Model is ready".into(),
        },
    );
    paths::model_status()
}

fn validate_download() -> AppResult<()> {
    let status = paths::model_status()?;
    if !status.installed {
        return Err(AppError::Download(format!(
            "Model download is incomplete: {} file(s) missing",
            status.missing_files.len()
        )));
    }
    let model_dir = paths::model_dir()?;
    let _: serde_json::Value =
        serde_json::from_reader(fs::File::open(model_dir.join("config.json"))?)?;
    tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|error| AppError::Download(format!("Tokenizer validation failed: {error}")))?;
    let weights_file = fs::File::open(model_dir.join("model-00000-of-00001.safetensors"))?;
    // SAFETY: the mapping is read-only and remains borrowed by `SafeTensors`
    // only for the duration of this validation scope.
    let weights = unsafe { memmap2::MmapOptions::new().map(&weights_file) }
        .map_err(|error| AppError::Download(format!("Model mapping failed: {error}")))?;
    safetensors::SafeTensors::deserialize(&weights)
        .map_err(|error| AppError::Download(format!("Model validation failed: {error}")))?;
    Ok(())
}

struct DownloadProgressHandler {
    app: AppHandle,
    file: String,
    index: usize,
    total_files: usize,
    file_total_bytes: u64,
    completed_bytes_before_file: u64,
    total_bytes: u64,
    last_sample: Mutex<(Instant, u64)>,
}

impl ProgressHandler for DownloadProgressHandler {
    fn on_progress(&self, event: &HfProgressEvent) {
        let HfProgressEvent::Download(event) = event else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                self.emit(0, *total_bytes, 0.0, "downloading")
            }
            DownloadEvent::Progress { files } => {
                for file in files {
                    let mut last = match self.last_sample.lock() {
                        Ok(last) => last,
                        Err(_) => return,
                    };
                    let now = Instant::now();
                    let elapsed = now.duration_since(last.0).as_secs_f64();
                    let speed = if elapsed > 0.0 {
                        file.bytes_completed.saturating_sub(last.1) as f64 / elapsed
                    } else {
                        0.0
                    };
                    *last = (now, file.bytes_completed);
                    drop(last);
                    let state = if file.status == FileStatus::Complete {
                        "fileComplete"
                    } else {
                        "downloading"
                    };
                    self.emit(file.bytes_completed, file.total_bytes, speed, state);
                }
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
            } => self.emit(
                *bytes_completed,
                *total_bytes,
                bytes_per_sec.unwrap_or(0.0),
                "downloading",
            ),
            DownloadEvent::Complete => self.emit(
                self.file_total_bytes,
                self.file_total_bytes,
                0.0,
                "fileComplete",
            ),
        }
    }
}

impl DownloadProgressHandler {
    fn emit(&self, completed: u64, total: u64, speed: f64, state: &str) {
        let completed_bytes = self.completed_bytes_before_file.saturating_add(completed);
        let percent = download_percent(completed_bytes, self.total_bytes);
        emit(
            &self.app,
            DownloadProgress {
                model_id: MODEL_ID.into(),
                state: state.into(),
                current_file: Some(self.file.clone()),
                file_index: self.index + 1,
                total_files: self.total_files,
                file_bytes_completed: completed,
                file_total_bytes: total,
                speed_bytes_per_sec: speed,
                percent,
                message: format!("Downloading {}", self.file),
            },
        );
    }
}

fn download_percent(completed_bytes: u64, total_bytes: u64) -> f64 {
    if total_bytes == 0 {
        return 0.0;
    }
    (completed_bytes as f64 / total_bytes as f64 * 100.0).clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::download_percent;

    #[test]
    fn download_percent_uses_byte_weight() {
        assert_eq!(download_percent(1, 101), 100.0 / 101.0);
        assert_eq!(download_percent(101, 101), 100.0);
    }

    #[test]
    fn download_percent_handles_empty_total() {
        assert_eq!(download_percent(0, 0), 0.0);
    }
}

fn emit(app: &AppHandle, progress: DownloadProgress) {
    let _ = app.emit("model-download-progress", progress);
}
