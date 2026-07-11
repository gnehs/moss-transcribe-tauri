use std::{fs, io::ErrorKind, path::PathBuf};

use crate::{
    error::{AppError, AppResult},
    models::{ModelStatus, MODEL_FILES, MODEL_ID, MODEL_REPO},
};

pub fn app_support_dir() -> AppResult<PathBuf> {
    dirs::data_dir()
        .map(|path| path.join("MOSS Transcribe Studio"))
        .ok_or_else(|| AppError::Io("Could not resolve the application support directory".into()))
}

pub fn model_dir() -> AppResult<PathBuf> {
    Ok(app_support_dir()?.join("models").join(MODEL_ID))
}

pub fn hf_cache_dir() -> AppResult<PathBuf> {
    Ok(app_support_dir()?.join("huggingface-cache"))
}

pub fn model_status() -> AppResult<ModelStatus> {
    let path = model_dir()?;
    let missing_files = MODEL_FILES
        .iter()
        .filter(|file| !path.join(file).is_file())
        .map(|file| (*file).to_string())
        .collect::<Vec<_>>();

    Ok(ModelStatus {
        id: MODEL_ID.into(),
        title: "MOSS-Transcribe-Diarize 0.9B".into(),
        repo: MODEL_REPO.into(),
        size_hint: "1.83 GB".into(),
        installed: missing_files.is_empty(),
        path: path.to_string_lossy().into_owned(),
        bytes_on_disk: directory_size(&path),
        files: MODEL_FILES.iter().map(|file| (*file).to_string()).collect(),
        missing_files,
    })
}

pub fn delete_model() -> AppResult<ModelStatus> {
    let path = model_dir()?;
    if path.exists() && !path.is_dir() {
        return Err(AppError::Model("The model path is not a directory".into()));
    }
    match fs::remove_dir_all(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    model_status()
}

fn directory_size(path: &std::path::Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            match entry.metadata() {
                Ok(metadata) if metadata.is_file() => metadata.len(),
                Ok(metadata) if metadata.is_dir() => directory_size(&path),
                _ => 0,
            }
        })
        .sum()
}
