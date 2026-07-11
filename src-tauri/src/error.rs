use serde::Serialize;
use thiserror::Error;

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug, Error, Serialize)]
#[serde(tag = "kind", content = "message")]
pub enum AppError {
    #[error("I/O error: {0}")]
    Io(String),
    #[error("Model error: {0}")]
    Model(String),
    #[error("Download error: {0}")]
    Download(String),
    #[error("Audio error: {0}")]
    Audio(String),
    #[error("Transcription error: {0}")]
    Transcription(String),
    #[error("Export error: {0}")]
    Export(String),
    #[error("Task was cancelled")]
    Cancelled,
}

impl From<std::io::Error> for AppError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(value: serde_json::Error) -> Self {
        Self::Io(value.to_string())
    }
}
