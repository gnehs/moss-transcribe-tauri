use serde::{Deserialize, Serialize};

pub const MODEL_ID: &str = "moss-transcribe-diarize";
pub const MODEL_REPO: &str = "OpenMOSS-Team/MOSS-Transcribe-Diarize";
pub const MODEL_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
    "model-00000-of-00001.safetensors",
];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub id: String,
    pub title: String,
    pub repo: String,
    pub size_hint: String,
    pub installed: bool,
    pub path: String,
    pub bytes_on_disk: u64,
    pub files: Vec<String>,
    pub missing_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub model_id: String,
    pub state: String,
    pub current_file: Option<String>,
    pub file_index: usize,
    pub total_files: usize,
    pub file_bytes_completed: u64,
    pub file_total_bytes: u64,
    pub speed_bytes_per_sec: f64,
    pub percent: f64,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FfmpegStatus {
    pub available: bool,
    pub version: Option<String>,
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeInfo {
    pub platform: String,
    pub architecture: String,
    pub mlx_available: bool,
    pub metal_device: Option<String>,
    pub app_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStage {
    Queued,
    Preparing,
    Encoding,
    Prefilling,
    Generating,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressEvent {
    pub task_id: String,
    pub stage: TaskStage,
    pub percent: f64,
    pub message: String,
    pub elapsed_ms: u64,
    pub audio_duration_ms: Option<u64>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub estimated_generated_tokens: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscribeOptions {
    pub prompt: Option<String>,
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub convert_to_traditional: bool,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            prompt: None,
            max_new_tokens: None,
            convert_to_traditional: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptSegment {
    pub start: f32,
    pub end: f32,
    pub speaker: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptResult {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    #[serde(default)]
    pub truncated: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportOptions {
    pub output_dir: Option<String>,
    pub write_txt: bool,
    pub write_json: bool,
    pub write_srt: bool,
}

#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct OutputPaths {
    pub txt_path: Option<String>,
    pub json_path: Option<String>,
    pub srt_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscribeFileRequest {
    pub task_id: String,
    pub audio_path: String,
    pub options: TranscribeOptions,
    pub export: ExportOptions,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionResponse {
    pub audio_path: String,
    pub audio_duration_ms: u64,
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub truncated: bool,
    pub outputs: OutputPaths,
}
