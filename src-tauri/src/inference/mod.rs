mod mel;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx_runtime;
mod processor;

use std::path::Path;

use serde::Deserialize;

use crate::{
    error::{AppError, AppResult},
    models::{ProgressEvent, TaskStage, TranscribeOptions, TranscriptResult},
};

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use crate::transcript::parse_transcript;
#[cfg(all(feature = "parity-trace", target_os = "macos", target_arch = "aarch64"))]
pub use mlx_runtime::{generate_native_parity_trace, NativeParityTrace};
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use mlx_runtime::{MlxRuntime, RuntimeProgress};

pub use mel::{chunk_audio, WhisperLogMel};
pub use processor::MossProcessor;

#[derive(Debug, Deserialize)]
struct MossConfig {
    audio_token_id: u32,
    audio_merge_size: usize,
    adaptor_input_dim: usize,
    text_config: TextConfig,
    audio_config: AudioConfig,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
}

#[derive(Debug, Deserialize)]
struct AudioConfig {
    num_mel_bins: usize,
    d_model: usize,
    encoder_layers: usize,
    encoder_attention_heads: usize,
    max_source_positions: usize,
}

/// Native MOSS inference owner. The MLX tensors remain on the Rust side and are
/// never exposed through Tauri IPC.
#[derive(Debug)]
pub struct MossTranscriber {
    processor: MossProcessor,
    mel: WhisperLogMel,
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    runtime: MlxRuntime,
}

impl MossTranscriber {
    pub fn load(model_dir: &Path) -> AppResult<Self> {
        let config: MossConfig =
            serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)
                .map_err(|error| AppError::Model(format!("Invalid model config: {error}")))?;
        validate_config(&config)?;
        let processor = MossProcessor::load(model_dir)?;
        let weights = model_dir.join("model-00000-of-00001.safetensors");
        if !weights.is_file() {
            return Err(AppError::Model("Model weights are missing".into()));
        }
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        let runtime = MlxRuntime::load(model_dir)?;
        Ok(Self {
            processor,
            mel: WhisperLogMel::new(),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            runtime,
        })
    }

    pub fn transcribe(
        &mut self,
        pcm: &[f32],
        options: &TranscribeOptions,
        progress: impl Fn(ProgressEvent),
    ) -> AppResult<TranscriptResult> {
        if options.max_new_tokens == 0 || options.max_new_tokens > 131_072 {
            return Err(AppError::Transcription(
                "max_new_tokens must be between 1 and 131072".into(),
            ));
        }
        let chunks = chunk_audio(pcm)?;
        progress(ProgressEvent {
            task_id: String::new(),
            stage: TaskStage::Encoding,
            percent: 10.0,
            message: format!("Preparing {} Whisper chunk(s)", chunks.len()),
            elapsed_ms: 0,
            audio_duration_ms: Some((pcm.len() as f64 / 16_000.0 * 1000.0) as u64),
            prompt_tokens: 0,
            generated_tokens: 0,
        });
        let mel_chunks = chunks
            .iter()
            .map(|chunk| self.mel.extract_chunk(&chunk.pcm))
            .collect::<AppResult<Vec<_>>>()?;
        let audio_feature_lengths = chunks
            .iter()
            .map(|chunk| chunk.audio_token_length)
            .collect::<Vec<_>>();
        let audio_tokens = chunks.iter().map(|chunk| chunk.audio_token_length).sum();
        let input_ids = self
            .processor
            .expanded_input_ids(audio_tokens, options.prompt.as_deref())?;
        let prompt_tokens = input_ids.len();

        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        {
            let generated = self.runtime.generate(
                &input_ids,
                &mel_chunks,
                &audio_feature_lengths,
                options.max_new_tokens,
                |runtime_progress| {
                    let (stage, percent, message, generated_tokens) = match runtime_progress {
                        RuntimeProgress::AudioEncoded => (
                            TaskStage::Prefilling,
                            45.0,
                            "Whisper and VQAdaptor encoding complete".to_string(),
                            0,
                        ),
                        RuntimeProgress::Prefill { completed, total } => {
                            let ratio = completed as f64 / total.max(1) as f64;
                            (
                                TaskStage::Prefilling,
                                45.0 + ratio * 25.0,
                                format!("Prefilling prompt tokens {completed}/{total}"),
                                0,
                            )
                        }
                        RuntimeProgress::Token { generated } => (
                            TaskStage::Generating,
                            70.0 + generated as f64 / options.max_new_tokens as f64 * 29.0,
                            format!("Generated {generated} tokens"),
                            generated,
                        ),
                    };
                    progress(ProgressEvent {
                        task_id: String::new(),
                        stage,
                        percent: percent.min(99.0),
                        message,
                        elapsed_ms: 0,
                        audio_duration_ms: Some((pcm.len() as f64 / 16_000.0 * 1000.0) as u64),
                        prompt_tokens,
                        generated_tokens,
                    });
                },
            )?;
            let text = self.processor.decode(&generated)?.trim().to_string();
            let segments = parse_transcript(&text);
            if segments.is_empty() {
                return Err(AppError::Transcription(
                    "MOSS did not produce any valid [start][Sxx]text[end] segments".into(),
                ));
            }
            return Ok(TranscriptResult {
                text,
                segments,
                prompt_tokens,
                generated_tokens: generated.len(),
            });
        }

        #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (mel_chunks, audio_feature_lengths, input_ids, prompt_tokens);
            Err(AppError::Transcription(
                "This build does not include the Apple Silicon MLX runtime".into(),
            ))
        }
    }
}

fn validate_config(config: &MossConfig) -> AppResult<()> {
    let valid = config.audio_token_id == 151_671
        && config.audio_merge_size == 4
        && config.adaptor_input_dim == 4096
        && config.text_config.vocab_size == 151_936
        && config.text_config.hidden_size == 1024
        && config.text_config.num_hidden_layers == 28
        && config.text_config.num_attention_heads == 16
        && config.text_config.num_key_value_heads == 8
        && config.text_config.head_dim == 128
        && config.text_config.max_position_embeddings == 131_072
        && config.audio_config.num_mel_bins == 80
        && config.audio_config.d_model == 1024
        && config.audio_config.encoder_layers == 24
        && config.audio_config.encoder_attention_heads == 16
        && config.audio_config.max_source_positions == 1500;
    if valid {
        Ok(())
    } else {
        Err(AppError::Model(
            "The downloaded model configuration is not the supported MOSS 0.9B architecture".into(),
        ))
    }
}
