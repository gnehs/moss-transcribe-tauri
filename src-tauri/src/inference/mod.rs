mod mel;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx_runtime;
mod processor;

use std::path::Path;

use serde::Deserialize;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use mlx_rs::{memory, Stream};

use crate::{
    error::{AppError, AppResult},
    models::{
        ProgressEvent, TaskStage, TranscribeOptions, TranscriptResult, TranscriptStreamEvent,
    },
};

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use crate::transcript::parse_transcript;
#[cfg(all(feature = "parity-trace", target_os = "macos", target_arch = "aarch64"))]
pub use mlx_runtime::{generate_native_parity_trace, NativeParityTrace};
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use mlx_runtime::{MlxRuntime, RuntimeProgress};

pub use mel::{chunk_audio, WhisperLogMel};
pub use processor::MossProcessor;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use processor::{EOS_TOKEN_ID, PAD_TOKEN_ID};

const LONG_AUDIO_MAX_NEW_TOKENS: usize = 65_536;
const ESTIMATED_OUTPUT_TOKEN_OVERHEAD: usize = 128;
const STREAM_DECODE_INTERVAL_TOKENS: usize = 8;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
const MLX_CACHE_LIMIT_BYTES: usize = 1024 * 1024 * 1024;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn begin_mlx_memory_session() -> AppResult<()> {
    let previous_limit = memory::set_cache_limit(MLX_CACHE_LIMIT_BYTES)
        .map_err(|error| AppError::Model(format!("Could not set the MLX cache limit: {error}")))?;
    memory::reset_peak()
        .map_err(|error| AppError::Model(format!("Could not reset MLX peak memory: {error}")))?;
    eprintln!(
        "MLX allocator cache limit: {:.1} MiB (previously {:.1} MiB)",
        bytes_to_mib(MLX_CACHE_LIMIT_BYTES),
        bytes_to_mib(previous_limit),
    );
    log_mlx_memory("before model load")
}

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
pub(crate) fn begin_mlx_memory_session() -> AppResult<()> {
    Ok(())
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn log_mlx_memory(stage: &str) -> AppResult<()> {
    let stats = memory::stats()
        .map_err(|error| AppError::Model(format!("Could not read MLX memory stats: {error}")))?;
    eprintln!(
        "MLX {stage}: active={:.1} MiB, cache={:.1} MiB, peak={:.1} MiB",
        bytes_to_mib(stats.active),
        bytes_to_mib(stats.cache),
        bytes_to_mib(stats.peak),
    );
    Ok(())
}

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
pub(crate) fn log_mlx_memory(_stage: &str) -> AppResult<()> {
    Ok(())
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn cleanup_mlx_memory() -> AppResult<()> {
    let before = memory::stats().ok();
    let stream = Stream::task_local_or_default();
    let synchronize_error = stream.synchronize().err();
    let clear_error = memory::clear_cache().err();
    let after = memory::stats()
        .map_err(|error| AppError::Model(format!("Could not read MLX memory stats: {error}")))?;

    if let Some(before) = before {
        eprintln!(
            "MLX after model unload: active={:.1} MiB, cache={:.1} -> {:.1} MiB, peak={:.1} MiB",
            bytes_to_mib(after.active),
            bytes_to_mib(before.cache),
            bytes_to_mib(after.cache),
            bytes_to_mib(after.peak),
        );
    } else {
        eprintln!(
            "MLX after model unload: active={:.1} MiB, cache={:.1} MiB, peak={:.1} MiB",
            bytes_to_mib(after.active),
            bytes_to_mib(after.cache),
            bytes_to_mib(after.peak),
        );
    }

    if let Some(error) = synchronize_error {
        return Err(AppError::Model(format!(
            "Could not synchronize MLX before cleanup: {error}"
        )));
    }
    if let Some(error) = clear_error {
        return Err(AppError::Model(format!(
            "Could not clear the MLX allocator cache: {error}"
        )));
    }
    Ok(())
}

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
pub(crate) fn cleanup_mlx_memory() -> AppResult<()> {
    Ok(())
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

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

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    max_new_tokens: usize,
}

/// Native MOSS inference owner. The MLX tensors remain on the Rust side and are
/// never exposed through Tauri IPC.
#[derive(Debug)]
pub struct MossTranscriber {
    processor: MossProcessor,
    mel: WhisperLogMel,
    max_new_tokens: usize,
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    runtime: MlxRuntime,
}

impl MossTranscriber {
    pub fn load(model_dir: &Path) -> AppResult<Self> {
        let config: MossConfig =
            serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)
                .map_err(|error| AppError::Model(format!("Invalid model config: {error}")))?;
        validate_config(&config)?;
        let generation: GenerationConfig = serde_json::from_reader(std::fs::File::open(
            model_dir.join("generation_config.json"),
        )?)
        .map_err(|error| AppError::Model(format!("Invalid generation config: {error}")))?;
        if generation.max_new_tokens == 0
            || generation.max_new_tokens > config.text_config.max_position_embeddings
        {
            return Err(AppError::Model(format!(
                "Invalid model generation limit: {}",
                generation.max_new_tokens
            )));
        }
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
            max_new_tokens: generation.max_new_tokens,
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            runtime,
        })
    }

    pub fn transcribe(
        &mut self,
        pcm: &[f32],
        options: &TranscribeOptions,
        progress: impl Fn(ProgressEvent),
        stream: impl Fn(TranscriptStreamEvent),
    ) -> AppResult<TranscriptResult> {
        if options
            .max_new_tokens
            .is_some_and(|limit| limit == 0 || limit > 131_072)
        {
            return Err(AppError::Transcription(
                "max_new_tokens must be between 1 and 131072 when provided".into(),
            ));
        }
        let chunks = chunk_audio(pcm)?;
        progress(ProgressEvent {
            task_id: String::new(),
            stage: TaskStage::Encoding,
            percent: 1.5,
            message: format!("Preparing {} Whisper chunk(s)", chunks.len()),
            elapsed_ms: 0,
            audio_duration_ms: Some((pcm.len() as f64 / 16_000.0 * 1000.0) as u64),
            prompt_tokens: 0,
            generated_tokens: 0,
            estimated_generated_tokens: 0,
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
        let estimated_generated_tokens = estimated_output_tokens(audio_tokens);
        let input_ids = self
            .processor
            .expanded_input_ids(audio_tokens, options.prompt.as_deref())?;
        let prompt_tokens = input_ids.len();
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        {
            let max_new_tokens = resolve_generation_limit(
                self.max_new_tokens,
                options.max_new_tokens,
                audio_tokens,
                prompt_tokens,
                131_072,
            )?;
            let mut streamed_tokens = Vec::with_capacity(max_new_tokens.min(4096));
            let generated = self.runtime.generate(
                &input_ids,
                &mel_chunks,
                &audio_feature_lengths,
                max_new_tokens,
                |runtime_progress| {
                    let (stage, percent, message, generated_tokens) = match runtime_progress {
                        RuntimeProgress::AudioEncoded => (
                            TaskStage::Prefilling,
                            2.0,
                            "Whisper and VQAdaptor encoding complete".to_string(),
                            0,
                        ),
                        RuntimeProgress::Prefill { completed, total } => {
                            let ratio = completed as f64 / total.max(1) as f64;
                            (
                                TaskStage::Prefilling,
                                2.0 + ratio,
                                format!("Prefilling prompt tokens {completed}/{total}"),
                                0,
                            )
                        }
                        RuntimeProgress::Token { generated, token } => {
                            streamed_tokens.push(token);
                            if generated % STREAM_DECODE_INTERVAL_TOKENS == 0
                                || token == EOS_TOKEN_ID
                                || token == PAD_TOKEN_ID
                            {
                                if let Ok(text) = self.processor.decode(&streamed_tokens) {
                                    let text = text.trim().to_string();
                                    stream(TranscriptStreamEvent {
                                        task_id: String::new(),
                                        segments: parse_transcript(&text),
                                        text,
                                        generated_tokens: generated,
                                    });
                                }
                            }
                            (
                                TaskStage::Generating,
                                generation_progress_percent(generated, audio_tokens),
                                format!("Generated {generated} tokens"),
                                generated,
                            )
                        }
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
                        estimated_generated_tokens,
                    });
                },
            )?;
            let truncated = generation_was_truncated(&generated);
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
                truncated,
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

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn generation_was_truncated(tokens: &[u32]) -> bool {
    !tokens
        .last()
        .is_some_and(|token| *token == EOS_TOKEN_ID || *token == PAD_TOKEN_ID)
}

fn generation_progress_percent(generated_tokens: usize, audio_tokens: usize) -> f64 {
    // MOSS emits roughly one transcript token per merged audio placeholder on
    // the first measured long-form sample. A small overhead allowance keeps
    // the bar below 99% until timestamps, speaker tags, and EOS are emitted.
    let estimated_output_tokens = estimated_output_tokens(audio_tokens);
    let ratio = generated_tokens as f64 / estimated_output_tokens as f64;
    3.0 + ratio.min(1.0) * 96.0
}

fn estimated_output_tokens(audio_tokens: usize) -> usize {
    audio_tokens
        .saturating_add(ESTIMATED_OUTPUT_TOKEN_OVERHEAD)
        .max(1)
}

fn resolve_generation_limit(
    model_default: usize,
    requested: Option<usize>,
    audio_tokens: usize,
    prompt_tokens: usize,
    context_size: usize,
) -> AppResult<usize> {
    let remaining_context = context_size.saturating_sub(prompt_tokens);
    // Audio placeholders represent roughly 80 ms each. Two output tokens per
    // audio token is a deliberately generous ceiling for dense multilingual
    // speech plus timestamps/speaker tags. It grows smoothly for long audio,
    // while the model's EOS normally ends generation far below the ceiling.
    let automatic_limit = model_default
        .max(audio_tokens.saturating_mul(2))
        .min(LONG_AUDIO_MAX_NEW_TOKENS);
    let limit = requested.unwrap_or(automatic_limit).min(remaining_context);
    if limit == 0 {
        Err(AppError::Transcription(
            "The prompt and audio fill the model context window".into(),
        ))
    } else {
        Ok(limit)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn distinguishes_normal_generation_stop_from_truncation() {
        assert!(!generation_was_truncated(&[42, EOS_TOKEN_ID]));
        assert!(!generation_was_truncated(&[42, PAD_TOKEN_ID]));
        assert!(generation_was_truncated(&[42, 43]));
        assert!(generation_was_truncated(&[]));
    }

    #[test]
    fn generation_limit_uses_model_default_and_remaining_context() {
        assert_eq!(
            resolve_generation_limit(5120, None, 750, 1_000, 131_072).unwrap(),
            5120
        );
        assert_eq!(
            resolve_generation_limit(5120, Some(800), 750, 1_000, 131_072).unwrap(),
            800
        );
        assert_eq!(
            resolve_generation_limit(5120, None, 45_000, 40_000, 131_072).unwrap(),
            65_536
        );
        assert_eq!(
            resolve_generation_limit(5120, None, 45_000, 130_000, 131_072).unwrap(),
            1072
        );
        assert!(resolve_generation_limit(5120, None, 750, 131_072, 131_072).is_err());
    }

    #[test]
    fn generation_progress_tracks_audio_sized_output_without_reaching_completion() {
        assert_eq!(estimated_output_tokens(4_688), 4_816);
        assert_eq!(generation_progress_percent(0, 4_688), 3.0);
        let measured_sample = generation_progress_percent(4_784, 4_688);
        assert!(measured_sample > 98.0 && measured_sample < 99.0);
        assert_eq!(generation_progress_percent(10_000, 4_688), 99.0);
    }
}
