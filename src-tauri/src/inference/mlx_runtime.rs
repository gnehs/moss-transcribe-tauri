//! Native MLX inference for MOSS-Transcribe-Diarize.
//!
//! The implementation deliberately owns the complete forward pass.  The Qwen
//! model shipped by `mlx-lm` embeds integer token ids internally, which cannot
//! represent MOSS' audio embedding injection, so using it directly would make
//! the multimodal model subtly incorrect.

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

use memmap2::MmapOptions;
use mlx_rs::{
    fast::{self, ScaledDotProductAttentionMask},
    nn,
    ops::{
        self,
        indexing::{TryIndexMutOp, TryIndexOp},
    },
    Array,
};
use safetensors::SafeTensors;
#[cfg(feature = "parity-trace")]
use serde::Serialize;
#[cfg(feature = "parity-trace")]
use sha2::{Digest, Sha256};

use crate::error::{AppError, AppResult};

use super::processor::{AUDIO_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID};
#[cfg(feature = "parity-trace")]
use super::{
    mel::{MelBatch, WhisperLogMel, CHUNK_SAMPLES},
    processor::MossProcessor,
};

const AUDIO_LAYERS: usize = 24;
const TEXT_LAYERS: usize = 28;
const AUDIO_DIM: i32 = 1024;
const AUDIO_HEADS: i32 = 16;
const AUDIO_HEAD_DIM: i32 = 64;
const TEXT_DIM: i32 = 1024;
const TEXT_HEADS: i32 = 16;
const TEXT_KV_HEADS: i32 = 8;
const TEXT_HEAD_DIM: i32 = 128;
const VOCAB_SIZE: i32 = 151_936;
const PREFILL_CHUNK_SIZE: usize = 4096;
// Match mlx-lm's block-allocated KVCache. Decode appends one token at a time,
// so reserving a modest block avoids rebuilding the complete cache per token.
const KV_CACHE_STEP: i32 = 256;
const N_MELS: usize = 80;
const N_FRAMES: usize = 3000;

#[cfg(feature = "parity-trace")]
const MODEL_REPO: &str = "OpenMOSS-Team/MOSS-Transcribe-Diarize";
#[cfg(feature = "parity-trace")]
const MODEL_REVISION: &str = "d7231bbae2587a4af278735eb765b318c4f64edd";
#[cfg(feature = "parity-trace")]
const MLX_AUDIO_REVISION: &str = "64e8416c303fb3b3463dab8eb4ebd78c55a87c1a";
#[cfg(feature = "parity-trace")]
const MLX_RS_REVISION: &str = "f4aa309c79b6be35255ca7d34157dfc10d9ed4c9";
#[cfg(feature = "parity-trace")]
const PARITY_MAX_NEW_TOKENS: usize = 4096;
#[cfg(feature = "parity-trace")]
const PARITY_PROBE_TOKENS: usize = 32;

/// Runtime-level progress, independent from Tauri's task metadata.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RuntimeProgress {
    AudioEncoded,
    Prefill { completed: usize, total: usize },
    Token { generated: usize, token: u32 },
}

/// JSON-only parity evidence. This type is available exclusively to the local
/// `moss-parity-trace` developer feature and is never registered as Tauri IPC.
#[cfg(feature = "parity-trace")]
#[derive(Debug, Serialize)]
pub struct NativeParityTrace {
    pub schema_version: u32,
    pub trace_format: &'static str,
    pub provenance: ParityProvenance,
    pub input: ParityInput,
    pub tensors: ParityTensors,
    pub comparison: ParityComparison,
    pub decode: ParityDecode,
}

#[cfg(feature = "parity-trace")]
#[derive(Debug, Serialize)]
pub struct ParityProvenance {
    pub model_repo: &'static str,
    pub model_revision: &'static str,
    pub mlx_audio_revision: &'static str,
    pub mlx_rs_revision: &'static str,
    pub runtime: &'static str,
}

#[cfg(feature = "parity-trace")]
#[derive(Debug, Serialize)]
pub struct ParityInput {
    pub sample_rate: u32,
    pub duration_seconds: f64,
    pub source_basename: String,
    pub sha256: String,
    pub prompt: Option<String>,
}

#[cfg(feature = "parity-trace")]
#[derive(Debug, Serialize)]
pub struct ParityTensors {
    pub log_mel: ParityTensorSummary,
    pub whisper_encoder: ParityTensorSummary,
    pub vq_adaptor: ParityTensorSummary,
    pub expanded_input_ids: ParityTensorSummary,
    pub fused_embeddings: ParityTensorSummary,
    pub first_token_logits: ParityTensorSummary,
    pub greedy_token_ids: ParityTensorSummary,
    pub final_transcript: ParityTensorSummary,
}

#[cfg(feature = "parity-trace")]
#[derive(Debug, Serialize)]
pub struct ParityTensorSummary {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_indices: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_values: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean: Option<f64>,
}

#[cfg(feature = "parity-trace")]
#[derive(Debug, Serialize)]
pub struct ParityComparison {
    pub float_tolerances: HashMap<&'static str, ParityTolerance>,
    pub exact: [&'static str; 3],
}

#[cfg(feature = "parity-trace")]
#[derive(Debug, Serialize)]
pub struct ParityTolerance {
    pub atol: f64,
    pub rtol: f64,
}

#[cfg(feature = "parity-trace")]
#[derive(Debug, Serialize)]
pub struct ParityDecode {
    pub expanded_input_ids: Vec<i64>,
    /// Compatibility alias for the original v1 Rust comparator.
    pub greedy_token_ids: Vec<i64>,
    pub first_token_id: i64,
    pub first_32_greedy_tokens: Vec<i64>,
    pub final_transcript: String,
}

#[cfg(feature = "parity-trace")]
#[derive(Default)]
struct RuntimeTraceCapture {
    log_mel: Option<ParityTensorSummary>,
    whisper_encoder: Option<ParityTensorSummary>,
    vq_adaptor: Option<ParityTensorSummary>,
    fused_embeddings: Option<ParityTensorSummary>,
    first_token_logits: Option<ParityTensorSummary>,
}

/// Fully loaded MOSS model. All tensors and caches stay inside MLX.
#[derive(Debug)]
pub(crate) struct MlxRuntime {
    whisper: WhisperEncoder,
    adaptor: VqAdaptor,
    qwen: Qwen3,
}

impl MlxRuntime {
    /// Load and strictly validate the single official MOSS safetensors shard.
    /// Unknown, missing, or incorrectly shaped tensors are rejected.
    pub(crate) fn load(model_dir: &Path) -> AppResult<Self> {
        let path = model_dir.join("model-00000-of-00001.safetensors");
        let mut tensors = TensorStore::load(&path)?;
        let whisper = WhisperEncoder::load(&mut tensors)?;
        let adaptor = VqAdaptor::load(&mut tensors)?;
        let qwen = Qwen3::load(&mut tensors)?;
        tensors.finish()?;
        Ok(Self {
            whisper,
            adaptor,
            qwen,
        })
    }

    /// Batch-encode one owned contiguous `[chunks, 80, 3000]` mel buffer,
    /// concatenate only the valid `Li * 4` encoder frames, inject the adapted
    /// audio embeddings, and run one decoder generation pass.
    ///
    /// The host buffer is released immediately after MLX copies it into an
    /// `Array`, before the audio encoder graph is built.
    pub(crate) fn generate(
        &mut self,
        input_ids: &[u32],
        mel_data: Vec<f32>,
        audio_feature_lengths: &[usize],
        max_new_tokens: usize,
        progress: impl FnMut(RuntimeProgress),
    ) -> AppResult<Vec<u32>> {
        self.generate_observed(
            input_ids,
            mel_data,
            audio_feature_lengths,
            max_new_tokens,
            progress,
            |_, _| Ok(()),
        )
    }

    fn generate_observed(
        &mut self,
        input_ids: &[u32],
        mel_data: Vec<f32>,
        audio_feature_lengths: &[usize],
        max_new_tokens: usize,
        mut progress: impl FnMut(RuntimeProgress),
        mut observe: impl FnMut(&'static str, &Array) -> AppResult<()>,
    ) -> AppResult<Vec<u32>> {
        let chunk_count = validate_mel_batch(&mel_data, audio_feature_lengths)?;

        let placeholder_count = input_ids
            .iter()
            .filter(|token| **token == AUDIO_TOKEN_ID)
            .count();
        let audio_token_count: usize = audio_feature_lengths.iter().sum();
        if placeholder_count != audio_token_count {
            return Err(AppError::Transcription(format!(
                "Audio placeholder count ({placeholder_count}) does not match encoded features ({audio_token_count})"
            )));
        }

        let mel = Array::from_slice(
            &mel_data,
            &[chunk_count as i32, N_MELS as i32, N_FRAMES as i32],
        );
        drop(mel_data);
        observe("log_mel", &mel)?;
        let encoded = self.whisper.forward(&mel)?;
        observe("whisper_encoder", &encoded)?;

        let mut valid_parts = Vec::with_capacity(audio_feature_lengths.len());
        for (chunk, token_length) in audio_feature_lengths.iter().copied().enumerate() {
            let frames = (token_length * 4) as i32;
            valid_parts.push(mlx(
                encoded.try_index((chunk as i32..chunk as i32 + 1, 0..frames, ..)),
                "slice valid Whisper frames",
            )?);
        }
        let audio = mlx(
            ops::concatenate_axis(&valid_parts, 1),
            "concatenate Whisper chunks",
        )?;
        let merged = mlx(
            audio.reshape(&[1, audio_token_count as i32, AUDIO_DIM * 4]),
            "merge four Whisper frames",
        )?;
        let audio_embeds = self.adaptor.forward(&merged)?;
        mlx(audio_embeds.eval(), "evaluate audio encoder")?;
        observe("vq_adaptor", &audio_embeds)?;
        progress(RuntimeProgress::AudioEncoded);

        let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
        let token_embeds = self.qwen.embed(&ids)?;
        let inputs_embeds = inject_audio_embeddings(&token_embeds, input_ids, &audio_embeds)?;
        observe("fused_embeddings", &inputs_embeds)?;

        self.qwen
            .generate(&inputs_embeds, max_new_tokens, progress, &mut observe)
    }
}

fn validate_mel_batch(mel_data: &[f32], audio_feature_lengths: &[usize]) -> AppResult<usize> {
    let chunk_size = N_MELS * N_FRAMES;
    if mel_data.is_empty()
        || mel_data.len() > i32::MAX as usize
        || !mel_data.len().is_multiple_of(chunk_size)
    {
        return Err(AppError::Transcription(format!(
            "Whisper features must have contiguous shape [chunks, {N_MELS}, {N_FRAMES}]"
        )));
    }
    let chunk_count = mel_data.len() / chunk_size;
    if chunk_count != audio_feature_lengths.len() {
        return Err(AppError::Transcription(
            "MLX requires one audio feature length per mel chunk".into(),
        ));
    }
    if audio_feature_lengths
        .iter()
        .any(|length| *length == 0 || *length > 375)
    {
        return Err(AppError::Transcription(
            "An audio chunk must contain between 1 and 375 merged tokens".into(),
        ));
    }
    Ok(chunk_count)
}

fn inject_audio_embeddings(
    token_embeds: &Array,
    token_ids: &[u32],
    audio_embeds: &Array,
) -> AppResult<Array> {
    let [batch, token_count, hidden_size] = shape3(token_embeds, "token embeddings")?;
    let [audio_batch, audio_token_count, audio_hidden_size] =
        shape3(audio_embeds, "audio embeddings")?;
    if batch != 1 || token_ids.len() != token_count as usize {
        return Err(AppError::Transcription(format!(
            "Audio injection requires one token ID per embedding; got embeddings {:?} and {} IDs",
            token_embeds.shape(),
            token_ids.len()
        )));
    }
    if audio_batch != 1 || audio_hidden_size != hidden_size {
        return Err(AppError::Transcription(format!(
            "Audio embeddings {:?} are incompatible with token embeddings {:?}",
            audio_embeds.shape(),
            token_embeds.shape()
        )));
    }

    // Build an explicit token -> audio-row mapping on the Rust side. Gathering
    // aligned rows and selecting with `where` avoids masked_scatter's flattened
    // source contract while all large embeddings remain in MLX memory.
    let (replacement_indices, audio_mask) =
        audio_replacement_map(token_ids, audio_token_count as usize)?;
    let indices = Array::from_slice(&replacement_indices, &[token_count]);
    let replacements = mlx(
        audio_embeds.take_axis(&indices, 1),
        "align audio embeddings with prompt tokens",
    )?;
    let mask = Array::from_slice(&audio_mask, &[1, token_count]);
    let mask = mlx(mask.expand_dims_axes(&[-1]), "expand audio selection mask")?;
    mlx(
        ops::r#where(&mask, &replacements, token_embeds),
        "inject audio embeddings",
    )
}

fn audio_replacement_map(
    token_ids: &[u32],
    audio_token_count: usize,
) -> AppResult<(Vec<u32>, Vec<bool>)> {
    let mut next_audio_row = 0_u32;
    let mut replacement_indices = Vec::with_capacity(token_ids.len());
    let mut audio_mask = Vec::with_capacity(token_ids.len());
    for token_id in token_ids {
        let is_audio = *token_id == AUDIO_TOKEN_ID;
        audio_mask.push(is_audio);
        replacement_indices.push(if is_audio {
            let row = next_audio_row;
            next_audio_row += 1;
            row
        } else {
            0
        });
    }
    if next_audio_row as usize != audio_token_count {
        return Err(AppError::Transcription(format!(
            "Audio placeholder count ({next_audio_row}) does not match audio embeddings ({audio_token_count})"
        )));
    }
    Ok((replacement_indices, audio_mask))
}

/// Generate deterministic native probes from the exact production forward
/// path. Missing model/audio inputs are errors; no placeholder trace is ever
/// emitted. This API is intentionally absent from production builds.
#[cfg(feature = "parity-trace")]
pub fn generate_native_parity_trace(
    model_dir: &Path,
    pcm: &[f32],
    source_basename: impl Into<String>,
    prompt: Option<&str>,
) -> AppResult<NativeParityTrace> {
    if pcm.len() <= CHUNK_SAMPLES {
        return Err(AppError::Transcription(format!(
            "Parity audio must be longer than 30 seconds; got {:.3} seconds",
            pcm.len() as f64 / 16_000.0
        )));
    }
    if !model_dir.join("config.json").is_file()
        || !model_dir.join("model-00000-of-00001.safetensors").is_file()
    {
        return Err(AppError::Model(format!(
            "Parity model snapshot is incomplete at {}",
            model_dir.display()
        )));
    }

    let mel = WhisperLogMel::new();
    let MelBatch {
        features,
        audio_feature_lengths,
    } = mel.extract_audio(pcm)?;
    let audio_tokens = audio_feature_lengths.iter().sum();
    let processor = MossProcessor::load(model_dir)?;
    let input_ids = processor.expanded_input_ids(audio_tokens, prompt)?;
    let mut runtime = MlxRuntime::load(model_dir)?;
    let mut captured = RuntimeTraceCapture::default();
    let generated = runtime.generate_observed(
        &input_ids,
        features,
        &audio_feature_lengths,
        PARITY_MAX_NEW_TOKENS,
        |_| {},
        |name, array| {
            let summary = summarize_float_array(array)?;
            match name {
                "log_mel" => captured.log_mel = Some(summary),
                "whisper_encoder" => captured.whisper_encoder = Some(summary),
                "vq_adaptor" => captured.vq_adaptor = Some(summary),
                "fused_embeddings" => captured.fused_embeddings = Some(summary),
                "first_token_logits" => captured.first_token_logits = Some(summary),
                _ => {
                    return Err(AppError::Transcription(format!(
                        "Unknown parity trace stage {name}"
                    )))
                }
            }
            Ok(())
        },
    )?;
    if generated.len() < PARITY_PROBE_TOKENS {
        return Err(AppError::Transcription(format!(
            "Parity decode ended after {} tokens; at least {PARITY_PROBE_TOKENS} are required",
            generated.len()
        )));
    }
    let transcript = processor.decode(&generated)?.trim().to_string();
    let expanded = input_ids
        .iter()
        .map(|value| i64::from(*value))
        .collect::<Vec<_>>();
    let greedy = generated
        .iter()
        .map(|value| i64::from(*value))
        .collect::<Vec<_>>();
    let tensors = ParityTensors {
        log_mel: require_capture(captured.log_mel, "log_mel")?,
        whisper_encoder: require_capture(captured.whisper_encoder, "whisper_encoder")?,
        vq_adaptor: require_capture(captured.vq_adaptor, "vq_adaptor")?,
        expanded_input_ids: summarize_i64(&expanded, vec![1, expanded.len()]),
        fused_embeddings: require_capture(captured.fused_embeddings, "fused_embeddings")?,
        first_token_logits: require_capture(captured.first_token_logits, "first_token_logits")?,
        greedy_token_ids: summarize_i64(&greedy, vec![greedy.len()]),
        final_transcript: summarize_transcript(&transcript),
    };

    let mut tolerances = HashMap::new();
    tolerances.insert(
        "log_mel",
        ParityTolerance {
            atol: 1.0e-4,
            rtol: 1.0e-4,
        },
    );
    for stage in ["whisper_encoder", "vq_adaptor", "fused_embeddings"] {
        tolerances.insert(
            stage,
            ParityTolerance {
                atol: 2.0e-3,
                rtol: 2.0e-3,
            },
        );
    }
    tolerances.insert(
        "first_token_logits",
        ParityTolerance {
            atol: 2.0e-2,
            rtol: 2.0e-3,
        },
    );

    Ok(NativeParityTrace {
        schema_version: 1,
        trace_format: "moss-native-json-probes-v1",
        provenance: ParityProvenance {
            model_repo: MODEL_REPO,
            model_revision: MODEL_REVISION,
            mlx_audio_revision: MLX_AUDIO_REVISION,
            mlx_rs_revision: MLX_RS_REVISION,
            runtime: "mlx-rs-native",
        },
        input: ParityInput {
            sample_rate: 16_000,
            duration_seconds: pcm.len() as f64 / 16_000.0,
            source_basename: source_basename.into(),
            sha256: sha256_f32(pcm),
            prompt: prompt.map(str::to_owned),
        },
        tensors,
        comparison: ParityComparison {
            float_tolerances: tolerances,
            exact: ["expanded_input_ids", "greedy_token_ids", "final_transcript"],
        },
        decode: ParityDecode {
            expanded_input_ids: expanded,
            greedy_token_ids: greedy.clone(),
            first_token_id: greedy[0],
            first_32_greedy_tokens: greedy[..PARITY_PROBE_TOKENS].to_vec(),
            final_transcript: transcript,
        },
    })
}

#[cfg(feature = "parity-trace")]
fn require_capture(
    value: Option<ParityTensorSummary>,
    name: &str,
) -> AppResult<ParityTensorSummary> {
    value.ok_or_else(|| AppError::Transcription(format!("Parity trace omitted stage {name}")))
}

#[cfg(feature = "parity-trace")]
fn summarize_float_array(array: &Array) -> AppResult<ParityTensorSummary> {
    let canonical = mlx(array.as_type::<f32>(), "cast parity tensor to float32")?;
    mlx(canonical.eval(), "evaluate parity tensor")?;
    let values = canonical
        .try_as_slice::<f32>()
        .map_err(|error| AppError::Transcription(format!("Read parity tensor: {error}")))?;
    if values.is_empty() {
        return Err(AppError::Transcription(
            "Parity tensors must not be empty".into(),
        ));
    }
    let indices = probe_indices(values.len());
    let sample_values = indices
        .iter()
        .map(|index| f64::from(values[*index]))
        .collect::<Vec<_>>();
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut hasher = Sha256::new();
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
        sum += f64::from(*value);
        hasher.update(value.to_le_bytes());
    }
    Ok(ParityTensorSummary {
        dtype: "float32".into(),
        shape: array.shape().iter().map(|value| *value as usize).collect(),
        sha256: hex_digest(hasher.finalize()),
        sample_indices: Some(indices),
        sample_values: Some(sample_values),
        min: Some(f64::from(min)),
        max: Some(f64::from(max)),
        mean: Some(sum / values.len() as f64),
    })
}

#[cfg(feature = "parity-trace")]
fn summarize_i64(values: &[i64], shape: Vec<usize>) -> ParityTensorSummary {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    ParityTensorSummary {
        dtype: "int64".into(),
        shape,
        sha256: hex_digest(hasher.finalize()),
        sample_indices: None,
        sample_values: None,
        min: None,
        max: None,
        mean: None,
    }
}

#[cfg(feature = "parity-trace")]
fn summarize_transcript(transcript: &str) -> ParityTensorSummary {
    let mut hasher = Sha256::new();
    let length = transcript.chars().count();
    for character in transcript.chars() {
        hasher.update(u32::from(character).to_le_bytes());
    }
    ParityTensorSummary {
        dtype: format!("<U{length}"),
        shape: Vec::new(),
        sha256: hex_digest(hasher.finalize()),
        sample_indices: None,
        sample_values: None,
        min: None,
        max: None,
        mean: None,
    }
}

#[cfg(feature = "parity-trace")]
fn probe_indices(length: usize) -> Vec<usize> {
    let count = length.min(64);
    if count <= 1 {
        return vec![0];
    }
    (0..count)
        .map(|index| ((index as f64 * (length - 1) as f64) / (count - 1) as f64).floor() as usize)
        .collect()
}

#[cfg(feature = "parity-trace")]
fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hex_digest(hasher.finalize())
}

#[cfg(feature = "parity-trace")]
fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[derive(Debug)]
struct TensorStore {
    tensors: HashMap<String, Array>,
}

impl TensorStore {
    fn load(path: &Path) -> AppResult<Self> {
        let file = File::open(path)?;
        // SAFETY: the read-only mapping cannot outlive `file` during manifest
        // validation, and the backing model file is not mutated by this app.
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|error| AppError::Model(format!("Could not map model weights: {error}")))?;
        let safetensors = SafeTensors::deserialize(&mmap)
            .map_err(|error| AppError::Model(format!("Invalid safetensors file: {error}")))?;
        validate_manifest(&safetensors)?;
        drop(safetensors);
        drop(mmap);

        let tensors = Array::load_safetensors(path).map_err(|error| {
            AppError::Model(format!("MLX could not load model weights: {error}"))
        })?;
        Ok(Self { tensors })
    }

    fn take(&mut self, name: &str) -> AppResult<Array> {
        self.tensors
            .remove(name)
            .ok_or_else(|| AppError::Model(format!("Missing validated tensor {name}")))
    }

    fn finish(self) -> AppResult<()> {
        if self.tensors.is_empty() {
            Ok(())
        } else {
            let mut keys = self.tensors.into_keys().collect::<Vec<_>>();
            keys.sort();
            Err(AppError::Model(format!(
                "Unconsumed model weights: {}",
                keys.join(", ")
            )))
        }
    }
}

fn validate_manifest(tensors: &SafeTensors<'_>) -> AppResult<()> {
    let expected = expected_manifest();
    let expected_names = expected.keys().cloned().collect::<HashSet<_>>();
    let actual_names = tensors
        .names()
        .into_iter()
        .map(str::to_owned)
        .collect::<HashSet<_>>();

    let mut missing = expected_names
        .difference(&actual_names)
        .cloned()
        .collect::<Vec<_>>();
    let mut unexpected = actual_names
        .difference(&expected_names)
        .cloned()
        .collect::<Vec<_>>();
    missing.sort();
    unexpected.sort();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(AppError::Model(format!(
            "Weight key mismatch; missing [{}], unexpected [{}]",
            summarize(&missing),
            summarize(&unexpected)
        )));
    }

    let mut mismatches = Vec::new();
    for (name, shape) in expected {
        let tensor = tensors.tensor(&name).map_err(|error| {
            AppError::Model(format!("Could not inspect tensor {name}: {error}"))
        })?;
        if tensor.shape() != shape {
            mismatches.push(format!("{name}: {:?}, expected {shape:?}", tensor.shape()));
        }
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(AppError::Model(format!(
            "Weight shape mismatch: {}",
            summarize(&mismatches)
        )))
    }
}

fn summarize(items: &[String]) -> String {
    const LIMIT: usize = 8;
    let suffix = if items.len() > LIMIT {
        format!(" … and {} more", items.len() - LIMIT)
    } else {
        String::new()
    };
    format!(
        "{}{}",
        items
            .iter()
            .take(LIMIT)
            .cloned()
            .collect::<Vec<_>>()
            .join(", "),
        suffix
    )
}

fn expected_manifest() -> HashMap<String, Vec<usize>> {
    let mut expected = HashMap::with_capacity(683);
    let mut add = |name: String, shape: &[usize]| {
        expected.insert(name, shape.to_vec());
    };

    add("model.whisper_encoder.conv1.weight".into(), &[1024, 80, 3]);
    add("model.whisper_encoder.conv1.bias".into(), &[1024]);
    add(
        "model.whisper_encoder.conv2.weight".into(),
        &[1024, 1024, 3],
    );
    add("model.whisper_encoder.conv2.bias".into(), &[1024]);
    add(
        "model.whisper_encoder.embed_positions.weight".into(),
        &[1500, 1024],
    );
    add("model.whisper_encoder.layer_norm.weight".into(), &[1024]);
    add("model.whisper_encoder.layer_norm.bias".into(), &[1024]);
    for layer in 0..AUDIO_LAYERS {
        let prefix = format!("model.whisper_encoder.layers.{layer}");
        add(format!("{prefix}.self_attn.q_proj.weight"), &[1024, 1024]);
        add(format!("{prefix}.self_attn.q_proj.bias"), &[1024]);
        add(format!("{prefix}.self_attn.k_proj.weight"), &[1024, 1024]);
        add(format!("{prefix}.self_attn.v_proj.weight"), &[1024, 1024]);
        add(format!("{prefix}.self_attn.v_proj.bias"), &[1024]);
        add(format!("{prefix}.self_attn.out_proj.weight"), &[1024, 1024]);
        add(format!("{prefix}.self_attn.out_proj.bias"), &[1024]);
        add(format!("{prefix}.self_attn_layer_norm.weight"), &[1024]);
        add(format!("{prefix}.self_attn_layer_norm.bias"), &[1024]);
        add(format!("{prefix}.fc1.weight"), &[4096, 1024]);
        add(format!("{prefix}.fc1.bias"), &[4096]);
        add(format!("{prefix}.fc2.weight"), &[1024, 4096]);
        add(format!("{prefix}.fc2.bias"), &[1024]);
        add(format!("{prefix}.final_layer_norm.weight"), &[1024]);
        add(format!("{prefix}.final_layer_norm.bias"), &[1024]);
    }

    add("model.vq_adaptor.layers.0.weight".into(), &[1024, 4096]);
    add("model.vq_adaptor.layers.0.bias".into(), &[1024]);
    add("model.vq_adaptor.layers.2.weight".into(), &[1024, 1024]);
    add("model.vq_adaptor.layers.2.bias".into(), &[1024]);
    add("model.vq_adaptor.layers.3.weight".into(), &[1024]);
    add("model.vq_adaptor.layers.3.bias".into(), &[1024]);

    add(
        "model.language_model.embed_tokens.weight".into(),
        &[VOCAB_SIZE as usize, TEXT_DIM as usize],
    );
    for layer in 0..TEXT_LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        add(format!("{prefix}.self_attn.q_proj.weight"), &[2048, 1024]);
        add(format!("{prefix}.self_attn.k_proj.weight"), &[1024, 1024]);
        add(format!("{prefix}.self_attn.v_proj.weight"), &[1024, 1024]);
        add(format!("{prefix}.self_attn.o_proj.weight"), &[1024, 2048]);
        add(format!("{prefix}.self_attn.q_norm.weight"), &[128]);
        add(format!("{prefix}.self_attn.k_norm.weight"), &[128]);
        add(format!("{prefix}.mlp.gate_proj.weight"), &[3072, 1024]);
        add(format!("{prefix}.mlp.up_proj.weight"), &[3072, 1024]);
        add(format!("{prefix}.mlp.down_proj.weight"), &[1024, 3072]);
        add(format!("{prefix}.input_layernorm.weight"), &[1024]);
        add(format!("{prefix}.post_attention_layernorm.weight"), &[1024]);
    }
    add("model.language_model.norm.weight".into(), &[1024]);
    debug_assert_eq!(expected.len(), 683);
    expected
}

#[derive(Debug)]
struct Linear {
    weight: Array,
    bias: Option<Array>,
}

impl Linear {
    fn load(store: &mut TensorStore, prefix: &str, bias: bool) -> AppResult<Self> {
        Ok(Self {
            weight: store.take(&format!("{prefix}.weight"))?,
            bias: bias
                .then(|| store.take(&format!("{prefix}.bias")))
                .transpose()?,
        })
    }

    fn forward(&self, input: &Array) -> AppResult<Array> {
        let output = mlx(ops::matmul(input, self.weight.t()), "linear projection")?;
        match &self.bias {
            Some(bias) => mlx(output.add(bias), "linear bias"),
            None => Ok(output),
        }
    }
}

#[derive(Debug)]
struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNorm {
    fn load(store: &mut TensorStore, prefix: &str) -> AppResult<Self> {
        Self::load_with_eps(store, prefix, 1.0e-5)
    }

    fn load_with_eps(store: &mut TensorStore, prefix: &str, eps: f32) -> AppResult<Self> {
        Ok(Self {
            weight: store.take(&format!("{prefix}.weight"))?,
            bias: store.take(&format!("{prefix}.bias"))?,
            eps,
        })
    }

    fn forward(&self, input: &Array) -> AppResult<Array> {
        mlx(
            fast::layer_norm(input, &self.weight, &self.bias, self.eps),
            "layer norm",
        )
    }
}

#[derive(Debug)]
struct WhisperAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl WhisperAttention {
    fn load(store: &mut TensorStore, prefix: &str) -> AppResult<Self> {
        Ok(Self {
            q_proj: Linear::load(store, &format!("{prefix}.q_proj"), true)?,
            k_proj: Linear::load(store, &format!("{prefix}.k_proj"), false)?,
            v_proj: Linear::load(store, &format!("{prefix}.v_proj"), true)?,
            out_proj: Linear::load(store, &format!("{prefix}.out_proj"), true)?,
        })
    }

    fn forward(&self, input: &Array) -> AppResult<Array> {
        let [batch, length, _] = shape3(input, "Whisper attention input")?;
        let q = mlx(
            self.q_proj
                .forward(input)?
                .reshape(&[batch, length, AUDIO_HEADS, AUDIO_HEAD_DIM])
                .and_then(|array| array.transpose_axes(&[0, 2, 1, 3])),
            "reshape Whisper queries",
        )?;
        let k = mlx(
            self.k_proj
                .forward(input)?
                .reshape(&[batch, length, AUDIO_HEADS, AUDIO_HEAD_DIM])
                .and_then(|array| array.transpose_axes(&[0, 2, 1, 3])),
            "reshape Whisper keys",
        )?;
        let v = mlx(
            self.v_proj
                .forward(input)?
                .reshape(&[batch, length, AUDIO_HEADS, AUDIO_HEAD_DIM])
                .and_then(|array| array.transpose_axes(&[0, 2, 1, 3])),
            "reshape Whisper values",
        )?;
        let attended = mlx(
            fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                (AUDIO_HEAD_DIM as f32).sqrt().recip(),
                None,
                None,
            ),
            "Whisper self attention",
        )?;
        let attended = mlx(
            attended
                .transpose_axes(&[0, 2, 1, 3])
                .and_then(|array| array.reshape(&[batch, length, AUDIO_DIM])),
            "merge Whisper attention heads",
        )?;
        self.out_proj.forward(&attended)
    }
}

#[derive(Debug)]
struct WhisperLayer {
    attention: WhisperAttention,
    attention_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
}

impl WhisperLayer {
    fn load(store: &mut TensorStore, layer: usize) -> AppResult<Self> {
        let prefix = format!("model.whisper_encoder.layers.{layer}");
        Ok(Self {
            attention: WhisperAttention::load(store, &format!("{prefix}.self_attn"))?,
            attention_norm: LayerNorm::load(store, &format!("{prefix}.self_attn_layer_norm"))?,
            fc1: Linear::load(store, &format!("{prefix}.fc1"), true)?,
            fc2: Linear::load(store, &format!("{prefix}.fc2"), true)?,
            final_norm: LayerNorm::load(store, &format!("{prefix}.final_layer_norm"))?,
        })
    }

    fn forward(&self, input: &Array) -> AppResult<Array> {
        let attention = self
            .attention
            .forward(&self.attention_norm.forward(input)?)?;
        let hidden = mlx(input.add(&attention), "Whisper attention residual")?;
        let feed_forward = self.fc1.forward(&self.final_norm.forward(&hidden)?)?;
        let feed_forward = mlx(nn::gelu(&feed_forward), "Whisper GELU")?;
        let feed_forward = self.fc2.forward(&feed_forward)?;
        mlx(hidden.add(&feed_forward), "Whisper MLP residual")
    }
}

#[derive(Debug)]
struct WhisperEncoder {
    conv1_weight: Array,
    conv1_bias: Array,
    conv2_weight: Array,
    conv2_bias: Array,
    positions: Array,
    layers: Vec<WhisperLayer>,
    norm: LayerNorm,
}

impl WhisperEncoder {
    fn load(store: &mut TensorStore) -> AppResult<Self> {
        let conv1_weight = mlx(
            store
                .take("model.whisper_encoder.conv1.weight")?
                .transpose_axes(&[0, 2, 1]),
            "convert Whisper conv1 to MLX channels-last weights",
        )?;
        let conv2_weight = mlx(
            store
                .take("model.whisper_encoder.conv2.weight")?
                .transpose_axes(&[0, 2, 1]),
            "convert Whisper conv2 to MLX channels-last weights",
        )?;
        let layers = (0..AUDIO_LAYERS)
            .map(|layer| WhisperLayer::load(store, layer))
            .collect::<AppResult<Vec<_>>>()?;
        Ok(Self {
            conv1_weight,
            conv1_bias: store.take("model.whisper_encoder.conv1.bias")?,
            conv2_weight,
            conv2_bias: store.take("model.whisper_encoder.conv2.bias")?,
            positions: store.take("model.whisper_encoder.embed_positions.weight")?,
            layers,
            norm: LayerNorm::load(store, "model.whisper_encoder.layer_norm")?,
        })
    }

    fn forward(&self, input: &Array) -> AppResult<Array> {
        let input = mlx(
            input.as_dtype(self.conv1_weight.dtype()),
            "cast mel features to model dtype",
        )?;
        let input = mlx(
            input.transpose_axes(&[0, 2, 1]),
            "transpose mel channels last",
        )?;
        let hidden = mlx(
            ops::conv1d(&input, &self.conv1_weight, 1, 1, 1, 1),
            "Whisper conv1",
        )?;
        let hidden = mlx(hidden.add(&self.conv1_bias), "Whisper conv1 bias")?;
        let hidden = mlx(nn::gelu(&hidden), "Whisper conv1 GELU")?;
        let hidden = mlx(
            ops::conv1d(&hidden, &self.conv2_weight, 2, 1, 1, 1),
            "Whisper conv2",
        )?;
        let hidden = mlx(hidden.add(&self.conv2_bias), "Whisper conv2 bias")?;
        let mut hidden = mlx(nn::gelu(&hidden), "Whisper conv2 GELU")?;
        if hidden.shape().get(1) != Some(&1500) {
            return Err(AppError::Transcription(format!(
                "Whisper convolution produced {:?}, expected [batch, 1500, 1024]",
                hidden.shape()
            )));
        }
        hidden = mlx(hidden.add(&self.positions), "add Whisper positions")?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        self.norm.forward(&hidden)
    }
}

#[derive(Debug)]
struct VqAdaptor {
    input: Linear,
    output: Linear,
    norm: LayerNorm,
}

impl VqAdaptor {
    fn load(store: &mut TensorStore) -> AppResult<Self> {
        Ok(Self {
            input: Linear::load(store, "model.vq_adaptor.layers.0", true)?,
            output: Linear::load(store, "model.vq_adaptor.layers.2", true)?,
            norm: LayerNorm::load_with_eps(store, "model.vq_adaptor.layers.3", 1.0e-6)?,
        })
    }

    fn forward(&self, input: &Array) -> AppResult<Array> {
        let hidden = self.input.forward(input)?;
        let hidden = mlx(nn::silu(&hidden), "VQ adaptor SiLU")?;
        self.norm.forward(&self.output.forward(&hidden)?)
    }
}

#[derive(Debug, Default)]
struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    reserve: i32,
}

impl KvCache {
    fn with_reserve(prompt_length: i32) -> AppResult<Self> {
        Ok(Self {
            reserve: initial_kv_capacity(prompt_length)?,
            ..Self::default()
        })
    }

    fn offset(&self) -> i32 {
        self.offset
    }

    fn append(&mut self, keys: Array, values: Array) -> AppResult<(Array, Array)> {
        let length = keys.dim(-2);
        let expected = [1, TEXT_KV_HEADS, length, TEXT_HEAD_DIM];
        if length <= 0 || keys.shape() != expected || values.shape() != expected {
            return Err(AppError::Transcription(format!(
                "Invalid GQA KV cache shape: keys {:?}, values {:?}, expected {expected:?}",
                keys.shape(),
                values.shape()
            )));
        }

        let previous = self.offset;
        let end = previous
            .checked_add(length)
            .ok_or_else(|| AppError::Transcription("Qwen KV cache offset overflowed".into()))?;
        let capacity = self.keys.as_ref().map_or(0, |cached| cached.dim(-2));

        if end > capacity {
            // Allocate enough complete blocks for this append. If an unusual
            // append crosses a boundary before consuming the old spare block,
            // discard that unused tail before extending, as mlx-lm does.
            let target_capacity = next_kv_capacity(previous, capacity, length)?.max(self.reserve);
            let added_capacity = target_capacity - previous;
            let key_zeros = mlx(
                ops::zeros_dtype(
                    &[1, TEXT_KV_HEADS, added_capacity, TEXT_HEAD_DIM],
                    keys.dtype(),
                ),
                "allocate KV key block",
            )?;
            let value_zeros = mlx(
                ops::zeros_dtype(
                    &[1, TEXT_KV_HEADS, added_capacity, TEXT_HEAD_DIM],
                    values.dtype(),
                ),
                "allocate KV value block",
            )?;

            let (expanded_keys, expanded_values) = match (&self.keys, &self.values) {
                (Some(cached_keys), Some(cached_values)) => {
                    let valid_keys = mlx(
                        cached_keys.try_index((.., .., ..previous, ..)),
                        "trim unused KV key capacity",
                    )?;
                    let valid_values = mlx(
                        cached_values.try_index((.., .., ..previous, ..)),
                        "trim unused KV value capacity",
                    )?;
                    (
                        mlx(
                            ops::concatenate_axis(&[valid_keys, key_zeros], -2),
                            "expand KV key cache",
                        )?,
                        mlx(
                            ops::concatenate_axis(&[valid_values, value_zeros], -2),
                            "expand KV value cache",
                        )?,
                    )
                }
                (None, None) => (key_zeros, value_zeros),
                _ => {
                    return Err(AppError::Transcription(
                        "Qwen KV key/value cache state is inconsistent".into(),
                    ));
                }
            };
            self.keys = Some(expanded_keys);
            self.values = Some(expanded_values);
        }

        let cached_keys = self
            .keys
            .as_mut()
            .ok_or_else(|| AppError::Transcription("Qwen KV key cache was not allocated".into()))?;
        mlx(
            cached_keys.try_index_mut((.., .., previous..end, ..), &keys),
            "update KV key cache",
        )?;
        let cached_values = self.values.as_mut().ok_or_else(|| {
            AppError::Transcription("Qwen KV value cache was not allocated".into())
        })?;
        mlx(
            cached_values.try_index_mut((.., .., previous..end, ..), &values),
            "update KV value cache",
        )?;
        self.offset = end;

        let valid_keys = mlx(
            cached_keys.try_index((.., .., ..end, ..)),
            "slice valid KV keys",
        )?;
        let valid_values = mlx(
            cached_values.try_index((.., .., ..end, ..)),
            "slice valid KV values",
        )?;
        Ok((valid_keys, valid_values))
    }
}

fn next_kv_capacity(previous: i32, capacity: i32, length: i32) -> AppResult<i32> {
    if previous < 0 || capacity < previous || length <= 0 {
        return Err(AppError::Transcription(format!(
            "Invalid KV cache allocation state: offset {previous}, capacity {capacity}, append {length}"
        )));
    }
    let end = previous
        .checked_add(length)
        .ok_or_else(|| AppError::Transcription("Qwen KV cache capacity overflowed".into()))?;
    if end <= capacity {
        return Ok(capacity);
    }
    let blocks = length
        .checked_add(KV_CACHE_STEP - 1)
        .ok_or_else(|| AppError::Transcription("Qwen KV cache capacity overflowed".into()))?
        / KV_CACHE_STEP;
    let added = blocks
        .checked_mul(KV_CACHE_STEP)
        .ok_or_else(|| AppError::Transcription("Qwen KV cache capacity overflowed".into()))?;
    previous
        .checked_add(added)
        .ok_or_else(|| AppError::Transcription("Qwen KV cache capacity overflowed".into()))
}

fn initial_kv_capacity(prompt_length: i32) -> AppResult<i32> {
    if prompt_length <= 0 {
        return Err(AppError::Transcription(format!(
            "Invalid Qwen prompt length for KV cache: {prompt_length}"
        )));
    }
    let requested = prompt_length
        .checked_add(KV_CACHE_STEP)
        .and_then(|value| value.checked_add(KV_CACHE_STEP - 1))
        .ok_or_else(|| AppError::Transcription("Qwen KV cache capacity overflowed".into()))?;
    let blocks = requested / KV_CACHE_STEP;
    blocks
        .checked_mul(KV_CACHE_STEP)
        .ok_or_else(|| AppError::Transcription("Qwen KV cache capacity overflowed".into()))
}

#[derive(Debug)]
struct QwenAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Array,
    k_norm: Array,
}

impl QwenAttention {
    fn load(store: &mut TensorStore, prefix: &str) -> AppResult<Self> {
        Ok(Self {
            q_proj: Linear::load(store, &format!("{prefix}.q_proj"), false)?,
            k_proj: Linear::load(store, &format!("{prefix}.k_proj"), false)?,
            v_proj: Linear::load(store, &format!("{prefix}.v_proj"), false)?,
            o_proj: Linear::load(store, &format!("{prefix}.o_proj"), false)?,
            q_norm: store.take(&format!("{prefix}.q_norm.weight"))?,
            k_norm: store.take(&format!("{prefix}.k_norm.weight"))?,
        })
    }

    fn forward(
        &self,
        input: &Array,
        cache: &mut KvCache,
        mask: Option<&Array>,
    ) -> AppResult<Array> {
        let [batch, length, _] = shape3(input, "Qwen attention input")?;
        let offset = cache.offset();
        let q = mlx(
            self.q_proj
                .forward(input)?
                .reshape(&[batch, length, TEXT_HEADS, TEXT_HEAD_DIM])
                .and_then(|array| array.transpose_axes(&[0, 2, 1, 3])),
            "reshape Qwen queries",
        )?;
        let k = mlx(
            self.k_proj
                .forward(input)?
                .reshape(&[batch, length, TEXT_KV_HEADS, TEXT_HEAD_DIM])
                .and_then(|array| array.transpose_axes(&[0, 2, 1, 3])),
            "reshape Qwen keys",
        )?;
        let v = mlx(
            self.v_proj
                .forward(input)?
                .reshape(&[batch, length, TEXT_KV_HEADS, TEXT_HEAD_DIM])
                .and_then(|array| array.transpose_axes(&[0, 2, 1, 3])),
            "reshape Qwen values",
        )?;
        let q = mlx(fast::rms_norm(&q, &self.q_norm, 1.0e-6), "Qwen q_norm")?;
        let k = mlx(fast::rms_norm(&k, &self.k_norm, 1.0e-6), "Qwen k_norm")?;
        let q = mlx(
            fast::rope(&q, TEXT_HEAD_DIM, false, 1_000_000.0, 1.0, offset, None),
            "Qwen query RoPE",
        )?;
        let k = mlx(
            fast::rope(&k, TEXT_HEAD_DIM, false, 1_000_000.0, 1.0, offset, None),
            "Qwen key RoPE",
        )?;
        let (k, v) = cache.append(k, v)?;
        let attention_mask = mask.map(ScaledDotProductAttentionMask::Array);
        let attended = mlx(
            fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                (TEXT_HEAD_DIM as f32).sqrt().recip(),
                attention_mask,
                None,
            ),
            "Qwen grouped-query attention",
        )?;
        let attended = mlx(
            attended
                .transpose_axes(&[0, 2, 1, 3])
                .and_then(|array| array.reshape(&[batch, length, TEXT_HEADS * TEXT_HEAD_DIM])),
            "merge Qwen attention heads",
        )?;
        self.o_proj.forward(&attended)
    }
}

#[derive(Debug)]
struct QwenLayer {
    attention: QwenAttention,
    gate: Linear,
    up: Linear,
    down: Linear,
    input_norm: Array,
    post_attention_norm: Array,
}

impl QwenLayer {
    fn load(store: &mut TensorStore, layer: usize) -> AppResult<Self> {
        let prefix = format!("model.language_model.layers.{layer}");
        Ok(Self {
            attention: QwenAttention::load(store, &format!("{prefix}.self_attn"))?,
            gate: Linear::load(store, &format!("{prefix}.mlp.gate_proj"), false)?,
            up: Linear::load(store, &format!("{prefix}.mlp.up_proj"), false)?,
            down: Linear::load(store, &format!("{prefix}.mlp.down_proj"), false)?,
            input_norm: store.take(&format!("{prefix}.input_layernorm.weight"))?,
            post_attention_norm: store
                .take(&format!("{prefix}.post_attention_layernorm.weight"))?,
        })
    }

    fn forward(
        &self,
        input: &Array,
        cache: &mut KvCache,
        mask: Option<&Array>,
    ) -> AppResult<Array> {
        let normalized = mlx(
            fast::rms_norm(input, &self.input_norm, 1.0e-6),
            "Qwen input RMS norm",
        )?;
        let attention = self.attention.forward(&normalized, cache, mask)?;
        let hidden = mlx(input.add(&attention), "Qwen attention residual")?;
        let normalized = mlx(
            fast::rms_norm(&hidden, &self.post_attention_norm, 1.0e-6),
            "Qwen post-attention RMS norm",
        )?;
        let gate = mlx(nn::silu(&self.gate.forward(&normalized)?), "Qwen MLP SiLU")?;
        let gated = mlx(
            gate.multiply(self.up.forward(&normalized)?),
            "Qwen gated MLP",
        )?;
        mlx(hidden.add(self.down.forward(&gated)?), "Qwen MLP residual")
    }
}

#[derive(Debug)]
struct Qwen3 {
    embeddings: Array,
    layers: Vec<QwenLayer>,
    norm: Array,
}

impl Qwen3 {
    fn load(store: &mut TensorStore) -> AppResult<Self> {
        let embeddings = store.take("model.language_model.embed_tokens.weight")?;
        let layers = (0..TEXT_LAYERS)
            .map(|layer| QwenLayer::load(store, layer))
            .collect::<AppResult<Vec<_>>>()?;
        Ok(Self {
            embeddings,
            layers,
            norm: store.take("model.language_model.norm.weight")?,
        })
    }

    fn embed(&self, token_ids: &Array) -> AppResult<Array> {
        mlx(
            self.embeddings.take_axis(token_ids, 0),
            "Qwen token embedding lookup",
        )
    }

    fn forward(&self, input: &Array, caches: &mut [KvCache]) -> AppResult<Array> {
        if caches.len() != self.layers.len() {
            return Err(AppError::Transcription(format!(
                "Qwen requires {} KV caches, got {}",
                self.layers.len(),
                caches.len()
            )));
        }
        let length = input.dim(1);
        let offset = caches.first().map_or(0, KvCache::offset);
        let mask = if length > 1 {
            Some(causal_mask(length, offset)?)
        } else {
            None
        };
        let mut hidden = input.clone();
        for ((layer, cache), index) in self
            .layers
            .iter()
            .zip(caches.iter_mut())
            .zip(0..TEXT_LAYERS)
        {
            if cache.offset() != offset {
                return Err(AppError::Transcription(format!(
                    "KV cache layer {index} has offset {}, expected {offset}",
                    cache.offset()
                )));
            }
            hidden = layer.forward(&hidden, cache, mask.as_ref())?;
        }
        mlx(
            fast::rms_norm(&hidden, &self.norm, 1.0e-6),
            "Qwen final RMS norm",
        )
    }

    fn logits(&self, hidden: &Array) -> AppResult<Array> {
        // MOSS ties lm_head.weight to embed_tokens.weight; the official shard
        // stores only the embedding tensor.
        mlx(
            ops::matmul(hidden, self.embeddings.t()),
            "tied Qwen LM head",
        )
    }

    fn generate(
        &self,
        inputs_embeds: &Array,
        max_new_tokens: usize,
        mut progress: impl FnMut(RuntimeProgress),
        observe: &mut impl FnMut(&'static str, &Array) -> AppResult<()>,
    ) -> AppResult<Vec<u32>> {
        let prompt_length = inputs_embeds.dim(1) as usize;
        if prompt_length == 0 {
            return Err(AppError::Transcription("Qwen prompt is empty".into()));
        }
        if prompt_length > 131_072 {
            return Err(AppError::Transcription(format!(
                "Prompt length {prompt_length} exceeds Qwen's 131072-token context"
            )));
        }

        let mut caches = (0..TEXT_LAYERS)
            .map(|_| KvCache::with_reserve(prompt_length as i32))
            .collect::<AppResult<Vec<_>>>()?;
        let mut last_hidden = None;
        for start in (0..prompt_length).step_by(PREFILL_CHUNK_SIZE) {
            let end = (start + PREFILL_CHUNK_SIZE).min(prompt_length);
            let chunk = mlx(
                inputs_embeds.try_index((.., start as i32..end as i32, ..)),
                "slice Qwen prefill chunk",
            )?;
            let hidden = self.forward(&chunk, &mut caches)?;
            mlx(hidden.eval(), "evaluate Qwen prefill")?;
            eval_caches(&caches)?;
            last_hidden = Some(hidden);
            progress(RuntimeProgress::Prefill {
                completed: end,
                total: prompt_length,
            });
        }

        let hidden = last_hidden.ok_or_else(|| {
            AppError::Transcription("Qwen prefill did not produce hidden states".into())
        })?;
        let last = mlx(hidden.try_index((.., -1, ..)), "select final prompt state")?;
        let first_logits = self.logits(&last)?;
        observe("first_token_logits", &first_logits)?;
        let mut next = greedy(&first_logits)?;
        let mut output = Vec::with_capacity(max_new_tokens.min(4096));

        while output.len() < max_new_tokens {
            output.push(next);
            progress(RuntimeProgress::Token {
                generated: output.len(),
                token: next,
            });
            if next == PAD_TOKEN_ID || next == EOS_TOKEN_ID {
                break;
            }
            if prompt_length + output.len() >= 131_072 {
                break;
            }
            let token = Array::from_slice(&[next], &[1, 1]);
            let embedded = self.embed(&token)?;
            let hidden = self.forward(&embedded, &mut caches)?;
            let logits = self.logits(&hidden)?;
            next = greedy(&logits)?;
            eval_caches(&caches)?;
        }
        Ok(output)
    }
}

fn causal_mask(length: i32, offset: i32) -> AppResult<Array> {
    let keys = mlx(
        Array::arange::<_, i32>(None, offset + length, None),
        "create causal key positions",
    )?;
    let queries = mlx(
        Array::arange::<_, i32>(offset, offset + length, None),
        "create causal query positions",
    )?;
    let keys = mlx(keys.expand_dims_axes(&[0]), "expand causal keys")?;
    let queries = mlx(queries.expand_dims_axes(&[1]), "expand causal queries")?;
    mlx(queries.ge(&keys), "build chunked causal mask")
}

fn greedy(logits: &Array) -> AppResult<u32> {
    let token = mlx(
        ops::indexing::argmax_axis(logits, -1, false),
        "greedy token selection",
    )?;
    token
        .try_item::<u32>()
        .map_err(|error| AppError::Transcription(format!("Read generated token: {error}")))
}

fn eval_caches(caches: &[KvCache]) -> AppResult<()> {
    for cache in caches {
        if let Some(keys) = &cache.keys {
            mlx(keys.eval(), "evaluate KV keys")?;
        }
        if let Some(values) = &cache.values {
            mlx(values.eval(), "evaluate KV values")?;
        }
    }
    Ok(())
}

fn shape3(array: &Array, context: &str) -> AppResult<[i32; 3]> {
    array.shape().try_into().map_err(|_| {
        AppError::Transcription(format!("{context} must be rank 3, got {:?}", array.shape()))
    })
}

fn mlx<T, E: std::fmt::Display>(result: Result<T, E>, context: &str) -> AppResult<T> {
    result.map_err(|error| AppError::Transcription(format!("{context}: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_has_every_official_weight_once() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), 683);
        assert!(!manifest.contains_key("lm_head.weight"));
        assert!(manifest.contains_key("model.whisper_encoder.layers.23.self_attn.q_proj.bias"));
        assert!(!manifest.contains_key("model.whisper_encoder.layers.23.self_attn.k_proj.bias"));
        assert!(manifest.contains_key("model.language_model.layers.27.self_attn.q_norm.weight"));
    }

    #[test]
    fn injects_one_hidden_row_per_audio_placeholder() {
        let token_ids = [7_u32, AUDIO_TOKEN_ID, 8, AUDIO_TOKEN_ID, 9];
        let (indices, mask) =
            audio_replacement_map(&token_ids, 2).expect("audio rows should align");
        assert_eq!(
            indices,
            vec![0, 0, 0, 1, 0],
            "audio placeholders must consume embedding rows in prompt order"
        );
        assert_eq!(mask, vec![false, true, false, true, false]);
        assert!(audio_replacement_map(&token_ids, 1).is_err());
    }

    #[test]
    fn validates_contiguous_mel_shape_and_chunk_token_lengths() {
        let chunk = N_MELS * N_FRAMES;
        assert_eq!(
            validate_mel_batch(&vec![0.0; chunk * 2], &[375, 13])
                .expect("two contiguous chunks should validate"),
            2
        );
        assert!(validate_mel_batch(&vec![0.0; chunk - 1], &[375]).is_err());
        assert!(validate_mel_batch(&vec![0.0; chunk * 2], &[375]).is_err());
        assert!(validate_mel_batch(&vec![0.0; chunk], &[0]).is_err());
        assert!(validate_mel_batch(&vec![0.0; chunk], &[376]).is_err());
    }

    #[test]
    fn kv_cache_reserves_blocks_without_growing_each_decode_token() {
        assert_eq!(next_kv_capacity(0, 0, 250).expect("first block"), 256);
        assert_eq!(next_kv_capacity(250, 256, 1).expect("use spare"), 256);
        assert_eq!(next_kv_capacity(251, 256, 1).expect("use spare"), 256);
        assert_eq!(next_kv_capacity(256, 256, 1).expect("grow block"), 512);

        // Crossing a boundary with spare capacity trims that unused tail and
        // attaches a complete new block after the valid prefix, like mlx-lm.
        assert_eq!(next_kv_capacity(250, 256, 10).expect("cross block"), 506);
    }

    #[test]
    fn kv_cache_reserves_the_full_chunked_prefill_and_decode_room() {
        let reserve = initial_kv_capacity(5_000).expect("reserve full prompt");
        assert_eq!(reserve, 5_376);
        let first = next_kv_capacity(0, 0, PREFILL_CHUNK_SIZE as i32)
            .expect("allocate first prefill chunk")
            .max(reserve);
        assert_eq!(first, reserve);
        let second = next_kv_capacity(4_096, first, 904).expect("append final prefill chunk");
        assert_eq!(second, reserve, "prefill must not copy the cache again");

        // The final prefill offset is 5000, so the reserved decode block also
        // avoids allocation for the first 376 generated tokens.
        assert_eq!(
            next_kv_capacity(5_000, reserve, 1).expect("decode spare"),
            reserve
        );
        assert_eq!(
            next_kv_capacity(5_375, reserve, 1).expect("last spare"),
            reserve
        );
        assert_eq!(
            next_kv_capacity(5_376, reserve, 1).expect("next block"),
            5_632
        );

        let hour_prompt_reserve = initial_kv_capacity(45_000).expect("reserve hour prompt");
        assert_eq!(hour_prompt_reserve, 45_312);
        assert_eq!(
            next_kv_capacity(40_960, hour_prompt_reserve, 4_040)
                .expect("append final hour prefill chunk"),
            hour_prompt_reserve
        );

        assert!(next_kv_capacity(2, 1, 1).is_err());
        assert!(next_kv_capacity(0, 0, 0).is_err());
        assert!(initial_kv_capacity(0).is_err());
    }

    #[cfg(feature = "parity-trace")]
    #[test]
    fn parity_probes_match_the_fixed_linspace_contract() {
        assert_eq!(probe_indices(1), vec![0]);
        assert_eq!(probe_indices(4), vec![0, 1, 2, 3]);
        let probes = probe_indices(1_000_003);
        assert_eq!(probes.len(), 64);
        assert_eq!(probes.first(), Some(&0));
        assert_eq!(probes.last(), Some(&1_000_002));
        assert!(probes.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[cfg(feature = "parity-trace")]
    #[test]
    fn parity_scalar_metadata_uses_numpy_compatible_types() {
        let ids = summarize_i64(&[1, 2, 3], vec![3]);
        assert_eq!(ids.dtype, "int64");
        assert_eq!(ids.shape, vec![3]);
        assert_eq!(ids.sha256.len(), 64);

        let transcript = summarize_transcript("你A🙂");
        assert_eq!(transcript.dtype, "<U3");
        assert!(transcript.shape.is_empty());
        assert_eq!(transcript.sha256.len(), 64);
    }
}
