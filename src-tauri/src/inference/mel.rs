use std::sync::Arc;

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::error::{AppError, AppResult};

pub const CHUNK_SAMPLES: usize = 30 * 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 80;
pub const N_FRAMES: usize = 3000;

#[derive(Clone)]
pub struct WhisperLogMel {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    filters: Vec<f32>,
}

impl std::fmt::Debug for WhisperLogMel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WhisperLogMel")
            .finish_non_exhaustive()
    }
}

impl WhisperLogMel {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let window = (0..N_FFT)
            .map(|index| {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / N_FFT as f32).cos()
            })
            .collect();
        Self {
            fft,
            window,
            filters: create_slaney_filters(),
        }
    }

    /// Returns a mel-major `[80, 3000]` Whisper feature matrix.
    pub fn extract_chunk(&self, input: &[f32]) -> AppResult<Vec<f32>> {
        let mut mel = Vec::with_capacity(N_MELS * N_FRAMES);
        self.extract_chunk_into(input, &mut mel)?;
        Ok(mel)
    }

    /// Extracts every 30-second chunk into one contiguous
    /// `[chunks, 80, 3000]` buffer.
    ///
    /// Complete chunks borrow directly from `pcm`; only the final short chunk
    /// is copied into a zero-padded scratch buffer.
    pub fn extract_audio(&self, pcm: &[f32]) -> AppResult<MelBatch> {
        if pcm.is_empty() {
            return Err(AppError::Transcription("Audio is empty".into()));
        }

        let chunk_count = pcm.len().div_ceil(CHUNK_SAMPLES);
        let feature_count = chunk_count
            .checked_mul(N_MELS * N_FRAMES)
            .ok_or_else(|| AppError::Transcription("Mel feature buffer is too large".into()))?;
        let mut features = Vec::with_capacity(feature_count);
        let mut audio_feature_lengths = Vec::with_capacity(chunk_count);

        for chunk in pcm.chunks(CHUNK_SAMPLES) {
            audio_feature_lengths.push(audio_token_length(chunk.len()));
            if chunk.len() == CHUNK_SAMPLES {
                self.extract_chunk_into(chunk, &mut features)?;
            } else {
                let mut padded = Vec::with_capacity(CHUNK_SAMPLES);
                padded.extend_from_slice(chunk);
                padded.resize(CHUNK_SAMPLES, 0.0);
                self.extract_chunk_into(&padded, &mut features)?;
            }
        }

        debug_assert_eq!(features.len(), feature_count);
        Ok(MelBatch {
            features,
            audio_feature_lengths,
        })
    }

    fn extract_chunk_into(&self, input: &[f32], mel: &mut Vec<f32>) -> AppResult<()> {
        if input.len() != CHUNK_SAMPLES {
            return Err(AppError::Transcription(format!(
                "Expected a padded 30 second chunk, got {} samples",
                input.len()
            )));
        }

        let padded = reflect_pad(input, N_FFT / 2);
        let mut power = vec![0.0f32; (N_FFT / 2 + 1) * N_FRAMES];
        let mut spectrum = vec![Complex32::default(); N_FFT];
        for frame in 0..N_FRAMES {
            let offset = frame * HOP_LENGTH;
            for index in 0..N_FFT {
                spectrum[index] = Complex32::new(padded[offset + index] * self.window[index], 0.0);
            }
            self.fft.process(&mut spectrum);
            for frequency in 0..=N_FFT / 2 {
                power[frequency * N_FRAMES + frame] = spectrum[frequency].norm_sqr();
            }
        }

        let mel_start = mel.len();
        mel.resize(mel_start + N_MELS * N_FRAMES, 0.0);
        let mut global_max = f32::NEG_INFINITY;
        for mel_index in 0..N_MELS {
            for frame in 0..N_FRAMES {
                let mut value = 0.0f32;
                for frequency in 0..=N_FFT / 2 {
                    value += self.filters[mel_index * (N_FFT / 2 + 1) + frequency]
                        * power[frequency * N_FRAMES + frame];
                }
                let value = value.max(1.0e-10).log10();
                mel[mel_start + mel_index * N_FRAMES + frame] = value;
                global_max = global_max.max(value);
            }
        }

        let floor = global_max - 8.0;
        for value in &mut mel[mel_start..] {
            *value = (value.max(floor) + 4.0) / 4.0;
        }
        Ok(())
    }
}

impl Default for WhisperLogMel {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct MelBatch {
    /// Row-major `[chunk_count, N_MELS, N_FRAMES]` features.
    pub features: Vec<f32>,
    /// Number of merged audio tokens retained from each padded chunk.
    pub audio_feature_lengths: Vec<usize>,
}

impl MelBatch {
    pub fn chunk_count(&self) -> usize {
        self.audio_feature_lengths.len()
    }

    pub fn audio_token_count(&self) -> usize {
        self.audio_feature_lengths.iter().sum()
    }
}

#[derive(Debug)]
pub struct AudioChunk {
    pub pcm: Vec<f32>,
    pub audio_token_length: usize,
}

/// Compatibility wrapper for older callers. Prefer
/// [`WhisperLogMel::extract_audio`] to avoid retaining padded PCM chunks.
pub fn chunk_audio(pcm: &[f32]) -> AppResult<Vec<AudioChunk>> {
    if pcm.is_empty() {
        return Err(AppError::Transcription("Audio is empty".into()));
    }
    Ok(pcm
        .chunks(CHUNK_SAMPLES)
        .map(|chunk| {
            let mut padded = chunk.to_vec();
            padded.resize(CHUNK_SAMPLES, 0.0);
            AudioChunk {
                pcm: padded,
                audio_token_length: audio_token_length(chunk.len()),
            }
        })
        .collect())
}

fn audio_token_length(sample_count: usize) -> usize {
    sample_count.div_ceil(HOP_LENGTH * 2 * 4)
}

fn reflect_pad(input: &[f32], padding: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len() + padding * 2);
    for index in (1..=padding).rev() {
        output.push(input[index]);
    }
    output.extend_from_slice(input);
    for index in 0..padding {
        output.push(input[input.len() - 2 - index]);
    }
    output
}

fn create_slaney_filters() -> Vec<f32> {
    let frequencies = N_FFT / 2 + 1;
    let hz_to_mel = |frequency: f64| {
        let linear_step = 200.0 / 3.0;
        let min_log_mel = 1000.0 / linear_step;
        if frequency < 1000.0 {
            frequency / linear_step
        } else {
            min_log_mel + (frequency / 1000.0).ln() / (6.4_f64.ln() / 27.0)
        }
    };
    let mel_to_hz = |mel: f64| {
        let linear_step = 200.0 / 3.0;
        let min_log_mel = 1000.0 / linear_step;
        if mel < min_log_mel {
            mel * linear_step
        } else {
            1000.0 * ((6.4_f64.ln() / 27.0) * (mel - min_log_mel)).exp()
        }
    };
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(8000.0);
    let edges = (0..N_MELS + 2)
        .map(|index| mel_to_hz(mel_min + (mel_max - mel_min) * index as f64 / (N_MELS + 1) as f64))
        .collect::<Vec<_>>();
    let mut filters = vec![0.0f32; N_MELS * frequencies];
    for mel in 0..N_MELS {
        let lower_width = edges[mel + 1] - edges[mel];
        let upper_width = edges[mel + 2] - edges[mel + 1];
        let normalization = 2.0 / (edges[mel + 2] - edges[mel]);
        for bin in 0..frequencies {
            let frequency = bin as f64 * 16_000.0 / N_FFT as f64;
            let lower = (frequency - edges[mel]) / lower_width;
            let upper = (edges[mel + 2] - frequency) / upper_width;
            filters[mel * frequencies + bin] =
                lower.min(upper).max(0.0) as f32 * normalization as f32;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_long_audio_without_overlap() {
        let pcm = vec![0.0; CHUNK_SAMPLES + 16_001];
        let chunks = chunk_audio(&pcm).expect("chunking fixture should succeed");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].audio_token_length, 375);
        assert_eq!(chunks[1].audio_token_length, 13);
        assert!(chunks.iter().all(|chunk| chunk.pcm.len() == CHUNK_SAMPLES));
    }

    #[test]
    fn extracts_contiguous_chunks_with_tail_padding_and_token_lengths() {
        let mut pcm = vec![0.0; CHUNK_SAMPLES + 16_001];
        *pcm.last_mut().expect("fixture is non-empty") = 0.5;
        let extractor = WhisperLogMel::new();
        let batch = extractor
            .extract_audio(&pcm)
            .expect("batched mel extraction should succeed");

        assert_eq!(batch.chunk_count(), 2);
        assert_eq!(batch.features.len(), 2 * N_MELS * N_FRAMES);
        assert_eq!(batch.audio_feature_lengths, vec![375, 13]);
        assert_eq!(batch.audio_token_count(), 388);

        let mut padded_tail = pcm[CHUNK_SAMPLES..].to_vec();
        padded_tail.resize(CHUNK_SAMPLES, 0.0);
        let expected_tail = extractor
            .extract_chunk(&padded_tail)
            .expect("explicitly padded tail should succeed");
        assert_eq!(&batch.features[N_MELS * N_FRAMES..], expected_tail);
    }

    #[test]
    fn exact_chunk_does_not_add_a_padding_chunk() {
        let batch = WhisperLogMel::new()
            .extract_audio(&vec![0.0; CHUNK_SAMPLES])
            .expect("exact chunk should succeed");
        assert_eq!(batch.chunk_count(), 1);
        assert_eq!(batch.features.len(), N_MELS * N_FRAMES);
        assert_eq!(batch.audio_feature_lengths, vec![375]);
    }

    #[test]
    fn silence_has_expected_shape_and_floor() {
        let mel = WhisperLogMel::new()
            .extract_chunk(&vec![0.0; CHUNK_SAMPLES])
            .expect("mel extraction should succeed");
        assert_eq!(mel.len(), N_MELS * N_FRAMES);
        assert!(mel.iter().all(|value| (*value + 1.5).abs() < 1.0e-5));
    }
}
