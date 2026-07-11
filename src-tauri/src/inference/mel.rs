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

        let mut mel = vec![0.0f32; N_MELS * N_FRAMES];
        let mut global_max = f32::NEG_INFINITY;
        for mel_index in 0..N_MELS {
            for frame in 0..N_FRAMES {
                let mut value = 0.0f32;
                for frequency in 0..=N_FFT / 2 {
                    value += self.filters[mel_index * (N_FFT / 2 + 1) + frequency]
                        * power[frequency * N_FRAMES + frame];
                }
                let value = value.max(1.0e-10).log10();
                mel[mel_index * N_FRAMES + frame] = value;
                global_max = global_max.max(value);
            }
        }

        let floor = global_max - 8.0;
        for value in &mut mel {
            *value = (value.max(floor) + 4.0) / 4.0;
        }
        Ok(mel)
    }
}

impl Default for WhisperLogMel {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct AudioChunk {
    pub pcm: Vec<f32>,
    pub audio_token_length: usize,
}

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
                audio_token_length: (chunk.len() - 1) / (HOP_LENGTH * 2 * 4) + 1,
            }
        })
        .collect())
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
    fn silence_has_expected_shape_and_floor() {
        let mel = WhisperLogMel::new()
            .extract_chunk(&vec![0.0; CHUNK_SAMPLES])
            .expect("mel extraction should succeed");
        assert_eq!(mel.len(), N_MELS * N_FRAMES);
        assert!(mel.iter().all(|value| (*value + 1.5).abs() < 1.0e-5));
    }
}
