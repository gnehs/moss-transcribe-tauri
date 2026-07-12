//! Opt-in end-to-end test against the real MOSS model and a fixed >30 second WAV.
//!
//! The test is ignored in ordinary CI because it requires the multi-gigabyte
//! model. When explicitly invoked, missing inputs are hard failures.

use std::path::PathBuf;

use moss_transcribe_tauri_lib::{MossTranscriber, TranscribeOptions};

#[test]
#[ignore = "requires MOSS_E2E_MODEL_DIR and MOSS_E2E_WAV"]
fn transcribes_long_audio_with_timestamps_and_speakers() {
    let model_dir = required_path("MOSS_E2E_MODEL_DIR");
    let wav_path = required_path("MOSS_E2E_WAV");
    let mut reader = hound::WavReader::open(&wav_path).expect("MOSS_E2E_WAV must be a valid WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
    assert_eq!(spec.channels, 1, "fixture must be mono");
    assert_eq!(spec.sample_format, hound::SampleFormat::Int);
    assert_eq!(spec.bits_per_sample, 16);
    let pcm = reader
        .samples::<i16>()
        .map(|sample| sample.expect("fixture PCM must be readable") as f32 / 32_768.0)
        .collect::<Vec<_>>();
    assert!(pcm.len() > 30 * 16_000, "fixture must exceed 30 seconds");

    let mut transcriber = MossTranscriber::load(&model_dir).expect("real MOSS model must load");
    let result = transcriber
        .transcribe(&pcm, &TranscribeOptions::default(), |_| {}, |_| {})
        .expect("real MOSS transcription must succeed");

    assert!(
        result.prompt_tokens > 375,
        "multi-chunk prompt was not built"
    );
    assert!(result.generated_tokens > 0);
    assert!(!result.text.trim().is_empty());
    assert!(!result.segments.is_empty());
    assert!(result.segments.iter().all(|segment| {
        segment.start >= 0.0
            && segment.end >= segment.start
            && segment.speaker.starts_with('S')
            && !segment.text.trim().is_empty()
    }));
}

fn required_path(name: &str) -> PathBuf {
    let path = std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} is required when running this ignored E2E test"));
    assert!(path.exists(), "{name} does not exist: {}", path.display());
    path
}
