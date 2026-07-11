//! Opt-in integration gate for Python mlx-audio versus native MLX-Rust traces.
//!
//! The native trace is generated in-process from the real model and WAV. Missing
//! inputs are errors when explicitly invoked, never a fake pass.

use std::path::{Path, PathBuf};

use moss_transcribe_tauri_lib::generate_native_parity_trace;
use serde_json::Value;

const MODEL_REVISION: &str = "d7231bbae2587a4af278735eb765b318c4f64edd";
const MLX_AUDIO_REVISION: &str = "64e8416c303fb3b3463dab8eb4ebd78c55a87c1a";
const MLX_RS_REVISION: &str = "f4aa309c79b6be35255ca7d34157dfc10d9ed4c9";
const REQUIRED_TENSORS: [&str; 8] = [
    "log_mel",
    "whisper_encoder",
    "vq_adaptor",
    "expanded_input_ids",
    "fused_embeddings",
    "first_token_logits",
    "greedy_token_ids",
    "final_transcript",
];

#[test]
#[ignore = "requires a generated reference fixture, MOSS_PARITY_MODEL_DIR, and MOSS_PARITY_WAV"]
fn pinned_mlx_audio_and_native_trace_agree() {
    let fixture_dir = std::env::var_os("MOSS_PARITY_FIXTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../tests/fixtures/parity/generated/long-audio")
        });
    let metadata = read_json(&fixture_dir.join("metadata.json"), "reference metadata");
    assert_fixture_contract(&fixture_dir, &metadata);

    let model_dir = required_path("MOSS_PARITY_MODEL_DIR");
    let wav_path = required_path("MOSS_PARITY_WAV");
    let pcm = read_canonical_wav(&wav_path);
    assert!(pcm.len() > 30 * 16_000, "parity WAV must exceed 30 seconds");
    let prompt = metadata["input"]["prompt"].as_str();
    let basename = wav_path
        .file_name()
        .and_then(|value| value.to_str())
        .expect("MOSS_PARITY_WAV basename must be valid UTF-8");
    let native = generate_native_parity_trace(&model_dir, &pcm, basename, prompt)
        .expect("native MOSS parity trace generation must succeed");
    let native = serde_json::to_value(native).expect("native trace must serialize");
    assert_native_trace(&metadata, &native);
}

fn required_path(name: &str) -> PathBuf {
    let path = std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} is required when running this ignored parity gate"));
    assert!(path.exists(), "{name} does not exist: {}", path.display());
    path
}

fn read_canonical_wav(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("MOSS_PARITY_WAV must be a valid WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "parity WAV must be 16 kHz");
    assert_eq!(spec.channels, 1, "parity WAV must be mono");
    assert_eq!(spec.sample_format, hound::SampleFormat::Int);
    assert_eq!(spec.bits_per_sample, 16, "parity WAV must be PCM16");
    reader
        .samples::<i16>()
        .map(|sample| sample.expect("parity WAV PCM must be readable") as f32 / 32_768.0)
        .collect()
}

fn read_json(path: &Path, label: &str) -> Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{label} is required at {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{label} must be valid JSON at {}: {error}", path.display()))
}

fn assert_fixture_contract(fixture_dir: &Path, metadata: &Value) {
    assert_eq!(metadata["schema_version"], 1, "unsupported fixture schema");
    assert_eq!(metadata["provenance"]["model_revision"], MODEL_REVISION);
    assert_eq!(
        metadata["provenance"]["mlx_audio_revision"],
        MLX_AUDIO_REVISION
    );
    assert_eq!(metadata["provenance"]["mlx_rs_revision"], MLX_RS_REVISION);
    assert_eq!(metadata["input"]["sample_rate"], 16_000);
    assert!(
        metadata["input"]["duration_seconds"]
            .as_f64()
            .unwrap_or_default()
            > 30.0,
        "fixture must represent audio longer than 30 seconds"
    );
    let tensors_file = metadata["tensors_file"]
        .as_str()
        .expect("tensors_file must be a string");
    assert!(
        fixture_dir.join(tensors_file).is_file(),
        "NPZ tensor evidence is missing"
    );
    for name in REQUIRED_TENSORS {
        let tensor = &metadata["tensors"][name];
        assert!(
            tensor.is_object(),
            "fixture omitted tensor metadata for {name}"
        );
        assert!(tensor["dtype"].is_string(), "{name} is missing dtype");
        assert!(tensor["shape"].is_array(), "{name} is missing shape");
        assert!(tensor["sha256"].is_string(), "{name} is missing SHA-256");
    }
    assert_eq!(
        metadata["decode"]["first_32_greedy_tokens"]
            .as_array()
            .map(Vec::len),
        Some(32)
    );
    assert!(
        metadata["decode"]["greedy_token_ids"]
            .as_array()
            .is_some_and(|tokens| tokens.len() >= 32),
        "fixture must contain the complete greedy decode"
    );
    assert!(metadata["decode"]["final_transcript"].is_string());
}

fn assert_native_trace(reference: &Value, native: &Value) {
    // Runtime trace hooks should emit tensor metadata/probes using the same shape
    // as metadata.json. Exact fields catch tokenization and decoding drift; float
    // probe values use the tolerance captured with the reference fixture.
    for name in REQUIRED_TENSORS {
        let expected = &reference["tensors"][name];
        let actual = &native["tensors"][name];
        assert_eq!(actual["shape"], expected["shape"], "{name} shape differs");
        assert_eq!(actual["dtype"], expected["dtype"], "{name} dtype differs");
    }
    assert_eq!(native["schema_version"], reference["schema_version"]);
    assert_eq!(native["provenance"]["model_revision"], MODEL_REVISION);
    assert_eq!(
        native["provenance"]["mlx_audio_revision"],
        MLX_AUDIO_REVISION
    );
    assert_eq!(native["provenance"]["mlx_rs_revision"], MLX_RS_REVISION);
    assert_eq!(native["input"]["sha256"], reference["input"]["sha256"]);
    assert_eq!(
        native["decode"]["expanded_input_ids"], reference["decode"]["expanded_input_ids"],
        "expanded_input_ids differs exactly"
    );
    assert_eq!(
        native["decode"]["first_32_greedy_tokens"], reference["decode"]["first_32_greedy_tokens"],
        "first_32_greedy_tokens differs exactly"
    );
    assert_eq!(
        native["decode"]["greedy_token_ids"], reference["decode"]["greedy_token_ids"],
        "complete greedy_token_ids differs exactly"
    );
    assert_eq!(
        native["decode"]["final_transcript"], reference["decode"]["final_transcript"],
        "final_transcript differs exactly"
    );
    for name in [
        "log_mel",
        "whisper_encoder",
        "vq_adaptor",
        "fused_embeddings",
        "first_token_logits",
    ] {
        assert_float_probes(name, reference, native);
    }
}

fn assert_float_probes(name: &str, reference: &Value, native: &Value) {
    let expected = &reference["tensors"][name];
    let actual = &native["tensors"][name];
    assert_eq!(
        actual["sample_indices"], expected["sample_indices"],
        "{name} probe layout differs"
    );
    let atol = reference["comparison"]["float_tolerances"][name]["atol"]
        .as_f64()
        .expect("missing atol");
    let rtol = reference["comparison"]["float_tolerances"][name]["rtol"]
        .as_f64()
        .expect("missing rtol");
    let expected_values = expected["sample_values"]
        .as_array()
        .expect("missing reference probes");
    let actual_values = actual["sample_values"]
        .as_array()
        .expect("missing native probes");
    assert_eq!(
        actual_values.len(),
        expected_values.len(),
        "{name} probe count differs"
    );
    for (index, (expected, actual)) in expected_values.iter().zip(actual_values).enumerate() {
        let expected = expected.as_f64().expect("reference probe must be numeric");
        let actual = actual.as_f64().expect("native probe must be numeric");
        let limit = atol + rtol * expected.abs();
        assert!(
            (actual - expected).abs() <= limit,
            "{name} probe {index}: actual {actual}, expected {expected}, tolerance {limit}"
        );
    }
}
