use serde_json::Value;

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
fn checked_in_fixture_schema_describes_every_required_stage() {
    let schema_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/fixtures/parity/fixture.schema.json"
    );
    let schema: Value = serde_json::from_str(
        &std::fs::read_to_string(schema_path).expect("parity fixture schema must be checked in"),
    )
    .expect("parity fixture schema must be valid JSON");

    assert_eq!(schema["properties"]["schema_version"]["const"], 1);
    for property in [
        "provenance",
        "input",
        "tensors_file",
        "tensors",
        "comparison",
        "decode",
    ] {
        assert!(
            schema["properties"].get(property).is_some(),
            "schema is missing {property}"
        );
    }
    let required_tensors = schema["properties"]["tensors"]["required"]
        .as_array()
        .expect("tensor schema must enumerate required tensor stages");
    for tensor in REQUIRED_TENSORS {
        assert!(
            required_tensors
                .iter()
                .any(|value| value.as_str() == Some(tensor)),
            "schema is missing required tensor stage {tensor}"
        );
    }
    let required_decode = schema["properties"]["decode"]["required"]
        .as_array()
        .expect("decode schema must enumerate exact outputs");
    for field in [
        "expanded_input_ids",
        "greedy_token_ids",
        "first_token_id",
        "first_32_greedy_tokens",
        "final_transcript",
    ] {
        assert!(
            required_decode
                .iter()
                .any(|value| value.as_str() == Some(field)),
            "schema is missing required decode field {field}"
        );
    }
}
