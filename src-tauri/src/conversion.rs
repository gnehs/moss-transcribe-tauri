use ferrous_opencc::{config::BuiltinConfig, OpenCC};

use crate::{
    error::{AppError, AppResult},
    models::TranscriptResult,
};

/// Convert transcript text to Taiwan Traditional Chinese, including common
/// Taiwan vocabulary such as `服务器` -> `伺服器`.
pub fn simplified_to_traditional(result: &mut TranscriptResult) -> AppResult<()> {
    let converter = OpenCC::from_config(BuiltinConfig::S2twp)
        .map_err(|error| AppError::Transcription(format!("Could not load OpenCC: {error}")))?;

    result.text = converter.convert(&result.text);
    for segment in &mut result.segments {
        segment.text = converter.convert(&segment.text);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TranscriptSegment;

    #[test]
    fn converts_result_text_and_segments_to_taiwan_traditional() {
        let mut result = TranscriptResult {
            text: "[0][S01]开放中文转换和服务器[1]".into(),
            segments: vec![TranscriptSegment {
                start: 0.0,
                end: 1.0,
                speaker: "S01".into(),
                text: "开放中文转换和服务器".into(),
            }],
            prompt_tokens: 0,
            generated_tokens: 0,
            truncated: false,
        };

        simplified_to_traditional(&mut result).expect("OpenCC should initialize");

        assert_eq!(result.text, "[0][S01]開放中文轉換和伺服器[1]");
        assert_eq!(result.segments[0].text, "開放中文轉換和伺服器");
    }
}
