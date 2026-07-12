use ferrous_opencc::{config::BuiltinConfig, OpenCC};

use crate::{
    error::{AppError, AppResult},
    models::{TranscriptResult, TranscriptStreamEvent},
};

pub struct TraditionalConverter {
    converter: OpenCC,
}

impl TraditionalConverter {
    pub fn new() -> AppResult<Self> {
        let converter = OpenCC::from_config(BuiltinConfig::S2twp)
            .map_err(|error| AppError::Transcription(format!("Could not load OpenCC: {error}")))?;
        Ok(Self { converter })
    }

    pub fn convert_result(&self, result: &mut TranscriptResult) {
        result.text = self.converter.convert(&result.text);
        for segment in &mut result.segments {
            segment.text = self.converter.convert(&segment.text);
        }
    }

    pub fn convert_stream(&self, event: &mut TranscriptStreamEvent) {
        event.text = self.converter.convert(&event.text);
        for segment in &mut event.segments {
            segment.text = self.converter.convert(&segment.text);
        }
    }
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

        TraditionalConverter::new()
            .expect("OpenCC should initialize")
            .convert_result(&mut result);

        assert_eq!(result.text, "[0][S01]開放中文轉換和伺服器[1]");
        assert_eq!(result.segments[0].text, "開放中文轉換和伺服器");
    }

    #[test]
    fn converts_stream_text_and_segments_to_taiwan_traditional() {
        let mut event = TranscriptStreamEvent {
            task_id: "task-1".into(),
            text: "[0][S01]台风登陆[1]".into(),
            segment_offset: 0,
            segments: vec![TranscriptSegment {
                start: 0.0,
                end: 1.0,
                speaker: "S01".into(),
                text: "台风登陆".into(),
            }],
            generated_tokens: 8,
        };

        TraditionalConverter::new()
            .expect("OpenCC should initialize")
            .convert_stream(&mut event);

        assert_eq!(event.text, "[0][S01]颱風登陸[1]");
        assert_eq!(event.segments[0].text, "颱風登陸");
    }
}
