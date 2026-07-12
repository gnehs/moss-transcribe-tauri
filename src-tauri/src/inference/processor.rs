use std::path::Path;

use tokenizers::{
    decoders::DecoderWrapper, models::ModelWrapper, normalizers::NormalizerWrapper,
    pre_tokenizers::PreTokenizerWrapper, processors::PostProcessorWrapper, DecodeStream, Tokenizer,
};

use crate::error::{AppError, AppResult};

pub const AUDIO_TOKEN_ID: u32 = 151_671;
pub const EOS_TOKEN_ID: u32 = 151_645;
pub const PAD_TOKEN_ID: u32 = 151_643;
pub const DEFAULT_PROMPT: &str = "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。";
const AUDIO_PAD_TOKEN: &str = "<|audio_pad|>";
const TOKENS_PER_SECOND: f32 = 12.5;
const MARKER_INTERVAL_SECONDS: usize = 5;

#[derive(Debug)]
pub struct MossProcessor {
    tokenizer: Tokenizer,
    digit_token_ids: [u32; 10],
}

pub(crate) struct MossDecodeStream<'a> {
    inner: DecodeStream<
        'a,
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
}

impl MossDecodeStream<'_> {
    pub(crate) fn step(&mut self, token: u32) -> AppResult<Option<String>> {
        self.inner.step(token).map_err(|error| {
            AppError::Transcription(format!("Token stream decode failed: {error}"))
        })
    }
}

impl MossProcessor {
    pub fn load(model_dir: &Path) -> AppResult<Self> {
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|error| AppError::Model(format!("Could not load tokenizer: {error}")))?;
        let mut digit_token_ids = [0; 10];
        for (digit, token_id) in digit_token_ids.iter_mut().enumerate() {
            let encoding = tokenizer
                .encode(digit.to_string(), false)
                .map_err(|error| AppError::Model(format!("Could not tokenize digit: {error}")))?;
            if encoding.get_ids().len() != 1 {
                return Err(AppError::Model(format!(
                    "Digit {digit} is not represented by a single token"
                )));
            }
            *token_id = encoding.get_ids()[0];
        }
        Ok(Self {
            tokenizer,
            digit_token_ids,
        })
    }

    pub fn expanded_input_ids(
        &self,
        audio_token_count: usize,
        prompt: Option<&str>,
    ) -> AppResult<Vec<u32>> {
        let prompt = prompt
            .filter(|prompt| !prompt.trim().is_empty())
            .unwrap_or(DEFAULT_PROMPT);
        let template = format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_start|>{AUDIO_PAD_TOKEN}<|audio_end|>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let (before, after) = template.split_once(AUDIO_PAD_TOKEN).ok_or_else(|| {
            AppError::Transcription("Audio placeholder is missing from the chat template".into())
        })?;
        let mut input_ids = self.encode(before)?;
        input_ids.extend(self.audio_span_ids(audio_token_count));
        input_ids.extend(self.encode(after)?);
        Ok(input_ids)
    }

    pub fn decode(&self, tokens: &[u32]) -> AppResult<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|error| AppError::Transcription(format!("Token decode failed: {error}")))
    }

    pub(crate) fn decode_stream(&self) -> MossDecodeStream<'_> {
        MossDecodeStream {
            inner: self.tokenizer.decode_stream(true),
        }
    }

    fn encode(&self, text: &str) -> AppResult<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| AppError::Transcription(format!("Prompt encode failed: {error}")))
    }

    fn audio_span_ids(&self, audio_token_count: usize) -> Vec<u32> {
        let tokens_per_marker = (TOKENS_PER_SECOND * MARKER_INTERVAL_SECONDS as f32) as usize;
        let duration = audio_token_count as f32 / TOKENS_PER_SECOND;
        let mut output = Vec::with_capacity(audio_token_count + duration as usize / 5 * 2);
        let mut consumed = 0usize;
        for second in
            (MARKER_INTERVAL_SECONDS..=duration.floor() as usize).step_by(MARKER_INTERVAL_SECONDS)
        {
            let position = second / MARKER_INTERVAL_SECONDS * tokens_per_marker;
            let segment_length = position.saturating_sub(consumed);
            output.extend(std::iter::repeat_n(AUDIO_TOKEN_ID, segment_length));
            consumed += segment_length;
            output.extend(
                second
                    .to_string()
                    .bytes()
                    .map(|digit| self.digit_token_ids[(digit - b'0') as usize]),
            );
        }
        output.extend(std::iter::repeat_n(
            AUDIO_TOKEN_ID,
            audio_token_count.saturating_sub(consumed),
        ));
        output
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::AddedToken;

    use super::*;

    #[test]
    fn full_chunk_span_preserves_all_audio_tokens_and_adds_markers() {
        let processor = MossProcessor {
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            digit_token_ids: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        };
        let span = processor.audio_span_ids(375);
        assert_eq!(
            span.iter()
                .filter(|token| **token == AUDIO_TOKEN_ID)
                .count(),
            375
        );
        assert_eq!(span.len(), 386);
        assert_eq!(span[62], 20); // `5`
    }

    #[test]
    fn decode_stream_preserves_context_between_tokens() {
        let mut tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
        tokenizer.add_tokens(&[
            AddedToken::from("Hello", false),
            AddedToken::from("world", false),
        ]);
        let hello = tokenizer.token_to_id("Hello").unwrap();
        let world = tokenizer.token_to_id("world").unwrap();
        let processor = MossProcessor {
            tokenizer,
            digit_token_ids: [0; 10],
        };
        let mut stream = processor.decode_stream();

        assert_eq!(stream.step(hello).unwrap().as_deref(), Some("Hello"));
        assert_eq!(stream.step(world).unwrap().as_deref(), Some(" world"));
    }
}
