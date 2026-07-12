use crate::models::TranscriptSegment;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    SeekStart,
    ReadStart,
    ExpectSpeakerOpen,
    ReadSpeaker,
    ReadText,
    ReadEnd,
    AfterEnd,
}

/// Streaming, single-pass parser for `[start][Sxx]text[end]` output.
///
/// An apparent end timestamp is accepted only when the next segment starts, or
/// when `close` is called. This preserves numeric bracket text such as `[123]`.
#[derive(Debug, Clone)]
pub struct TranscriptStreamParser {
    state: State,
    token: String,
    text: String,
    pending_after_end: String,
    start: Option<f32>,
    end: Option<f32>,
    end_token: String,
    speaker: Option<String>,
}

impl Default for TranscriptStreamParser {
    fn default() -> Self {
        Self {
            state: State::SeekStart,
            token: String::new(),
            text: String::new(),
            pending_after_end: String::new(),
            start: None,
            end: None,
            end_token: String::new(),
            speaker: None,
        }
    }
}

impl TranscriptStreamParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, chunk: &str) -> Vec<TranscriptSegment> {
        let mut segments = Vec::new();
        for character in chunk.chars() {
            match self.state {
                State::SeekStart => self.seek_start(character),
                State::ReadStart => self.read_start(character),
                State::ExpectSpeakerOpen => self.expect_speaker_open(character),
                State::ReadSpeaker => self.read_speaker(character),
                State::ReadText => self.read_text(character),
                State::ReadEnd => self.read_end(character),
                State::AfterEnd => self.after_end(character, &mut segments),
            }
        }
        segments
    }

    pub fn close(&mut self) -> Vec<TranscriptSegment> {
        let mut segments = Vec::new();
        if self.state == State::AfterEnd {
            self.emit_segment(&mut segments);
        }
        self.reset();
        segments
    }

    /// Returns the currently complete trailing segment without consuming the
    /// parser state. Segments already returned by `feed` are not repeated.
    pub fn snapshot(&self) -> Vec<TranscriptSegment> {
        let mut snapshot = self.clone();
        snapshot.close()
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn seek_start(&mut self, character: char) {
        if character == '[' {
            self.token.clear();
            self.state = State::ReadStart;
        }
    }

    fn read_start(&mut self, character: char) {
        if character == ']' {
            if let Some(start) = parse_timestamp(&self.token) {
                self.start = Some(start);
                self.token.clear();
                self.state = State::ExpectSpeakerOpen;
            } else {
                self.reset();
            }
            return;
        }

        if is_timestamp_char(character) && self.token.len() < 32 {
            self.token.push(character);
            return;
        }

        self.reset();
        if character == '[' {
            self.state = State::ReadStart;
        }
    }

    fn expect_speaker_open(&mut self, character: char) {
        if character == '[' {
            self.token.clear();
            self.state = State::ReadSpeaker;
        } else if !character.is_whitespace() {
            self.reset();
        }
    }

    fn read_speaker(&mut self, character: char) {
        if character == ']' {
            if is_valid_speaker(&self.token) {
                self.speaker = Some(std::mem::take(&mut self.token));
                self.text.clear();
                self.state = State::ReadText;
            } else {
                self.reset();
            }
            return;
        }

        if is_speaker_char(character) && self.token.len() < 16 {
            self.token.push(character);
            return;
        }

        self.reset();
        if character == '[' {
            self.state = State::ReadStart;
        }
    }

    fn read_text(&mut self, character: char) {
        if character == '[' {
            self.token.clear();
            self.state = State::ReadEnd;
        } else {
            self.text.push(character);
        }
    }

    fn read_end(&mut self, character: char) {
        if character == ']' {
            let end = parse_timestamp(&self.token);
            if end.is_some_and(|end| self.start.is_some_and(|start| end >= start)) {
                self.end = end;
                self.end_token.clone_from(&self.token);
                self.pending_after_end.clear();
                self.state = State::AfterEnd;
            } else {
                self.restore_end_candidate(Some(character));
            }
            self.token.clear();
            return;
        }

        if is_timestamp_char(character) && self.token.len() < 32 {
            self.token.push(character);
            return;
        }

        self.restore_end_candidate(Some(character));
        self.token.clear();
    }

    fn restore_end_candidate(&mut self, trailing: Option<char>) {
        self.text.push('[');
        self.text.push_str(&self.token);
        if let Some(trailing) = trailing {
            self.text.push(trailing);
        }
        self.state = State::ReadText;
    }

    fn after_end(&mut self, character: char, segments: &mut Vec<TranscriptSegment>) {
        if character == '[' {
            self.emit_segment(segments);
            self.token.clear();
            self.state = State::ReadStart;
            return;
        }

        if character.is_whitespace() {
            self.pending_after_end.push(character);
            return;
        }

        self.text.push('[');
        self.text.push_str(&self.end_token);
        self.text.push(']');
        self.text.push_str(&self.pending_after_end);
        self.text.push(character);
        self.pending_after_end.clear();
        self.end = None;
        self.end_token.clear();
        self.state = State::ReadText;
    }

    fn emit_segment(&mut self, segments: &mut Vec<TranscriptSegment>) {
        if let (Some(start), Some(end), Some(speaker)) = (self.start, self.end, self.speaker.take())
        {
            let text = self.text.trim();
            if !text.is_empty() {
                segments.push(TranscriptSegment {
                    start,
                    end,
                    speaker,
                    text: text.to_string(),
                });
            }
        }
        self.reset();
    }
}

pub fn parse_transcript(raw: &str) -> Vec<TranscriptSegment> {
    let mut parser = TranscriptStreamParser::new();
    let mut segments = parser.feed(raw);
    segments.extend(parser.close());
    segments
}

fn parse_timestamp(value: &str) -> Option<f32> {
    if value.is_empty()
        || value.chars().filter(|character| *character == '.').count() > 1
        || !value.chars().any(|character| character.is_ascii_digit())
    {
        return None;
    }
    let parsed = value.parse::<f32>().ok()?;
    (parsed.is_finite() && parsed >= 0.0).then_some(parsed)
}

fn is_valid_speaker(value: &str) -> bool {
    value.len() >= 2
        && value.starts_with('S')
        && value[1..]
            .chars()
            .all(|character| character.is_ascii_digit())
}

fn is_timestamp_char(character: char) -> bool {
    character.is_ascii_digit() || character == '.'
}

fn is_speaker_char(character: char) -> bool {
    character == 'S' || character.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_adjacent_moss_segments_across_chunks() {
        let mut parser = TranscriptStreamParser::new();
        let mut segments = parser.feed("[0.48][S01]Welcome everyone[1.");
        segments.extend(parser.feed("66][12.26][S02]Pipeline ready[13.81]"));
        segments.extend(parser.close());

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, "S01");
        assert_eq!(segments[1].text, "Pipeline ready");
    }

    #[test]
    fn preserves_numeric_brackets_in_text() {
        let segments = parse_transcript("[0][S01]version [123] is literal[2]");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "version [123] is literal");
    }

    #[test]
    fn skips_truncated_or_reversed_final_segments() {
        let segments = parse_transcript("[2.0][S01]bad[1.0]");
        assert!(segments.is_empty());

        let segments = parse_transcript("[3.0][S02]good[4.0][5.0][S03]truncated");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "good");
    }

    #[test]
    fn snapshot_does_not_consume_or_duplicate_stream_state() {
        let mut parser = TranscriptStreamParser::new();
        assert!(parser.feed("[0][S01]first[1]").is_empty());
        assert_eq!(parser.snapshot()[0].text, "first");
        assert_eq!(parser.snapshot()[0].text, "first");

        let completed = parser.feed("[2][S02]second[3]");
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].text, "first");
        assert_eq!(parser.snapshot()[0].text, "second");
    }
}
