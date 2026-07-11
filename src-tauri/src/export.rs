use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

use uuid::Uuid;

use crate::{
    error::{AppError, AppResult},
    models::{ExportOptions, OutputPaths, TranscriptResult},
};

pub fn export_result(
    audio_path: &Path,
    options: &ExportOptions,
    result: &TranscriptResult,
) -> AppResult<OutputPaths> {
    let output_dir = resolve_output_dir(audio_path, options.output_dir.as_deref())?;
    fs::create_dir_all(&output_dir)?;
    let stem = safe_stem(audio_path);

    let mut rendered = Vec::new();
    if options.write_txt {
        rendered.push(("txt", render_txt(result).into_bytes()));
    }
    if options.write_json {
        let json = serde_json::to_vec_pretty(result)
            .map_err(|error| AppError::Export(error.to_string()))?;
        rendered.push(("json", json));
    }
    if options.write_srt {
        rendered.push(("srt", render_srt(result).into_bytes()));
    }

    let mut published = Vec::<(String, PathBuf)>::new();
    for (extension, bytes) in rendered {
        let destination = output_dir.join(format!("{stem}.{extension}"));
        write_atomic(&destination, &bytes)?;
        published.push((extension.to_string(), destination));
    }

    let find = |extension: &str| {
        published
            .iter()
            .find(|(candidate, _)| candidate == extension)
            .map(|(_, path)| path.to_string_lossy().into_owned())
    };

    Ok(OutputPaths {
        txt_path: find("txt"),
        json_path: find("json"),
        srt_path: find("srt"),
    })
}

fn resolve_output_dir(audio_path: &Path, requested: Option<&str>) -> AppResult<PathBuf> {
    if let Some(requested) = requested.filter(|path| !path.trim().is_empty()) {
        return Ok(PathBuf::from(requested));
    }

    audio_path
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| AppError::Export("Could not resolve the input file directory".into()))
}

fn safe_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("transcript")
        .to_string()
}

fn render_txt(result: &TranscriptResult) -> String {
    if result.segments.is_empty() {
        return result.text.trim().to_string();
    }

    result
        .segments
        .iter()
        .map(|segment| {
            format!(
                "[{:.2}–{:.2}] [{}] {}",
                segment.start, segment.end, segment.speaker, segment.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_srt(result: &TranscriptResult) -> String {
    result
        .segments
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            format!(
                "{}\n{} --> {}\n[{}] {}\n",
                index + 1,
                format_srt_time(segment.start),
                format_srt_time(segment.end),
                segment.speaker,
                segment.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_srt_time(seconds: f32) -> String {
    let total_ms = (seconds.max(0.0) as f64 * 1000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let minutes = total_ms % 3_600_000 / 60_000;
    let secs = total_ms % 60_000 / 1_000;
    let millis = total_ms % 1_000;
    format!("{hours:02}:{minutes:02}:{secs:02},{millis:03}")
}

fn write_atomic(destination: &Path, bytes: &[u8]) -> AppResult<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| AppError::Export("Output path has no parent directory".into()))?;
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("transcript");
    let temporary = parent.join(format!(".{file_name}.tmp-{}", Uuid::new_v4()));

    let result = (|| -> AppResult<()> {
        let mut file = File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, destination)?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TranscriptSegment;

    fn fixture() -> TranscriptResult {
        TranscriptResult {
            text: "[0.48][S01]Hello[1.66]".into(),
            segments: vec![TranscriptSegment {
                start: 0.48,
                end: 1.66,
                speaker: "S01".into(),
                text: "Hello".into(),
            }],
            prompt_tokens: 12,
            generated_tokens: 8,
            truncated: false,
        }
    }

    #[test]
    fn renders_speaker_in_srt() {
        let srt = render_srt(&fixture());
        assert!(srt.contains("00:00:00,480 --> 00:00:01,660"));
        assert!(srt.contains("[S01] Hello"));
    }
}
