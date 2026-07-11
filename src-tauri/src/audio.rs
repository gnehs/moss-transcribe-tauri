use std::{
    ffi::OsString,
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    process::Command,
};

use uuid::Uuid;

use crate::{
    error::{AppError, AppResult},
    models::FfmpegStatus,
};

pub const SAMPLE_RATE: u32 = 16_000;
const MIN_SAMPLES: u32 = 320;
const SUPPORTED_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "m4a", "aac", "flac", "ogg", "mp4", "mov", "mkv", "webm",
];

#[cfg(target_os = "macos")]
const MACOS_FFMPEG_PATHS: &[&str] = &[
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/opt/local/bin/ffmpeg",
];

#[derive(Debug)]
pub struct DecodedAudio {
    pub pcm: Vec<f32>,
    pub duration_ms: u64,
}

struct TemporaryWav(PathBuf);

impl Drop for TemporaryWav {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub fn ffmpeg_status() -> FfmpegStatus {
    match resolve_ffmpeg() {
        Some(ffmpeg) => FfmpegStatus {
            available: true,
            version: Some(ffmpeg.version),
            path: Some(ffmpeg.program.to_string_lossy().into_owned()),
        },
        None => FfmpegStatus {
            available: false,
            version: None,
            path: None,
        },
    }
}

pub fn decode_to_pcm(input: &Path) -> AppResult<DecodedAudio> {
    validate_input(input)?;
    let ffmpeg = resolve_ffmpeg().ok_or_else(|| {
        AppError::Audio("找不到 FFmpeg。請安裝 FFmpeg 後在設定中重新檢查。".to_string())
    })?;
    let output_path = temporary_wav_path()?;
    let temporary = TemporaryWav(output_path.clone());

    let output = Command::new(&ffmpeg.program)
        .args(normalize_args(input, &output_path))
        .output()
        .map_err(|error| {
            if error.kind() == ErrorKind::NotFound {
                AppError::Audio("FFmpeg is unavailable".into())
            } else {
                AppError::Audio(format!("Could not run FFmpeg: {error}"))
            }
        })?;

    if !output.status.success() {
        return Err(AppError::Audio(format!(
            "FFmpeg failed: {}",
            summarize_stderr(&output.stderr, input, &output_path)
        )));
    }

    let mut reader = hound::WavReader::open(&temporary.0)
        .map_err(|error| AppError::Audio(format!("Could not read normalized WAV: {error}")))?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.sample_rate != SAMPLE_RATE
        || spec.sample_format != hound::SampleFormat::Int
        || spec.bits_per_sample != 16
    {
        return Err(AppError::Audio(
            "FFmpeg returned an unsupported WAV format".into(),
        ));
    }

    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.map(|value| value as f32 / 32768.0))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| AppError::Audio(format!("Invalid PCM data: {error}")))?;
    if samples.len() < MIN_SAMPLES as usize {
        return Err(AppError::Audio(
            "The selected file contains too little audio".into(),
        ));
    }

    Ok(DecodedAudio {
        duration_ms: samples_to_ms(samples.len()),
        pcm: samples,
    })
}

fn validate_input(path: &Path) -> AppResult<()> {
    if !path.is_file() {
        return Err(AppError::Audio(
            "The selected media file does not exist".into(),
        ));
    }
    let supported = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            SUPPORTED_EXTENSIONS
                .iter()
                .any(|candidate| extension.eq_ignore_ascii_case(candidate))
        });
    if !supported {
        return Err(AppError::Audio(
            "The selected media format is not supported".into(),
        ));
    }
    Ok(())
}

fn normalize_args(input: &Path, output: &Path) -> Vec<OsString> {
    let mut args = ["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"]
        .into_iter()
        .map(OsString::from)
        .collect::<Vec<_>>();
    args.push(input.as_os_str().to_os_string());
    args.extend(
        [
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
        ]
        .into_iter()
        .map(OsString::from),
    );
    args.push(output.as_os_str().to_os_string());
    args
}

fn temporary_wav_path() -> AppResult<PathBuf> {
    let directory = std::env::temp_dir().join("moss-transcribe-studio");
    fs::create_dir_all(&directory)?;
    Ok(directory.join(format!("{}.wav", Uuid::new_v4())))
}

fn samples_to_ms(samples: usize) -> u64 {
    (samples as f64 / SAMPLE_RATE as f64 * 1000.0).round() as u64
}

#[derive(Debug)]
struct ResolvedFfmpeg {
    program: PathBuf,
    version: String,
}

fn resolve_ffmpeg() -> Option<ResolvedFfmpeg> {
    let mut candidates = vec![PathBuf::from("ffmpeg")];
    #[cfg(target_os = "macos")]
    candidates.extend(MACOS_FFMPEG_PATHS.iter().map(PathBuf::from));

    candidates.into_iter().find_map(|program| {
        let output = Command::new(&program).arg("-version").output().ok()?;
        if !output.status.success() {
            return None;
        }
        let version = String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .unwrap_or("ffmpeg")
            .to_string();
        Some(ResolvedFfmpeg { program, version })
    })
}

fn summarize_stderr(stderr: &[u8], input: &Path, output: &Path) -> String {
    let text = String::from_utf8_lossy(stderr)
        .replace(input.to_string_lossy().as_ref(), "<input>")
        .replace(output.to_string_lossy().as_ref(), "<output>");
    let summary = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(4)
        .collect::<Vec<_>>()
        .join("\n");
    if summary.is_empty() {
        "no diagnostic output".into()
    } else {
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_safe_ffmpeg_arguments_without_a_shell() {
        let args = normalize_args(Path::new("/tmp/input file.m4a"), Path::new("/tmp/out.wav"))
            .into_iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(args[0], "-nostdin");
        assert!(args.contains(&"/tmp/input file.m4a".to_string()));
        assert_eq!(args.last().map(String::as_str), Some("/tmp/out.wav"));
    }
}
