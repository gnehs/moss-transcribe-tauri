use std::{
    ffi::OsString,
    io::{ErrorKind, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{OnceLock, RwLock},
};

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

pub fn ffmpeg_status() -> FfmpegStatus {
    match refresh_ffmpeg() {
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
    let ffmpeg = resolve_ffmpeg_cached().ok_or_else(|| {
        AppError::Audio("找不到 FFmpeg。請安裝 FFmpeg 後在設定中重新檢查。".to_string())
    })?;
    let mut child = Command::new(&ffmpeg.program)
        .args(normalize_args(input))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            if error.kind() == ErrorKind::NotFound {
                AppError::Audio("FFmpeg is unavailable".into())
            } else {
                AppError::Audio(format!("Could not run FFmpeg: {error}"))
            }
        })?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| AppError::Audio("Could not read FFmpeg audio output".into()))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| AppError::Audio("Could not read FFmpeg diagnostics".into()))?;
    let stderr_reader = std::thread::spawn(move || {
        const DIAGNOSTIC_LIMIT: usize = 64 * 1024;
        let mut diagnostics = Vec::with_capacity(DIAGNOSTIC_LIMIT);
        let mut buffer = [0_u8; 8 * 1024];
        while let Ok(read) = stderr.read(&mut buffer) {
            if read == 0 {
                break;
            }
            let remaining = DIAGNOSTIC_LIMIT.saturating_sub(diagnostics.len());
            diagnostics.extend_from_slice(&buffer[..read.min(remaining)]);
        }
        diagnostics
    });

    let mut samples = Vec::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut pending_byte = None;
    loop {
        let read = match stdout.read(&mut buffer) {
            Ok(read) => read,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stderr_reader.join();
                return Err(AppError::Audio(format!(
                    "Could not read FFmpeg output: {error}"
                )));
            }
        };
        if read == 0 {
            break;
        }
        let mut bytes = &buffer[..read];
        if let Some(first) = pending_byte.take() {
            samples.push(i16::from_le_bytes([first, bytes[0]]) as f32 / 32768.0);
            bytes = &bytes[1..];
        }
        let mut pairs = bytes.chunks_exact(2);
        samples.extend(
            pairs
                .by_ref()
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0),
        );
        pending_byte = pairs.remainder().first().copied();
    }

    let status = child
        .wait()
        .map_err(|error| AppError::Audio(format!("Could not wait for FFmpeg: {error}")))?;
    let diagnostics = stderr_reader
        .join()
        .map_err(|_| AppError::Audio("Could not collect FFmpeg diagnostics".into()))?;
    if !status.success() {
        return Err(AppError::Audio(format!(
            "FFmpeg failed: {}",
            summarize_stderr(&diagnostics, input)
        )));
    }
    if pending_byte.is_some() {
        return Err(AppError::Audio(
            "FFmpeg returned an incomplete PCM sample".into(),
        ));
    }
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

fn normalize_args(input: &Path) -> Vec<OsString> {
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
            "s16le",
            "pipe:1",
        ]
        .into_iter()
        .map(OsString::from),
    );
    args
}

fn samples_to_ms(samples: usize) -> u64 {
    (samples as f64 / SAMPLE_RATE as f64 * 1000.0).round() as u64
}

#[derive(Debug, Clone)]
struct ResolvedFfmpeg {
    program: PathBuf,
    version: String,
}

static FFMPEG_CACHE: OnceLock<RwLock<Option<ResolvedFfmpeg>>> = OnceLock::new();

fn ffmpeg_cache() -> &'static RwLock<Option<ResolvedFfmpeg>> {
    FFMPEG_CACHE.get_or_init(|| RwLock::new(None))
}

fn resolve_ffmpeg_cached() -> Option<ResolvedFfmpeg> {
    if let Some(ffmpeg) = ffmpeg_cache().read().ok().and_then(|cached| cached.clone()) {
        return Some(ffmpeg);
    }
    refresh_ffmpeg()
}

fn refresh_ffmpeg() -> Option<ResolvedFfmpeg> {
    let resolved = find_ffmpeg();
    if let Ok(mut cached) = ffmpeg_cache().write() {
        *cached = resolved.clone();
    }
    resolved
}

fn find_ffmpeg() -> Option<ResolvedFfmpeg> {
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

fn summarize_stderr(stderr: &[u8], input: &Path) -> String {
    let text = String::from_utf8_lossy(stderr).replace(input.to_string_lossy().as_ref(), "<input>");
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
        let args = normalize_args(Path::new("/tmp/input file.m4a"))
            .into_iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(args[0], "-nostdin");
        assert!(args.contains(&"/tmp/input file.m4a".to_string()));
        assert_eq!(args.last().map(String::as_str), Some("pipe:1"));
    }

    #[test]
    fn decodes_ffmpeg_pcm_pipe_without_a_temporary_output_file() {
        if find_ffmpeg().is_none() {
            return;
        }
        let input =
            std::env::temp_dir().join(format!("moss-audio-pipe-{}.wav", uuid::Uuid::new_v4()));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer =
            hound::WavWriter::create(&input, spec).expect("fixture should be writable");
        for index in 0..800 {
            writer
                .write_sample((index as i16).wrapping_mul(31))
                .expect("fixture sample should be writable");
        }
        writer.finalize().expect("fixture should finalize");

        let decoded = decode_to_pcm(&input).expect("FFmpeg pipe decode should succeed");
        let _ = std::fs::remove_file(input);
        assert_eq!(decoded.pcm.len(), 800);
        assert_eq!(decoded.duration_ms, 50);
    }
}
