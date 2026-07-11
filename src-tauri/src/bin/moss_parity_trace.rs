//! Local-only native MOSS parity trace generator.

use std::{
    env,
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use moss_transcribe_tauri_lib::generate_native_parity_trace;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Native MOSS parity trace was not generated: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let arguments = Arguments::parse(env::args_os().skip(1))?;
    if arguments.output.exists() {
        return Err(format!(
            "output must not already exist: {}",
            arguments.output.display()
        ));
    }
    let pcm = read_canonical_wav(&arguments.wav)?;
    if pcm.len() <= 30 * 16_000 {
        return Err(format!(
            "WAV must be longer than 30 seconds; got {:.3} seconds",
            pcm.len() as f64 / 16_000.0
        ));
    }
    let basename = arguments
        .wav
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "WAV basename is not valid UTF-8".to_string())?;
    let trace = generate_native_parity_trace(
        &arguments.model_dir,
        &pcm,
        basename,
        arguments.prompt.as_deref(),
    )
    .map_err(|error| error.to_string())?;

    if let Some(parent) = arguments
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&arguments.output)
        .map_err(|error| format!("could not create {}: {error}", arguments.output.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &trace)
        .map_err(|error| format!("could not serialize trace: {error}"))?;
    writer
        .write_all(b"\n")
        .map_err(|error| format!("could not finish trace: {error}"))?;
    writer
        .flush()
        .map_err(|error| format!("could not flush trace: {error}"))?;
    println!(
        "Wrote native parity trace to {}",
        arguments.output.display()
    );
    Ok(())
}

fn read_canonical_wav(path: &Path) -> Result<Vec<f32>, String> {
    if !path.is_file() {
        return Err(format!("WAV is missing: {}", path.display()));
    }
    let mut reader = hound::WavReader::open(path)
        .map_err(|error| format!("WAV is invalid at {}: {error}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000
        || spec.channels != 1
        || spec.sample_format != hound::SampleFormat::Int
        || spec.bits_per_sample != 16
    {
        return Err(format!(
            "WAV must be canonical mono 16 kHz PCM16; got {} Hz, {} channel(s), {:?}/{} bit",
            spec.sample_rate, spec.channels, spec.sample_format, spec.bits_per_sample
        ));
    }
    reader
        .samples::<i16>()
        .map(|sample| {
            sample
                .map(|value| value as f32 / 32_768.0)
                .map_err(|error| format!("WAV PCM is unreadable: {error}"))
        })
        .collect()
}

struct Arguments {
    model_dir: PathBuf,
    wav: PathBuf,
    output: PathBuf,
    prompt: Option<String>,
}

impl Arguments {
    fn parse(arguments: impl IntoIterator<Item = std::ffi::OsString>) -> Result<Self, String> {
        let mut arguments = arguments.into_iter();
        let mut model_dir = None;
        let mut wav = None;
        let mut output = None;
        let mut prompt = None;
        while let Some(flag) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("{} requires a value", flag.to_string_lossy()))?;
            match flag.to_str() {
                Some("--model-dir") => model_dir = Some(PathBuf::from(value)),
                Some("--wav") => wav = Some(PathBuf::from(value)),
                Some("--output") => output = Some(PathBuf::from(value)),
                Some("--prompt") => {
                    prompt = Some(
                        value
                            .into_string()
                            .map_err(|_| "--prompt must be valid UTF-8".to_string())?,
                    )
                }
                _ => {
                    return Err(format!(
                        "unknown argument {}; expected --model-dir, --wav, --output, or --prompt",
                        flag.to_string_lossy()
                    ))
                }
            }
        }
        Ok(Self {
            model_dir: model_dir.ok_or_else(usage)?,
            wav: wav.ok_or_else(usage)?,
            output: output.ok_or_else(usage)?,
            prompt,
        })
    }
}

fn usage() -> String {
    "usage: moss-parity-trace --model-dir <snapshot> --wav <mono-16k-pcm16.wav> --output <trace.json> [--prompt <text>]".into()
}
