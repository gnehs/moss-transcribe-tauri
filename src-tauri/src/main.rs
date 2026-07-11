fn main() {
    if let Err(error) = moss_transcribe_tauri_lib::run() {
        eprintln!("MOSS Transcribe Studio failed to start: {error}");
        std::process::exit(1);
    }
}
