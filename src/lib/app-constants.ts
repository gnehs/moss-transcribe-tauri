import type { TaskDraft } from "@/types/transcription";

export const mossModelRepository = "OpenMOSS-Team/MOSS-Transcribe-Diarize";

export const audioFilters = [
  {
    name: "Audio and video",
    extensions: ["wav", "mp3", "m4a", "aac", "flac", "ogg", "mp4", "mov", "mkv", "webm"],
  },
];

export const defaultTaskDraft: TaskDraft = {
  inputPaths: [],
  outputDir: "",
  outputs: { txt: true, json: true, srt: true },
  prompt: "",
  maxNewTokens: 4096,
};
