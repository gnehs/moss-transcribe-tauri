export type TaskId = string;

export type TaskStatus =
  | "queued"
  | "preparing"
  | "encoding"
  | "prefilling"
  | "generating"
  | "completed"
  | "failed";

export type TimedTaskStage = Extract<
  TaskStatus,
  "preparing" | "encoding" | "prefilling" | "generating"
>;

export type StageTimings = Partial<Record<TimedTaskStage, number>>;

export type OutputOptions = {
  txt: boolean;
  json: boolean;
  srt: boolean;
};

export type TaskOptions = {
  outputDir: string;
  outputs: OutputOptions;
  prompt: string | null;
  convertToTraditional: boolean;
};

export type CommandError = {
  code: string;
  message: string;
  retryable: boolean;
  details?: Record<string, unknown> | null;
};

export type TranscriptSegment = {
  start: number;
  end: number;
  speaker: string;
  text: string;
};

export type TaskSummary = {
  id: TaskId;
  inputPath: string;
  fileName: string;
  status: TaskStatus;
  percent: number;
  message?: string | null;
  createdAt: string;
  updatedAt: string;
  revision: number;
};

export type TaskDetail = TaskSummary & {
  elapsedMs: number;
  audioDurationSeconds?: number | null;
  promptTokens: number;
  generatedTokens: number;
  text?: string | null;
  segments: TranscriptSegment[];
  txtPath?: string | null;
  jsonPath?: string | null;
  srtPath?: string | null;
  error?: CommandError | null;
  options: TaskOptions;
};

export type TranscriptionProgress = {
  taskId: TaskId;
  stage: Exclude<TaskStatus, "queued">;
  percent: number;
  message: string;
  elapsedMs: number;
  audioDurationMs: number | null;
  promptTokens: number;
  generatedTokens: number;
  estimatedGeneratedTokens: number;
};

export type TranscriptStreamEvent = {
  taskId: TaskId;
  text: string;
  segmentOffset: number;
  segments: TranscriptSegment[];
  generatedTokens: number;
};

export type DownloadProgress = {
  modelId: string;
  state: string;
  currentFile: string | null;
  fileIndex: number;
  totalFiles: number;
  fileBytesCompleted: number;
  fileTotalBytes: number;
  speedBytesPerSec: number;
  percent: number;
  message: string;
};

export type TranscriptionResult = {
  audioPath: string;
  audioDurationMs: number;
  text: string;
  segments: TranscriptSegment[];
  promptTokens: number;
  generatedTokens: number;
  truncated: boolean;
  outputs: {
    txtPath: string | null;
    jsonPath: string | null;
    srtPath: string | null;
  };
};

export type TranscriptionTask = TaskSummary & {
  options: TaskOptions;
  progress: TranscriptionProgress | null;
  result: TranscriptionResult | null;
  stream: TranscriptStreamEvent | null;
  error: string | null;
  startedAt: number | null;
  completedAt: number | null;
  stageStartedAt: number | null;
  stageTimings: StageTimings;
};

export type TaskDraft = {
  inputPaths: string[];
  outputDir: string;
  outputs: OutputOptions;
  prompt: string;
  convertToTraditional: boolean;
};

export type ModelStatus = {
  id: string;
  title: string;
  repo: string;
  sizeHint: string;
  installed: boolean;
  path: string;
  bytesOnDisk: number;
  files: string[];
  missingFiles: string[];
};

export type FfmpegStatus = {
  available: boolean;
  path?: string | null;
  version?: string | null;
  error?: CommandError | null;
};

export type SystemInfo = {
  platform: string;
  architecture: string;
  mlxAvailable: boolean;
  metalDevice?: string | null;
  appVersion: string;
};

export type Settings = Record<string, unknown>;

export type AppSnapshot = {
  tasks: TaskSummary[];
  model: ModelStatus;
  ffmpeg: FfmpegStatus;
  system: SystemInfo;
  settings: Settings;
};

export type TaskChangedEvent = {
  schemaVersion: 1;
  sequence: number;
  taskId: TaskId;
  task: TaskSummary;
  promptTokens?: number;
  generatedTokens?: number;
};

export type ModelChangedEvent = {
  schemaVersion: 1;
  sequence: number;
  status: ModelStatus;
  file?: string | null;
};

export type SystemChangedEvent = {
  schemaVersion: 1;
  sequence: number;
  ffmpeg: FfmpegStatus;
};
