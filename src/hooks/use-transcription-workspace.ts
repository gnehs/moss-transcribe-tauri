import { useCallback, useEffect, useRef, useState } from "react";
import { msg } from "@lingui/core/macro";
import { Channel, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";
import {
  isPermissionGranted,
  requestPermission,
  sendNotification,
} from "@tauri-apps/plugin-notification";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { toast } from "sonner";

import { audioFilters, defaultTaskDraft } from "@/lib/app-constants";
import { i18n } from "@/i18n";
import { basename, formatInvokeError, uniquePaths } from "@/lib/format";
import type {
  DownloadProgress,
  FfmpegStatus,
  ModelStatus,
  SystemInfo,
  TaskDraft,
  TaskStatus,
  TimedTaskStage,
  TranscriptionProgress,
  TranscriptionResult,
  TranscriptStreamEvent,
  TranscriptionTask,
} from "@/types/transcription";

const timedTaskStages = new Set<TaskStatus>([
  "preparing",
  "encoding",
  "prefilling",
  "generating",
]);

function isTimedTaskStage(status: TaskStatus): status is TimedTaskStage {
  return timedTaskStages.has(status);
}

function transitionTaskStage(
  task: TranscriptionTask,
  nextStage: TaskStatus,
  changedAt: number
) {
  if (task.status === nextStage)
    return {
      stageStartedAt: task.stageStartedAt,
      stageTimings: task.stageTimings,
    };

  const stageTimings = { ...(task.stageTimings ?? {}) };
  if (isTimedTaskStage(task.status) && task.stageStartedAt != null) {
    stageTimings[task.status] =
      (stageTimings[task.status] ?? 0) +
      Math.max(0, changedAt - task.stageStartedAt);
  }

  return {
    stageStartedAt: isTimedTaskStage(nextStage) ? changedAt : null,
    stageTimings,
  };
}

const supportedExtensions = new Set(
  audioFilters.flatMap((filter) =>
    filter.extensions.map((extension) => extension.toLowerCase())
  )
);

const taskDraftStorageKey = "moss-transcribe.task-draft-preferences";

function createDefaultTaskDraft(): TaskDraft {
  return {
    ...defaultTaskDraft,
    outputs: { ...defaultTaskDraft.outputs },
  };
}

function readStoredTaskDraft(): TaskDraft {
  const fallback = createDefaultTaskDraft();

  try {
    const stored = window.localStorage.getItem(taskDraftStorageKey);
    if (!stored) return fallback;

    const parsed: unknown = JSON.parse(stored);
    if (!parsed || typeof parsed !== "object") return fallback;

    const preferences = parsed as Record<string, unknown>;
    const outputs = preferences.outputs;
    const storedOutputs =
      outputs && typeof outputs === "object"
        ? (outputs as Record<string, unknown>)
        : {};

    return {
      ...fallback,
      outputDir:
        typeof preferences.outputDir === "string"
          ? preferences.outputDir
          : fallback.outputDir,
      outputs: {
        txt:
          typeof storedOutputs.txt === "boolean"
            ? storedOutputs.txt
            : fallback.outputs.txt,
        json:
          typeof storedOutputs.json === "boolean"
            ? storedOutputs.json
            : fallback.outputs.json,
        srt:
          typeof storedOutputs.srt === "boolean"
            ? storedOutputs.srt
            : fallback.outputs.srt,
      },
      prompt:
        typeof preferences.prompt === "string"
          ? preferences.prompt
          : fallback.prompt,
      convertToTraditional:
        typeof preferences.convertToTraditional === "boolean"
          ? preferences.convertToTraditional
          : fallback.convertToTraditional,
    };
  } catch {
    return fallback;
  }
}

function saveTaskDraftPreferences(taskDraft: TaskDraft) {
  try {
    window.localStorage.setItem(
      taskDraftStorageKey,
      JSON.stringify({
        outputDir: taskDraft.outputDir,
        outputs: taskDraft.outputs,
        prompt: taskDraft.prompt,
        convertToTraditional: taskDraft.convertToTraditional,
      })
    );
  } catch {
    // Storage can be unavailable or full; task creation should still work in memory.
  }
}

function createTaskId() {
  return window.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

function isSupportedMediaPath(path: string) {
  const extension = path.split(".").pop()?.toLowerCase();
  return Boolean(extension && supportedExtensions.has(extension));
}

function pathsFromDialogSelection(selected: string | string[] | null) {
  return typeof selected === "string" ? [selected] : (selected ?? []);
}

function modelFallback(): ModelStatus {
  return {
    id: "moss-transcribe-diarize",
    title: "MOSS Transcribe Diarize",
    repo: "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    sizeHint: "",
    installed: false,
    path: "",
    bytesOnDisk: 0,
    files: [],
    missingFiles: [],
  };
}

function systemFallback(): SystemInfo {
  return {
    platform: "macos",
    architecture: "aarch64",
    mlxAvailable: false,
    metalDevice: null,
    appVersion: "—",
  };
}

async function notifyTranscriptionComplete(inputPath: string) {
  try {
    let permissionGranted = await isPermissionGranted();
    if (!permissionGranted) {
      permissionGranted = (await requestPermission()) === "granted";
    }
    if (!permissionGranted) return;

    sendNotification({
      title: "MOSS Transcribe Studio",
      body: i18n._(msg`${basename(inputPath)} 已完成`),
    });
  } catch {
    // Notifications are best-effort and must not affect task completion.
  }
}

export function useTranscriptionWorkspace() {
  const [tasks, setTasks] = useState<TranscriptionTask[]>([]);
  const [taskDraft, setTaskDraft] = useState<TaskDraft>(readStoredTaskDraft);
  const [model, setModel] = useState<ModelStatus>(modelFallback);
  const [ffmpeg, setFfmpeg] = useState<FfmpegStatus>({ available: false });
  const [system, setSystem] = useState<SystemInfo>(systemFallback);
  const [isTaskDialogOpen, setTaskDialogOpen] = useState(false);
  const [isDraggingFiles, setIsDraggingFiles] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] =
    useState<DownloadProgress | null>(null);
  const [isConfirmingTasks, setIsConfirmingTasks] = useState(false);
  const [deletingModel, setDeletingModel] = useState(false);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const runningTaskIdRef = useRef<string | null>(null);
  const openTaskDialogRef = useRef<(paths: string[]) => void>(() => {});

  useEffect(() => {
    saveTaskDraftPreferences(taskDraft);
  }, [
    taskDraft.convertToTraditional,
    taskDraft.outputDir,
    taskDraft.outputs,
    taskDraft.prompt,
  ]);

  const refreshRuntime = useCallback(async () => {
    const [nextModel, nextFfmpeg, nextSystem] = await Promise.allSettled([
      invoke<ModelStatus>("get_model_status"),
      invoke<FfmpegStatus>("get_ffmpeg_status"),
      invoke<SystemInfo>("get_runtime_info"),
    ]);
    if (nextModel.status === "fulfilled") setModel(nextModel.value);
    if (nextFfmpeg.status === "fulfilled") setFfmpeg(nextFfmpeg.value);
    if (nextSystem.status === "fulfilled") setSystem(nextSystem.value);
  }, []);

  const openTaskDialog = useCallback((paths: string[]) => {
    const acceptedPaths = uniquePaths(paths.filter(isSupportedMediaPath));
    const rejectedCount = paths.length - acceptedPaths.length;
    if (acceptedPaths.length === 0) {
      toast.error(i18n._(msg`沒有可加入的支援檔案`));
      return;
    }
    if (rejectedCount > 0)
      toast.warning(i18n._(msg`已略過 ${rejectedCount} 個不支援的檔案`));
    setTaskDraft((current) => ({ ...current, inputPaths: acceptedPaths }));
    setTaskDialogOpen(true);
  }, []);

  const runQueuedTask = useCallback(async (task: TranscriptionTask) => {
    runningTaskIdRef.current = task.id;
    const startedAt = Date.now();
    setTasks((current) =>
      current.map((item) =>
        item.id === task.id
          ? {
              ...item,
              status: "preparing",
              startedAt,
              completedAt: null,
              stageStartedAt: startedAt,
              stageTimings: {},
              error: null,
              result: null,
              stream: null,
              progress: null,
            }
          : item
      )
    );
    try {
      const onStream = new Channel<TranscriptStreamEvent>();
      onStream.onmessage = (partial) => {
        if (partial.taskId !== task.id) return;
        setTasks((current) =>
          current.map((item) =>
            item.id === task.id
              ? {
                  ...item,
                  stream: partial,
                  updatedAt: new Date().toISOString(),
                }
              : item
          )
        );
      };
      const result = await invoke<TranscriptionResult>("transcribe_file", {
        onStream,
        request: {
          taskId: task.id,
          audioPath: task.inputPath,
          options: {
            prompt: task.options.prompt,
            convertToTraditional: task.options.convertToTraditional,
          },
          export: {
            outputDir: task.options.outputDir || null,
            writeTxt: task.options.outputs.txt,
            writeJson: task.options.outputs.json,
            writeSrt: task.options.outputs.srt,
          },
        },
      });
      const completedAt = Date.now();
      setTasks((current) =>
        current.map((item) =>
          item.id === task.id
            ? {
                ...item,
                ...transitionTaskStage(item, "completed", completedAt),
                status: "completed",
                percent: 100,
                progress: null,
                result,
                stream: null,
                completedAt,
                updatedAt: new Date().toISOString(),
              }
            : item
        )
      );
      if (result.truncated) {
        toast.warning(i18n._(msg`結果不完整`), {
          description: i18n._(
            msg`已保留目前逐字稿。請將較長音訊分段後，重新轉錄遺失的部分。`
          ),
        });
      } else {
        toast.success(i18n._(msg`${basename(task.inputPath)} 已完成`));
      }
      void notifyTranscriptionComplete(task.inputPath);
    } catch (error) {
      const message = formatInvokeError(error);
      const completedAt = Date.now();
      setTasks((current) =>
        current.map((item) =>
          item.id === task.id
            ? {
                ...item,
                ...transitionTaskStage(item, "failed", completedAt),
                status: "failed",
                error: message,
                progress: null,
                completedAt,
                updatedAt: new Date().toISOString(),
              }
            : item
        )
      );
      toast.error(i18n._(msg`${basename(task.inputPath)} 失敗`), {
        description: message,
      });
    } finally {
      runningTaskIdRef.current = null;
    }
  }, []);

  useEffect(() => {
    void refreshRuntime();
    const unlisteners = Promise.all([
      listen<TranscriptionProgress>("transcription-progress", (event) => {
        const progress = event.payload;
        if (!progress.taskId || progress.taskId !== runningTaskIdRef.current)
          return;
        const changedAt = Date.now();
        setTasks((current) =>
          current.map((task) =>
            task.id === progress.taskId
              ? {
                  ...task,
                  ...transitionTaskStage(
                    task,
                    progress.stage as TaskStatus,
                    changedAt
                  ),
                  status: progress.stage as TaskStatus,
                  percent: Math.max(0, Math.min(100, progress.percent)),
                  message: progress.message,
                  progress,
                  updatedAt: new Date().toISOString(),
                }
              : task
          )
        );
      }),
      listen<DownloadProgress>("model-download-progress", (event) => {
        setDownloadProgress(event.payload);
        if (event.payload.state === "complete") void refreshRuntime();
      }),
    ]);
    return () => {
      void unlisteners.then((items) => items.forEach((unlisten) => unlisten()));
    };
  }, [refreshRuntime]);

  useEffect(() => {
    openTaskDialogRef.current = openTaskDialog;
  }, [openTaskDialog]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    void getCurrentWebview()
      .onDragDropEvent((event) => {
        if (event.payload.type === "enter" || event.payload.type === "over") {
          setIsDraggingFiles(true);
          return;
        }
        setIsDraggingFiles(false);
        if (event.payload.type === "drop")
          openTaskDialogRef.current(event.payload.paths);
      })
      .then((handler) => {
        unlisten = handler;
      })
      .catch(() => setIsDraggingFiles(false));
    return () => unlisten?.();
  }, []);

  useEffect(() => {
    if (runningTaskIdRef.current) return;
    const next = tasks.find((task) => task.status === "queued");
    if (next) void runQueuedTask(next);
  }, [runQueuedTask, tasks]);

  const pickFilesForTasks = useCallback(async () => {
    const selected = await open({ multiple: true, filters: audioFilters });
    const paths = pathsFromDialogSelection(selected);
    if (paths.length > 0) openTaskDialog(paths);
  }, [openTaskDialog]);

  async function pickTaskOutputDir() {
    const selected = await open({ directory: true, multiple: false });
    if (typeof selected === "string")
      setTaskDraft((current) => ({ ...current, outputDir: selected }));
  }

  async function confirmTaskDraft() {
    if (!taskDraft.inputPaths.length || isConfirmingTasks) return;
    if (
      !taskDraft.outputs.txt &&
      !taskDraft.outputs.json &&
      !taskDraft.outputs.srt
    ) {
      toast.error(i18n._(msg`至少要勾選一個輸出格式。`));
      return;
    }
    setIsConfirmingTasks(true);
    try {
      if (!model.installed) await downloadModel();
      const now = new Date().toISOString();
      const nextTasks = taskDraft.inputPaths.map(
        (inputPath, index): TranscriptionTask => ({
          id: createTaskId(),
          inputPath,
          fileName: basename(inputPath),
          status: "queued",
          percent: 0,
          message: null,
          createdAt: now,
          updatedAt: now,
          revision: index,
          options: {
            outputDir: taskDraft.outputDir,
            outputs: taskDraft.outputs,
            prompt: taskDraft.prompt.trim() || null,
            convertToTraditional: taskDraft.convertToTraditional,
          },
          progress: null,
          result: null,
          stream: null,
          error: null,
          startedAt: null,
          completedAt: null,
          stageStartedAt: null,
          stageTimings: {},
        })
      );
      setTasks((current) => [...current, ...nextTasks]);
      setTaskDialogOpen(false);
      toast.success(i18n._(msg`已加入 ${nextTasks.length} 個任務`));
    } catch (error) {
      toast.error(formatInvokeError(error));
    } finally {
      setIsConfirmingTasks(false);
    }
  }

  async function downloadModel(redownload = false) {
    setIsDownloading(true);
    setDownloadProgress(null);
    try {
      const next = await invoke<ModelStatus>(
        redownload ? "redownload_model" : "download_model"
      );
      setModel(next);
      toast.success(
        redownload ? i18n._(msg`模型已重新下載`) : i18n._(msg`模型已可使用`)
      );
    } catch (error) {
      toast.error(formatInvokeError(error));
      throw error;
    } finally {
      setIsDownloading(false);
    }
  }

  async function recheckFfmpeg() {
    try {
      const next = await invoke<FfmpegStatus>("get_ffmpeg_status");
      setFfmpeg(next);
      toast.success(
        next.available
          ? i18n._(msg`FFmpeg 可用`)
          : i18n._(msg`仍未偵測到 FFmpeg`)
      );
    } catch (error) {
      toast.error(formatInvokeError(error));
    }
  }

  async function deleteModel() {
    if (deletingModel || runningTaskIdRef.current) return false;
    setDeletingModel(true);
    try {
      await invoke("delete_model");
      await refreshRuntime();
      return true;
    } catch (error) {
      toast.error(formatInvokeError(error));
      return false;
    } finally {
      setDeletingModel(false);
    }
  }

  async function revealModel() {
    if (!model.path) return;
    try {
      await revealItemInDir(model.path);
    } catch (error) {
      toast.error(formatInvokeError(error));
    }
  }

  function retryTask(taskId: string) {
    setTasks((current) =>
      current.map((task) =>
        task.id === taskId
          ? {
              ...task,
              status: "queued",
              percent: 0,
              error: null,
              progress: null,
              result: null,
              stream: null,
              startedAt: null,
              completedAt: null,
              stageStartedAt: null,
              stageTimings: {},
            }
          : task
      )
    );
  }

  function removeTask(taskId: string) {
    if (taskId === runningTaskIdRef.current) return;
    setTasks((current) => current.filter((task) => task.id !== taskId));
  }

  function clearFinishedTasks() {
    setTasks((current) =>
      current.filter((task) => task.status !== "completed")
    );
  }

  return {
    tasks,
    taskDraft,
    model,
    ffmpeg,
    system,
    isTaskDialogOpen,
    isDraggingFiles,
    isDownloading,
    downloadProgress,
    isConfirmingTasks,
    deletingModel,
    selectedTaskId,
    setTaskDraft,
    setTaskDialogOpen,
    setSelectedTaskId,
    pickFilesForTasks,
    pickTaskOutputDir,
    confirmTaskDraft,
    downloadModel,
    deleteModel,
    retryTask,
    removeTask,
    clearFinishedTasks,
    revealModel,
    refreshRuntime,
    recheckFfmpeg,
  };
}
