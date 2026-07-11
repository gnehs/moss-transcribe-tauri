import { useCallback, useEffect, useRef, useState } from "react";
import { msg } from "@lingui/core/macro";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";
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
  TranscriptionProgress,
  TranscriptionResult,
  TranscriptionTask,
} from "@/types/transcription";

const supportedExtensions = new Set(
  audioFilters.flatMap((filter) => filter.extensions.map((extension) => extension.toLowerCase())),
);

function createTaskId() {
  return window.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

function isSupportedMediaPath(path: string) {
  const extension = path.split(".").pop()?.toLowerCase();
  return Boolean(extension && supportedExtensions.has(extension));
}

function pathsFromDialogSelection(selected: string | string[] | null) {
  return typeof selected === "string" ? [selected] : selected ?? [];
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

export function useTranscriptionWorkspace() {
  const [tasks, setTasks] = useState<TranscriptionTask[]>([]);
  const [taskDraft, setTaskDraft] = useState<TaskDraft>(defaultTaskDraft);
  const [model, setModel] = useState<ModelStatus>(modelFallback);
  const [ffmpeg, setFfmpeg] = useState<FfmpegStatus>({ available: false });
  const [system, setSystem] = useState<SystemInfo>(systemFallback);
  const [isTaskDialogOpen, setTaskDialogOpen] = useState(false);
  const [isDraggingFiles, setIsDraggingFiles] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null);
  const [isConfirmingTasks, setIsConfirmingTasks] = useState(false);
  const [deletingModel, setDeletingModel] = useState(false);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const runningTaskIdRef = useRef<string | null>(null);
  const openTaskDialogRef = useRef<(paths: string[]) => void>(() => {});

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
    if (rejectedCount > 0) toast.warning(i18n._(msg`已略過 ${rejectedCount} 個不支援的檔案`));
    setTaskDraft((current) => ({ ...current, inputPaths: acceptedPaths }));
    setTaskDialogOpen(true);
  }, []);

  const runQueuedTask = useCallback(async (task: TranscriptionTask) => {
    runningTaskIdRef.current = task.id;
    setTasks((current) => current.map((item) => item.id === task.id ? {
      ...item,
      status: "preparing",
      startedAt: Date.now(),
      error: null,
      result: null,
      progress: null,
    } : item));
    try {
      const result = await invoke<TranscriptionResult>("transcribe_file", {
        request: {
          taskId: task.id,
          audioPath: task.inputPath,
          options: {
            prompt: task.options.prompt,
            maxNewTokens: task.options.maxNewTokens,
          },
          export: {
            outputDir: task.options.outputDir || null,
            writeTxt: task.options.outputs.txt,
            writeJson: task.options.outputs.json,
            writeSrt: task.options.outputs.srt,
          },
        },
      });
      setTasks((current) => current.map((item) => item.id === task.id ? {
        ...item,
        status: "completed",
        percent: 100,
        progress: null,
        result,
        completedAt: Date.now(),
        updatedAt: new Date().toISOString(),
      } : item));
      toast.success(i18n._(msg`${basename(task.inputPath)} 已完成`));
    } catch (error) {
      const message = formatInvokeError(error);
      setTasks((current) => current.map((item) => item.id === task.id ? {
        ...item,
        status: "failed",
        error: message,
        progress: null,
        completedAt: Date.now(),
        updatedAt: new Date().toISOString(),
      } : item));
      toast.error(i18n._(msg`${basename(task.inputPath)} 失敗`), { description: message });
    } finally {
      runningTaskIdRef.current = null;
    }
  }, []);

  useEffect(() => {
    void refreshRuntime();
    const unlisteners = Promise.all([
      listen<TranscriptionProgress>("transcription-progress", (event) => {
        const progress = event.payload;
        if (!progress.taskId || progress.taskId !== runningTaskIdRef.current) return;
        setTasks((current) => current.map((task) => task.id === progress.taskId ? {
          ...task,
          status: progress.stage as TaskStatus,
          percent: Math.max(0, Math.min(100, progress.percent)),
          message: progress.message,
          progress,
          updatedAt: new Date().toISOString(),
        } : task));
      }),
      listen<DownloadProgress>("model-download-progress", (event) => {
        setDownloadProgress(event.payload);
        if (event.payload.state === "complete") void refreshRuntime();
      }),
    ]);
    return () => { void unlisteners.then((items) => items.forEach((unlisten) => unlisten())); };
  }, [refreshRuntime]);

  useEffect(() => { openTaskDialogRef.current = openTaskDialog; }, [openTaskDialog]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    void getCurrentWebview().onDragDropEvent((event) => {
      if (event.payload.type === "enter" || event.payload.type === "over") {
        setIsDraggingFiles(true);
        return;
      }
      setIsDraggingFiles(false);
      if (event.payload.type === "drop") openTaskDialogRef.current(event.payload.paths);
    }).then((handler) => { unlisten = handler; }).catch(() => setIsDraggingFiles(false));
    return () => unlisten?.();
  }, []);

  useEffect(() => {
    if (runningTaskIdRef.current) return;
    const next = tasks.find((task) => task.status === "queued");
    if (next) void runQueuedTask(next);
  }, [runQueuedTask, tasks]);

  async function pickFilesForTasks() {
    const selected = await open({ multiple: true, filters: audioFilters });
    const paths = pathsFromDialogSelection(selected);
    if (paths.length > 0) openTaskDialog(paths);
  }

  async function pickTaskOutputDir() {
    const selected = await open({ directory: true, multiple: false });
    if (typeof selected === "string") setTaskDraft((current) => ({ ...current, outputDir: selected }));
  }

  async function confirmTaskDraft() {
    if (!taskDraft.inputPaths.length || isConfirmingTasks) return;
    setIsConfirmingTasks(true);
    try {
      if (!model.installed) await downloadModel();
      const now = new Date().toISOString();
      const nextTasks = taskDraft.inputPaths.map((inputPath, index): TranscriptionTask => ({
        id: createTaskId(), inputPath, fileName: basename(inputPath), status: "queued", percent: 0,
        message: null, createdAt: now, updatedAt: now, revision: index, options: {
          outputDir: taskDraft.outputDir, outputs: taskDraft.outputs,
          prompt: taskDraft.prompt.trim() || null, maxNewTokens: taskDraft.maxNewTokens,
        }, progress: null, result: null, error: null, startedAt: null, completedAt: null,
      }));
      setTasks((current) => [...current, ...nextTasks]);
      setTaskDialogOpen(false);
      toast.success(i18n._(msg`已加入 ${nextTasks.length} 個任務`));
    } catch (error) {
      toast.error(formatInvokeError(error));
    } finally { setIsConfirmingTasks(false); }
  }

  async function downloadModel(redownload = false) {
    setIsDownloading(true);
    setDownloadProgress(null);
    try {
      const next = await invoke<ModelStatus>(redownload ? "redownload_model" : "download_model");
      setModel(next);
      toast.success(redownload ? i18n._(msg`模型已重新下載`) : i18n._(msg`模型已可使用`));
    } catch (error) {
      toast.error(formatInvokeError(error));
      throw error;
    } finally { setIsDownloading(false); }
  }

  async function recheckFfmpeg() {
    try {
      const next = await invoke<FfmpegStatus>("get_ffmpeg_status");
      setFfmpeg(next);
      toast.success(next.available ? i18n._(msg`FFmpeg 可用`) : i18n._(msg`仍未偵測到 FFmpeg`));
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
    } catch (error) { toast.error(formatInvokeError(error)); return false; }
    finally { setDeletingModel(false); }
  }

  function retryTask(taskId: string) {
    setTasks((current) => current.map((task) => task.id === taskId ? {
      ...task, status: "queued", percent: 0, error: null, progress: null, result: null,
      startedAt: null, completedAt: null,
    } : task));
  }

  function removeTask(taskId: string) {
    if (taskId === runningTaskIdRef.current) return;
    setTasks((current) => current.filter((task) => task.id !== taskId));
  }

  function clearFinishedTasks() {
    setTasks((current) => current.filter((task) => task.status !== "completed"));
  }

  return {
    tasks, taskDraft, model, ffmpeg, system, isTaskDialogOpen, isDraggingFiles,
    isDownloading, downloadProgress, isConfirmingTasks, deletingModel, selectedTaskId,
    setTaskDraft, setTaskDialogOpen, setSelectedTaskId, pickFilesForTasks, pickTaskOutputDir,
    confirmTaskDraft, downloadModel, deleteModel, retryTask, removeTask, clearFinishedTasks,
    refreshRuntime, recheckFfmpeg,
  };
}
