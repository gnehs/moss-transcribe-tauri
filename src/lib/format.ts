import { msg } from "@lingui/core/macro";

import { i18n } from "@/i18n";

export function uniquePaths(paths: string[]) {
  return Array.from(new Set(paths));
}

export function basename(path: string) {
  return path.split(/[\\/]/).pop() ?? path;
}

export function toNumber(value: string, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";

  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let index = 0;

  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }

  const text = value >= 10 || index === 0 ? value.toFixed(0) : value.toFixed(1);
  return `${text} ${units[index]}`;
}

export function formatTimestamp(ms: number) {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function formatDuration(ms: number | null | undefined) {
  if (!Number.isFinite(ms) || ms == null || ms < 0) return "--";

  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  if (totalSeconds < 60) return i18n._(msg`${totalSeconds} 秒`);

  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return i18n._(msg`${hours} 小時 ${minutes} 分`);
  }

  return i18n._(msg`${minutes} 分 ${seconds.toString().padStart(2, "0")} 秒`);
}

export function formatTiming(ms: number | null | undefined) {
  if (!Number.isFinite(ms) || ms == null || ms < 0) return "--";
  if (ms < 1000) return `${Math.round(ms)} ms`;

  const seconds = ms / 1000;
  const formattedSeconds = seconds < 10 ? seconds.toFixed(2) : seconds.toFixed(1);
  return i18n._(msg`${formattedSeconds} 秒`);
}

export function formatInvokeError(error: unknown) {
  if (typeof error === "string") return error;

  if (error && typeof error === "object") {
    const value = error as { message?: string };
    return value.message ?? JSON.stringify(error);
  }

  return i18n._(msg`未知錯誤`);
}
