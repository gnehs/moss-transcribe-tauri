import type { ReactNode } from "react";
import { Trans } from "@lingui/react/macro";

import { Badge } from "@/components/ui/badge";
import type { FfmpegStatus } from "@/types/transcription";

export function AppToolbar({
  ffmpeg,
  title,
  subtitle,
  actions,
  utilities,
}: {
  ffmpeg: FfmpegStatus;
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  utilities?: ReactNode;
}) {
  const isMacOS = navigator.userAgent.includes("Mac OS X");

  return (
    <header
      className="window-toolbar"
      data-macos={isMacOS || undefined}
      data-tauri-drag-region
    >
      <div className="window-toolbar-copy" data-tauri-drag-region>
        <h1
          className="truncate font-heading text-base font-medium"
          data-tauri-drag-region
        >
          {title}
        </h1>
        {subtitle ? (
          <p
            className="truncate text-sm text-muted-foreground"
            data-tauri-drag-region
          >
            {subtitle}
          </p>
        ) : null}
      </div>
      {actions || !ffmpeg.available ? (
        <div className="window-toolbar-actions">
          <div className="window-toolbar-primary-actions">{actions}</div>
          {!ffmpeg.available ? (
            <Badge variant="destructive"><Trans>缺少 FFmpeg</Trans></Badge>
          ) : null}
        </div>
      ) : null}
      <div className="window-toolbar-utilities">{utilities}</div>
    </header>
  );
}
