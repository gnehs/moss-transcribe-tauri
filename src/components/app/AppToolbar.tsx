import type { ReactNode } from "react";
import { Trans } from "@lingui/react/macro";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
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
      className={cn(
        "bg-muted/50 flex min-h-12 items-center gap-3 border-b px-[18px] select-none max-[720px]:grid max-[720px]:grid-cols-[minmax(0,1fr)_auto] max-[720px]:px-3 max-[720px]:py-3",
        isMacOS && "pl-[86px]"
      )}
      data-macos={isMacOS || undefined}
      data-tauri-drag-region
    >
      <div className="min-w-0" data-tauri-drag-region>
        <h1
          className="font-heading truncate text-base font-medium"
          data-tauri-drag-region
        >
          {title}
        </h1>
        {subtitle ? (
          <p
            className="text-muted-foreground truncate text-sm"
            data-tauri-drag-region
          >
            {subtitle}
          </p>
        ) : null}
      </div>
      {actions || !ffmpeg.available ? (
        <div className="ml-auto flex min-w-0 flex-wrap items-center justify-end gap-2 max-[720px]:col-span-2 max-[720px]:row-start-2 max-[720px]:ml-0 max-[720px]:w-full max-[720px]:justify-start">
          <div className="flex min-w-0 items-center gap-2 max-[720px]:flex-wrap">
            {actions}
          </div>
          {!ffmpeg.available ? (
            <Badge variant="destructive">
              <Trans>缺少 FFmpeg</Trans>
            </Badge>
          ) : null}
        </div>
      ) : null}
      <div
        className={cn(
          "flex min-w-0 items-center gap-2",
          !actions && ffmpeg.available && "ml-auto",
          "max-[720px]:col-start-2 max-[720px]:row-start-1"
        )}
      >
        {utilities}
      </div>
    </header>
  );
}
