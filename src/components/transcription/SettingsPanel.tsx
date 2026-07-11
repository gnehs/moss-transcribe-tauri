import { i18n } from "@lingui/core";
import { msg } from "@lingui/core/macro";
import { Trans } from "@lingui/react/macro";
import { DownloadIcon, EllipsisIcon, FolderOpenIcon, RefreshCwIcon, Trash2Icon } from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Progress } from "@/components/ui/progress";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { activateLocale, locales, type Locale } from "@/i18n";
import { mossModelRepository } from "@/lib/app-constants";
import { formatBytes } from "@/lib/format";
import type {
  DownloadProgress,
  FfmpegStatus,
  ModelStatus,
  SystemInfo,
} from "@/types/transcription";

type SettingsPanelProps = {
  model: ModelStatus;
  ffmpeg: FfmpegStatus;
  system: SystemInfo;
  downloadProgress: DownloadProgress | null;
  isDownloading: boolean;
  deletingModel: boolean;
  onDownload: () => void;
  onRedownload: () => void;
  onDelete: () => void;
  onRevealModel: () => void;
  onRefreshFfmpeg: () => void;
};

export function SettingsPanel({
  model,
  ffmpeg,
  system,
  downloadProgress,
  isDownloading,
  deletingModel,
  onDownload,
  onRedownload,
  onDelete,
  onRevealModel,
  onRefreshFfmpeg,
}: SettingsPanelProps) {
  const [isChangingLocale, setIsChangingLocale] = useState(false);
  const activeLocale = (i18n.locale in locales ? i18n.locale : "zh-Hant") as Locale;
  const ready = model.installed;

  async function changeLocale(value: string | null) {
    if (!value || value === activeLocale) return;
    setIsChangingLocale(true);
    try {
      await activateLocale(value as Locale);
    } finally {
      setIsChangingLocale(false);
    }
  }

  return (
    <div className="settings-grid">
      <section className="settings-section">
        <div className="settings-section-header">
          <h2 className="settings-section-title"><Trans>介面語言</Trans></h2>
          <p className="settings-section-description"><Trans>立即套用並記住偏好設定。</Trans></p>
        </div>
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor="interface-language"><Trans>語言</Trans></FieldLabel>
            <Select value={activeLocale} disabled={isChangingLocale} onValueChange={changeLocale}>
              <SelectTrigger id="interface-language" className="w-full">
                <SelectValue>{locales[activeLocale]}</SelectValue>
              </SelectTrigger>
              <SelectContent alignItemWithTrigger={false}>
                <SelectGroup>
                  {Object.entries(locales).map(([value, label]) => (
                    <SelectItem key={value} value={value}>{label}</SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
            <FieldDescription><Trans>變更後會立即套用。</Trans></FieldDescription>
          </Field>
        </FieldGroup>
      </section>

      <section className="settings-section">
        <div className="settings-section-header">
          <h2 className="settings-section-title"><Trans>模型</Trans></h2>
          <p className="settings-section-description"><Trans>此 app 固定使用一個說話者分離轉錄模型。</Trans></p>
        </div>
        <Card size="sm">
          <CardHeader>
            <CardTitle className="min-w-0 truncate">{mossModelRepository}</CardTitle>
            <CardDescription className="flex items-center gap-2">
              <span>{formatBytes(model.bytesOnDisk)}</span>
              {model.missingFiles.length ? (
                <>
                  <span aria-hidden="true">·</span>
                  <span><Trans>缺少 {model.missingFiles.length} 個檔案</Trans></span>
                </>
              ) : null}
            </CardDescription>
            <CardAction className="flex items-center gap-1">
              <Badge variant={ready ? "secondary" : "outline"}>
                {ready ? <Trans>已下載</Trans> : isDownloading ? <Trans>下載中</Trans> : <Trans>未下載</Trans>}
              </Badge>
              <DropdownMenu>
                <DropdownMenuTrigger
                  render={<Button variant="ghost" size="icon-sm" />}
                >
                  <EllipsisIcon />
                  <span className="sr-only"><Trans>模型操作</Trans></span>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuGroup>
                    <DropdownMenuItem
                      disabled={!model.path || (!ready && model.bytesOnDisk <= 0)}
                      onClick={() => { void onRevealModel(); }}
                    >
                      <FolderOpenIcon data-icon="inline-start" />
                      <Trans>在 Finder 顯示</Trans>
                    </DropdownMenuItem>
                    {ready ? (
                      <>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          disabled={isDownloading || deletingModel}
                          onClick={onRedownload}
                        >
                          <RefreshCwIcon data-icon="inline-start" />
                          <Trans>重新下載</Trans>
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          variant="destructive"
                          disabled={isDownloading || deletingModel}
                          onClick={onDelete}
                        >
                          <Trash2Icon data-icon="inline-start" />
                          <Trans>刪除</Trans>
                        </DropdownMenuItem>
                      </>
                    ) : null}
                  </DropdownMenuGroup>
                </DropdownMenuContent>
              </DropdownMenu>
            </CardAction>
          </CardHeader>
          {isDownloading || !ready ? (
            <CardContent className="flex flex-col gap-3">
              {isDownloading && downloadProgress ? (
                <div className="flex flex-col gap-1">
                  <Progress value={downloadProgress.percent} aria-label={i18n._(msg`模型下載進度`)} />
                  <div className="flex justify-between gap-2 text-xs text-muted-foreground">
                    <span className="truncate">{downloadProgress.currentFile ?? downloadProgress.message}</span>
                    <span>{downloadProgress.percent.toFixed(0)}%</span>
                  </div>
                </div>
              ) : null}
              {!ready ? (
                <div className="flex flex-wrap gap-2">
                  <Button disabled={isDownloading || deletingModel} onClick={onDownload}>
                    <DownloadIcon data-icon="inline-start" /><Trans>下載模型</Trans>
                  </Button>
                </div>
              ) : null}
            </CardContent>
          ) : null}
        </Card>
      </section>

      <section className="settings-section">
        <div className="settings-section-header">
          <h2 className="settings-section-title"><Trans>工具狀態</Trans></h2>
          <p className="settings-section-description"><Trans>音訊和影片會先由本機工具轉為可處理的格式。</Trans></p>
        </div>
        <Card size="sm">
          <CardHeader>
            <CardTitle>FFmpeg</CardTitle>
            <CardDescription className="truncate">
              {ffmpeg.version ?? ffmpeg.path ?? <Trans>未偵測到 FFmpeg</Trans>}
            </CardDescription>
            <CardAction>
              <Badge variant={ffmpeg.available ? "secondary" : "destructive"}>
                {ffmpeg.available ? <Trans>可用</Trans> : <Trans>缺少</Trans>}
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {!ffmpeg.available ? (
              <p className="text-sm text-muted-foreground"><Trans>請安裝 FFmpeg 後重新檢查。</Trans></p>
            ) : null}
            <Button variant="outline" onClick={onRefreshFfmpeg}>
              <RefreshCwIcon data-icon="inline-start" /><Trans>重新檢查</Trans>
            </Button>
          </CardContent>
        </Card>
      </section>

      <section className="settings-section">
        <div className="settings-section-header">
          <h2 className="settings-section-title"><Trans>執行環境</Trans></h2>
          <p className="settings-section-description"><Trans>MOSS 在 Apple Silicon 上透過 MLX 與 Metal 執行。</Trans></p>
        </div>
        <div className="flex flex-col gap-2 text-sm">
          <div className="flex justify-between gap-3"><span className="text-muted-foreground">MLX</span><span>{system.mlxAvailable ? <Trans>可用</Trans> : <Trans>不可用</Trans>}</span></div>
          <Separator />
          <div className="flex justify-between gap-3"><span className="text-muted-foreground">Metal</span><span className="truncate">{system.metalDevice ?? <Trans>未偵測到</Trans>}</span></div>
          <Separator />
          <div className="flex justify-between gap-3"><span className="text-muted-foreground">App version</span><span>{system.appVersion}</span></div>
        </div>
      </section>
    </div>
  );
}
