import { i18n } from "@lingui/core";
import { msg } from "@lingui/core/macro";
import { Trans } from "@lingui/react/macro";
import { useTheme } from "next-themes";
import {
  DownloadIcon,
  EllipsisIcon,
  FolderOpenIcon,
  HardDriveIcon,
  RefreshCwIcon,
  Trash2Icon,
} from "lucide-react";
import type { ReactNode } from "react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
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
import { Spinner } from "@/components/ui/spinner";
import { activateLocale, locales, type Locale } from "@/i18n";
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

type Theme = "system" | "light" | "dark";

function isTheme(value: string | null | undefined): value is Theme {
  return value === "system" || value === "light" || value === "dark";
}

function SettingsSection({
  title,
  description,
  children,
}: {
  title: ReactNode;
  description: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="flex min-w-0 flex-col gap-4">
      <div className="flex min-w-0 flex-col gap-1">
        <h2 className="font-heading text-lg leading-snug font-semibold">
          {title}
        </h2>
        <p className="text-muted-foreground text-sm/relaxed">{description}</p>
      </div>
      {children}
    </section>
  );
}

function ModelCard({
  model,
  downloadProgress,
  isDownloading,
  deletingModel,
  onDownload,
  onRedownload,
  onDelete,
  onRevealModel,
}: {
  model: ModelStatus;
  downloadProgress: DownloadProgress | null;
  isDownloading: boolean;
  deletingModel: boolean;
  onDownload: () => void;
  onRedownload: () => void;
  onDelete: () => void;
  onRevealModel: () => void;
}) {
  const ready = model.installed;
  const downloadSize = model.sizeHint || formatBytes(model.bytesOnDisk);

  return (
    <Card size="sm">
      <CardHeader className="gap-2">
        <CardTitle className="flex min-w-0 items-center gap-2 text-base font-semibold">
          <span className="min-w-0 truncate">{model.title || model.repo}</span>
        </CardTitle>
        <CardDescription className="text-sm/relaxed">
          <Trans>支援說話者分離，適合單檔與批次轉錄。</Trans>
        </CardDescription>
        <CardAction>
          <DropdownMenu>
            <DropdownMenuTrigger
              render={<Button variant="ghost" size="icon-sm" />}
            >
              <EllipsisIcon />
              <span className="sr-only">
                <Trans>模型操作</Trans>
              </span>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuGroup>
                <DropdownMenuItem
                  disabled={!model.path || (!ready && model.bytesOnDisk <= 0)}
                  onClick={() => {
                    void onRevealModel();
                  }}
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
                  </>
                ) : null}
              </DropdownMenuGroup>
            </DropdownMenuContent>
          </DropdownMenu>
        </CardAction>
      </CardHeader>

      <CardContent className="flex flex-col gap-3 empty:hidden">
        {isDownloading && downloadProgress ? (
          <div className="flex flex-col gap-1">
            <Progress
              value={downloadProgress.percent}
              aria-label={i18n._(msg`模型下載進度`)}
            />
            <div className="text-muted-foreground flex justify-between gap-2 text-xs">
              <span className="truncate">
                {downloadProgress.currentFile ?? downloadProgress.message}
              </span>
              <span>{downloadProgress.percent.toFixed(0)}%</span>
            </div>
          </div>
        ) : null}
        {!ready && model.missingFiles.length ? (
          <p className="text-muted-foreground m-0 text-xs">
            <Trans>目前還缺少 {model.missingFiles.length} 個檔案。</Trans>
          </p>
        ) : null}
      </CardContent>

      <CardFooter className="justify-between gap-3 py-2 max-[420px]:flex-col max-[420px]:items-stretch">
        <div className="text-muted-foreground flex min-w-0 items-center gap-2 text-sm">
          <HardDriveIcon className="size-4 shrink-0" aria-hidden="true" />
          <span className="truncate">
            <Trans>約 {downloadSize}</Trans>
          </span>
        </div>
        {ready ? (
          <Button
            variant="outline"
            size="sm"
            className="max-[420px]:w-full"
            disabled={isDownloading || deletingModel}
            onClick={onDelete}
          >
            <Trash2Icon data-icon="inline-start" />
            {deletingModel ? <Trans>移除中</Trans> : <Trans>移除</Trans>}
          </Button>
        ) : (
          <Button
            size="sm"
            className="max-[420px]:w-full"
            disabled={isDownloading || deletingModel}
            onClick={onDownload}
          >
            {isDownloading ? (
              <Spinner data-icon="inline-start" />
            ) : (
              <DownloadIcon data-icon="inline-start" />
            )}
            {isDownloading ? <Trans>下載中</Trans> : <Trans>下載模型</Trans>}
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}

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
  const { setTheme, theme } = useTheme();
  const activeLocale = (
    i18n.locale in locales ? i18n.locale : "zh-Hant"
  ) as Locale;
  const activeTheme = isTheme(theme) ? theme : "system";

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
    <div className="flex min-h-0 flex-col gap-6">
      <SettingsSection
        title={<Trans>介面</Trans>}
        description={<Trans>立即套用並記住偏好設定。</Trans>}
      >
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor="interface-language">
              <Trans>語言</Trans>
            </FieldLabel>
            <Select
              value={activeLocale}
              disabled={isChangingLocale}
              onValueChange={changeLocale}
            >
              <SelectTrigger id="interface-language" className="w-full">
                <SelectValue>{locales[activeLocale]}</SelectValue>
              </SelectTrigger>
              <SelectContent alignItemWithTrigger={false}>
                <SelectGroup>
                  {Object.entries(locales).map(([value, label]) => (
                    <SelectItem key={value} value={value}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </Field>
          <Field>
            <FieldLabel htmlFor="interface-theme">
              <Trans>主題</Trans>
            </FieldLabel>
            <Select
              value={activeTheme}
              onValueChange={(value) => {
                if (isTheme(value)) {
                  setTheme(value);
                }
              }}
            >
              <SelectTrigger id="interface-theme" className="w-full">
                <SelectValue>
                  {activeTheme === "system" ? (
                    <Trans>跟隨系統</Trans>
                  ) : activeTheme === "light" ? (
                    <Trans>亮色</Trans>
                  ) : (
                    <Trans>暗色</Trans>
                  )}
                </SelectValue>
              </SelectTrigger>
              <SelectContent alignItemWithTrigger={false}>
                <SelectGroup>
                  <SelectItem value="system">
                    <Trans>跟隨系統</Trans>
                  </SelectItem>
                  <SelectItem value="light">
                    <Trans>亮色</Trans>
                  </SelectItem>
                  <SelectItem value="dark">
                    <Trans>暗色</Trans>
                  </SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>
          </Field>
        </FieldGroup>
      </SettingsSection>

      <SettingsSection
        title={<Trans>模型</Trans>}
        description={<Trans>管理 MOSS 模型下載與檔案。</Trans>}
      >
        <ModelCard
          model={model}
          downloadProgress={downloadProgress}
          isDownloading={isDownloading}
          deletingModel={deletingModel}
          onDownload={onDownload}
          onRedownload={onRedownload}
          onDelete={onDelete}
          onRevealModel={onRevealModel}
        />
      </SettingsSection>

      <SettingsSection
        title={<Trans>工具狀態</Trans>}
        description={<Trans>音訊和影片會先由本機工具轉為可處理的格式。</Trans>}
      >
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
              <p className="text-muted-foreground text-sm">
                <Trans>請安裝 FFmpeg 後重新檢查。</Trans>
              </p>
            ) : null}
            <Button variant="outline" onClick={onRefreshFfmpeg}>
              <RefreshCwIcon data-icon="inline-start" />
              <Trans>重新檢查</Trans>
            </Button>
          </CardContent>
        </Card>
      </SettingsSection>

      <SettingsSection
        title={<Trans>執行環境</Trans>}
        description={
          <Trans>MOSS 在 Apple Silicon 上透過 MLX 與 Metal 執行。</Trans>
        }
      >
        <div className="flex flex-col gap-2 text-sm">
          <div className="flex justify-between gap-3">
            <span className="text-muted-foreground">MLX</span>
            <span>
              {system.mlxAvailable ? (
                <Trans>可用</Trans>
              ) : (
                <Trans>不可用</Trans>
              )}
            </span>
          </div>
          <Separator />
          <div className="flex justify-between gap-3">
            <span className="text-muted-foreground">Metal</span>
            <span className="truncate">
              {system.metalDevice ?? <Trans>未偵測到</Trans>}
            </span>
          </div>
          <Separator />
          <div className="flex justify-between gap-3">
            <span className="text-muted-foreground">App version</span>
            <span>{system.appVersion}</span>
          </div>
        </div>
      </SettingsSection>
    </div>
  );
}
