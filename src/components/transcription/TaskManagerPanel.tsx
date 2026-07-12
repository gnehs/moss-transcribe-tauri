import type { Dispatch, ReactNode, SetStateAction } from "react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { msg } from "@lingui/core/macro";
import { useLingui } from "@lingui/react";
import { Trans } from "@lingui/react/macro";
import {
  AudioLinesIcon,
  Clock3Icon,
  FileAudioIcon,
  FileOutputIcon,
  FolderOpenIcon,
  HashIcon,
  HourglassIcon,
  ListPlusIcon,
  RotateCcwIcon,
  TimerIcon,
  Trash2Icon,
  TriangleAlertIcon,
  type LucideIcon,
} from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import {
  Field,
  FieldContent,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldTitle,
} from "@/components/ui/field";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
  InputGroupTextarea,
} from "@/components/ui/input-group";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CircleIndicator } from "@/components/transcription/CircleIndicator";
import {
  basename,
  formatDuration,
  formatElapsedClock,
  formatTimestamp,
} from "@/lib/format";
import { cn } from "@/lib/utils";
import type {
  DownloadProgress,
  TaskDraft,
  TaskStatus,
  TimedTaskStage,
  TranscriptionTask,
} from "@/types/transcription";

const stageProgressRanges: Array<{ stage: TimedTaskStage; range: string }> = [
  { stage: "preparing", range: "0–1%" },
  { stage: "encoding", range: "1–2%" },
  { stage: "prefilling", range: "2–3%" },
  { stage: "generating", range: "3–99%" },
];

function StatusLabel({ status }: { status: TaskStatus }) {
  switch (status) {
    case "queued":
      return <Trans>排隊中</Trans>;
    case "preparing":
      return <Trans>準備中</Trans>;
    case "encoding":
      return <Trans>編碼中</Trans>;
    case "prefilling":
      return <Trans>預填中</Trans>;
    case "generating":
      return <Trans>產生中</Trans>;
    case "completed":
      return <Trans>完成</Trans>;
    case "failed":
      return <Trans>失敗</Trans>;
  }
}

function statusVariant(status: TaskStatus) {
  if (status === "failed") return "destructive" as const;
  if (status === "completed") return "secondary" as const;
  return status === "queued" ? ("outline" as const) : ("default" as const);
}

function taskElapsedMs(task: TranscriptionTask, now: number) {
  if (task.startedAt == null) return null;
  return Math.max(0, (task.completedAt ?? now) - task.startedAt);
}

function taskStageElapsedMs(
  task: TranscriptionTask,
  stage: TimedTaskStage,
  now: number
) {
  const completed = task.stageTimings?.[stage] ?? 0;
  if (task.status === stage && task.stageStartedAt != null) {
    return completed + Math.max(0, now - task.stageStartedAt);
  }
  return completed > 0 ? completed : null;
}

function taskEtaMs(task: TranscriptionTask, now: number) {
  if (task.status !== "generating") return null;
  const generatedTokens = task.progress?.generatedTokens ?? 0;
  const estimatedTokens = task.progress?.estimatedGeneratedTokens ?? 0;
  const generatingElapsedMs = taskStageElapsedMs(task, "generating", now) ?? 0;
  if (
    generatedTokens < 64 ||
    estimatedTokens <= 0 ||
    generatingElapsedMs < 2_000
  )
    return null;

  const tokensPerMs = generatedTokens / generatingElapsedMs;
  return Math.max(0, estimatedTokens - generatedTokens) / tokensPerMs;
}

function TaskEtaBadge({ task, now }: { task: TranscriptionTask; now: number }) {
  if (
    !["preparing", "encoding", "prefilling", "generating"].includes(task.status)
  )
    return null;
  const etaMs = taskEtaMs(task, now);
  return (
    <>
      <HourglassIcon data-icon="inline-start" className="size-4" />
      {etaMs == null ? (
        <Trans>ETA 計算中</Trans>
      ) : (
        <>ETA {formatElapsedClock(etaMs)}</>
      )}
    </>
  );
}

function TaskOutputCheckbox({
  id,
  label,
  description,
  checked,
  disabled,
  onCheckedChange,
}: {
  id: string;
  label: ReactNode;
  description: ReactNode;
  checked: boolean;
  disabled?: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  const labelId = `${id}-label`;

  return (
    <Field orientation="horizontal">
      <FieldContent>
        <FieldTitle id={labelId}>{label}</FieldTitle>
        <FieldDescription>{description}</FieldDescription>
      </FieldContent>
      <Checkbox
        id={id}
        aria-labelledby={labelId}
        disabled={disabled}
        checked={checked}
        onCheckedChange={(next) => onCheckedChange(next)}
      />
    </Field>
  );
}

function TaskOutputSwitch({
  id,
  label,
  description,
  checked,
  disabled,
  onCheckedChange,
}: {
  id: string;
  label: ReactNode;
  description: ReactNode;
  checked: boolean;
  disabled?: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  const labelId = `${id}-label`;

  return (
    <Field orientation="horizontal">
      <FieldContent>
        <FieldTitle id={labelId}>{label}</FieldTitle>
        <FieldDescription>{description}</FieldDescription>
      </FieldContent>
      <Switch
        id={id}
        aria-labelledby={labelId}
        disabled={disabled}
        checked={checked}
        onCheckedChange={onCheckedChange}
      />
    </Field>
  );
}

function TaskDetailSectionTitle({
  icon: Icon,
  children,
}: {
  icon?: LucideIcon;
  children: ReactNode;
}) {
  return (
    <div className="flex min-w-0 items-center gap-2 text-sm leading-snug font-semibold">
      {Icon ? <Icon className="size-4 shrink-0" /> : null}
      {children}
    </div>
  );
}

function TaskDetailStat({
  icon: Icon,
  label,
  value,
  detail,
  className,
}: {
  icon: LucideIcon;
  label: ReactNode;
  value: ReactNode;
  detail?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex min-w-0 items-start gap-2.5 py-2 leading-none",
        className
      )}
    >
      <div
        className="bg-foreground/10 text-foreground grid size-[30px] shrink-0 place-items-center rounded-md [&_svg]:size-4"
        aria-hidden="true"
      >
        <Icon />
      </div>
      <div className="min-w-0">
        <div className="text-muted-foreground truncate text-xs leading-snug">
          {label}
        </div>
        <div
          className={cn(
            "text-base leading-snug font-semibold tabular-nums",
            detail && "flex flex-wrap items-center gap-x-2 gap-y-0.5"
          )}
        >
          {value}
          {detail ? (
            <div className="text-muted-foreground flex min-w-0 items-center gap-1 text-xs leading-snug font-normal">
              {detail}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function TaskManagerPanel({
  tasks,
  taskDraft,
  isTaskDialogOpen,
  isDraggingFiles,
  isConfirmingTasks,
  downloadProgress,
  selectedTaskId,
  onPickFiles,
  onPickOutputDir,
  onTaskDraftChange,
  onTaskDialogOpenChange,
  onConfirmTaskDraft,
  onRetryTask,
  onRemoveTask,
  onSelectedTaskChange,
}: {
  tasks: TranscriptionTask[];
  taskDraft: TaskDraft;
  isTaskDialogOpen: boolean;
  isDraggingFiles: boolean;
  isConfirmingTasks: boolean;
  downloadProgress: DownloadProgress | null;
  selectedTaskId: string | null;
  onPickFiles: () => void;
  onPickOutputDir: () => void;
  onTaskDraftChange: Dispatch<SetStateAction<TaskDraft>>;
  onTaskDialogOpenChange: (open: boolean) => void;
  onConfirmTaskDraft: () => void;
  onRetryTask: (taskId: string) => void;
  onRemoveTask: (taskId: string) => void;
  onSelectedTaskChange: (taskId: string | null) => void;
}) {
  const { i18n } = useLingui();
  const selectedTask = tasks.find((task) => task.id === selectedTaskId) ?? null;
  const [now, setNow] = useState(Date.now());
  const hasActiveTasks = tasks.some((task) =>
    ["preparing", "encoding", "prefilling", "generating"].includes(task.status)
  );
  const enabledOutputFormats =
    (taskDraft.outputs.txt ? 1 : 0) +
    (taskDraft.outputs.json ? 1 : 0) +
    (taskDraft.outputs.srt ? 1 : 0);
  const hasSelectedOutput = enabledOutputFormats > 0;
  const canSubmitTask = taskDraft.inputPaths.length > 0 && hasSelectedOutput;
  const isSingleOutputSelected = enabledOutputFormats === 1;
  const isModelDownloadPending = isConfirmingTasks;
  const modelDownloadPercent = Math.max(
    0,
    Math.min(100, downloadProgress?.percent ?? 0)
  );

  const onOutputCheckedChange = (
    key: keyof TaskDraft["outputs"],
    checked: boolean
  ) => {
    if (!checked && isSingleOutputSelected) return;

    onTaskDraftChange((current) => ({
      ...current,
      outputs: { ...current.outputs, [key]: checked },
    }));
  };

  useEffect(() => {
    if (!hasActiveTasks) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [hasActiveTasks]);

  return (
    <>
      <section
        className="flex h-full min-h-0 flex-col"
        aria-label={i18n._(msg`轉錄任務`)}
      >
        <ScrollArea className="min-h-0 flex-1" viewportClassName="scroll-fade">
          {tasks.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="pl-[18px] max-[720px]:pl-3">
                    <Trans>檔案</Trans>
                  </TableHead>
                  <TableHead>
                    <Trans>狀態</Trans>
                  </TableHead>
                  <TableHead>
                    <Trans>進度</Trans>
                  </TableHead>
                  <TableHead>
                    <Trans>設定</Trans>
                  </TableHead>
                  <TableHead className="pr-[18px] text-right max-[720px]:pr-3">
                    <Trans>動作</Trans>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {tasks.map((task) => (
                  <TableRow
                    key={task.id}
                    className="data-[selected=true]:bg-primary/5 cursor-pointer"
                    data-selected={selectedTask?.id === task.id}
                    onClick={() => onSelectedTaskChange(task.id)}
                  >
                    <TableCell className="w-[42%] max-w-0 min-w-0 pl-[18px] max-[720px]:min-w-[180px] max-[720px]:pl-3">
                      <div className="truncate font-medium">
                        {task.fileName}
                      </div>
                      <div className="text-muted-foreground truncate text-xs">
                        {task.options.outputDir || <Trans>來源資料夾</Trans>}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          task.result?.truncated
                            ? "destructive"
                            : statusVariant(task.status)
                        }
                      >
                        {task.result?.truncated ? (
                          <Trans>結果不完整</Trans>
                        ) : (
                          <StatusLabel status={task.status} />
                        )}
                      </Badge>
                    </TableCell>
                    <TableCell className="min-w-[170px]">
                      <div className="mb-1.5 flex flex-wrap items-center justify-between gap-1 tabular-nums">
                        <div className="flex items-center gap-1 opacity-50">
                          <TaskEtaBadge task={task} now={now} />
                        </div>
                        <div>{`${task.percent.toFixed(0)}%`}</div>
                      </div>
                      <Progress
                        value={task.percent}
                        aria-label={i18n._(msg`任務進度`)}
                      />
                    </TableCell>
                    <TableCell className="max-w-[180px] max-[720px]:min-w-[180px]">
                      <div className="truncate">
                        {Object.entries(task.options.outputs)
                          .filter(([, enabled]) => enabled)
                          .map(([format]) => format.toUpperCase())
                          .join(" · ") || <Trans>不輸出檔案</Trans>}
                      </div>
                    </TableCell>
                    <TableCell className="pr-[18px] text-right max-[720px]:pr-3">
                      <div className="flex items-center justify-end gap-2">
                        {task.status === "failed" || task.result?.truncated ? (
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            onClick={(event) => {
                              event.stopPropagation();
                              onRetryTask(task.id);
                            }}
                          >
                            <RotateCcwIcon data-icon="inline-start" />
                            <span className="sr-only">
                              <Trans>重試</Trans>
                            </span>
                          </Button>
                        ) : null}
                        {task.status !== "preparing" &&
                        task.status !== "encoding" &&
                        task.status !== "prefilling" &&
                        task.status !== "generating" ? (
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            onClick={(event) => {
                              event.stopPropagation();
                              onRemoveTask(task.id);
                            }}
                          >
                            <Trash2Icon data-icon="inline-start" />
                            <span className="sr-only">
                              <Trans>移除</Trans>
                            </span>
                          </Button>
                        ) : null}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <Empty className="h-full min-h-[360px] justify-between gap-6 pb-6">
              <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-4">
                <EmptyHeader>
                  <EmptyMedia className="size-16 rounded-xl" variant="icon">
                    <FileAudioIcon className="size-8" />
                  </EmptyMedia>
                  <EmptyTitle className="text-2xl leading-tight font-semibold">
                    <Trans>尚無任務</Trans>
                  </EmptyTitle>
                  <EmptyDescription>
                    <Trans>拖放檔案至此來建立任務</Trans>
                  </EmptyDescription>
                </EmptyHeader>
                <EmptyContent>
                  <Button onClick={onPickFiles}>
                    <ListPlusIcon data-icon="inline-start" />
                    {isDraggingFiles ? (
                      <Trans>放開以加入任務</Trans>
                    ) : (
                      <Trans>選擇檔案</Trans>
                    )}
                  </Button>
                </EmptyContent>
              </div>
              <p className="text-muted-foreground m-0 text-center text-sm/relaxed">
                wav、mp3、m4a、aac、flac、ogg、mp4、mov、mkv、webm
              </p>
            </Empty>
          )}
        </ScrollArea>
      </section>

      <TaskDetailSheet
        task={selectedTask}
        now={now}
        onRetryTask={onRetryTask}
        onOpenChange={(open) => !open && onSelectedTaskChange(null)}
      />

      <Dialog
        open={isTaskDialogOpen}
        onOpenChange={(open) => {
          if (!isModelDownloadPending) onTaskDialogOpenChange(open);
        }}
      >
        <DialogContent
          className="w-[min(680px,calc(100vw-2rem))] max-w-[min(680px,calc(100vw-2rem))]"
          showCloseButton={!isModelDownloadPending}
        >
          {isModelDownloadPending ? (
            <>
              <DialogHeader>
                <DialogTitle>
                  <Trans>正在下載模型</Trans>
                </DialogTitle>
                <DialogDescription>
                  <Trans>模型下載完成後，任務會自動加入佇列。</Trans>
                </DialogDescription>
              </DialogHeader>
              <div className="flex flex-col gap-3 py-2">
                <Progress
                  value={modelDownloadPercent}
                  aria-label={i18n._(msg`模型下載進度`)}
                />
                <div className="text-muted-foreground flex min-w-0 items-center justify-between gap-4 text-sm">
                  <span className="min-w-0 truncate">
                    {downloadProgress?.currentFile ?? (
                      <Trans>正在準備下載</Trans>
                    )}
                  </span>
                  <span className="shrink-0 tabular-nums">
                    {modelDownloadPercent.toFixed(0)}%
                  </span>
                </div>
              </div>
            </>
          ) : (
            <>
              <DialogHeader>
                <DialogTitle>
                  <Trans>新增轉錄任務</Trans>
                </DialogTitle>
              </DialogHeader>
              <div className="scroll-fade flex max-h-[min(620px,calc(100vh-220px))] min-h-0 flex-col gap-4 overflow-x-hidden overflow-y-auto p-1">
                <FieldGroup>
                  <Field>
                    <FieldLabel htmlFor="task-output-dir">
                      <span className="flex w-full items-center justify-between gap-2">
                        <Trans>輸出資料夾</Trans>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(event) => {
                            event.preventDefault();
                            onTaskDraftChange((current) => ({
                              ...current,
                              outputDir: "",
                            }));
                          }}
                          title={i18n._(msg`重置輸出資料夾`)}
                        >
                          <RotateCcwIcon className="size-3.5" />
                          <Trans>重置</Trans>
                        </Button>
                      </span>
                    </FieldLabel>
                    <InputGroup>
                      <InputGroupInput
                        id="task-output-dir"
                        readOnly
                        value={taskDraft.outputDir}
                        placeholder={i18n._(msg`預設使用來源檔案所在資料夾`)}
                      />
                      <InputGroupAddon align="inline-end">
                        <InputGroupButton onClick={onPickOutputDir}>
                          <FolderOpenIcon data-icon="inline-start" />
                          <Trans>選取</Trans>
                        </InputGroupButton>
                      </InputGroupAddon>
                    </InputGroup>
                  </Field>
                  <TaskOutputCheckbox
                    id="task-write-txt"
                    label={<Trans>輸出 TXT</Trans>}
                    description={<Trans>純文字轉錄稿。</Trans>}
                    checked={taskDraft.outputs.txt}
                    disabled={isSingleOutputSelected && taskDraft.outputs.txt}
                    onCheckedChange={(checked) =>
                      onOutputCheckedChange("txt", checked)
                    }
                  />
                  <TaskOutputCheckbox
                    id="task-write-json"
                    label={<Trans>輸出 JSON</Trans>}
                    description={<Trans>保留說話者與時間區段。</Trans>}
                    checked={taskDraft.outputs.json}
                    disabled={isSingleOutputSelected && taskDraft.outputs.json}
                    onCheckedChange={(checked) =>
                      onOutputCheckedChange("json", checked)
                    }
                  />
                  <TaskOutputCheckbox
                    id="task-write-srt"
                    label={<Trans>輸出 SRT</Trans>}
                    description={<Trans>建立字幕檔。</Trans>}
                    checked={taskDraft.outputs.srt}
                    disabled={isSingleOutputSelected && taskDraft.outputs.srt}
                    onCheckedChange={(checked) =>
                      onOutputCheckedChange("srt", checked)
                    }
                  />
                  {!hasSelectedOutput ? (
                    <FieldDescription className="text-destructive">
                      <Trans>至少要勾選一個輸出格式。</Trans>
                    </FieldDescription>
                  ) : null}
                  <TaskOutputSwitch
                    id="task-convert-traditional"
                    label={<Trans>簡體轉繁體</Trans>}
                    description={<Trans>將轉錄結果轉為台灣繁體中文。</Trans>}
                    checked={taskDraft.convertToTraditional}
                    onCheckedChange={(checked) =>
                      onTaskDraftChange((current) => ({
                        ...current,
                        convertToTraditional: checked,
                      }))
                    }
                  />
                  <Field>
                    <FieldLabel htmlFor="task-prompt">
                      <Trans>提示詞</Trans>
                    </FieldLabel>
                    <InputGroup>
                      <InputGroupTextarea
                        id="task-prompt"
                        value={taskDraft.prompt}
                        placeholder={i18n._(
                          msg`留空使用 MOSS 預設提示詞；也可加入專有名詞`
                        )}
                        onChange={(event) =>
                          onTaskDraftChange((current) => ({
                            ...current,
                            prompt: event.target.value,
                          }))
                        }
                      />
                    </InputGroup>
                  </Field>
                </FieldGroup>
                <ScrollArea className="h-40 w-full min-w-0 rounded-lg border">
                  <div className="flex min-h-full min-w-0 flex-col">
                    {taskDraft.inputPaths.map((path) => (
                      <div
                        key={path}
                        className="flex min-w-0 items-center gap-2 overflow-hidden border-b px-2.5 py-2 text-sm last:border-b-0"
                      >
                        <FileAudioIcon className="text-muted-foreground size-4 shrink-0" />
                        <span
                          className="min-w-0 flex-1 truncate"
                          title={basename(path)}
                        >
                          {basename(path)}
                        </span>
                      </div>
                    ))}
                    {taskDraft.inputPaths.length === 0 ? (
                      <div className="text-muted-foreground px-3 py-2 text-sm">
                        <Trans>尚未新增檔案</Trans>
                      </div>
                    ) : null}
                  </div>
                </ScrollArea>
              </div>
              <DialogFooter>
                <Button
                  variant="outline"
                  onClick={() => onTaskDialogOpenChange(false)}
                >
                  <Trans>取消</Trans>
                </Button>
                <Button
                  disabled={!canSubmitTask || isConfirmingTasks}
                  onClick={onConfirmTaskDraft}
                >
                  <ListPlusIcon data-icon="inline-start" />
                  {isConfirmingTasks ? (
                    <Trans>加入中</Trans>
                  ) : (
                    <Trans>加入任務</Trans>
                  )}
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

function TaskDetailSheet({
  task,
  now,
  onRetryTask,
  onOpenChange,
}: {
  task: TranscriptionTask | null;
  now: number;
  onRetryTask: (taskId: string) => void;
  onOpenChange: (open: boolean) => void;
}) {
  const result = task?.result;
  const transcript = result ?? task?.stream;
  const [activeTab, setActiveTab] = useState("statistics");
  const scrollViewportRef = useRef<HTMLDivElement>(null);
  const elapsedMs = task ? taskElapsedMs(task, now) : null;
  const audioDurationMs =
    task?.progress?.audioDurationMs ?? (result ? result.audioDurationMs : null);
  const taskPercent = task ? Math.max(0, Math.min(100, task.percent)) : 0;
  const outputFormats = task
    ? Object.entries(task.options.outputs)
        .filter(([, enabled]) => enabled)
        .map(([format]) => format.toUpperCase())
    : [];

  useEffect(() => {
    setActiveTab("statistics");
  }, [task?.id]);

  useLayoutEffect(() => {
    if (activeTab !== "transcript") return;
    const viewport = scrollViewportRef.current;
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
  }, [activeTab, task?.id, task?.stream?.generatedTokens]);

  return (
    <Sheet open={Boolean(task)} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="border-border gap-0 border-l data-[side=right]:w-[min(720px,100vw)] data-[side=right]:sm:max-w-[min(720px,100vw)]"
      >
        {task ? (
          <Tabs
            key={task.id}
            value={activeTab}
            onValueChange={setActiveTab}
            className="min-h-0 min-w-0 flex-1 gap-0"
          >
            <SheetHeader className="bg-foreground/[0.03] gap-3 px-6 pt-5 pr-12 pb-4 max-[720px]:px-4">
              <div className="flex min-w-0 items-center gap-3">
                <div className="min-w-0">
                  <div className="text-muted-foreground text-xs leading-tight font-semibold tracking-[0.04em]">
                    <Trans>任務詳情</Trans>
                  </div>
                  <SheetTitle className="mt-0.5 truncate text-lg leading-snug">
                    {task.fileName}
                  </SheetTitle>
                </div>
              </div>
              <SheetDescription className="text-muted-foreground flex min-w-0 flex-col gap-2 text-xs">
                <span className="flex min-w-0 items-center gap-2">
                  <CircleIndicator
                    progress={taskPercent}
                    size={20}
                    thickness={10}
                    color="var(--foreground)"
                    trackColor="var(--ring)"
                  />
                  <span className="text-foreground shrink-0 font-medium tabular-nums">
                    {taskPercent.toFixed(0)}%
                  </span>
                  <span aria-hidden="true">·</span>
                  <span className="text-foreground shrink-0">
                    {result?.truncated ? (
                      <Trans>結果不完整</Trans>
                    ) : (
                      <StatusLabel status={task.status} />
                    )}
                  </span>
                  <span aria-hidden="true">·</span>
                  <span className="min-w-0 shrink-0">
                    {formatDuration(audioDurationMs)}
                  </span>
                  <span aria-hidden="true">·</span>
                  <span className="min-w-0 truncate">
                    {outputFormats.length ? (
                      outputFormats.join(" · ")
                    ) : (
                      <Trans>不輸出檔案</Trans>
                    )}
                  </span>
                </span>
              </SheetDescription>
            </SheetHeader>
            <div className="bg-foreground/[0.03] px-6 max-[720px]:px-4">
              <TabsList
                variant="line"
                className="w-full justify-start gap-[18px] rounded-none p-0"
              >
                <TabsTrigger
                  className="flex-none px-0 pt-2 pb-2"
                  value="statistics"
                >
                  <Trans>統計資訊</Trans>
                </TabsTrigger>
                <TabsTrigger
                  className="flex-none px-0 pt-2 pb-2"
                  value="transcript"
                >
                  <Trans>逐字稿</Trans>
                </TabsTrigger>
              </TabsList>
            </div>
            <Separator />
            <ScrollArea
              className="min-h-0 flex-1 overflow-hidden px-6 pt-5 pb-6 max-[720px]:px-4"
              viewportClassName="scroll-fade"
              viewportRef={scrollViewportRef}
            >
              <div className="flex min-w-0 flex-col">
                <TabsContent
                  value="statistics"
                  className="flex min-w-0 flex-col gap-5 text-sm outline-none"
                >
                  <div className="grid grid-cols-2 max-[720px]:grid-cols-1">
                    <TaskDetailStat
                      icon={TimerIcon}
                      label={<Trans>任務耗時</Trans>}
                      value={formatDuration(elapsedMs)}
                      detail={<TaskEtaBadge task={task} now={now} />}
                    />
                    <TaskDetailStat
                      icon={AudioLinesIcon}
                      label={<Trans>音訊長度</Trans>}
                      value={
                        audioDurationMs == null
                          ? "—"
                          : formatDuration(audioDurationMs)
                      }
                    />
                    <TaskDetailStat
                      icon={HashIcon}
                      label="Prompt tokens"
                      value={
                        task.progress?.promptTokens ?? result?.promptTokens ?? 0
                      }
                    />
                    <TaskDetailStat
                      icon={HashIcon}
                      label="Generated tokens"
                      value={
                        task.progress?.generatedTokens ??
                        result?.generatedTokens ??
                        0
                      }
                    />
                  </div>
                  <div className="flex min-w-0 flex-col gap-2.5">
                    <TaskDetailSectionTitle icon={Clock3Icon}>
                      <Trans>階段耗時</Trans>
                    </TaskDetailSectionTitle>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="text-muted-foreground h-8 py-0 text-xs">
                            <Trans>階段</Trans>
                          </TableHead>
                          <TableHead className="text-muted-foreground h-8 py-0 text-xs">
                            <Trans>進度範圍</Trans>
                          </TableHead>
                          <TableHead className="text-muted-foreground h-8 py-0 text-right text-xs">
                            <Trans>耗時</Trans>
                          </TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {stageProgressRanges.map(({ stage, range }) => (
                          <TableRow key={stage}>
                            <TableCell className="py-2.5 font-medium">
                              <StatusLabel status={stage} />
                            </TableCell>
                            <TableCell className="py-2.5 font-mono">
                              {range}
                            </TableCell>
                            <TableCell className="py-2.5 text-right font-mono">
                              {formatElapsedClock(
                                taskStageElapsedMs(task, stage, now)
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                  {result ? (
                    <div className="flex min-w-0 flex-col gap-2.5">
                      <TaskDetailSectionTitle icon={FileOutputIcon}>
                        <Trans>輸出檔案</Trans>
                      </TaskDetailSectionTitle>
                      <div className="border-t">
                        {[
                          ["TXT", result.outputs.txtPath],
                          ["JSON", result.outputs.jsonPath],
                          ["SRT", result.outputs.srtPath],
                        ].map(([format, path]) => (
                          <div
                            key={format}
                            className="flex min-w-0 items-center gap-2.5 border-b px-0 py-2.5 text-[0.8125rem] last:border-b-0"
                          >
                            <span className="text-muted-foreground min-w-[42px] text-[0.6875rem] font-semibold tracking-[0.04em]">
                              {format}
                            </span>
                            <span className="text-foreground min-w-0 truncate">
                              {path ?? <Trans>未輸出</Trans>}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </TabsContent>
                <TabsContent
                  value="transcript"
                  className="flex min-w-0 flex-col gap-5 text-sm outline-none"
                >
                  {transcript ? (
                    transcript.segments.length > 0 ? (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>
                              <Trans>時間</Trans>
                            </TableHead>
                            <TableHead>
                              <Trans>說話者</Trans>
                            </TableHead>
                            <TableHead>
                              <Trans>文字</Trans>
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {transcript.segments.map((segment, index) => (
                            <TableRow key={`${segment.start}-${index}`}>
                              <TableCell className="text-muted-foreground text-xs whitespace-nowrap">
                                {formatTimestamp(segment.start * 1000)} –{" "}
                                {formatTimestamp(segment.end * 1000)}
                              </TableCell>
                              <TableCell className="font-mono text-xs whitespace-nowrap">
                                {segment.speaker}
                              </TableCell>
                              <TableCell className="whitespace-pre-wrap">
                                {segment.text}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    ) : (
                      <p className="m-0 leading-relaxed break-words whitespace-pre-wrap">
                        {transcript.text || <Trans>沒有逐字稿內容</Trans>}
                      </p>
                    )
                  ) : (
                    <p className="text-muted-foreground text-sm">
                      <Trans>逐字稿會在模型生成時顯示。</Trans>
                    </p>
                  )}
                </TabsContent>
                {task.error || result?.truncated || task.status === "failed" ? (
                  <div className="flex min-w-0 flex-col gap-3 border-t pt-4">
                    {task.error ? (
                      <p className="text-destructive text-sm">{task.error}</p>
                    ) : null}
                    {result?.truncated ? (
                      <Alert variant="destructive">
                        <TriangleAlertIcon />
                        <AlertTitle>
                          <Trans>結果不完整</Trans>
                        </AlertTitle>
                        <AlertDescription>
                          <Trans>
                            模型已達單次轉錄上限。已保留目前逐字稿；請將較長音訊分段後，重新轉錄遺失的部分。
                          </Trans>
                        </AlertDescription>
                      </Alert>
                    ) : null}
                    {task.status === "failed" || result?.truncated ? (
                      <Button
                        variant="outline"
                        onClick={() => onRetryTask(task.id)}
                      >
                        <RotateCcwIcon data-icon="inline-start" />
                        <Trans>重新執行</Trans>
                      </Button>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </ScrollArea>
          </Tabs>
        ) : null}
      </SheetContent>
    </Sheet>
  );
}
