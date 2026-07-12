import type { Dispatch, SetStateAction } from "react";
import { useEffect, useState } from "react";
import { msg } from "@lingui/core/macro";
import { useLingui } from "@lingui/react";
import { Trans } from "@lingui/react/macro";
import {
  Clock3Icon, FileAudioIcon, FolderOpenIcon, HourglassIcon, ListPlusIcon, RotateCcwIcon, Trash2Icon, TriangleAlertIcon,
} from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Field, FieldContent, FieldDescription, FieldGroup, FieldLabel, FieldTitle } from "@/components/ui/field";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput, InputGroupTextarea } from "@/components/ui/input-group";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { basename, formatDuration, formatElapsedClock, formatTimestamp } from "@/lib/format";
import type { TaskDraft, TaskStatus, TimedTaskStage, TranscriptionTask } from "@/types/transcription";

const stageProgressRanges: Array<{ stage: TimedTaskStage; range: string }> = [
  { stage: "preparing", range: "0–1%" },
  { stage: "encoding", range: "1–2%" },
  { stage: "prefilling", range: "2–3%" },
  { stage: "generating", range: "3–99%" },
];

function StatusLabel({ status }: { status: TaskStatus }) {
  switch (status) {
    case "queued": return <Trans>排隊中</Trans>;
    case "preparing": return <Trans>準備中</Trans>;
    case "encoding": return <Trans>編碼中</Trans>;
    case "prefilling": return <Trans>預填中</Trans>;
    case "generating": return <Trans>產生中</Trans>;
    case "completed": return <Trans>完成</Trans>;
    case "failed": return <Trans>失敗</Trans>;
  }
}

function statusVariant(status: TaskStatus) {
  if (status === "failed") return "destructive" as const;
  if (status === "completed") return "secondary" as const;
  return status === "queued" ? "outline" as const : "default" as const;
}

function taskElapsedMs(task: TranscriptionTask, now: number) {
  if (task.startedAt == null) return null;
  return Math.max(0, (task.completedAt ?? now) - task.startedAt);
}

function taskStageElapsedMs(task: TranscriptionTask, stage: TimedTaskStage, now: number) {
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
  if (generatedTokens < 64 || estimatedTokens <= 0 || generatingElapsedMs < 2_000) return null;

  const tokensPerMs = generatedTokens / generatingElapsedMs;
  return Math.max(0, estimatedTokens - generatedTokens) / tokensPerMs;
}

function formatSrtTimestamp(seconds: number) {
  const totalMs = Math.round(Math.max(0, seconds) * 1000);
  const hours = Math.floor(totalMs / 3_600_000);
  const minutes = Math.floor((totalMs % 3_600_000) / 60_000);
  const remainingSeconds = Math.floor((totalMs % 60_000) / 1_000);
  const milliseconds = totalMs % 1_000;
  return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")},${milliseconds.toString().padStart(3, "0")}`;
}

function renderSrtPreview(task: TranscriptionTask) {
  const segments = task.result?.segments ?? [];
  return segments
    .map((segment, index) => `${index + 1}\n${formatSrtTimestamp(segment.start)} --> ${formatSrtTimestamp(segment.end)}\n[${segment.speaker}] ${segment.text}\n`)
    .join("\n");
}

function TaskEtaBadge({ task, now }: { task: TranscriptionTask; now: number }) {
  if (!["preparing", "encoding", "prefilling", "generating"].includes(task.status)) return null;
  const etaMs = taskEtaMs(task, now);
  return (
    <>
      <HourglassIcon data-icon="inline-start" className="size-4" />
      {etaMs == null ? <Trans>ETA 計算中</Trans> : <>ETA {formatElapsedClock(etaMs)}</>}
    </>
  );
}

export function TaskManagerPanel({
  tasks, taskDraft, isTaskDialogOpen, isDraggingFiles, isConfirmingTasks,
  selectedTaskId, onPickFiles, onPickOutputDir, onTaskDraftChange,
  onTaskDialogOpenChange, onConfirmTaskDraft, onRetryTask, onRemoveTask, onSelectedTaskChange,
}: {
  tasks: TranscriptionTask[];
  taskDraft: TaskDraft;
  isTaskDialogOpen: boolean;
  isDraggingFiles: boolean;
  isConfirmingTasks: boolean;
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
  const hasActiveTasks = tasks.some((task) => ["preparing", "encoding", "prefilling", "generating"].includes(task.status));

  useEffect(() => {
    if (!hasActiveTasks) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [hasActiveTasks]);

  return (
    <>
      <section className="task-workspace" aria-label={i18n._(msg`轉錄任務`)}>
        <ScrollArea className="task-table-wrap" viewportClassName="scroll-fade">
          {tasks.length ? (
            <Table className="task-table">
              <TableHeader><TableRow>
                <TableHead><Trans>檔案</Trans></TableHead><TableHead><Trans>狀態</Trans></TableHead><TableHead><Trans>進度</Trans></TableHead><TableHead><Trans>設定</Trans></TableHead><TableHead className="text-right"><Trans>動作</Trans></TableHead>
              </TableRow></TableHeader>
              <TableBody>{tasks.map((task) => (
                <TableRow key={task.id} className="task-row" data-selected={selectedTask?.id === task.id} onClick={() => onSelectedTaskChange(task.id)}>
                  <TableCell className="task-name-cell"><div className="truncate font-medium">{task.fileName}</div><div className="truncate text-xs text-muted-foreground">{task.options.outputDir || <Trans>來源資料夾</Trans>}</div></TableCell>
                  <TableCell><Badge variant={task.result?.truncated ? "destructive" : statusVariant(task.status)}>{task.result?.truncated ? <Trans>結果不完整</Trans> : <StatusLabel status={task.status} />}</Badge></TableCell>
                  <TableCell className="task-progress-cell">

                    <div className="mb-1.5 flex flex-wrap justify-between gap-1 items-center tabular-nums">
                      <div className="flex items-center gap-1 opacity-50">
                        <TaskEtaBadge task={task} now={now} />
                      </div>
                      <div>
                        {`${task.percent.toFixed(0)}%`}
                      </div>
                    </div>
                    <Progress value={task.percent} aria-label={i18n._(msg`任務進度`)} />
                  </TableCell>
                  <TableCell className="task-options-cell"><div className="truncate">{Object.entries(task.options.outputs).filter(([, enabled]) => enabled).map(([format]) => format.toUpperCase()).join(" · ") || <Trans>不輸出檔案</Trans>}</div></TableCell>
                  <TableCell className="text-right"><div className="task-row-actions">
                    {task.status === "failed" || task.result?.truncated ? <Button variant="ghost" size="icon-sm" onClick={(event) => { event.stopPropagation(); onRetryTask(task.id); }}><RotateCcwIcon data-icon="inline-start" /><span className="sr-only"><Trans>重試</Trans></span></Button> : null}
                    {task.status !== "preparing" && task.status !== "encoding" && task.status !== "prefilling" && task.status !== "generating" ? <Button variant="ghost" size="icon-sm" onClick={(event) => { event.stopPropagation(); onRemoveTask(task.id); }}><Trash2Icon data-icon="inline-start" /><span className="sr-only"><Trans>移除</Trans></span></Button> : null}
                  </div></TableCell>
                </TableRow>
              ))}</TableBody>
            </Table>
          ) : (
            <Empty className="task-empty-state"><div className="task-empty-main"><EmptyHeader><EmptyMedia variant="icon"><FileAudioIcon /></EmptyMedia><EmptyTitle><Trans>尚無任務</Trans></EmptyTitle><EmptyDescription><Trans>拖放檔案至此來建立任務</Trans></EmptyDescription></EmptyHeader><EmptyContent><Button onClick={onPickFiles}><ListPlusIcon data-icon="inline-start" />{isDraggingFiles ? <Trans>放開以加入任務</Trans> : <Trans>選擇檔案</Trans>}</Button></EmptyContent></div><p className="task-empty-supported">wav、mp3、m4a、aac、flac、ogg、mp4、mov、mkv、webm</p></Empty>
          )}
        </ScrollArea>
      </section>

      <TaskDetailSheet task={selectedTask} now={now} onRetryTask={onRetryTask} onOpenChange={(open) => !open && onSelectedTaskChange(null)} />

      <Dialog open={isTaskDialogOpen} onOpenChange={onTaskDialogOpenChange}>
        <DialogContent className="task-dialog">
          <DialogHeader><DialogTitle><Trans>新增轉錄任務</Trans></DialogTitle><DialogDescription><Trans>{taskDraft.inputPaths.length} 個檔案將依序使用固定的 MOSS 模型處理。</Trans></DialogDescription></DialogHeader>
          <div className="task-dialog-body scroll-fade"><FieldGroup>
            <Field><FieldLabel><Trans>模型</Trans></FieldLabel><InputGroup><InputGroupInput readOnly value="OpenMOSS-Team/MOSS-Transcribe-Diarize" /><InputGroupAddon align="inline-end"><Trans>固定</Trans></InputGroupAddon></InputGroup></Field>
            <Field><FieldLabel htmlFor="task-output-dir"><Trans>輸出資料夾</Trans></FieldLabel><InputGroup><InputGroupInput id="task-output-dir" readOnly value={taskDraft.outputDir} placeholder={i18n._(msg`預設使用來源檔案所在資料夾`)} /><InputGroupAddon align="inline-end"><InputGroupButton onClick={onPickOutputDir}><FolderOpenIcon data-icon="inline-start" /><Trans>選取</Trans></InputGroupButton></InputGroupAddon></InputGroup></Field>
            <Field orientation="horizontal"><FieldContent><FieldTitle id="task-write-txt"><Trans>輸出 TXT</Trans></FieldTitle><FieldDescription><Trans>純文字轉錄稿。</Trans></FieldDescription></FieldContent><Switch aria-labelledby="task-write-txt" checked={taskDraft.outputs.txt} onCheckedChange={(checked) => onTaskDraftChange((current) => ({ ...current, outputs: { ...current.outputs, txt: checked } }))} /></Field>
            <Field orientation="horizontal"><FieldContent><FieldTitle id="task-write-json"><Trans>輸出 JSON</Trans></FieldTitle><FieldDescription><Trans>保留說話者與時間區段。</Trans></FieldDescription></FieldContent><Switch aria-labelledby="task-write-json" checked={taskDraft.outputs.json} onCheckedChange={(checked) => onTaskDraftChange((current) => ({ ...current, outputs: { ...current.outputs, json: checked } }))} /></Field>
            <Field orientation="horizontal"><FieldContent><FieldTitle id="task-write-srt"><Trans>輸出 SRT</Trans></FieldTitle><FieldDescription><Trans>建立字幕檔。</Trans></FieldDescription></FieldContent><Switch aria-labelledby="task-write-srt" checked={taskDraft.outputs.srt} onCheckedChange={(checked) => onTaskDraftChange((current) => ({ ...current, outputs: { ...current.outputs, srt: checked } }))} /></Field>
            <Field orientation="horizontal"><FieldContent><FieldTitle id="task-convert-traditional"><Trans>簡體轉繁體</Trans></FieldTitle><FieldDescription><Trans>將轉錄結果轉為台灣繁體中文，套用於畫面與所有匯出檔。</Trans></FieldDescription></FieldContent><Switch aria-labelledby="task-convert-traditional" checked={taskDraft.convertToTraditional} onCheckedChange={(checked) => onTaskDraftChange((current) => ({ ...current, convertToTraditional: checked }))} /></Field>
            <Field><FieldLabel htmlFor="task-prompt"><Trans>提示詞</Trans></FieldLabel><InputGroup><InputGroupTextarea id="task-prompt" value={taskDraft.prompt} placeholder={i18n._(msg`留空使用 MOSS 預設提示詞；也可加入專有名詞`)} onChange={(event) => onTaskDraftChange((current) => ({ ...current, prompt: event.target.value }))} /></InputGroup></Field>
          </FieldGroup><Separator />
          <div className="task-draft-files">{taskDraft.inputPaths.map((path) => <div key={path} className="task-draft-file"><FileAudioIcon /><span className="truncate">{basename(path)}</span></div>)}</div></div>
          <DialogFooter><Button variant="outline" onClick={() => onTaskDialogOpenChange(false)}><Trans>取消</Trans></Button><Button disabled={!taskDraft.inputPaths.length || isConfirmingTasks} onClick={onConfirmTaskDraft}><ListPlusIcon data-icon="inline-start" />{isConfirmingTasks ? <Trans>加入中</Trans> : <Trans>加入任務</Trans>}</Button></DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function TaskDetailSheet({ task, now, onRetryTask, onOpenChange }: {
  task: TranscriptionTask | null;
  now: number;
  onRetryTask: (taskId: string) => void;
  onOpenChange: (open: boolean) => void;
}) {
  const { i18n } = useLingui();
  const result = task?.result;
  const elapsedMs = task ? taskElapsedMs(task, now) : null;
  const audioDurationMs = task?.progress?.audioDurationMs ?? (result ? result.audioDurationMs : null);
  const hasSrtOutput = Boolean(task?.options.outputs.srt || result?.outputs.srtPath);

  return (
    <Sheet open={Boolean(task)} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="gap-0 data-[side=right]:w-[min(720px,100vw)] data-[side=right]:sm:max-w-[min(720px,100vw)]">
        <SheetHeader className="pr-12">
          <SheetTitle><Trans>任務詳情</Trans></SheetTitle>
          <SheetDescription className="truncate">{task?.fileName}</SheetDescription>
        </SheetHeader>
        <Separator />
        {task ? (
          <ScrollArea className="task-detail-content" viewportClassName="scroll-fade">
            <div className="task-detail-stack">
              <div className="task-detail-overview">
                <Badge variant={result?.truncated ? "destructive" : statusVariant(task.status)}>{result?.truncated ? <Trans>結果不完整</Trans> : <StatusLabel status={task.status} />}</Badge>
                <div className="task-detail-overview-progress">
                  <Progress value={task.percent} aria-label={i18n._(msg`任務進度`)} />
                  <div className="flex justify-between gap-3 text-xs text-muted-foreground">
                    <span>{task.percent.toFixed(0)}%</span>
                    <span className="truncate">{task.message ?? ""}</span>
                  </div>
                </div>
              </div>
              <Tabs key={task.id} defaultValue="statistics" className="task-detail-tabs">
                <TabsList variant="line" className="task-detail-tabs-list">
                  <TabsTrigger value="statistics"><Trans>統計資訊</Trans></TabsTrigger>
                  <TabsTrigger value="transcript"><Trans>逐字稿</Trans></TabsTrigger>
                  {hasSrtOutput ? <TabsTrigger value="subtitle"><Trans>字幕</Trans></TabsTrigger> : null}
                </TabsList>
                <TabsContent value="statistics" className="task-detail-tab-content">
                  <div className="task-detail-stat-grid text-sm">
                    <div><div className="text-muted-foreground"><Trans>任務耗時</Trans></div><div className="flex flex-wrap items-center gap-1 tabular-nums"><Clock3Icon className="size-4" />{formatDuration(elapsedMs)}<TaskEtaBadge task={task} now={now} /></div></div>
                    <div><div className="text-muted-foreground"><Trans>音訊長度</Trans></div><div>{audioDurationMs == null ? "—" : formatDuration(audioDurationMs)}</div></div>
                    <div><div className="text-muted-foreground">Prompt tokens</div><div className="font-mono">{task.progress?.promptTokens ?? result?.promptTokens ?? 0}</div></div>
                    <div><div className="text-muted-foreground">Generated tokens</div><div className="font-mono">{task.progress?.generatedTokens ?? result?.generatedTokens ?? 0}</div></div>
                  </div>
                  <div className="flex flex-col gap-2">
                    <div className="font-medium"><Trans>階段耗時</Trans></div>
                    <Table>
                      <TableHeader><TableRow><TableHead><Trans>階段</Trans></TableHead><TableHead><Trans>進度範圍</Trans></TableHead><TableHead className="text-right"><Trans>耗時</Trans></TableHead></TableRow></TableHeader>
                      <TableBody>{stageProgressRanges.map(({ stage, range }) => (
                        <TableRow key={stage}>
                          <TableCell><StatusLabel status={stage} /></TableCell>
                          <TableCell className="font-mono">{range}</TableCell>
                          <TableCell className="text-right font-mono">{formatElapsedClock(taskStageElapsedMs(task, stage, now))}</TableCell>
                        </TableRow>
                      ))}</TableBody>
                    </Table>
                  </div>
                  {result ? (
                    <div className="flex flex-col gap-1 text-xs text-muted-foreground">
                      <div className="font-medium text-foreground"><Trans>輸出檔案</Trans></div>
                      <span>TXT: {result.outputs.txtPath ?? <Trans>未輸出</Trans>}</span>
                      <span>JSON: {result.outputs.jsonPath ?? <Trans>未輸出</Trans>}</span>
                      <span>SRT: {result.outputs.srtPath ?? <Trans>未輸出</Trans>}</span>
                    </div>
                  ) : null}
                </TabsContent>
                <TabsContent value="transcript" className="task-detail-tab-content">
                  {result ? (
                    result.segments.length > 0 ? (
                      <Table>
                        <TableHeader><TableRow><TableHead><Trans>時間</Trans></TableHead><TableHead><Trans>說話者</Trans></TableHead><TableHead><Trans>文字</Trans></TableHead></TableRow></TableHeader>
                        <TableBody>
                          {result.segments.map((segment, index) => (
                            <TableRow key={`${segment.start}-${index}`}>
                              <TableCell className="whitespace-nowrap text-xs text-muted-foreground">{formatTimestamp(segment.start * 1000)} – {formatTimestamp(segment.end * 1000)}</TableCell>
                              <TableCell className="whitespace-nowrap font-mono text-xs">{segment.speaker}</TableCell>
                              <TableCell className="whitespace-pre-wrap">{segment.text}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    ) : <p className="task-transcript-text">{result.text || <Trans>沒有逐字稿內容</Trans>}</p>
                  ) : <p className="text-sm text-muted-foreground"><Trans>任務完成後會顯示逐字稿。</Trans></p>}
                </TabsContent>
                {hasSrtOutput ? (
                  <TabsContent value="subtitle" className="task-detail-tab-content">
                    {result ? (
                      <pre className="srt-preview-text font-mono text-xs">{renderSrtPreview(task) || <Trans>沒有可用的字幕內容</Trans>}</pre>
                    ) : <p className="text-sm text-muted-foreground"><Trans>任務完成後會顯示字幕。</Trans></p>}
                  </TabsContent>
                ) : null}
              </Tabs>
              {task.error ? <p className="text-sm text-destructive">{task.error}</p> : null}
              {result?.truncated ? (
                <Alert variant="destructive">
                  <TriangleAlertIcon />
                  <AlertTitle><Trans>結果不完整</Trans></AlertTitle>
                  <AlertDescription><Trans>模型已達單次轉錄上限。已保留目前逐字稿；請將較長音訊分段後，重新轉錄遺失的部分。</Trans></AlertDescription>
                </Alert>
              ) : null}
              {task.status === "failed" || result?.truncated ? (
                <Button variant="outline" onClick={() => onRetryTask(task.id)}>
                  <RotateCcwIcon data-icon="inline-start" /><Trans>重新執行</Trans>
                </Button>
              ) : null}
            </div>
          </ScrollArea>
        ) : null}
      </SheetContent>
    </Sheet>
  );
}
