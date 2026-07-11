import type { Dispatch, SetStateAction } from "react";
import { useEffect, useState } from "react";
import { msg } from "@lingui/core/macro";
import { useLingui } from "@lingui/react";
import { Trans } from "@lingui/react/macro";
import {
  FileAudioIcon, FolderOpenIcon, ListPlusIcon, RotateCcwIcon, Trash2Icon,
} from "lucide-react";

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
import { Textarea } from "@/components/ui/textarea";
import { basename, formatDuration, formatTimestamp } from "@/lib/format";
import type { TaskDraft, TaskStatus, TranscriptionTask } from "@/types/transcription";

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

  useEffect(() => {
    if (!tasks.some((task) => ["preparing", "encoding", "prefilling", "generating"].includes(task.status))) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [tasks]);

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
                  <TableCell><Badge variant={statusVariant(task.status)}><StatusLabel status={task.status} /></Badge></TableCell>
                  <TableCell className="task-progress-cell"><div className="task-progress-stack"><Progress value={task.percent} /><div className="task-progress-meta"><span>{task.percent.toFixed(0)}%</span><span className="truncate">{task.message ?? ""}</span></div></div></TableCell>
                  <TableCell className="task-options-cell"><div className="truncate">{Object.entries(task.options.outputs).filter(([, enabled]) => enabled).map(([format]) => format.toUpperCase()).join(" · ") || <Trans>不輸出檔案</Trans>}</div><div className="truncate text-xs text-muted-foreground"><Trans>最多 {task.options.maxNewTokens} tokens</Trans></div></TableCell>
                  <TableCell className="text-right"><div className="task-row-actions">
                    {task.status === "failed" ? <Button variant="ghost" size="icon-sm" onClick={(event) => { event.stopPropagation(); onRetryTask(task.id); }}><RotateCcwIcon data-icon="inline-start" /><span className="sr-only"><Trans>重試</Trans></span></Button> : null}
                    {task.status !== "preparing" && task.status !== "encoding" && task.status !== "prefilling" && task.status !== "generating" ? <Button variant="ghost" size="icon-sm" onClick={(event) => { event.stopPropagation(); onRemoveTask(task.id); }}><Trash2Icon data-icon="inline-start" /><span className="sr-only"><Trans>移除</Trans></span></Button> : null}
                  </div></TableCell>
                </TableRow>
              ))}</TableBody>
            </Table>
          ) : (
            <Empty className="task-empty-state"><div className="task-empty-main"><EmptyHeader><EmptyMedia variant="icon"><FileAudioIcon /></EmptyMedia><EmptyTitle><Trans>尚無任務</Trans></EmptyTitle><EmptyDescription><Trans>新增檔案後會自動排隊。</Trans></EmptyDescription></EmptyHeader><EmptyContent><Button onClick={onPickFiles}><ListPlusIcon data-icon="inline-start" />{isDraggingFiles ? <Trans>放開以加入任務</Trans> : <Trans>拖放或選取檔案</Trans>}</Button></EmptyContent></div><p className="task-empty-supported">wav、mp3、m4a、aac、flac、ogg、mp4、mov、mkv、webm</p></Empty>
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
            <Field><FieldLabel htmlFor="task-prompt"><Trans>提示詞</Trans></FieldLabel><InputGroup><InputGroupTextarea id="task-prompt" value={taskDraft.prompt} placeholder={i18n._(msg`留空使用 MOSS 預設提示詞；也可加入專有名詞`)} onChange={(event) => onTaskDraftChange((current) => ({ ...current, prompt: event.target.value }))} /></InputGroup></Field>
            <Field><FieldLabel htmlFor="task-max-tokens"><Trans>最大新 Token 數</Trans></FieldLabel><InputGroup><InputGroupInput id="task-max-tokens" type="number" min={1} max={4096} value={taskDraft.maxNewTokens} onChange={(event) => onTaskDraftChange((current) => ({ ...current, maxNewTokens: Math.min(4096, Math.max(1, Number(event.target.value) || 1)) }))} /><InputGroupAddon align="inline-end"><Trans>最多 4096</Trans></InputGroupAddon></InputGroup></Field>
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
  const elapsedMs = task?.progress?.elapsedMs ?? (task?.startedAt ? now - task.startedAt : 0);
  const audioDurationMs = task?.progress?.audioDurationMs ?? (result ? result.audioDurationMs : null);

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
            <div className="task-result-stack">
              <div className="flex items-center justify-between gap-2">
                <Badge variant={statusVariant(task.status)}><StatusLabel status={task.status} /></Badge>
                <span className="text-xs text-muted-foreground"><Trans>已用 {formatDuration(elapsedMs)}</Trans></span>
              </div>
              <div className="flex flex-col gap-1">
                <Progress value={task.percent} aria-label={i18n._(msg`任務進度`)} />
                <div className="flex justify-between gap-3 text-xs text-muted-foreground">
                  <span>{task.percent.toFixed(0)}%</span>
                  <span className="truncate">{task.message ?? ""}</span>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-3 text-sm">
                <div><div className="text-muted-foreground"><Trans>音訊長度</Trans></div><div>{audioDurationMs == null ? "—" : formatDuration(audioDurationMs)}</div></div>
                <div><div className="text-muted-foreground">Prompt tokens</div><div className="font-mono">{task.progress?.promptTokens ?? result?.promptTokens ?? 0}</div></div>
                <div><div className="text-muted-foreground">Generated tokens</div><div className="font-mono">{task.progress?.generatedTokens ?? result?.generatedTokens ?? 0}</div></div>
              </div>
              {task.error ? <p className="text-sm text-destructive">{task.error}</p> : null}
              {task.status === "failed" ? (
                <Button variant="outline" onClick={() => onRetryTask(task.id)}>
                  <RotateCcwIcon data-icon="inline-start" /><Trans>重新執行</Trans>
                </Button>
              ) : null}
              {result ? (
                <>
                  <Textarea readOnly value={result.text} className="min-h-36 resize-none" />
                  <div className="flex flex-col gap-1 text-xs text-muted-foreground">
                    <span>TXT: {result.outputs.txtPath ?? <Trans>未輸出</Trans>}</span>
                    <span>JSON: {result.outputs.jsonPath ?? <Trans>未輸出</Trans>}</span>
                    <span>SRT: {result.outputs.srtPath ?? <Trans>未輸出</Trans>}</span>
                  </div>
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
                </>
              ) : null}
            </div>
          </ScrollArea>
        ) : null}
      </SheetContent>
    </Sheet>
  );
}
