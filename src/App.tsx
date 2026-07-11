import { FileUpIcon, ListPlusIcon, Settings2Icon, Trash2Icon } from "lucide-react";
import { Trans } from "@lingui/react/macro";
import { Toaster } from "sonner";

import { AppToolbar } from "@/components/app/AppToolbar";
import { SettingsPanel } from "@/components/transcription/SettingsPanel";
import { TaskManagerPanel } from "@/components/transcription/TaskManagerPanel";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useTranscriptionWorkspace } from "@/hooks/use-transcription-workspace";
import "./App.css";

function App() {
  const workspace = useTranscriptionWorkspace();
  const hasFinishedTasks = workspace.tasks.some((task) => task.status === "completed");

  return (
    <TooltipProvider>
      <main className="app-shell">
        <Toaster richColors closeButton position="top-right" />
        <section className="app-window">
          {workspace.isDraggingFiles ? (
            <div className="file-drop-overlay" role="status" aria-live="polite">
              <div className="file-drop-overlay-content">
                <div className="file-drop-overlay-icon"><FileUpIcon /></div>
                <strong><Trans>把檔案拖到這裡</Trans></strong>
                <span><Trans>放開即可建立轉錄任務</Trans></span>
              </div>
            </div>
          ) : null}
           <AppToolbar
            ffmpeg={workspace.ffmpeg}
            title="MOSS Transcribe Studio"
            actions={workspace.tasks.length ? (
              <>
                {hasFinishedTasks ? (
                  <Button variant="outline" size="sm" onClick={workspace.clearFinishedTasks}>
                    <Trash2Icon data-icon="inline-start" /><Trans>清除已完成</Trans>
                  </Button>
                ) : null}
                <Button size="sm" onClick={workspace.pickFilesForTasks}>
                  <ListPlusIcon data-icon="inline-start" /><Trans>新增任務</Trans>
                </Button>
              </>
            ) : undefined}
            utilities={(
              <Sheet>
                <SheetTrigger render={<Button variant="outline" size="sm" />}>
                  <Settings2Icon data-icon="inline-start" /><Trans>設定</Trans>
                </SheetTrigger>
                <SheetContent side="right" className="settings-sheet data-[side=right]:w-[min(560px,100vw)] data-[side=right]:sm:max-w-[min(560px,100vw)]">
                  <SheetHeader>
                    <SheetTitle><Trans>設定</Trans></SheetTitle>
                    <SheetDescription><Trans>管理 MOSS 模型、本機工具與執行環境。</Trans></SheetDescription>
                  </SheetHeader>
                  <div className="settings-sheet-body scroll-fade">
                    <SettingsPanel
                      model={workspace.model}
                      ffmpeg={workspace.ffmpeg}
                      system={workspace.system}
                      downloadProgress={workspace.downloadProgress}
                      isDownloading={workspace.isDownloading}
                      deletingModel={workspace.deletingModel}
                      onDownload={() => { void workspace.downloadModel().catch(() => {}); }}
                      onRedownload={() => { void workspace.downloadModel(true).catch(() => {}); }}
                      onDelete={() => { void workspace.deleteModel(); }}
                      onRefreshFfmpeg={() => { void workspace.recheckFfmpeg(); }}
                    />
                  </div>
                </SheetContent>
              </Sheet>
            )}
          />
          <div className="main-content is-task-view">
            <TaskManagerPanel
              tasks={workspace.tasks}
              taskDraft={workspace.taskDraft}
              isConfirmingTasks={workspace.isConfirmingTasks}
              isTaskDialogOpen={workspace.isTaskDialogOpen}
              isDraggingFiles={workspace.isDraggingFiles}
              selectedTaskId={workspace.selectedTaskId}
              onPickFiles={workspace.pickFilesForTasks}
              onPickOutputDir={workspace.pickTaskOutputDir}
              onTaskDraftChange={workspace.setTaskDraft}
              onTaskDialogOpenChange={workspace.setTaskDialogOpen}
              onConfirmTaskDraft={workspace.confirmTaskDraft}
              onRetryTask={workspace.retryTask}
              onRemoveTask={workspace.removeTask}
              onSelectedTaskChange={workspace.setSelectedTaskId}
            />
          </div>
        </section>
      </main>
    </TooltipProvider>
  );
}

export default App;
