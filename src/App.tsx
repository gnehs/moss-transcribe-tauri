import {
  FileUpIcon,
  ListPlusIcon,
  Settings2Icon,
  Trash2Icon,
} from "lucide-react";
import { Trans } from "@lingui/react/macro";
import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { openUrl } from "@tauri-apps/plugin-opener";

import { AppToolbar } from "@/components/app/AppToolbar";
import { AboutWindow } from "@/components/transcription/AboutWindow";
import { SettingsPanel } from "@/components/transcription/SettingsPanel";
import { TaskManagerPanel } from "@/components/transcription/TaskManagerPanel";
import { Button } from "@/components/ui/button";
import { Toaster } from "@/components/ui/sonner";
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
import { githubUrl } from "@/lib/app-constants";

function App() {
  const isAboutWindow =
    new URLSearchParams(window.location.search).get("window") === "about";

  return isAboutWindow ? <AboutWindow /> : <MainWindow />;
}

function MainWindow() {
  const workspace = useTranscriptionWorkspace();
  const [isSettingsOpen, setSettingsOpen] = useState(false);
  const hasFinishedTasks = workspace.tasks.some(
    (task) => task.status === "completed"
  );
  const shouldShowDragOverlay =
    workspace.isDraggingFiles && workspace.tasks.length > 0;

  useEffect(() => {
    const unlisteners = Promise.all([
      listen("open-settings", () => setSettingsOpen(true)),
      listen("new-task", () => {
        void workspace.pickFilesForTasks().catch(() => {});
      }),
      listen("open-github", () => {
        void openUrl(githubUrl).catch(() => {});
      }),
    ]);

    return () => {
      void unlisteners
        .then((items) => items.forEach((unlisten) => unlisten()))
        .catch(() => {});
    };
  }, [workspace.pickFilesForTasks]);

  return (
    <TooltipProvider>
      <main className="bg-background min-h-screen">
        <Toaster richColors closeButton position="bottom-right" />
        <section className="bg-background flex h-screen min-h-0 flex-col overflow-hidden">
          {shouldShowDragOverlay ? (
            <div
              className="border-primary/40 bg-primary/10 ring-primary/20 fixed inset-3 z-50 grid place-items-center rounded-xl border-2 border-dashed text-center ring-1 ring-inset"
              role="status"
              aria-live="polite"
            >
              <div className="flex flex-col items-center gap-2 text-center">
                <div className="bg-primary/10 text-primary mb-2 grid size-16 place-items-center rounded-lg">
                  <FileUpIcon className="size-8" />
                </div>
                <strong className="font-heading text-2xl leading-tight font-semibold">
                  <Trans>把檔案拖到這裡</Trans>
                </strong>
                <span className="text-muted-foreground text-sm">
                  <Trans>放開即可建立轉錄任務</Trans>
                </span>
              </div>
            </div>
          ) : null}
          <AppToolbar
            ffmpeg={workspace.ffmpeg}
            title="MOSS Transcribe Studio"
            actions={
              workspace.tasks.length ? (
                <>
                  {hasFinishedTasks ? (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={workspace.clearFinishedTasks}
                    >
                      <Trash2Icon data-icon="inline-start" />
                      <Trans>清除已完成</Trans>
                    </Button>
                  ) : null}
                  <Button size="sm" onClick={workspace.pickFilesForTasks}>
                    <ListPlusIcon data-icon="inline-start" />
                    <Trans>新增任務</Trans>
                  </Button>
                </>
              ) : undefined
            }
            utilities={
              <Sheet
                open={isSettingsOpen}
                onOpenChange={(open) => setSettingsOpen(open)}
              >
                <SheetTrigger render={<Button variant="outline" size="sm" />}>
                  <Settings2Icon data-icon="inline-start" />
                  <Trans>設定</Trans>
                </SheetTrigger>
                <SheetContent
                  side="right"
                  className="gap-0 data-[side=right]:w-[min(560px,100vw)] data-[side=right]:sm:max-w-[min(560px,100vw)]"
                >
                  <SheetHeader>
                    <SheetTitle className="text-xl font-semibold">
                      <Trans>設定</Trans>
                    </SheetTitle>
                    <SheetDescription>
                      <Trans>管理 MOSS 模型、本機工具與執行環境。</Trans>
                    </SheetDescription>
                  </SheetHeader>
                  <div className="scroll-fade min-h-0 flex-1 overflow-y-auto px-4 pt-2 pb-6">
                    <SettingsPanel
                      model={workspace.model}
                      ffmpeg={workspace.ffmpeg}
                      system={workspace.system}
                      downloadProgress={workspace.downloadProgress}
                      isDownloading={workspace.isDownloading}
                      deletingModel={workspace.deletingModel}
                      onDownload={() => {
                        void workspace.downloadModel().catch(() => {});
                      }}
                      onRedownload={() => {
                        void workspace.downloadModel(true).catch(() => {});
                      }}
                      onDelete={() => {
                        void workspace.deleteModel();
                      }}
                      onRevealModel={() => {
                        void workspace.revealModel();
                      }}
                      onRefreshFfmpeg={() => {
                        void workspace.recheckFfmpeg();
                      }}
                    />
                  </div>
                </SheetContent>
              </Sheet>
            }
          />
          <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
            <TaskManagerPanel
              tasks={workspace.tasks}
              taskDraft={workspace.taskDraft}
              isConfirmingTasks={workspace.isConfirmingTasks}
              downloadProgress={workspace.downloadProgress}
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
