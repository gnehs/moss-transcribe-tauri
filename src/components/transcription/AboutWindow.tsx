import { getVersion } from "@tauri-apps/api/app";
import { openUrl } from "@tauri-apps/plugin-opener";
import { Trans } from "@lingui/react/macro";
import { CoffeeIcon, ExternalLinkIcon, GitBranchIcon } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { appName, coffeeUrl, githubUrl } from "@/lib/app-constants";

const UNKNOWN_VALUE = "—";

export function AboutWindow() {
  const [version, setVersion] = useState(UNKNOWN_VALUE);
  const commitSha = __APP_COMMIT_SHA__
    ? __APP_COMMIT_SHA__.slice(0, 7)
    : UNKNOWN_VALUE;

  useEffect(() => {
    let cancelled = false;

    void getVersion()
      .then((appVersion) => {
        if (!cancelled) {
          setVersion(appVersion?.replace(/^v/i, "") || UNKNOWN_VALUE);
        }
      })
      .catch(() => {
        if (!cancelled) setVersion(UNKNOWN_VALUE);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const openExternalUrl = (url: string) => {
    void openUrl(url).catch(() => {});
  };

  return (
    <main className="bg-background text-foreground min-h-screen px-6 py-6">
      <section className="mx-auto flex w-full max-w-lg flex-col gap-5">
        <header className="space-y-1">
          <p className="text-muted-foreground text-sm font-medium">{appName}</p>
          <h1 className="font-heading text-3xl font-semibold tracking-tight">
            <Trans>關於</Trans>
          </h1>
          <p className="text-muted-foreground text-sm">
            <Trans>查看專案連結與建置資訊。</Trans>
          </p>
        </header>

        <dl className="divide-border border-border divide-y rounded-xl border">
          <div className="flex items-center justify-between px-4 py-3 text-sm">
            <dt className="text-muted-foreground">
              <Trans>版本</Trans>
            </dt>
            <dd>{version}</dd>
          </div>
          <div className="flex items-center justify-between px-4 py-3 text-sm">
            <dt className="text-muted-foreground">
              <Trans>Commit SHA</Trans>
            </dt>
            <dd className="font-mono text-xs">{commitSha}</dd>
          </div>
        </dl>

        <div className="bg-muted/50 flex items-center gap-4 rounded-xl border p-4">
          <CoffeeIcon className="text-primary size-6 shrink-0" />
          <p className="min-w-0 flex-1 text-sm font-medium">
            <Trans>這個 APP 有幫到你嗎？考慮請我喝杯咖啡吧！</Trans>
          </p>
          <Button
            size="sm"
            className="ml-auto"
            onClick={() => openExternalUrl(coffeeUrl)}
          >
            <CoffeeIcon data-icon="inline-start" />
            <Trans>請我喝杯咖啡</Trans>
          </Button>
        </div>

        <Button
          variant="outline"
          className="w-full"
          onClick={() => openExternalUrl(githubUrl)}
        >
          <GitBranchIcon data-icon="inline-start" />
          <Trans>GitHub</Trans>
          <ExternalLinkIcon data-icon="inline-end" />
        </Button>
      </section>
    </main>
  );
}
