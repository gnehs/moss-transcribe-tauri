import { msg } from "@lingui/core/macro";
import { useLingui } from "@lingui/react";
import { Trans } from "@lingui/react/macro";
import { openUrl } from "@tauri-apps/plugin-opener";
import { CoffeeIcon, XIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { coffeeUrl } from "@/lib/app-constants";

export function SupportBanner({ onClose }: { onClose: () => void }) {
  const { i18n } = useLingui();

  return (
    <aside className="bg-muted/40 border-border/70 flex w-full max-w-2xl flex-wrap items-center gap-x-3 gap-y-1.5 rounded-lg border px-3 py-2.5 text-left">
      <CoffeeIcon
        className="text-muted-foreground size-4 shrink-0"
        aria-hidden="true"
      />
      <p className="text-muted-foreground m-0 min-w-0 flex-1 text-sm/relaxed">
        <Trans>這個 APP 有幫到你嗎？考慮請我喝杯咖啡吧！</Trans>
      </p>
      <Button
        variant="ghost"
        size="sm"
        className="text-foreground"
        onClick={() => void openUrl(coffeeUrl).catch(() => {})}
      >
        <CoffeeIcon data-icon="inline-start" />
        <Trans>請我喝杯咖啡</Trans>
      </Button>
      <Button
        variant="ghost"
        size="icon-xs"
        className="text-muted-foreground"
        aria-label={i18n._(msg`關閉`)}
        onClick={onClose}
      >
        <XIcon />
      </Button>
    </aside>
  );
}
