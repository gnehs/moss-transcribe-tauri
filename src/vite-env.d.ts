/// <reference types="vite/client" />

declare const __APP_COMMIT_SHA__: string;

declare module "*.po" {
  import type { Messages } from "@lingui/core";

  export const messages: Messages;
}
