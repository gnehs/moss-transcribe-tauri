import React from "react";
import ReactDOM from "react-dom/client";
import { i18n } from "@lingui/core";
import { I18nProvider } from "@lingui/react";
import { ThemeProvider } from "next-themes";
import App from "./App";
import { dynamicActivate, getInitialLocale } from "./i18n";
import "./index.css";

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement
);

async function bootstrap() {
  await dynamicActivate(getInitialLocale());
  root.render(
    <React.StrictMode>
      <ThemeProvider
        attribute="class"
        defaultTheme="system"
        disableTransitionOnChange
        storageKey="moss-transcribe.theme"
      >
        <I18nProvider i18n={i18n}>
          <App />
        </I18nProvider>
      </ThemeProvider>
    </React.StrictMode>
  );
}

void bootstrap();
