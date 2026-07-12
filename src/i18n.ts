import { i18n } from "@lingui/core";
import { msg } from "@lingui/core/macro";
import { invoke } from "@tauri-apps/api/core";

export type Locale = "zh-Hant" | "zh-Hans" | "en" | "ja";

export const locales: Record<Locale, string> = {
  "zh-Hant": "繁體中文",
  "zh-Hans": "简体中文",
  en: "English",
  ja: "日本語",
};

export const defaultLocale: Locale = "zh-Hant";
export const localeStorageKey = "moss-transcribe.locale";
const nativeMenuTextMessages = {
  about: msg`關於 MOSS Transcribe Studio`,
  settings: msg`設定`,
  file: msg`檔案`,
  newTask: msg`新增任務`,
  edit: msg`編輯`,
  window: msg`視窗`,
  help: msg`說明`,
  github: msg`GitHub`,
};

export function isLocale(value: string | null | undefined): value is Locale {
  return (
    value !== null &&
    value !== undefined &&
    Object.prototype.hasOwnProperty.call(locales, value)
  );
}

export function getStoredLocale(): Locale {
  if (typeof window === "undefined") {
    return defaultLocale;
  }

  try {
    const storedLocale = window.localStorage.getItem(localeStorageKey);
    return isLocale(storedLocale) ? storedLocale : defaultLocale;
  } catch {
    return defaultLocale;
  }
}

export function getInitialLocale(): Locale {
  return getStoredLocale();
}

export function saveLocale(locale: Locale): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(localeStorageKey, locale);
  } catch {
    // Ignore storage failures so locale activation still succeeds.
  }
}

function applyLocaleToDocument(locale: Locale): void {
  if (typeof document === "undefined") {
    return;
  }

  document.documentElement.lang = locale;
  document.documentElement.dataset.locale = locale;
}

export async function dynamicActivate(locale: Locale): Promise<void> {
  const { messages } = await import(`./locales/${locale}/messages.po`);

  i18n.load(locale, messages);
  i18n.activate(locale);
  applyLocaleToDocument(locale);
}

export async function syncNativeMenuText(): Promise<void> {
  if (
    typeof window === "undefined" ||
    new URLSearchParams(window.location.search).get("window") === "about"
  ) {
    return;
  }

  try {
    await invoke("set_native_menu_text", {
      menu: {
        about: i18n._(nativeMenuTextMessages.about),
        settings: i18n._(nativeMenuTextMessages.settings),
        file: i18n._(nativeMenuTextMessages.file),
        newTask: i18n._(nativeMenuTextMessages.newTask),
        edit: i18n._(nativeMenuTextMessages.edit),
        window: i18n._(nativeMenuTextMessages.window),
        help: i18n._(nativeMenuTextMessages.help),
        github: i18n._(nativeMenuTextMessages.github),
      },
    });
  } catch {
    // Ignore invocation failures when running the web app outside Tauri.
  }
}

export async function activateLocale(locale: Locale): Promise<void> {
  await dynamicActivate(locale);
  saveLocale(locale);
  await syncNativeMenuText();
}

export async function activateStoredLocale(): Promise<Locale> {
  const locale = getStoredLocale();
  await activateLocale(locale);
  return locale;
}

export { i18n };
