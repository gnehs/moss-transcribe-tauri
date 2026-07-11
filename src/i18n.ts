import { i18n } from "@lingui/core";

export type Locale = "zh-Hant" | "zh-Hans" | "en" | "ja";

export const locales: Record<Locale, string> = {
  "zh-Hant": "繁體中文",
  "zh-Hans": "简体中文",
  en: "English",
  ja: "日本語",
};

export const defaultLocale: Locale = "zh-Hant";
export const localeStorageKey = "moss-transcribe.locale";

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

export async function activateLocale(locale: Locale): Promise<void> {
  await dynamicActivate(locale);
  saveLocale(locale);
}

export async function activateStoredLocale(): Promise<Locale> {
  const locale = getStoredLocale();
  await activateLocale(locale);
  return locale;
}

export { i18n };
