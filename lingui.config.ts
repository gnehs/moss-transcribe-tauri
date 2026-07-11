import { defineConfig } from "@lingui/cli";

export default defineConfig({
  locales: ["zh-Hant", "zh-Hans", "en", "ja"],
  sourceLocale: "zh-Hant",
  catalogs: [
    {
      path: "<rootDir>/src/locales/{locale}/messages",
      include: ["<rootDir>/src"],
    },
  ],
});
