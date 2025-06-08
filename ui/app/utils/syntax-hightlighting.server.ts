import { codeToHtml } from "shiki";
import { JSONParseError } from "./common";

export type SupportedLanguage = "json";

export async function highlightCode(
  code: string,
  args?: { lang?: SupportedLanguage },
) {
  const { lang = "json" } = args || {};
  return await codeToHtml(code, {
    lang,
    theme: "github-light",
    // TODO: Support dark mode when the app supports it
    // themes: {
    //   light: "github-light",
    //   dark: "github-dark",
    // },
  });
}

export async function processJson(object: object, objectRef: string) {
  let raw: string | null = null;
  let html: string | null = null;
  try {
    raw = JSON.stringify(object, null, 2);
    html = await highlightCode(raw).catch((error) => {
      console.error(
        `Syntax error highlighting error for ${objectRef}. Using raw JSON instead.`,
        error,
      );
      return null;
    });
  } catch {
    throw new JSONParseError(`Failed to parse JSON for ${objectRef}`);
  }

  return { raw, html };
}
