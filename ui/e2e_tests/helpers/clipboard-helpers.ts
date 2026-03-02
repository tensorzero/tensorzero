import type { Page } from "@playwright/test";

/**
 * Mocks navigator.clipboard in the page context.
 * Needed because clipboard API permissions aren't available in CI Docker.
 */
export async function installClipboardMock(page: Page) {
  await page.evaluate(() => {
    const storage = { text: "" };
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: (text: string) => {
          storage.text = text;
          return Promise.resolve();
        },
        readText: () => Promise.resolve(storage.text),
      },
      writable: true,
      configurable: true,
    });
    (
      window as unknown as { __clipboardStorage: typeof storage }
    ).__clipboardStorage = storage;
  });
}

/**
 * Reads text from the mocked clipboard. Must call installClipboardMock first.
 */
export async function readMockClipboard(page: Page): Promise<string> {
  return page.evaluate(
    () =>
      (window as unknown as { __clipboardStorage: { text: string } })
        .__clipboardStorage.text,
  );
}
