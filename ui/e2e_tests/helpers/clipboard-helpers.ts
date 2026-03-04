import type { Page } from "@playwright/test";

/**
 * Mocks navigator.clipboard in the page context via addInitScript.
 * Must be called BEFORE page.goto so the mock is available when React renders.
 *
 * Also overrides window.isSecureContext for non-secure origins (e.g. 0.0.0.0
 * in CI Docker) where the clipboard API would otherwise be unavailable.
 */
export async function installClipboardMock(page: Page) {
  await page.addInitScript(() => {
    const storage = { text: "" };

    // Override isSecureContext for non-secure origins (CI Docker uses 0.0.0.0)
    if (!window.isSecureContext) {
      Object.defineProperty(window, "isSecureContext", {
        value: true,
        writable: false,
        configurable: true,
      });
    }

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
