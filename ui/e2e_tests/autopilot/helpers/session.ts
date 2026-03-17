import { expect } from "@playwright/test";
import { v7 } from "uuid";

/**
 * Create a new autopilot session and return the session ID.
 * Sends a message through the UI, waits for redirect, then extracts
 * the session ID from the URL.
 */
export async function createSession(
  page: import("@playwright/test").Page,
  label = "Test session",
): Promise<string> {
  await page.goto("/autopilot/sessions/new");
  await page.waitForLoadState("networkidle");
  const messageInput = page.getByRole("textbox");
  await messageInput.fill(`${label} ${v7()}`);
  const sendButton = page.getByRole("button", { name: "Send message" });
  await expect(sendButton).toBeEnabled({ timeout: 10000 });
  await sendButton.click();

  await expect(page).toHaveURL(/\/autopilot\/sessions\/[a-f0-9-]+$/, {
    timeout: 30000,
  });

  const sessionId = page
    .url()
    .match(/\/autopilot\/sessions\/([a-f0-9-]+)$/)?.[1];
  if (!sessionId) throw new Error("Could not extract session ID from URL");

  return sessionId;
}
