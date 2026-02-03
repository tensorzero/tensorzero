import { test, expect } from "@playwright/test";
import { v7 } from "uuid";

test("should interrupt an active session", async ({ page }) => {
  // Increase timeout for this test since it involves LLM responses
  test.setTimeout(120000);

  // Navigate to autopilot sessions
  await page.goto("/autopilot/sessions");

  // Wait for the page to load
  await expect(
    page.getByRole("heading", { name: "Autopilot Sessions" }),
  ).toBeVisible();

  // Click to create a new session
  await page.getByRole("button", { name: /new session/i }).click();

  // Wait for the new session page
  await expect(page).toHaveURL(/\/autopilot\/sessions\/new/);

  // Find the message textarea and type a message that will trigger tool calls
  const messageInput = page.getByRole("textbox");
  // We randomize this message so it misses the provider proxy cache and we have time to click the stop button
  await messageInput.fill(
    `What functions are available in my TensorZero config? ${v7()}`,
  );

  // Send the message
  await page.getByRole("button", { name: "Send message" }).click();

  // Wait for redirect to the actual session page
  await expect(page).toHaveURL(/\/autopilot\/sessions\/[a-f0-9-]+$/, {
    timeout: 30000,
  });

  // Wait for the stop button to appear (session is processing)
  const stopButton = page.getByRole("button", {
    name: /stop session/i,
  });
  await expect(stopButton).toBeVisible({ timeout: 30000 });

  // Click the stop button
  await stopButton.click();

  // Verify the success toast appears
  await expect(
    page.getByRole("status").filter({ hasText: "Session interrupted" }),
  ).toBeVisible({
    timeout: 10000,
  });

  // Verify the status update message appears in the event stream
  await expect(page.getByText("Interrupted session")).toBeVisible({
    timeout: 10000,
  });

  // Verify the send button reappears after interruption (session returns to idle)
  const sendButton = page.getByRole("button", { name: /send message/i });
  await expect(sendButton).toBeVisible({ timeout: 30000 });

  // Verify the status indicator shows "Ready" (which is the label for "idle" status)
  await expect(page.getByText("Ready")).toBeVisible({ timeout: 10000 });

  // Verify no error message is shown - the session should be cleanly idle, not failed
  await expect(page.getByText("Something went wrong")).not.toBeVisible();
});

test("should create a session, send a message, approve tool calls, and get a response", async ({
  page,
}) => {
  // Increase timeout for this test since it involves LLM responses
  test.setTimeout(120000);

  // Navigate to autopilot sessions
  await page.goto("/autopilot/sessions");

  // Wait for the page to load
  await expect(
    page.getByRole("heading", { name: "Autopilot Sessions" }),
  ).toBeVisible();

  // Click to create a new session
  await page.getByRole("button", { name: /new session/i }).click();

  // Wait for the new session page
  await expect(page).toHaveURL(/\/autopilot\/sessions\/new/);

  // Find the message textarea and type a message asking about available functions
  const messageInput = page.getByRole("textbox");
  await messageInput.fill(
    "What functions are available in my TensorZero config?",
  );

  // Send the message (button has aria-label="Send message")
  await page.getByRole("button", { name: "Send message" }).click();

  // Wait for redirect to the actual session page
  await expect(page).toHaveURL(/\/autopilot\/sessions\/[a-f0-9-]+$/, {
    timeout: 30000,
  });

  // Verify the message appears in the conversation
  await expect(
    page.getByText("What functions are available in my TensorZero config?"),
  ).toBeVisible();

  // Wait for and approve tool calls as they appear
  // Keep approving until we see the final response containing "basic_test"
  const maxApprovalAttempts = 10;
  for (let i = 0; i < maxApprovalAttempts; i++) {
    // Check if we already have the expected response
    const hasResponse = await page
      .getByText("basic_test", { exact: false })
      .first()
      .isVisible()
      .catch(() => false);

    if (hasResponse) {
      break;
    }

    // Look for an approve button and click it if found
    const approveButton = page.getByRole("button", { name: "Approve" }).first();
    const isApproveVisible = await approveButton.isVisible().catch(() => false);

    if (isApproveVisible) {
      await approveButton.click();
      // Wait a bit for the tool to execute and new content to appear
      await page.waitForTimeout(2000);
    } else {
      // No approve button visible, wait a bit and check again
      await page.waitForTimeout(2000);
    }
  }

  // Verify the response contains "basic_test" (a function from the test fixtures)
  await expect(
    page.getByText("basic_test", { exact: false }).first(),
  ).toBeVisible({ timeout: 60000 });
});

test.describe("Autopilot New Session Button", () => {
  test("should navigate to new session from existing session page", async ({
    page,
  }) => {
    // 1. Go to autopilot sessions list
    await page.goto("/autopilot/sessions");
    await page.waitForLoadState("networkidle");

    // 2. Click the first session in the table
    const firstSessionLink = page
      .locator("table a[href^='/autopilot/sessions/']")
      .first();
    await firstSessionLink.waitFor({ state: "visible" });
    await firstSessionLink.click();

    // 3. Wait for session page to load
    await page.waitForLoadState("networkidle");

    // 4. Click "New Session" link in the header
    const newSessionLink = page.getByRole("link", { name: "New Session" });
    await newSessionLink.waitFor({ state: "visible" });
    await newSessionLink.click();

    // 5. Verify navigation to /autopilot/sessions/new and page stays there
    await page.waitForURL("**/autopilot/sessions/new", { timeout: 10000 });
    await page.waitForLoadState("networkidle");
    expect(page.url()).toMatch(/\/autopilot\/sessions\/new$/);
  });
});
