import { test, expect } from "@playwright/test";

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

test.describe("Chat input validation", () => {
  test("send button disabled when empty, enabled with text", async ({
    page,
  }) => {
    await page.goto("/autopilot/sessions/new");
    await page.waitForLoadState("networkidle");

    const textarea = page.getByRole("textbox");
    const sendButton = page.getByRole("button", { name: "Send message" });

    // Empty → disabled
    await expect(sendButton).toBeDisabled();

    // Whitespace only → still disabled
    await textarea.fill("   ");
    await expect(sendButton).toBeDisabled();

    // Real text → enabled
    await textarea.fill("Hello");
    await expect(sendButton).toBeEnabled();

    // Clear → disabled again
    await textarea.fill("");
    await expect(sendButton).toBeDisabled();
  });

  test("Shift+Enter inserts newline without sending", async ({ page }) => {
    await page.goto("/autopilot/sessions/new");
    await page.waitForLoadState("networkidle");

    const textarea = page.getByRole("textbox");
    await textarea.fill("Line 1");
    await textarea.press("Shift+Enter");
    await textarea.type("Line 2");

    // Should still be on new session page (not submitted)
    await expect(page).toHaveURL(/\/autopilot\/sessions\/new/);
    await expect(textarea).toHaveValue("Line 1\nLine 2");
  });
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

test.describe("YOLO mode", () => {
  test("toggle is visible and defaults to off", async ({ page }) => {
    await page.goto("/autopilot/sessions/new");

    const yoloToggle = page
      .locator("label")
      .filter({ hasText: "YOLO Mode" })
      .getByRole("switch");
    await expect(yoloToggle).toBeVisible();
    await expect(yoloToggle).toHaveAttribute("data-state", "unchecked");
  });

  test("reads persisted state from localStorage on page load", async ({
    page,
  }) => {
    await page.goto("/autopilot/sessions/new");
    await page.evaluate(() =>
      localStorage.setItem("tensorzero-yolo-mode", "true"),
    );
    await page.reload();

    const yoloToggle = page
      .locator("label")
      .filter({ hasText: "YOLO Mode" })
      .getByRole("switch");
    await expect(yoloToggle).toBeVisible();
    await expect(yoloToggle).toHaveAttribute("data-state", "checked");
  });

  test("clicking toggle enables auto-approval of tool calls", async ({
    page,
  }) => {
    test.setTimeout(120000);

    await page.goto("/autopilot/sessions/new");

    const yoloLabel = page.locator("label").filter({ hasText: "YOLO Mode" });
    const yoloToggle = yoloLabel.getByRole("switch");
    await expect(yoloToggle).toBeVisible();
    await yoloLabel.click();
    await expect(yoloToggle).toHaveAttribute("data-state", "checked");

    const messageInput = page.getByRole("textbox");
    await messageInput.fill(
      "What functions are available in my TensorZero config?",
    );
    await page.getByRole("button", { name: "Send message" }).click();

    await expect(page).toHaveURL(/\/autopilot\/sessions\/[a-f0-9-]+$/, {
      timeout: 30000,
    });

    // "Tool Result" appearing confirms tool was auto-approved and executed
    await expect(page.getByText("Tool Result").first()).toBeVisible({
      timeout: 60000,
    });
  });
});
