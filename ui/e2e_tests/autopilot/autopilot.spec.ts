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

test.describe("Tool call authorization deduplication", () => {
  test("spam-clicking Approve sends only one request", async ({ page }) => {
    test.setTimeout(120000);

    // Track requests by tool_call_event_id to scope assertion to the clicked tool call
    const requestsByToolCallId = new Map<string, number>();
    let firstToolCallId: string | null = null;

    await page.route("**/events/authorize", async (route) => {
      const body = route.request().postDataJSON();
      const toolCallId = body?.tool_call_event_id;
      if (toolCallId) {
        if (!firstToolCallId) {
          firstToolCallId = toolCallId;
        }
        requestsByToolCallId.set(
          toolCallId,
          (requestsByToolCallId.get(toolCallId) ?? 0) + 1,
        );
      }
      await route.continue();
    });

    await page.goto("/autopilot/sessions/new");

    const messageInput = page.getByRole("textbox");
    await messageInput.fill(
      "What functions are available in my TensorZero config?",
    );
    await page.getByRole("button", { name: "Send message" }).click();

    await expect(page).toHaveURL(/\/autopilot\/sessions\/[a-f0-9-]+$/, {
      timeout: 30000,
    });

    // Wait for Approve button to appear
    const approveButton = page.getByRole("button", { name: "Approve" }).first();
    await expect(approveButton).toBeVisible({ timeout: 60000 });

    // Get button position before clicking - use mouse.click at fixed coordinates
    // to avoid flakiness if the button is removed/disabled after first click
    const boundingBox = await approveButton.boundingBox();
    if (!boundingBox) {
      throw new Error("Approve button bounding box not found");
    }
    const clickX = boundingBox.x + boundingBox.width / 2;
    const clickY = boundingBox.y + boundingBox.height / 2;

    // Spam-click at the fixed position rapidly
    await Promise.all([
      page.mouse.click(clickX, clickY),
      page.mouse.click(clickX, clickY),
      page.mouse.click(clickX, clickY),
    ]);

    // Wait for request to complete
    await page.waitForTimeout(2000);

    // Only count requests for the first tool call (the one we spam-clicked)
    const requestCount = firstToolCallId
      ? (requestsByToolCallId.get(firstToolCallId) ?? 0)
      : 0;
    expect(
      requestCount,
      "Spam-clicking Approve should send only one request for the same tool call",
    ).toBe(1);
  });

  test("spam-clicking Reject sends only one request", async ({ page }) => {
    test.setTimeout(120000);

    // Track requests by tool_call_event_id to scope assertion to the clicked tool call
    const requestsByToolCallId = new Map<string, number>();
    let firstToolCallId: string | null = null;

    await page.route("**/events/authorize", async (route) => {
      const body = route.request().postDataJSON();
      const toolCallId = body?.tool_call_event_id;
      if (toolCallId) {
        if (!firstToolCallId) {
          firstToolCallId = toolCallId;
        }
        requestsByToolCallId.set(
          toolCallId,
          (requestsByToolCallId.get(toolCallId) ?? 0) + 1,
        );
      }
      await route.continue();
    });

    await page.goto("/autopilot/sessions/new");

    const messageInput = page.getByRole("textbox");
    await messageInput.fill(
      "What functions are available in my TensorZero config?",
    );
    await page.getByRole("button", { name: "Send message" }).click();

    await expect(page).toHaveURL(/\/autopilot\/sessions\/[a-f0-9-]+$/, {
      timeout: 30000,
    });

    // Wait for Reject button to appear
    const rejectButton = page.getByRole("button", { name: "Reject" }).first();
    await expect(rejectButton).toBeVisible({ timeout: 60000 });

    // Click Reject to show confirmation
    await rejectButton.click();

    // Wait for and spam-click the confirm button (button has aria-label="Confirm rejection")
    const confirmButton = page
      .getByRole("button", { name: "Confirm rejection" })
      .first();
    await expect(confirmButton).toBeVisible();

    // Get button position before clicking - use mouse.click at fixed coordinates
    // to avoid flakiness if the button is removed/disabled after first click
    const boundingBox = await confirmButton.boundingBox();
    if (!boundingBox) {
      throw new Error("Confirm rejection button bounding box not found");
    }
    const clickX = boundingBox.x + boundingBox.width / 2;
    const clickY = boundingBox.y + boundingBox.height / 2;

    // Spam-click at the fixed position rapidly
    await Promise.all([
      page.mouse.click(clickX, clickY),
      page.mouse.click(clickX, clickY),
      page.mouse.click(clickX, clickY),
    ]);

    // Wait for request to complete
    await page.waitForTimeout(2000);

    // Only count requests for the first tool call (the one we spam-clicked)
    const requestCount = firstToolCallId
      ? (requestsByToolCallId.get(firstToolCallId) ?? 0)
      : 0;
    expect(
      requestCount,
      "Spam-clicking Reject confirm should send only one request for the same tool call",
    ).toBe(1);
  });
});
