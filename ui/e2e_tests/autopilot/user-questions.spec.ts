import { test, expect } from "@playwright/test";
import { v7 } from "uuid";
import { insertEvent, queryEventPayloads } from "./helpers/db";

// ── Types ──────────────────────────────────────────────────────────────

type AnswerPayload = {
  type: string;
  user_questions_event_id: string;
  responses: Record<
    string,
    { type: string; selected?: string[]; text?: string }
  >;
};

// ── Payload builders ───────────────────────────────────────────────────

function buildMultipleChoicePayload() {
  const q1Id = v7();
  const opt1Id = v7();
  const opt2Id = v7();

  const payload = {
    type: "user_questions" as const,
    questions: [
      {
        id: q1Id,
        header: "Auth",
        question: "Which authentication method should we use?",
        type: "multiple_choice" as const,
        options: [
          {
            id: opt1Id,
            label: "JWT",
            description: "JSON Web Tokens for stateless auth",
          },
          {
            id: opt2Id,
            label: "OAuth",
            description: "OAuth 2.0 with external provider",
          },
        ],
        multi_select: false,
      },
    ],
  };

  return { payload, q1Id, opt1Id, opt2Id };
}

function buildMultiSelectPayload() {
  const q1Id = v7();
  const opt1Id = v7();
  const opt2Id = v7();
  const opt3Id = v7();

  const payload = {
    type: "user_questions" as const,
    questions: [
      {
        id: q1Id,
        header: "Features",
        question: "Which features should we include?",
        type: "multiple_choice" as const,
        options: [
          { id: opt1Id, label: "SSR", description: "Server-side rendering" },
          { id: opt2Id, label: "i18n", description: "Internationalization" },
          {
            id: opt3Id,
            label: "PWA",
            description: "Progressive web app support",
          },
        ],
        multi_select: true,
      },
    ],
  };

  return { payload, q1Id, opt1Id, opt2Id, opt3Id };
}

function buildFreeResponsePayload(
  question = "What specific requirements do you have?",
) {
  const q1Id = v7();

  const payload = {
    type: "user_questions" as const,
    questions: [
      {
        id: q1Id,
        header: "Details",
        question,
        type: "free_response" as const,
      },
    ],
  };

  return { payload, q1Id };
}

function buildMultiQuestionPayload() {
  const q1Id = v7();
  const q2Id = v7();
  const opt1Id = v7();
  const opt2Id = v7();

  const payload = {
    type: "user_questions" as const,
    questions: [
      {
        id: q1Id,
        header: "Framework",
        question: "Which framework do you prefer?",
        type: "multiple_choice" as const,
        options: [
          { id: opt1Id, label: "React", description: "React with TypeScript" },
          {
            id: opt2Id,
            label: "Vue",
            description: "Vue.js with Composition API",
          },
        ],
        multi_select: false,
      },
      {
        id: q2Id,
        header: "Notes",
        question: "Any additional requirements?",
        type: "free_response" as const,
      },
    ],
  };

  return { payload, q1Id, q2Id, opt1Id, opt2Id };
}

// ── Helpers ────────────────────────────────────────────────────────────

/**
 * Create a new autopilot session and return the session ID.
 * Sends a message through the UI, waits for redirect, then interrupts
 * so the worker stops generating events and the session is quiescent.
 */
async function createAndInterruptSession(
  page: import("@playwright/test").Page,
): Promise<string> {
  await page.goto("/autopilot/sessions/new");
  const messageInput = page.getByRole("textbox");
  await messageInput.fill(`Test question flow ${v7()}`);
  await page.getByRole("button", { name: "Send message" }).click();

  await expect(page).toHaveURL(/\/autopilot\/sessions\/[a-f0-9-]+$/, {
    timeout: 30000,
  });

  const sessionId = page
    .url()
    .match(/\/autopilot\/sessions\/([a-f0-9-]+)$/)?.[1];
  if (!sessionId) throw new Error("Could not extract session ID from URL");

  const stopButton = page.getByRole("button", { name: /stop session/i });
  await expect(stopButton).toBeVisible({ timeout: 30000 });
  await stopButton.click();
  await expect(page.getByText("Interrupted session")).toBeVisible({
    timeout: 10000,
  });

  return sessionId;
}

/**
 * Query the first user_questions_answers payload for a session.
 * Asserts at least one answer exists before returning.
 */
function queryFirstAnswerPayload(sessionId: string): AnswerPayload {
  const payloads = queryEventPayloads(sessionId, "user_questions_answers");
  expect(payloads.length).toBeGreaterThanOrEqual(1);
  return payloads[0] as AnswerPayload;
}

// ── Tests ──────────────────────────────────────────────────────────────

test.describe("User questions", () => {
  test("should submit a multiple choice answer", async ({ page }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    // Insert a user_questions event via the database
    const eventId = v7();
    const { payload, q1Id, opt1Id } = buildMultipleChoicePayload();
    insertEvent(eventId, sessionId, payload);

    // Wait for the PendingQuestionCard to appear in the footer
    await expect(
      page.getByText("Which authentication method should we use?"),
    ).toBeVisible({ timeout: 15000 });

    // Single-question cards show "Question" header (no step tabs)
    await expect(page.getByText("Question").first()).toBeVisible();

    await page.getByText("JWT").click();
    await page.getByRole("button", { name: /submit/i }).click();

    // Verify "Answered" event appears in the stream
    await expect(page.getByText("Answered")).toBeVisible({ timeout: 10000 });

    // Verify the response was persisted correctly
    const answer = queryFirstAnswerPayload(sessionId);
    expect(answer.type).toBe("user_questions_answers");
    expect(answer.user_questions_event_id).toBe(eventId);
    expect(answer.responses[q1Id]).toBeDefined();
    expect(answer.responses[q1Id].type).toBe("multiple_choice");
    expect(answer.responses[q1Id].selected).toContain(opt1Id);
  });

  test("should submit a free response answer", async ({ page }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    const eventId = v7();
    const { payload, q1Id } = buildFreeResponsePayload();
    insertEvent(eventId, sessionId, payload);

    await expect(
      page.getByText("What specific requirements do you have?"),
    ).toBeVisible({ timeout: 15000 });

    const textarea = page.getByPlaceholder("Type your response...");
    await textarea.fill("We need RBAC with role-based permissions");

    await page.getByRole("button", { name: /submit/i }).click();

    await expect(page.getByText("Answered")).toBeVisible({ timeout: 10000 });

    const answer = queryFirstAnswerPayload(sessionId);
    expect(answer.user_questions_event_id).toBe(eventId);
    expect(answer.responses[q1Id].type).toBe("free_response");
    expect(answer.responses[q1Id].text).toBe(
      "We need RBAC with role-based permissions",
    );
  });

  test("should submit multi-select question with multiple selected options", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    const eventId = v7();
    const { payload, q1Id, opt1Id, opt2Id } = buildMultiSelectPayload();
    insertEvent(eventId, sessionId, payload);

    await expect(
      page.getByText("Which features should we include?"),
    ).toBeVisible({ timeout: 15000 });

    // Multi-select questions show hint text
    await expect(page.getByText("Select all that apply")).toBeVisible();

    // Select two of three options
    await page.getByRole("button", { name: /SSR/ }).click();
    await page.getByRole("button", { name: /i18n/ }).click();

    await page.getByRole("button", { name: /submit/i }).click();

    await expect(page.getByText("Answered")).toBeVisible({ timeout: 10000 });

    // Verify exactly the two selected options were persisted
    const answer = queryFirstAnswerPayload(sessionId);
    expect(answer.responses[q1Id].type).toBe("multiple_choice");
    expect(answer.responses[q1Id].selected).toContain(opt1Id);
    expect(answer.responses[q1Id].selected).toContain(opt2Id);
    expect(answer.responses[q1Id].selected).toHaveLength(2);
  });

  test("should navigate multi-question flow and submit all answers", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    const eventId = v7();
    const { payload, q1Id, q2Id, opt1Id } = buildMultiQuestionPayload();
    insertEvent(eventId, sessionId, payload);

    // Wait for Q1 (multiple choice)
    await expect(page.getByText("Which framework do you prefer?")).toBeVisible({
      timeout: 15000,
    });

    // Multi-question cards show "Questions" header (plural) with step tabs
    await expect(page.getByText("Questions").first()).toBeVisible();

    await page
      .getByRole("button", { name: "React React with TypeScript" })
      .click();

    // Navigate to Q2 (free response)
    await page.getByRole("button", { name: /next/i }).click();

    await expect(page.getByText("Any additional requirements?")).toBeVisible();

    const textarea = page.getByPlaceholder("Type your response...");
    await textarea.fill("Must support SSR");

    await page.getByRole("button", { name: /submit/i }).click();

    await expect(page.getByText("Answered")).toBeVisible({ timeout: 10000 });

    // Verify both answers persisted
    const answer = queryFirstAnswerPayload(sessionId);
    expect(answer.user_questions_event_id).toBe(eventId);
    expect(answer.responses[q1Id].type).toBe("multiple_choice");
    expect(answer.responses[q1Id].selected).toContain(opt1Id);
    expect(answer.responses[q2Id].type).toBe("free_response");
    expect(answer.responses[q2Id].text).toBe("Must support SSR");
  });

  test("should preserve selections when navigating back", async ({ page }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    const eventId = v7();
    const { payload, opt1Id } = buildMultiQuestionPayload();
    insertEvent(eventId, sessionId, payload);

    await expect(page.getByText("Which framework do you prefer?")).toBeVisible({
      timeout: 15000,
    });

    await page
      .getByRole("button", { name: "React React with TypeScript" })
      .click();

    await page.getByRole("button", { name: /next/i }).click();
    await expect(page.getByText("Any additional requirements?")).toBeVisible();

    await page.getByRole("button", { name: /back/i }).click();
    await expect(
      page.getByText("Which framework do you prefer?"),
    ).toBeVisible();

    // Verify React is still selected (button should have selected styling)
    const reactButton = page.getByRole("button", {
      name: "React React with TypeScript",
    });
    await expect(reactButton).toHaveClass(/border-purple-500/);

    // Complete the flow and verify the preserved selection in DB
    await page.getByRole("button", { name: /next/i }).click();
    const textarea = page.getByPlaceholder("Type your response...");
    await textarea.fill("Needs dark mode");
    await page.getByRole("button", { name: /submit/i }).click();

    await expect(page.getByText("Answered")).toBeVisible({ timeout: 10000 });

    const answer = queryFirstAnswerPayload(sessionId);
    const mcResponse = Object.values(answer.responses).find(
      (r) => r.type === "multiple_choice",
    );
    expect(mcResponse?.selected).toContain(opt1Id);
  });

  test("should skip questions when dismiss button is clicked", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    const eventId = v7();
    const { payload, q1Id } = buildMultipleChoicePayload();
    insertEvent(eventId, sessionId, payload);

    await expect(
      page.getByText("Which authentication method should we use?"),
    ).toBeVisible({ timeout: 15000 });

    // Dismiss instead of answering
    await page.getByRole("button", { name: "Dismiss questions" }).click();

    // Skip submits type: "skipped" for each question
    await expect(page.getByText("Skipped")).toBeVisible({ timeout: 10000 });

    const answer = queryFirstAnswerPayload(sessionId);
    expect(answer.user_questions_event_id).toBe(eventId);
    expect(answer.responses[q1Id].type).toBe("skipped");
  });

  test("should show Action Required badge for pending questions", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    const eventId = v7();
    const { payload } = buildMultipleChoicePayload();
    insertEvent(eventId, sessionId, payload);

    // Pending questions show "Action Required" badge in the event stream
    await expect(page.getByText("Action Required")).toBeVisible({
      timeout: 15000,
    });

    // The event stream title should be "Question"
    const questionEvent = page.getByText("Question").first();
    await expect(questionEvent).toBeVisible();
  });

  test("should disable Submit when free response is empty", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    const eventId = v7();
    const { payload } = buildFreeResponsePayload("Describe your requirements");
    insertEvent(eventId, sessionId, payload);

    await expect(page.getByText("Describe your requirements")).toBeVisible({
      timeout: 15000,
    });

    // Submit should be disabled with empty textarea
    const submitButton = page.getByRole("button", { name: /submit/i });
    await expect(submitButton).toBeDisabled();

    // Whitespace-only input should still be disabled
    const textarea = page.getByPlaceholder("Type your response...");
    await textarea.fill("   ");
    await expect(submitButton).toBeDisabled();

    // Real content should enable Submit
    await textarea.fill("Need RBAC");
    await expect(submitButton).toBeEnabled();
  });

  test("spam-clicking Submit sends only one answer request", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    // Intercept answer-questions requests to count them
    let answerRequestCount = 0;
    await page.route("**/events/answer-questions", async (route) => {
      answerRequestCount++;
      await route.continue();
    });

    const eventId = v7();
    const { payload } = buildMultipleChoicePayload();
    insertEvent(eventId, sessionId, payload);

    await expect(
      page.getByText("Which authentication method should we use?"),
    ).toBeVisible({ timeout: 15000 });

    await page.getByText("JWT").click();

    const submitButton = page.getByRole("button", { name: /submit/i });
    await expect(submitButton).toBeEnabled();
    const boundingBox = await submitButton.boundingBox();
    if (!boundingBox) throw new Error("Submit button bounding box not found");

    const clickX = boundingBox.x + boundingBox.width / 2;
    const clickY = boundingBox.y + boundingBox.height / 2;

    await Promise.all([
      page.mouse.click(clickX, clickY),
      page.mouse.click(clickX, clickY),
      page.mouse.click(clickX, clickY),
    ]);

    await page.waitForTimeout(3000);

    expect(
      answerRequestCount,
      "Spam-clicking Submit should send only one request",
    ).toBe(1);
  });
});
