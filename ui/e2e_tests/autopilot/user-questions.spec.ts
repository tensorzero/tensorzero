import { test, expect } from "@playwright/test";
import { v7 } from "uuid";
import { insertEvent, queryEventPayloads } from "./helpers/db";

/**
 * Build a user_questions event payload with multiple choice questions.
 * Returns the payload and the IDs for later reference.
 */
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

/**
 * Build a multi-question payload with one MC question and one free response.
 */
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

/**
 * Create a new autopilot session and return the session ID.
 * Sends a message through the UI, waits for redirect, then interrupts.
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

  // Interrupt the session so the worker stops generating events
  const stopButton = page.getByRole("button", { name: /stop session/i });
  await expect(stopButton).toBeVisible({ timeout: 30000 });
  await stopButton.click();
  await expect(page.getByText("Interrupted session")).toBeVisible({
    timeout: 10000,
  });

  return sessionId;
}

test.describe("User questions", () => {
  test("should display a multiple choice question and submit the selected answer", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    // Insert a user_questions event with a single MC question
    const eventId = v7();
    const { payload, q1Id, opt1Id } = buildMultipleChoicePayload();
    insertEvent(eventId, sessionId, payload);

    // Wait for the PendingQuestionCard to appear in the footer
    await expect(
      page.getByText("Which authentication method should we use?"),
    ).toBeVisible({ timeout: 15000 });

    // The "Question" header should be visible
    await expect(page.getByText("Question").first()).toBeVisible();

    // Select the JWT option
    await page.getByText("JWT").click();

    // Submit the answer
    await page.getByRole("button", { name: /submit/i }).click();

    // Verify "Question · Answered" event appears in the stream
    await expect(page.getByText("Answered")).toBeVisible({ timeout: 10000 });

    // Verify the response was recorded in the database
    const answers = queryEventPayloads(sessionId, "user_questions_answers");
    expect(answers.length).toBeGreaterThanOrEqual(1);

    const answer = answers[0] as {
      type: string;
      user_questions_event_id: string;
      responses: Record<
        string,
        { type: string; selected?: string[]; text?: string }
      >;
    };
    expect(answer.type).toBe("user_questions_answers");
    expect(answer.user_questions_event_id).toBe(eventId);
    expect(answer.responses[q1Id]).toBeDefined();
    expect(answer.responses[q1Id].type).toBe("multiple_choice");
    expect(answer.responses[q1Id].selected).toContain(opt1Id);
  });

  test("should display a free response question and submit the typed answer", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    // Insert a user_questions event with a free response question
    const eventId = v7();
    const q1Id = v7();
    const payload = {
      type: "user_questions" as const,
      questions: [
        {
          id: q1Id,
          header: "Details",
          question: "What specific requirements do you have?",
          type: "free_response" as const,
        },
      ],
    };
    insertEvent(eventId, sessionId, payload);

    // Wait for the PendingQuestionCard to appear
    await expect(
      page.getByText("What specific requirements do you have?"),
    ).toBeVisible({ timeout: 15000 });

    // Type a response
    const textarea = page.getByPlaceholder("Type your response...");
    await textarea.fill("We need RBAC with role-based permissions");

    // Submit
    await page.getByRole("button", { name: /submit/i }).click();

    // Verify answered event appears
    await expect(page.getByText("Answered")).toBeVisible({ timeout: 10000 });

    // Verify the response in the database
    const answers = queryEventPayloads(sessionId, "user_questions_answers");
    expect(answers.length).toBeGreaterThanOrEqual(1);

    const answer = answers[0] as {
      type: string;
      user_questions_event_id: string;
      responses: Record<
        string,
        { type: string; selected?: string[]; text?: string }
      >;
    };
    expect(answer.user_questions_event_id).toBe(eventId);
    expect(answer.responses[q1Id].type).toBe("free_response");
    expect(answer.responses[q1Id].text).toBe(
      "We need RBAC with role-based permissions",
    );
  });

  test("should navigate between steps in a multi-question flow and submit all answers", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    // Insert a user_questions event with two questions
    const eventId = v7();
    const { payload, q1Id, q2Id, opt1Id } = buildMultiQuestionPayload();
    insertEvent(eventId, sessionId, payload);

    // Wait for the first question to appear
    await expect(page.getByText("Which framework do you prefer?")).toBeVisible({
      timeout: 15000,
    });

    // Multi-question cards show "Questions" header and step tabs
    await expect(page.getByText("Questions").first()).toBeVisible();

    // Select React for the first question (use role to avoid matching description text)
    await page
      .getByRole("button", { name: "React React with TypeScript" })
      .click();

    // Click Next to move to the second question
    await page.getByRole("button", { name: /next/i }).click();

    // Wait for the free response step to appear
    await expect(page.getByText("Any additional requirements?")).toBeVisible();

    // Type a free response
    const textarea = page.getByPlaceholder("Type your response...");
    await textarea.fill("Must support SSR");

    // Submit all answers
    await page.getByRole("button", { name: /submit/i }).click();

    // Verify answered event appears
    await expect(page.getByText("Answered")).toBeVisible({ timeout: 10000 });

    // Verify both answers in the database
    const answers = queryEventPayloads(sessionId, "user_questions_answers");
    expect(answers.length).toBeGreaterThanOrEqual(1);

    const answer = answers[0] as {
      type: string;
      user_questions_event_id: string;
      responses: Record<
        string,
        { type: string; selected?: string[]; text?: string }
      >;
    };
    expect(answer.user_questions_event_id).toBe(eventId);

    // First question: multiple choice
    expect(answer.responses[q1Id].type).toBe("multiple_choice");
    expect(answer.responses[q1Id].selected).toContain(opt1Id);

    // Second question: free response
    expect(answer.responses[q2Id].type).toBe("free_response");
    expect(answer.responses[q2Id].text).toBe("Must support SSR");
  });

  test("should skip questions when dismiss button is clicked", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    // Insert a user_questions event
    const eventId = v7();
    const { payload, q1Id } = buildMultipleChoicePayload();
    insertEvent(eventId, sessionId, payload);

    // Wait for the question card to appear
    await expect(
      page.getByText("Which authentication method should we use?"),
    ).toBeVisible({ timeout: 15000 });

    // Click the dismiss/skip button
    await page.getByRole("button", { name: "Dismiss questions" }).click();

    // Verify skipped event appears (skip submits as type: "skipped")
    await expect(page.getByText("Skipped")).toBeVisible({ timeout: 10000 });

    // Verify the skipped response in the database
    const answers = queryEventPayloads(sessionId, "user_questions_answers");
    expect(answers.length).toBeGreaterThanOrEqual(1);

    const answer = answers[0] as {
      type: string;
      user_questions_event_id: string;
      responses: Record<string, { type: string }>;
    };
    expect(answer.user_questions_event_id).toBe(eventId);
    expect(answer.responses[q1Id].type).toBe("skipped");
  });

  test("should show user_questions event in the event stream with Action Required badge", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    const eventId = v7();
    const { payload } = buildMultipleChoicePayload();
    insertEvent(eventId, sessionId, payload);

    // Wait for the event to appear in the stream
    await expect(page.getByText("Action Required")).toBeVisible({
      timeout: 15000,
    });

    // The event stream should show "Question" as the event title
    // (expandable event in the stream, separate from the footer card)
    const questionEvent = page.getByText("Question").first();
    await expect(questionEvent).toBeVisible();
  });

  test("spam-clicking Submit sends only one answer request", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createAndInterruptSession(page);

    // Track answer-questions requests (set up after navigation so the
    // route handler survives for the rest of the test)
    let answerRequestCount = 0;
    await page.route("**/events/answer-questions", async (route) => {
      answerRequestCount++;
      await route.continue();
    });

    const eventId = v7();
    const { payload } = buildMultipleChoicePayload();
    insertEvent(eventId, sessionId, payload);

    // Wait for question card
    await expect(
      page.getByText("Which authentication method should we use?"),
    ).toBeVisible({ timeout: 15000 });

    // Select an option
    await page.getByText("JWT").click();

    // Get submit button position before clicking
    const submitButton = page.getByRole("button", { name: /submit/i });
    await expect(submitButton).toBeEnabled();
    const boundingBox = await submitButton.boundingBox();
    if (!boundingBox) throw new Error("Submit button bounding box not found");

    const clickX = boundingBox.x + boundingBox.width / 2;
    const clickY = boundingBox.y + boundingBox.height / 2;

    // Spam-click at the fixed position
    await Promise.all([
      page.mouse.click(clickX, clickY),
      page.mouse.click(clickX, clickY),
      page.mouse.click(clickX, clickY),
    ]);

    // Wait for submission to complete
    await page.waitForTimeout(3000);

    expect(
      answerRequestCount,
      "Spam-clicking Submit should send only one request",
    ).toBe(1);
  });
});
