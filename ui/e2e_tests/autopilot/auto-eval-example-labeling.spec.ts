import { test, expect } from "@playwright/test";
import { v7 } from "uuid";
import { insertEvent, queryEventPayloads } from "./helpers/db";
import { createSession } from "./helpers/session";

// ── Types ──────────────────────────────────────────────────────────────

type AnswerExample = {
  label_answer: { type: string; selected?: string[] };
  explanation_answer?: { type: string; text?: string };
};

type AnswerPayload = {
  type: string;
  auto_eval_example_labeling_event_id: string;
  examples: AnswerExample[];
};

// ── Payload builders ───────────────────────────────────────────────────

function buildSingleExamplePayload() {
  const labelQId = v7();
  const explQId = v7();
  const optYes = v7();
  const optNo = v7();

  const payload = {
    type: "auto_eval_example_labeling" as const,
    examples: [
      {
        maybe_excerpted_prompt: {
          type: "json" as const,
          label: "Input",
          data: {
            system: "You are playing 20 questions. The secret word is: piano.",
            messages: [
              {
                role: "user",
                content: [{ type: "text", text: "Is it a living thing?" }],
              },
            ],
          },
        },
        maybe_excerpted_response: {
          type: "json" as const,
          label: "Output",
          data: [{ type: "text", text: "No, it is not a living thing." }],
        },
        source: {
          type: "inference" as const,
          id: "00000000-0000-0000-0000-000000000001",
        },
        label_question: {
          id: labelQId,
          header: "Example 1",
          question: "Did the model answer correctly?",
          options: [
            { id: optYes, label: "Yes", description: "Correct answer" },
            { id: optNo, label: "No", description: "Incorrect answer" },
          ],
        },
        explanation_question: {
          id: explQId,
          header: "Rationale",
          question: "Explain your rating",
        },
      },
    ],
  };

  return { payload, labelQId, explQId, optYes, optNo };
}

function buildMultiExamplePayload() {
  const q1Id = v7();
  const q2Id = v7();
  const q3Id = v7();
  const eq1Id = v7();
  const eq2Id = v7();
  const opt1Yes = v7();
  const opt1No = v7();
  const opt2Yes = v7();
  const opt2No = v7();
  const opt3Yes = v7();
  const opt3No = v7();

  const payload = {
    type: "auto_eval_example_labeling" as const,
    examples: [
      {
        maybe_excerpted_prompt: {
          type: "json" as const,
          label: "Input",
          data: {
            system: "Secret word: piano.",
            messages: [
              {
                role: "user",
                content: [{ type: "text", text: "Is it a living thing?" }],
              },
            ],
          },
        },
        maybe_excerpted_response: {
          type: "json" as const,
          label: "Output",
          data: [{ type: "text", text: "No." }],
        },
        source: {
          type: "inference" as const,
          id: "00000000-0000-0000-0000-000000000001",
        },
        label_question: {
          id: q1Id,
          header: "Example 1",
          question: "Did the model answer correctly for example 1?",
          options: [
            { id: opt1Yes, label: "Yes", description: "Correct" },
            { id: opt1No, label: "No", description: "Incorrect" },
          ],
        },
        explanation_question: {
          id: eq1Id,
          header: "Rationale",
          question: "Explain your rating for example 1",
        },
      },
      {
        maybe_excerpted_prompt: {
          type: "json" as const,
          label: "Input",
          data: {
            system: "Secret word: soccer ball.",
            messages: [
              {
                role: "user",
                content: [{ type: "text", text: "Is it man-made?" }],
              },
            ],
          },
        },
        maybe_excerpted_response: {
          type: "json" as const,
          label: "Output",
          data: [{ type: "text", text: "Yes, it is man-made." }],
        },
        source: {
          type: "inference" as const,
          id: "00000000-0000-0000-0000-000000000002",
        },
        label_question: {
          id: q2Id,
          header: "Example 2",
          question: "Did the model answer correctly for example 2?",
          options: [
            { id: opt2Yes, label: "Yes", description: "Correct" },
            { id: opt2No, label: "No", description: "Incorrect" },
          ],
        },
        explanation_question: {
          id: eq2Id,
          header: "Rationale",
          question: "Explain your rating for example 2",
        },
      },
      {
        maybe_excerpted_prompt: null,
        maybe_excerpted_response: {
          type: "json" as const,
          label: "Output",
          data: [{ type: "text", text: "Mount Everest is very tall." }],
        },
        source: {
          type: "inference" as const,
          id: "00000000-0000-0000-0000-000000000003",
        },
        label_question: {
          id: q3Id,
          header: "Example 3",
          question: "Did the model answer correctly for example 3?",
          options: [
            { id: opt3Yes, label: "Yes", description: "Correct" },
            { id: opt3No, label: "No", description: "Incorrect" },
          ],
        },
        // No explanation question for example 3
      },
    ],
  };

  return {
    payload,
    q1Id,
    q2Id,
    q3Id,
    eq1Id,
    eq2Id,
    opt1Yes,
    opt1No,
    opt2Yes,
    opt2No,
    opt3Yes,
    opt3No,
  };
}

function buildNoExplanationPayload() {
  const labelQId = v7();
  const optYes = v7();
  const optNo = v7();

  const payload = {
    type: "auto_eval_example_labeling" as const,
    examples: [
      {
        maybe_excerpted_prompt: {
          type: "json" as const,
          label: "Input",
          data: {
            messages: [
              {
                role: "user",
                content: [{ type: "text", text: "What is 2+2?" }],
              },
            ],
          },
        },
        maybe_excerpted_response: {
          type: "json" as const,
          label: "Output",
          data: [{ type: "text", text: "4" }],
        },
        source: {
          type: "inference" as const,
          id: "00000000-0000-0000-0000-000000000004",
        },
        label_question: {
          id: labelQId,
          header: "Correctness",
          question: "Is the answer correct?",
          options: [
            { id: optYes, label: "Yes", description: "Correct answer" },
            { id: optNo, label: "No", description: "Incorrect answer" },
          ],
        },
        // No explanation_question
      },
    ],
  };

  return { payload, labelQId, optYes, optNo };
}

function buildMarkdownPayload() {
  const labelQId = v7();
  const optYes = v7();
  const optNo = v7();

  const payload = {
    type: "auto_eval_example_labeling" as const,
    examples: [
      {
        maybe_excerpted_prompt: {
          type: "markdown" as const,
          label: "Prompt",
          text: "Review the following code for security issues.",
        },
        maybe_excerpted_response: {
          type: "markdown" as const,
          label: "Response",
          text: "## Review\n\nNo security issues found. The code follows best practices.",
        },
        source: {
          type: "synthetic" as const,
          full_prompt: null,
          full_response: {
            type: "markdown" as const,
            label: "Response",
            text: "## Review\n\nNo security issues found.",
          },
        },
        label_question: {
          id: labelQId,
          header: "Quality",
          question: "Is the review accurate?",
          options: [
            { id: optYes, label: "Yes", description: "Accurate review" },
            { id: optNo, label: "No", description: "Inaccurate review" },
          ],
        },
      },
    ],
  };

  return { payload, labelQId, optYes, optNo };
}

// ── Helpers ────────────────────────────────────────────────────────────

function queryFirstAutoEvalAnswerPayload(sessionId: string): AnswerPayload {
  const payloads = queryEventPayloads(
    sessionId,
    "auto_eval_example_labeling_answers",
  );
  expect(payloads.length).toBeGreaterThanOrEqual(1);
  return payloads[0] as AnswerPayload;
}

// ── Tests ──────────────────────────────────────────────────────────────

test.describe("Auto-eval example labeling", () => {
  test("should submit a single example label", async ({ page }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const eventId = v7();
    const {
      payload,
      labelQId: _labelQId,
      optYes,
    } = buildSingleExamplePayload();
    insertEvent(eventId, sessionId, payload);

    // Wait for the auto-eval card to appear
    await expect(page.getByText("Did the model answer correctly?")).toBeVisible(
      { timeout: 15000 },
    );

    // Card title should indicate example labeling
    await expect(
      page.getByText("Label examples to improve evaluator accuracy"),
    ).toBeVisible();

    // Select "Yes"
    await page.getByRole("button", { name: "Yes" }).click();

    // Submit
    await page.getByRole("button", { name: /submit/i }).click();

    // Verify the answer event appears in the stream
    await expect(page.getByText("Example Label")).toBeVisible({
      timeout: 10000,
    });

    // Verify the response was persisted correctly
    const answer = queryFirstAutoEvalAnswerPayload(sessionId);
    expect(answer.type).toBe("auto_eval_example_labeling_answers");
    expect(answer.auto_eval_example_labeling_event_id).toBe(eventId);
    expect(answer.examples).toHaveLength(1);
    expect(answer.examples[0].label_answer.type).toBe("multiple_choice");
    expect(answer.examples[0].label_answer.selected).toContain(optYes);
  });

  test("should submit a single example with explanation", async ({ page }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const eventId = v7();
    const {
      payload,
      labelQId: _labelQId,
      explQId: _explQId,
      optNo,
    } = buildSingleExamplePayload();
    insertEvent(eventId, sessionId, payload);

    await expect(page.getByText("Did the model answer correctly?")).toBeVisible(
      { timeout: 15000 },
    );

    // Select "No"
    await page.getByRole("button", { name: "No" }).click();

    // Fill in the explanation textarea
    const textarea = page.getByRole("textbox", {
      name: "Explain your rating",
    });
    await textarea.fill("The model gave too much detail instead of yes/no");

    await page.getByRole("button", { name: /submit/i }).click();

    await expect(page.getByText("Example Label")).toBeVisible({
      timeout: 10000,
    });

    const answer = queryFirstAutoEvalAnswerPayload(sessionId);
    expect(answer.auto_eval_example_labeling_event_id).toBe(eventId);
    expect(answer.examples).toHaveLength(1);
    expect(answer.examples[0].label_answer.type).toBe("multiple_choice");
    expect(answer.examples[0].label_answer.selected).toContain(optNo);
    expect(answer.examples[0].explanation_answer?.type).toBe("free_response");
    expect(answer.examples[0].explanation_answer?.text).toBe(
      "The model gave too much detail instead of yes/no",
    );
  });

  test("should submit single example without explanation question", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const eventId = v7();
    const {
      payload,
      labelQId: _labelQId,
      optYes,
    } = buildNoExplanationPayload();
    insertEvent(eventId, sessionId, payload);

    await expect(page.getByText("Is the answer correct?")).toBeVisible({
      timeout: 15000,
    });

    // No explanation textarea should be present
    await expect(
      page.getByRole("textbox", { name: /explain/i }),
    ).not.toBeVisible();

    await page.getByRole("button", { name: "Yes" }).click();
    await page.getByRole("button", { name: /submit/i }).click();

    await expect(page.getByText("Example Label")).toBeVisible({
      timeout: 10000,
    });

    const answer = queryFirstAutoEvalAnswerPayload(sessionId);
    expect(answer.examples).toHaveLength(1);
    expect(answer.examples[0].label_answer.type).toBe("multiple_choice");
    expect(answer.examples[0].label_answer.selected).toContain(optYes);
  });

  test("should navigate multi-example flow and submit all labels", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const eventId = v7();
    const {
      payload,
      q1Id: _q1Id,
      q2Id: _q2Id,
      q3Id: _q3Id,
      eq1Id: _eq1Id,
      opt1Yes,
      opt2No,
      opt3Yes,
    } = buildMultiExamplePayload();
    insertEvent(eventId, sessionId, payload);

    // Wait for first example question
    await expect(
      page.getByText("Did the model answer correctly for example 1?"),
    ).toBeVisible({ timeout: 15000 });

    // Should show step counter
    await expect(page.getByText("0/3 labeled")).toBeVisible();

    // Label example 1 as "Yes"
    await page.getByRole("button", { name: "Yes" }).click();
    await expect(page.getByText("1/3 labeled")).toBeVisible();

    // Fill explanation for example 1
    const textarea1 = page.getByRole("textbox", {
      name: "Explain your rating for example 1",
    });
    await textarea1.fill("Correct yes/no answer");

    // Navigate to example 2
    await page.getByRole("button", { name: /next/i }).click();
    await expect(
      page.getByText("Did the model answer correctly for example 2?"),
    ).toBeVisible();

    // Label example 2 as "No"
    await page.getByRole("button", { name: "No" }).click();
    await expect(page.getByText("2/3 labeled")).toBeVisible();

    // Navigate to example 3
    await page.getByRole("button", { name: /next/i }).click();
    await expect(
      page.getByText("Did the model answer correctly for example 3?"),
    ).toBeVisible();

    // Example 3 has no explanation textarea
    await expect(
      page.getByRole("textbox", { name: /explain/i }),
    ).not.toBeVisible();

    // Label example 3 as "Yes"
    await page.getByRole("button", { name: "Yes" }).click();
    await expect(page.getByText("3/3 labeled")).toBeVisible();

    // Submit (last example shows Submit instead of Next)
    await page.getByRole("button", { name: /submit/i }).click();

    // Verify answer event in the stream (multi-example shows "Example Labels" plural)
    await expect(page.getByText("Example Labels")).toBeVisible({
      timeout: 10000,
    });

    // Verify persisted responses
    const answer = queryFirstAutoEvalAnswerPayload(sessionId);
    expect(answer.auto_eval_example_labeling_event_id).toBe(eventId);
    expect(answer.examples).toHaveLength(3);

    // Example 1: Yes with explanation
    expect(answer.examples[0].label_answer.type).toBe("multiple_choice");
    expect(answer.examples[0].label_answer.selected).toContain(opt1Yes);
    expect(answer.examples[0].explanation_answer?.type).toBe("free_response");
    expect(answer.examples[0].explanation_answer?.text).toBe(
      "Correct yes/no answer",
    );

    // Example 2: No, explanation skipped
    expect(answer.examples[1].label_answer.type).toBe("multiple_choice");
    expect(answer.examples[1].label_answer.selected).toContain(opt2No);

    // Example 3: Yes, no explanation question
    expect(answer.examples[2].label_answer.type).toBe("multiple_choice");
    expect(answer.examples[2].label_answer.selected).toContain(opt3Yes);
  });

  test("should preserve selections when navigating back", async ({ page }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const eventId = v7();
    const {
      payload,
      q1Id: _q1Id,
      q2Id: _q2Id,
      q3Id: _q3Id,
      opt1Yes,
      opt2No,
      opt3Yes,
    } = buildMultiExamplePayload();
    insertEvent(eventId, sessionId, payload);

    await expect(
      page.getByText("Did the model answer correctly for example 1?"),
    ).toBeVisible({ timeout: 15000 });

    // Label example 1
    await page.getByRole("button", { name: "Yes" }).click();

    // Go to example 2
    await page.getByRole("button", { name: /next/i }).click();
    await expect(
      page.getByText("Did the model answer correctly for example 2?"),
    ).toBeVisible();

    // Label example 2
    await page.getByRole("button", { name: "No" }).click();

    // Go back to example 1
    await page.getByRole("button", { name: /back/i }).click();
    await expect(
      page.getByText("Did the model answer correctly for example 1?"),
    ).toBeVisible();

    // Go forward again and continue to example 3
    await page.getByRole("button", { name: /next/i }).click();
    await page.getByRole("button", { name: /next/i }).click();
    await expect(
      page.getByText("Did the model answer correctly for example 3?"),
    ).toBeVisible();

    // Label example 3
    await page.getByRole("button", { name: "Yes" }).click();

    // Submit and verify all preserved
    await page.getByRole("button", { name: /submit/i }).click();

    await expect(page.getByText("Example Labels")).toBeVisible({
      timeout: 10000,
    });

    const answer = queryFirstAutoEvalAnswerPayload(sessionId);
    expect(answer.examples).toHaveLength(3);
    expect(answer.examples[0].label_answer.selected).toContain(opt1Yes);
    expect(answer.examples[1].label_answer.selected).toContain(opt2No);
    expect(answer.examples[2].label_answer.selected).toContain(opt3Yes);
  });

  test("should dismiss and skip all examples", async ({ page }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const eventId = v7();
    const {
      payload,
      q1Id: _q1Id,
      q2Id: _q2Id,
      q3Id: _q3Id,
      eq1Id: _eq1Id,
      eq2Id: _eq2Id,
    } = buildMultiExamplePayload();
    insertEvent(eventId, sessionId, payload);

    await expect(
      page.getByText("Did the model answer correctly for example 1?"),
    ).toBeVisible({ timeout: 15000 });

    // Dismiss instead of answering
    await page.getByRole("button", { name: "Dismiss" }).click();

    // Verify the answer event appears in the stream
    await expect(page.getByText("Submitted")).toBeVisible({ timeout: 10000 });

    const answer = queryFirstAutoEvalAnswerPayload(sessionId);
    expect(answer.auto_eval_example_labeling_event_id).toBe(eventId);
    expect(answer.examples).toHaveLength(3);

    // All label questions should be skipped
    expect(answer.examples[0].label_answer.type).toBe("skipped");
    expect(answer.examples[1].label_answer.type).toBe("skipped");
    expect(answer.examples[2].label_answer.type).toBe("skipped");

    // All explanation questions should be skipped
    expect(answer.examples[0].explanation_answer?.type).toBe("skipped");
    expect(answer.examples[1].explanation_answer?.type).toBe("skipped");
  });

  test("should disable Submit until all examples are labeled", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const eventId = v7();
    const { payload } = buildMultiExamplePayload();
    insertEvent(eventId, sessionId, payload);

    await expect(
      page.getByText("Did the model answer correctly for example 1?"),
    ).toBeVisible({ timeout: 15000 });

    // On first example, Next should be disabled before selection
    const nextButton = page.getByRole("button", { name: /next/i });
    await expect(nextButton).toBeDisabled();

    // Select an option to enable Next
    await page.getByRole("button", { name: "Yes" }).click();
    await expect(nextButton).toBeEnabled();

    // Navigate to example 2
    await nextButton.click();
    await expect(
      page.getByText("Did the model answer correctly for example 2?"),
    ).toBeVisible();

    // Next should be disabled again
    await expect(nextButton).toBeDisabled();

    await page.getByRole("button", { name: "No" }).click();
    await nextButton.click();

    // On example 3 (last), Submit should be disabled until labeled
    await expect(
      page.getByText("Did the model answer correctly for example 3?"),
    ).toBeVisible();
    const submitButton = page.getByRole("button", { name: /submit/i });
    await expect(submitButton).toBeDisabled();

    await page.getByRole("button", { name: "Yes" }).click();
    await expect(submitButton).toBeEnabled();
  });

  test("should show Action Required badge for pending auto-eval", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const eventId = v7();
    const { payload } = buildSingleExamplePayload();
    insertEvent(eventId, sessionId, payload);

    // The event stream title should be "Example Labeling"
    await expect(page.getByText("Example Labeling")).toBeVisible({
      timeout: 15000,
    });
  });

  test("should render markdown content blocks", async ({ page }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const eventId = v7();
    const { payload, labelQId: _labelQId, optYes } = buildMarkdownPayload();
    insertEvent(eventId, sessionId, payload);

    await expect(page.getByText("Is the review accurate?")).toBeVisible({
      timeout: 15000,
    });

    // Select and submit
    await page.getByRole("button", { name: "Yes" }).click();
    await page.getByRole("button", { name: /submit/i }).click();

    await expect(page.getByText("Example Label")).toBeVisible({
      timeout: 10000,
    });

    const answer = queryFirstAutoEvalAnswerPayload(sessionId);
    expect(answer.examples).toHaveLength(1);
    expect(answer.examples[0].label_answer.type).toBe("multiple_choice");
    expect(answer.examples[0].label_answer.selected).toContain(optYes);
  });

  test("spam-clicking Submit sends only one answer request", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    // Intercept answer requests to count them
    let answerRequestCount = 0;
    await page.route(
      "**/events/answer-auto-eval-example-labeling",
      async (route) => {
        answerRequestCount++;
        await route.continue();
      },
    );

    const eventId = v7();
    const { payload } = buildSingleExamplePayload();
    insertEvent(eventId, sessionId, payload);

    await expect(page.getByText("Did the model answer correctly?")).toBeVisible(
      { timeout: 15000 },
    );

    await page.getByRole("button", { name: "Yes" }).click();

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

  test("should allow changing selection before submitting", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const eventId = v7();
    const { payload, labelQId: _labelQId, optNo } = buildSingleExamplePayload();
    insertEvent(eventId, sessionId, payload);

    await expect(page.getByText("Did the model answer correctly?")).toBeVisible(
      { timeout: 15000 },
    );

    // Select "Yes" first, then change to "No"
    await page.getByRole("button", { name: "Yes" }).click();
    await page.getByRole("button", { name: "No" }).click();

    await page.getByRole("button", { name: /submit/i }).click();

    await expect(page.getByText("Example Label")).toBeVisible({
      timeout: 10000,
    });

    const answer = queryFirstAutoEvalAnswerPayload(sessionId);
    expect(answer.examples).toHaveLength(1);
    expect(answer.examples[0].label_answer.type).toBe("multiple_choice");
    expect(answer.examples[0].label_answer.selected).toEqual([optNo]);
  });

  test("should click step tabs to jump between examples", async ({ page }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const eventId = v7();
    const { payload } = buildMultiExamplePayload();
    insertEvent(eventId, sessionId, payload);

    await expect(
      page.getByText("Did the model answer correctly for example 1?"),
    ).toBeVisible({ timeout: 15000 });

    // Click on the "Example 3" step tab directly
    await page.getByRole("button", { name: /Example 3/ }).click();
    await expect(
      page.getByText("Did the model answer correctly for example 3?"),
    ).toBeVisible();

    // Click on the "Example 1" step tab
    await page.getByRole("button", { name: /Example 1/ }).click();
    await expect(
      page.getByText("Did the model answer correctly for example 1?"),
    ).toBeVisible();
  });

  test("should skip explanation when left empty", async ({ page }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const eventId = v7();
    const {
      payload,
      labelQId: _labelQId,
      explQId: _explQId,
      optYes,
    } = buildSingleExamplePayload();
    insertEvent(eventId, sessionId, payload);

    await expect(page.getByText("Did the model answer correctly?")).toBeVisible(
      { timeout: 15000 },
    );

    // Select label but don't fill explanation
    await page.getByRole("button", { name: "Yes" }).click();
    await page.getByRole("button", { name: /submit/i }).click();

    await expect(page.getByText("Example Label")).toBeVisible({
      timeout: 10000,
    });

    const answer = queryFirstAutoEvalAnswerPayload(sessionId);
    expect(answer.examples).toHaveLength(1);
    expect(answer.examples[0].label_answer.type).toBe("multiple_choice");
    expect(answer.examples[0].label_answer.selected).toContain(optYes);
    // Explanation should be skipped when left empty
    expect(answer.examples[0].explanation_answer?.type).toBe("skipped");
  });
});
