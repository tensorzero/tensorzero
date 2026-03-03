import { test, expect } from "@playwright/test";
import { v7 } from "uuid";
import { insertEvent } from "./helpers/db";
import { createSession } from "./helpers/session";

// ── Payload builders ───────────────────────────────────────────────────

function buildFeedbackByVariantToolEvents() {
  const toolCallEventId = v7();

  const toolCallPayload = {
    type: "tool_call" as const,
    name: "get_feedback_by_variant",
    arguments: {
      metric_name: "exact_match",
      function_name: "basic_test",
      variant_names: null,
    },
    side_info: {
      tool_call_event_id: toolCallEventId,
      session_id: "placeholder",
      config_snapshot_hash: "abc123",
      optimization: {
        poll_interval_secs: 60,
        max_wait_secs: 86400,
      },
    },
  };

  const feedbackData = [
    {
      variant_name: "gpt4o_mini_variant",
      mean: 0.72,
      variance: 0.0225,
      count: 150,
    },
    {
      variant_name: "claude_variant",
      mean: 0.85,
      variance: 0.0144,
      count: 120,
    },
    {
      variant_name: "baseline_variant",
      mean: 0.45,
      variance: 0.0484,
      count: 90,
    },
  ];

  const toolResultPayload = {
    type: "tool_result" as const,
    tool_call_event_id: toolCallEventId,
    outcome: {
      type: "success" as const,
      result: JSON.stringify(feedbackData),
    },
  };

  return { toolCallPayload, toolResultPayload, toolCallEventId };
}

// ── Tests ──────────────────────────────────────────────────────────────

test.describe("Variant Performance Visualization", () => {
  test("should render variant performance chart from tool result", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const { toolCallPayload, toolResultPayload } =
      buildFeedbackByVariantToolEvents();

    // Insert tool_call then tool_result events
    const toolCallId = v7();
    const toolResultId = v7();
    insertEvent(toolCallId, sessionId, toolCallPayload);
    insertEvent(toolResultId, sessionId, toolResultPayload);

    // Wait for the tool result card to appear (auto-expanded)
    await expect(
      page.getByText("get_feedback_by_variant", { exact: false }),
    ).toBeVisible({ timeout: 15000 });

    // Verify the metric and function name labels are shown
    await expect(page.getByText("exact_match")).toBeVisible({ timeout: 5000 });
    await expect(page.getByText("basic_test")).toBeVisible();

    // Verify the chart is rendered (ChartContainer adds data-chart attribute)
    const chart = page.locator("[data-chart]");
    await expect(chart).toHaveCount(1);

    // Verify SVG chart elements are rendered
    const svgElement = page.locator("svg.recharts-surface");
    await expect(svgElement).toHaveCount(1);

    // Verify bar chart rectangles are rendered (3 variants = 3 bars)
    const bars = page.locator(".recharts-bar-rectangle");
    await expect(bars).toHaveCount(3);

    // Verify the "All time" x-axis label (cumulative granularity)
    await expect(
      chart.locator("text").filter({ hasText: "All time" }),
    ).toBeVisible();
  });

  test("should render single variant without legend", async ({ page }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const toolCallEventId = v7();
    const toolCallPayload = {
      type: "tool_call" as const,
      name: "get_feedback_by_variant",
      arguments: {
        metric_name: "accuracy",
        function_name: "simple_fn",
        variant_names: null,
      },
      side_info: {
        tool_call_event_id: toolCallEventId,
        session_id: "placeholder",
        config_snapshot_hash: "abc123",
        optimization: {
          poll_interval_secs: 60,
          max_wait_secs: 86400,
        },
      },
    };

    const toolResultPayload = {
      type: "tool_result" as const,
      tool_call_event_id: toolCallEventId,
      outcome: {
        type: "success" as const,
        result: JSON.stringify([
          {
            variant_name: "only_variant",
            mean: 0.92,
            variance: null,
            count: 50,
          },
        ]),
      },
    };

    insertEvent(v7(), sessionId, toolCallPayload);
    insertEvent(v7(), sessionId, toolResultPayload);

    // Wait for the tool result to appear
    await expect(
      page.getByText("get_feedback_by_variant", { exact: false }),
    ).toBeVisible({ timeout: 15000 });

    // Single variant should render one bar
    await expect(page.locator("[data-chart]")).toHaveCount(1);
    const bars = page.locator(".recharts-bar-rectangle");
    await expect(bars).toHaveCount(1);
  });
});
