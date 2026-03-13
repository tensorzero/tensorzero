import { test, expect } from "@playwright/test";
import { v7 } from "uuid";
import { insertEvent } from "./helpers/db";
import { createSession } from "./helpers/session";

function buildFeedbackByVariantToolEvents(sessionId: string) {
  const toolCallEventId = v7();
  const toolCallId = `call_test_${toolCallEventId.replace(/-/g, "").slice(0, 24)}`;

  const toolCallPayload = {
    type: "tool_call" as const,
    tool_call_id: toolCallId,
    name: "get_feedback_by_variant",
    arguments: {
      metric_name: "exact_match",
      function_name: "basic_test",
      variant_names: null,
    },
    side_info: {
      tool_call_event_id: toolCallEventId,
      session_id: sessionId,
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
    tool_call_id: toolCallId,
    tool_call_event_id: toolCallEventId,
    tool_call_name: "get_feedback_by_variant",
    tool_call_arguments: toolCallPayload.arguments,
    outcome: {
      type: "success" as const,
      result: JSON.stringify(feedbackData),
    },
  };

  return { toolCallPayload, toolResultPayload };
}

test.describe("Variant Performance Visualization", () => {
  test("should render variant performance chart from tool result", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    const { toolCallPayload, toolResultPayload } =
      buildFeedbackByVariantToolEvents(sessionId);

    insertEvent(v7(), sessionId, toolCallPayload);
    insertEvent(v7(), sessionId, toolResultPayload);

    // Wait for the tool result card to appear (auto-expanded)
    await expect(
      page.getByText("get_feedback_by_variant", { exact: false }),
    ).toBeVisible({ timeout: 15000 });

    // Verify the chart is rendered with bars
    const chart = page.locator("[data-chart]");
    await expect(chart).toHaveCount(1);
    await expect(page.locator(".recharts-bar-rectangle")).toHaveCount(3);

    // Verify cumulative x-axis label
    await expect(
      chart.locator("text").filter({ hasText: "All time" }),
    ).toBeVisible();
  });
});
