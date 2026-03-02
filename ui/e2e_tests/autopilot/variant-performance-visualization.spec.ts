import { test, expect } from "@playwright/test";
import { v7 } from "uuid";
import { insertEvent } from "./helpers/db";

// ── Payload builders ───────────────────────────────────────────────────

function buildVariantPerformancesVisualization() {
  const toolExecutionId = v7();

  const payload = {
    type: "visualization" as const,
    tool_execution_id: toolExecutionId,
    visualization: {
      type: "variant_performances" as const,
      function_name: "basic_test",
      metric_name: "exact_match",
      time_granularity: "cumulative",
      performances: [
        {
          period_start: "1970-01-01T00:00:00Z",
          variant_name: "gpt4o_mini_variant",
          count: 150,
          avg_metric: 0.72,
          stdev: 0.15,
          ci_error: 0.024,
        },
        {
          period_start: "1970-01-01T00:00:00Z",
          variant_name: "claude_variant",
          count: 120,
          avg_metric: 0.85,
          stdev: 0.12,
          ci_error: 0.021,
        },
        {
          period_start: "1970-01-01T00:00:00Z",
          variant_name: "baseline_variant",
          count: 90,
          avg_metric: 0.45,
          stdev: 0.22,
          ci_error: 0.045,
        },
      ],
    },
  };

  return { payload, toolExecutionId };
}

// ── Helpers ────────────────────────────────────────────────────────────

async function createSession(
  page: import("@playwright/test").Page,
): Promise<string> {
  await page.goto("/autopilot/sessions/new");
  await page.waitForLoadState("networkidle");
  const messageInput = page.getByRole("textbox");
  await messageInput.fill(`Test variant perf viz ${v7()}`);
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

// ── Tests ──────────────────────────────────────────────────────────────

test.describe("Variant Performance Visualization", () => {
  test("should render variant performance chart from visualization event", async ({
    page,
  }) => {
    test.setTimeout(120000);

    const sessionId = await createSession(page);

    // Insert a visualization event directly into the database
    const eventId = v7();
    const { payload } = buildVariantPerformancesVisualization();
    insertEvent(eventId, sessionId, payload);

    // Wait for the visualization card to appear in the event stream
    const vizCard = page.getByText("Variant Performance");
    await expect(vizCard).toBeVisible({ timeout: 15000 });

    // Click on the card to expand it
    await vizCard.click();

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

    const eventId = v7();
    const payload = {
      type: "visualization" as const,
      tool_execution_id: v7(),
      visualization: {
        type: "variant_performances" as const,
        function_name: "simple_fn",
        metric_name: "accuracy",
        time_granularity: "cumulative",
        performances: [
          {
            period_start: "1970-01-01T00:00:00Z",
            variant_name: "only_variant",
            count: 50,
            avg_metric: 0.92,
          },
        ],
      },
    };
    insertEvent(eventId, sessionId, payload);

    const vizCard = page.getByText("Variant Performance");
    await expect(vizCard).toBeVisible({ timeout: 15000 });
    await vizCard.click();

    // Single variant should render one bar
    await expect(page.locator("[data-chart]")).toHaveCount(1);
    const bars = page.locator(".recharts-bar-rectangle");
    await expect(bars).toHaveCount(1);
  });
});
