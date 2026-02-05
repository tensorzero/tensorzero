import { test, expect } from "@playwright/test";
import { uniqueDatasetName } from "./helpers";

// Helper to create test datapoints for the basic_test function via API
async function createTestDatapoints(
  gatewayUrl: string,
  datasetName: string,
  count: number,
): Promise<string[]> {
  const messages = [
    "Hello",
    "How are you?",
    "Tell me a joke",
    "What is the weather?",
    "Good morning",
    "Goodbye",
    "Help me",
    "Thanks",
    "What can you do?",
    "Test message",
  ];

  const datapoints = Array.from({ length: count }, (_, i) => {
    const messageText = messages[i % messages.length];
    return {
      type: "chat",
      function_name: "basic_test",
      input: {
        system: { assistant_name: "TestBot" },
        messages: [
          {
            role: "user",
            content: [{ type: "text", text: messageText }],
          },
        ],
      },
      // Reference output equals input text for exact_match testing
      output: [{ type: "text", text: messageText }],
      tags: { source: "e2e_topk_visualization_test" },
    };
  });

  const response = await fetch(
    `${gatewayUrl}/v1/datasets/${datasetName}/datapoints`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ datapoints }),
    },
  );

  if (!response.ok) {
    throw new Error(
      `Failed to create datapoints: ${response.status} ${await response.text()}`,
    );
  }

  const body = await response.json();
  return body.ids;
}

test.describe("TopK Evaluation Visualization", () => {
  // Get gateway URL from environment or default (matches CLIENT_GATEWAY_URL in Rust tests)
  const gatewayUrl = process.env.CLIENT_GATEWAY_URL || "http://localhost:3040";

  test("should run topk_evaluation and render the visualization", async ({
    page,
  }, testInfo) => {
    // Long timeout since this involves LLM responses and evaluation runs
    test.setTimeout(180000);

    // Create test datapoints dynamically
    // Use deterministic name to enable prompt/model caching
    const datasetName = uniqueDatasetName(testInfo, "topk_viz");
    const datapointIds = await createTestDatapoints(
      gatewayUrl,
      datasetName,
      10,
    );
    expect(datapointIds.length).toBe(10);

    // Wait for ClickHouse to commit the datapoints
    await page.waitForTimeout(1000);

    // Navigate to autopilot sessions
    await page.goto("/autopilot/sessions");

    // Wait for the page to load
    await expect(
      page.getByRole("heading", { name: "Autopilot Sessions" }),
    ).toBeVisible();

    // Wait for the page to stabilize (sessions table to finish loading)
    // The table shows skeleton rows while loading, then actual content or empty state
    await page.waitForLoadState("networkidle");

    // Click to create a new session
    await page.getByRole("button", { name: /new session/i }).click();

    // Wait for the new session page
    await expect(page).toHaveURL(/\/autopilot\/sessions\/new/);

    // Ask the agent to run the topk_evaluation tool with test fixtures
    // Using the test_topk_evaluation_no_error evaluation which uses dummy models
    const messageInput = page.getByRole("textbox");
    await messageInput.fill(`Call the topk_evaluation tool with:
- evaluation_name: "test_topk_evaluation_no_error"
- dataset_name: "${datasetName}"
- variant_names: ["echo", "empty", "empty2"]
- k_min: 1
- max_datapoints: 10`);

    // Send the message
    await page.getByRole("button", { name: "Send message" }).click();

    // Wait for redirect to the actual session page
    await expect(page).toHaveURL(/\/autopilot\/sessions\/[a-f0-9-]+$/, {
      timeout: 30000,
    });

    // Wait for and approve tool calls as they appear
    const maxApprovalAttempts = 20;
    for (let i = 0; i < maxApprovalAttempts; i++) {
      // Check if we already have the visualization card
      const hasVisualizationCard = await page
        .getByText("Top-K Evaluation Results")
        .isVisible()
        .catch(() => false);

      if (hasVisualizationCard) {
        break;
      }

      // Look for an approve button and click it if found
      const approveButton = page
        .getByRole("button", { name: "Approve" })
        .first();
      const isApproveVisible = await approveButton
        .isVisible()
        .catch(() => false);

      if (isApproveVisible) {
        await approveButton.click();
        // Wait for the tool to execute
        await page.waitForTimeout(3000);
      } else {
        // No approve button visible, wait and check again
        await page.waitForTimeout(3000);
      }
    }

    // Wait for the visualization card to appear
    const visualizationCard = page.getByText("Top-K Evaluation Results");
    await expect(visualizationCard).toBeVisible({ timeout: 60000 });

    // Wait for the final assistant message to appear after the visualization
    // Using .last() to get the most recent assistant message (after the visualization)
    const lastAssistantMessage = page.getByText("Assistant").last();
    // Scroll to make the assistant message visible in the viewport (and in video recordings)
    await lastAssistantMessage.scrollIntoViewIfNeeded();
    await expect(lastAssistantMessage).toBeVisible({ timeout: 60000 });

    // Click on the card to expand it
    await visualizationCard.click();

    // Verify the visualization renders with both chart titles inside the expanded card
    await expect(page.getByText("Mean Performance by Variant")).toBeVisible({
      timeout: 10000,
    });
    await expect(
      page.getByText("Number of Evaluations by Variant"),
    ).toBeVisible();

    // Verify the charts are rendered (ChartContainer adds data-chart attribute)
    const charts = page.locator("[data-chart]");
    await expect(charts).toHaveCount(2);

    // Scroll the second chart into view to ensure all content is visible in video
    const secondChart = page.locator("[data-chart]").nth(1);
    await secondChart.scrollIntoViewIfNeeded();

    // Verify variant names appear on the x-axis (below the second chart)
    await expect(secondChart.getByText("echo", { exact: true })).toBeVisible();
    await expect(secondChart.getByText("empty", { exact: true })).toBeVisible();
    await expect(
      secondChart.getByText("empty2", { exact: true }),
    ).toBeVisible();

    // Verify SVG chart elements are rendered
    const svgElements = page.locator("svg.recharts-surface");
    await expect(svgElements).toHaveCount(2);

    // Verify bar chart rectangles are rendered (3 variants × 2 charts = 6 bars)
    // Note the top chart uses custom "bars" which are actually dots
    const bars = page.locator(".recharts-bar-rectangle");
    await expect(bars).toHaveCount(6);

    // Verify the final assistant message is still visible at the end of the test
    await expect(lastAssistantMessage).toBeVisible();
  });
});
