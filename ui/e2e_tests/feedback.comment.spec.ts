import { test, expect } from "@playwright/test";
import { z } from "zod";

let INFERENCE_ID: string;

console.log(process.env.TENSORZERO_GATEWAY_URL);

test.beforeEach(async () => {
  const response = await fetch(
    `${process.env.TENSORZERO_GATEWAY_URL}/inference`,
    {
      method: "POST",
      body: JSON.stringify({
        model_name: "",
        input: {
          messages: [],
        },
      }),
    },
  );

  console.log(await response.json());

  const { inference_id } = z
    .object({
      inference_id: z.string(),
    })
    .parse(await response.json());

  INFERENCE_ID = inference_id;
});

test("adding feedback should correctly show latest/overwritten", async ({
  page,
}) => {
  await page.goto(`/observability/inferences/${INFERENCE_ID}`);

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // Count existing feedback to know starting state
  // const existingFeedbackRows = page.locator("tbody tr");
  // const initialCount = await existingFeedbackRows.count();

  // Step 3: Add first comment
  await page.getByText("Add feedback").click();

  // Select comment metric
  await page.getByText("Select a metric").click();
  const commentMetricLocator = page
    .locator('div[role="dialog"]')
    .locator('div[cmdk-item=""]')
    .filter({
      hasText: "comment",
    });
  await commentMetricLocator.waitFor({ state: "visible" });
  await commentMetricLocator.click();

  // Enter first comment with timestamp to ensure uniqueness
  const timestamp = Date.now();
  const firstComment = `Test comment 1 - ${timestamp}`;
  await page.locator("#comment-input").fill(firstComment);

  // Submit first feedback
  await page.getByText("Submit Feedback").click();

  await page.waitForURL((url) => url.searchParams.has("newFeedbackId"), {
    timeout: 10000,
  });

  // Step 4: Verify the first comment was added
  // Wait for the feedback to appear in the table
  await expect(page.getByText(firstComment)).toBeVisible();

  // Get new count after first comment
  // const afterFirstCount = await page.locator("tbody tr").count();

  // Step 5: Add second comment
  await page.getByText("Add feedback").click();

  // Select comment metric again
  await page.getByText("Select a metric").click();
  const commentMetricLocator2 = page
    .locator('div[role="dialog"]')
    .locator('div[cmdk-item=""]')
    .filter({
      hasText: "comment",
    });
  await commentMetricLocator2.waitFor({ state: "visible" });
  await commentMetricLocator2.click();

  // Enter second comment
  const secondComment = `Test comment 2 - ${timestamp}`;
  await page.locator("#comment-input").fill(secondComment);

  // Submit second feedback
  await page.getByText("Submit Feedback").click();

  await page.waitForURL((url) => url.searchParams.has("newFeedbackId"), {
    timeout: 10000,
  });

  // Step 6: Assert the feedback count increased by 2 total
  // await expect(feedbackRows).toHaveCount(initialCount + 2);

  // Verify both comments are visible
  await expect(page.getByText(firstComment)).toBeVisible();
  await expect(page.getByText(secondComment)).toBeVisible();

  // Verify badges are now shown
  await expect(page.getByText("Latest")).toBeVisible();
  await expect(page.getByText("Overwritten")).toBeVisible();

  // The newer comment should have "Latest" badge
  // The older comment should have "Overwritten" badge and be visually dimmed
  const latestBadge = page.getByText("Latest");
  const overwrittenBadge = page.getByText("Overwritten");

  await expect(latestBadge).toBeVisible();
  await expect(overwrittenBadge).toBeVisible();
});
