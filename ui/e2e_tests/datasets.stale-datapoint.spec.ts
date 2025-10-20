import { test, expect } from "@playwright/test";
import { v7 } from "uuid";
import { createDatapointFromInference } from "./helpers/datapoint-helpers";

test("should show stale badge and disable edit/delete buttons on stale datapoint", async ({
  page,
}) => {
  const datasetName =
    "test_stale_dataset_" + Math.random().toString(36).substring(2, 15);

  // Step 1: Create a new datapoint from an inference
  await createDatapointFromInference(page, {
    inferenceId: "0196374b-0d7a-7a22-b2d2-598a14f2eacc",
    datasetName,
  });

  // Step 2: Capture the original datapoint ID from URL
  const originalUrl = page.url();
  const originalDatapointId = originalUrl.split("/datapoint/")[1].split("?")[0]; // Remove query params

  // Step 3: Edit and save the datapoint to create a new version
  await page.getByRole("button", { name: "Edit" }).click();

  // Edit the input (first contenteditable element)
  const inputTopic = v7();
  const inputMessage = `Tell me about ${inputTopic}`;
  await page.locator("div[contenteditable='true']").first().fill(inputMessage);

  // Edit the output (last contenteditable element)
  const outputTopic = v7();
  const output = `Here's information about ${outputTopic}`;
  await page.locator("div[contenteditable='true']").last().fill(output);

  // Save the datapoint
  await page.getByRole("button", { name: "Save" }).click();

  // Wait for redirect to new datapoint page
  await page.waitForURL((url) => !url.pathname.includes(originalDatapointId), {
    timeout: 10000,
  });
  await page.waitForLoadState("networkidle");

  // Verify no errors occurred
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();

  // Verify we've been redirected to a new datapoint ID
  const newUrl = page.url();
  const newDatapointId = newUrl.split("/datapoint/")[1].split("?")[0];
  expect(newDatapointId).not.toBe(originalDatapointId);

  // Step 4: Navigate back to the original (now stale) datapoint
  await page.goto(`/datasets/${datasetName}/datapoint/${originalDatapointId}`);
  await page.waitForLoadState("networkidle");

  // Step 5: Verify stale behavior
  // 5a. Page loads successfully (not 404)
  await expect(page.getByText("Datapoint", { exact: true })).toBeVisible();
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();

  // 5b. "Stale" badge is visible in the header
  const staleBadge = page.getByText("Stale", { exact: true });
  await expect(staleBadge).toBeVisible();

  // 5c. Verify badge has help cursor (via CSS class)
  await expect(staleBadge).toHaveClass(/cursor-help/);

  // 5d. Hover over badge to show tooltip
  await staleBadge.hover();
  await page.waitForTimeout(300); // Wait for tooltip delay (200ms) + buffer

  // Verify tooltip content
  await expect(
    page.getByText("This datapoint has since been edited or deleted."),
  ).toBeVisible();

  // 5e. Edit button is disabled
  const editButton = page.getByRole("button", { name: "Edit" });
  await expect(editButton).toBeDisabled();

  // 5f. Delete button is disabled
  const deleteButton = page.locator("button").filter({
    has: page.locator('svg[class*="lucide-trash"]'),
  });
  await expect(deleteButton).toBeDisabled();
});

test("should show stale badge in evaluation page basic info", async ({
  page,
}) => {
  // Use an existing stale datapoint (marked as stale in fixtures)
  // Navigate to evaluation page with a stale datapoint
  // Datapoint 01939a16-b258-71e1-a467-183001c1952c is marked as stale in the fixtures
  await page.goto(
    "/evaluations/entity_extraction/01939a16-b258-71e1-a467-183001c1952c?evaluation_run_ids=0196368f-19bd-7082-a677-1c0bf346ff24",
  );
  await page.waitForLoadState("networkidle");

  // Verify page loads successfully
  await expect(page.getByRole("heading", { name: "Input" })).toBeVisible();

  // Verify stale badge appears in Basic Info section
  // The badge should appear next to the Datapoint ID in the Basic Info
  const staleBadge = page.getByText("Stale", { exact: true });
  await expect(staleBadge).toBeVisible();

  // Verify badge has help cursor
  await expect(staleBadge).toHaveClass(/cursor-help/);

  // Hover to show tooltip
  await staleBadge.hover();
  await page.waitForTimeout(300); // Wait for 200ms delay + buffer

  // Verify tooltip content
  await expect(
    page.getByText("This datapoint has since been edited or deleted."),
  ).toBeVisible();
});
