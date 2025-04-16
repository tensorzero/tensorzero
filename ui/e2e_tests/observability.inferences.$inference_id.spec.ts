import { test, expect } from "@playwright/test";

test("should show the inference detail page", async ({ page }) => {
  await page.goto(
    "/observability/inferences/0195aef8-3eaa-7dc2-9376-8dde217649e8",
  );
  // The episode ID should be visible
  await expect(
    page.getByText("0195aef8-3eaa-7dc2-9376-8de1d8c6536b"),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should display inferences with image content", async ({ page }) => {
  await page.goto(
    "/observability/inferences/0195e31b-68d6-7001-b7b7-fde770175c65",
  );
  // Assert that there are 2 images displayed and they render
  const images = page.locator("img");
  await expect(images).toHaveCount(2);

  // Verify both images are visible in the viewport
  const firstImage = images.nth(0);
  const secondImage = images.nth(1);
  await expect(firstImage).toBeVisible();
  await expect(secondImage).toBeVisible();

  // Verify images have loaded correctly
  await expect(firstImage).toHaveJSProperty("complete", true);
  await expect(secondImage).toHaveJSProperty("complete", true);

  // Verify that images display in the modelInference section too
  // Click on the modelInference section
  await page.getByText("0195e31b-703c-74a3-bbdd-e252ca10a86d").click();
  // Assert that the images are visible
  const newImages = page.locator("img");
  await expect(newImages).toHaveCount(4);
  const firstNewImage = newImages.nth(0);
  const secondNewImage = newImages.nth(1);
  const thirdNewImage = newImages.nth(2);
  const fourthNewImage = newImages.nth(3);
  await expect(firstNewImage).toBeVisible();
  await expect(secondNewImage).toBeVisible();
  await expect(thirdNewImage).toBeVisible();
  await expect(fourthNewImage).toBeVisible();
});

test("tag navigation works by evaluation_name", async ({ page }) => {
  await page.goto(
    "/observability/inferences/0195f845-949b-76c0-b9d4-68b3fd799b50",
  );

  // Wait for page to load
  await page.waitForLoadState("networkidle");

  // Use a more specific selector for the code element with entity_extraction
  // Look for a table cell containing the exact text "entity_extraction"
  const entityExtractionCell = page
    .locator("code")
    .filter({ hasText: /^entity_extraction$/ })
    .first();

  // Wait for the element to be visible
  await entityExtractionCell.waitFor({ state: "visible" });

  // Click the element
  await entityExtractionCell.click();

  // Assert that the page is /evaluations/entity_extraction
  await expect(page).toHaveURL("/evaluations/entity_extraction");
});

test("tag navigation works by datapoint_id", async ({ page }) => {
  await page.goto(
    "/observability/inferences/0195f845-949b-76c0-b9d4-68b3fd799b50",
  );

  // Wait for page to load completely
  await page.waitForLoadState("networkidle");

  // Use a more specific selector and ensure it's visible before clicking
  const datapointElement = page.getByText("tensorzero::datapoint_id");
  await datapointElement.waitFor({ state: "visible" });

  // Force the click to ensure it happens correctly
  await datapointElement.click({ force: true });

  // Wait for navigation to complete
  await page.waitForURL("**/datasets/foo/datapoint/**");

  // Assert the URL
  await expect(page).toHaveURL(
    "/datasets/foo/datapoint/019368c7-d150-7ba0-819a-88a2cec33663",
  );
});

test("should be able to add float feedback via the inference page", async ({
  page,
}) => {
  await page.goto(
    "/observability/inferences/0195f845-a261-72d2-8686-774b967d938e",
  );
  // Click on the Add feedback button
  await page.getByText("Add feedback").click();

  // Click "Select a metric"
  await page.getByText("Select a metric").click();

  // Explicitly wait for the item to be visible before clicking
  const metricItemLocator = page
    .locator('div[role="dialog"]')
    .locator('div[cmdk-item=""]')
    .filter({
      hasText:
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports",
    });
  await metricItemLocator.waitFor({ state: "visible" });
  // Click on the metric in the command list
  await metricItemLocator.click();

  await page.locator("body").click();

  // Fill in the value using the correct role and label
  // generate a random float between 0 and 1
  const randomFloat = Math.random();
  // Assert that the feedback value is visible in its table cell
  // Truncate the float to 3 decimal places
  const truncatedFloat = Math.floor(randomFloat * 1000) / 1000;
  await page
    .getByRole("spinbutton", { name: "Value" })
    .fill(truncatedFloat.toString());
  // Click the submit button
  await page.getByText("Submit Feedback").click();

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // sleep for 500ms
  await page.waitForTimeout(500);

  await expect(
    page.getByRole("cell", { name: truncatedFloat.toString() }),
  ).toBeVisible();
});

test("should be able to add boolean feedback via the inference page", async ({
  page,
}) => {
  await page.goto(
    "/observability/inferences/0195f845-a261-72d2-8686-774b967d938e",
  );
  // Click on the Add feedback button
  await page.getByText("Add feedback").click();

  // Click "Select a metric"
  await page.getByText("Select a metric").click();

  // Explicitly wait for the item to be visible before clicking
  const metricItemLocator = page
    .locator('div[role="dialog"]')
    .locator('div[cmdk-item=""]')
    .filter({
      hasText:
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match",
    });
  await metricItemLocator.waitFor({ state: "visible" });
  // Click on the metric in the command list
  await metricItemLocator.click();

  // Click the radio button for "true"
  await page.getByRole("radio", { name: "true" }).click();

  // Click the submit button
  await page.getByText("Submit Feedback").click();

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // sleep for 500ms
  await page.waitForTimeout(500);

  // Get the search param `newFeedbackId` from the url
  const newFeedbackId = new URL(page.url()).searchParams.get("newFeedbackId");
  if (!newFeedbackId) {
    throw new Error("newFeedbackId is not present in the url");
  }
  // Assert that the feedback value is visible in its table cell
  await expect(page.getByRole("cell", { name: newFeedbackId })).toBeVisible();
});

test("should be able to add json demonstration feedback via the inference page", async ({
  page,
}) => {
  await page.goto(
    "/observability/inferences/0195f845-a261-72d2-8686-774b967d938e",
  );
  // Click on the Add feedback button
  await page.getByText("Add feedback").click();

  // Click "Select a metric"
  await page.getByText("Select a metric").click();

  // Explicitly wait for the item to be visible before clicking
  const metricItemLocator = page
    .locator('div[role="dialog"]')
    .locator('div[cmdk-item=""]')
    .filter({
      hasText: "demonstration",
    });
  await metricItemLocator.waitFor({ state: "visible" });
  // Click on the metric in the command list
  await metricItemLocator.click();

  // Generate a random float between 0 and 1 with 3 decimal places
  const randomFloat = Math.floor(Math.random() * 1000) / 1000;
  // fill in
  // Locate the dialog first
  const dialog = page.locator('div[role="dialog"]');
  // Locate the textbox within the dialog and fill it
  await dialog.getByRole("textbox").waitFor({ state: "visible" });
  const json = `{"thinking": "hmm", "score": ${randomFloat}}`;
  await dialog.getByRole("textbox").fill(json);

  // Click the submit button
  await page.getByText("Submit Feedback").click();

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // sleep for 1 second
  await page.waitForTimeout(1000);

  // Get the search param `newFeedbackId` from the url
  const newFeedbackId = new URL(page.url()).searchParams.get("newFeedbackId");
  if (!newFeedbackId) {
    throw new Error("newFeedbackId is not present in the url");
  }
  // Assert that the feedback value is visible in its table cell
  await expect(page.getByRole("cell", { name: newFeedbackId })).toBeVisible();
});

test("should be able to add chat demonstration feedback via the inference page", async ({
  page,
}) => {
  await page.goto(
    "/observability/inferences/0195c498-70e3-71f3-bbd9-a2db26b8d349",
  );
  // Click on the Add feedback button
  await page.getByText("Add feedback").click();

  // Click "Select a metric"
  await page.getByText("Select a metric").click();

  // Explicitly wait for the item to be visible before clicking
  const metricItemLocator = page
    .locator('div[role="dialog"]')
    .locator('div[cmdk-item=""]')
    .filter({
      hasText: "demonstration",
    });
  await metricItemLocator.waitFor({ state: "visible" });
  // Click on the metric in the command list
  await metricItemLocator.click();

  // Generate a random float between 0 and 1 with 3 decimal places
  const randomFloat = Math.floor(Math.random() * 1000) / 1000;
  // Locate the dialog first
  const dialog = page.locator('div[role="dialog"]');
  // Locate the textbox within the dialog and fill it
  await dialog.getByRole("textbox").waitFor({ state: "visible" });
  // This doesn't have to be a JSON, it can be any string
  const json = `{"thinking": "hmm", "score": ${randomFloat}}`;
  await dialog.getByRole("textbox").fill(json);

  // Click the submit button
  await page.getByText("Submit Feedback").click();

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // sleep for 1 second
  await page.waitForTimeout(1000);

  // Get the search param `newFeedbackId` from the url
  const newFeedbackId = new URL(page.url()).searchParams.get("newFeedbackId");
  if (!newFeedbackId) {
    throw new Error("newFeedbackId is not present in the url");
  }
  // Assert that the feedback value is visible in its table cell
  await expect(page.getByRole("cell", { name: newFeedbackId })).toBeVisible();
});

test("should be able to add comment feedback via the episode page", async ({
  page,
}) => {
  await page.goto(
    "/observability/episodes/0195f845-a261-72d2-8686-775bc929a221",
  );
  // Click on the Add feedback button
  await page.getByText("Add feedback").click();

  // Click "Select a metric"
  await page.getByText("Select a metric").click();

  // Explicitly wait for the item to be visible before clicking
  const metricItemLocator = page
    .locator('div[role="dialog"]')
    .locator('div[cmdk-item=""]')
    .filter({
      hasText: "comment",
    });
  await metricItemLocator.waitFor({ state: "visible" });
  // Click on the metric in the command list
  await metricItemLocator.click();

  // Generate a random float between 0 and 1 with 3 decimal places
  const randomFloat = Math.floor(Math.random() * 1000) / 1000;
  // Locate the dialog first
  const dialog = page.locator('div[role="dialog"]');
  // Locate the textbox within the dialog and fill it
  await dialog.getByRole("textbox").waitFor({ state: "visible" });
  // This doesn't have to be a JSON, it can be any string
  const json = `{"thinking": "hmm", "score": ${randomFloat}}`;
  await dialog.getByRole("textbox").fill(json);

  // Click the submit button
  await page.getByText("Submit Feedback").click();

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // sleep for 1 second
  await page.waitForTimeout(1000);

  // Get the search param `newFeedbackId` from the url
  const newFeedbackId = new URL(page.url()).searchParams.get("newFeedbackId");
  if (!newFeedbackId) {
    throw new Error("newFeedbackId is not present in the url");
  }
  // Assert that the feedback value is visible in its table cell
  await expect(page.getByRole("cell", { name: newFeedbackId })).toBeVisible();
  // Assert that the comment is visible in the comment section
  await expect(page.getByText(json)).toBeVisible();
});
