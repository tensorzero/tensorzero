import { test, expect } from "@playwright/test";

test("should show the inference detail page", async ({ page }) => {
  const inference_id = "0196367a-842d-74c2-9e62-67e058632503";
  await page.goto(`/observability/inferences/${inference_id}`);
  // The episode ID should be visible
  await expect(
    page.getByText("0196367a-842d-74c2-9e62-67f07369b6ad"),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

// Tests an inference stored under the 'file' content block in the db
test("should display inferences with new image content", async ({ page }) => {
  await page.goto(
    "/observability/inferences/0196fdd6-25f1-72ba-8dc0-be7a0d9df2c5",
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

  // Wait for the page to load
  await page.waitForTimeout(500);

  // Verify that images display in the modelInference section too
  // Click on the modelInference section
  await page.getByText("0196fdd7-5287-78a4-b36b-dcafd5d541fd").click();
  // Wait for 500ms
  await page.waitForTimeout(500);
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

// Tests an inference stored under the 'image' content block in the db
test("should display inferences with old image content", async ({ page }) => {
  await page.goto(
    "/observability/inferences/0196372f-1b4b-7013-a446-511e312a3c30",
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

  // Wait for the page to load
  await page.waitForTimeout(500);

  // Verify that images display in the modelInference section too
  // Click on the modelInference section
  await page.getByText("0196372f-2b63-7ed1-9a5a-9d0fa69c43e9").click();
  // Wait for 500ms
  await page.waitForTimeout(500);
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
    "/observability/inferences/0196368f-1b05-7181-b50c-e2ea0acea312",
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
    "/observability/inferences/0196368f-1ae7-7e21-9027-f120f73d8ce0",
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
    "/datasets/foo/datapoint/01936b20-e838-7322-956f-cd5a5d56f5fa",
  );
});

test("should be able to add float feedback via the inference page", async ({
  page,
}) => {
  await page.goto(
    "/observability/inferences/0196368f-1aeb-7f92-a62b-bdc595d0a626",
  );
  // Wait for the page to load
  await page.waitForLoadState("networkidle");
  // Click on the Add feedback button
  await page.getByText("Add feedback").click();

  // Click "Select a metric"
  await page.getByText("Select a metric").click();

  // Explicitly wait for the item to be visible before clicking
  const metricItemLocator = page
    .locator('div[role="dialog"]')
    .locator('div[cmdk-item=""]')
    .filter({
      hasText: "jaccard_similarity",
    });
  await metricItemLocator.waitFor({ state: "visible" });
  // Click on the metric in the command list
  await metricItemLocator.click();

  await page.locator("body").click();

  // Fill in the value using the correct role and label
  // Generate a random float between 0 and 1, avoiding .225 which seems to occur frequently
  // Use a different approach to generate the random number
  const randomValue = (0.1 + Math.random() * 0.8).toFixed(3);
  const randomFloat = parseFloat(randomValue);

  await page
    .getByRole("spinbutton", { name: "Value" })
    .fill(randomFloat.toString());

  // Click the submit button
  await page.getByText("Submit Feedback").click();

  await page.waitForURL((url) => url.searchParams.has("newFeedbackId"), {
    timeout: 10000,
  });

  // Verify the feedback value is visible in the table cell
  await expect(
    page.getByRole("cell", { name: randomFloat.toString() }),
  ).toBeVisible();
});

test("should be able to add boolean feedback via the inference page", async ({
  page,
}) => {
  await page.goto(
    "/observability/inferences/0196368f-1ae7-7e21-9027-f120f73d8ce0",
  );
  // Wait for the page to load
  await page.waitForLoadState("networkidle");
  // Click on the Add feedback button
  await page.getByText("Add feedback").click();

  // Click "Select a metric"
  await page.getByText("Select a metric").click();

  // Explicitly wait for the item to be visible before clicking
  const metricItemLocator = page
    .locator('div[role="dialog"]')
    .locator('div[cmdk-item=""]')
    .filter({
      hasText: "exact_match",
    });
  await metricItemLocator.waitFor({ state: "visible" });
  // Click on the metric in the command list
  await metricItemLocator.click();

  // Wait for the radio button to be visible
  await page.getByRole("radio", { name: "true" }).waitFor({ state: "visible" });

  // Click the radio button for "true"
  await page.getByRole("radio", { name: "true" }).click();

  // Click the submit button
  await page.getByText("Submit Feedback").click();

  await page.waitForURL((url) => url.searchParams.has("newFeedbackId"), {
    timeout: 10000,
  });

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
    "/observability/inferences/0196368e-5933-7632-814c-2cd498b961de",
  );
  // Wait for the page to load
  await page.waitForLoadState("networkidle");
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
  const json = `{"score": ${randomFloat}}`;
  await dialog.getByRole("textbox").fill(json);

  // Click the submit button
  await page.getByText("Submit Feedback").click();

  await page.waitForURL((url) => url.searchParams.has("newFeedbackId"), {
    timeout: 10000,
  });

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
    "/observability/inferences/0196374b-0d7d-7422-b6dc-e94c572cc79b",
  );
  // Click on the Add feedback button
  await page.getByText("Add feedback").click();
  // Sleep for a little bit to ensure the dialog is open
  await page.waitForTimeout(500);

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

  // Locate the dialog first
  const dialog = page.locator('div[role="dialog"]');
  // Locate the textbox within the dialog and fill it
  await dialog.getByRole("textbox").waitFor({ state: "visible" });
  // This doesn't have to be a JSON, it can be any string
  const demonstration = "hop on pop";
  await dialog.getByRole("textbox").fill(demonstration);

  // Click the submit button
  await page.getByText("Submit Feedback").click();

  await page.waitForURL((url) => url.searchParams.has("newFeedbackId"), {
    timeout: 10000,
  });

  // Get the search param `newFeedbackId` from the url
  const newFeedbackId = new URL(page.url()).searchParams.get("newFeedbackId");
  if (!newFeedbackId) {
    throw new Error("newFeedbackId is not present in the url");
  }
  // Assert that the feedback value is visible in its table cell
  await expect(page.getByRole("cell", { name: newFeedbackId })).toBeVisible();
});
