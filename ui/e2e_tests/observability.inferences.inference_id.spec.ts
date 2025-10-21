import { test, expect } from "@playwright/test";
import { createDatapointFromInference } from "./helpers/datapoint-helpers";

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
    .locator("span")
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

test.describe("should be able to add demonstration feedback via Try with X flows", () => {
  const testData = [
    {
      buttonText: "Try with variant",
      inference: "0196374b-0d7d-7422-b6dc-e94c572cc79b",
      option: "initial_prompt_gpt4o_mini",
    },
    {
      buttonText: "Try with model",
      inference: "019926fd-1a06-7fe2-b7f4-2318de2f2046",
      // NOTE(bret): This option was chosen because it came
      // from `getUsedVariants('tensorzero::default')`
      option: "openai::gpt-4o-mini",
    },
  ];
  testData.forEach(({ buttonText, inference, option }) => {
    test(buttonText, async ({ page }) => {
      await page.goto(`/observability/inferences/${inference}`);

      // Wait for the page to load
      await page.waitForLoadState("networkidle");

      // Click on "Try with variant" button
      await page.getByText(buttonText).click();

      // Wait for the dropdown menu to appear and select a variant
      // Look for variant options and click on one that's not the current variant
      const variantOption = page.getByRole("menuitem").filter({
        // NOTE(bret): Get exact match, in the case that the
        // option is also a substring of another option
        has: page.locator(`text="${option}"`),
      });

      await variantOption.waitFor({ state: "visible" });
      await variantOption.click();

      // Wait for the variant response modal to open and show results
      await page.getByRole("dialog").waitFor({ state: "visible" });

      // Wait for the variant inference to complete and show the "Add as Demonstration" button
      const demoButton = page.getByText("Add as Demonstration");
      await demoButton.waitFor({ state: "visible", timeout: 15000 });

      // Click the "Add as Demonstration" button
      await demoButton.click();

      // Wait for the modal to close and the page to redirect with newFeedbackId
      await page.waitForURL((url) => url.searchParams.has("newFeedbackId"), {
        timeout: 10000,
      });

      // Verify the modal is closed (no longer visible)
      await expect(page.getByRole("dialog")).not.toBeVisible();

      // Verify the toast appears (look for "Feedback Added" text)
      await expect(
        page
          .getByRole("region", { name: /notifications/i })
          .getByText("Feedback Added"),
      ).toBeVisible();

      // Get the feedback ID from URL and verify it appears in the feedback table
      const newFeedbackId = new URL(page.url()).searchParams.get(
        "newFeedbackId",
      );
      if (!newFeedbackId) {
        throw new Error("newFeedbackId is not present in the url");
      }

      // Assert that the feedback ID is visible in the feedback table
      await expect(
        page.getByRole("cell", { name: newFeedbackId }),
      ).toBeVisible();
    });
  });
});

test.describe("should navigate to inference from Try with X modal and verify tags", () => {
  const testData = [
    {
      buttonText: "Try with variant",
      inference: "0196374b-0d7d-7422-b6dc-e94c572cc79b",
      option: "initial_prompt_gpt4o_mini",
    },
    {
      buttonText: "Try with model",
      inference: "019926fd-1a06-7fe2-b7f4-2318de2f2046",
      option: "openai::gpt-4o-mini",
    },
  ];
  testData.forEach(({ buttonText, inference, option }) => {
    test(buttonText, async ({ page }) => {
      await page.goto(`/observability/inferences/${inference}`);

      // Wait for the page to load
      await page.waitForLoadState("networkidle");

      // Click on "Try with variant/model" button
      await page.getByText(buttonText).click();

      // Wait for the dropdown menu to appear and select a variant
      const variantOption = page.getByRole("menuitem").filter({
        has: page.locator(`text="${option}"`),
      });

      await variantOption.waitFor({ state: "visible" });
      await variantOption.click();

      // Wait for the variant response modal to open and show results
      await page.getByRole("dialog").waitFor({ state: "visible" });

      // Wait for the inference to complete and get the inference ID link
      const inferenceLink = page
        .getByRole("dialog")
        .getByText(/Inference ID:/)
        .locator("..")
        .getByRole("link");
      await inferenceLink.waitFor({ state: "visible", timeout: 5000 });

      // Click the inference link to navigate to the inference page
      await inferenceLink.click();

      // Verify the modal is closed after clicking the link
      await expect(page.getByRole("dialog")).not.toBeVisible();

      // Wait for navigation to the new inference page
      await page.waitForURL("/observability/inferences/**", {
        timeout: 5000,
      });

      // Verify we're on an inference detail page
      await expect(page.getByText("Inference", { exact: true })).toBeVisible();

      // Verify the tags section shows our UI tags
      await expect(page.getByText("tensorzero::internal")).toBeVisible();
      await expect(page.getByText("tensorzero::ui")).toBeVisible();

      // Verify the tag values are "true"
      const internalTagValue = page
        .locator("tr")
        .filter({ hasText: "tensorzero::internal" })
        .getByText("true");
      await expect(internalTagValue).toBeVisible();

      const uiTagValue = page
        .locator("tr")
        .filter({ hasText: "tensorzero::ui" })
        .getByText("true");
      await expect(uiTagValue).toBeVisible();
    });
  });
});

test("should be able to add a datapoint from the inference page", async ({
  page,
}) => {
  // NOTE: this datapoint has auxiliary_content as "" so was failing to insert into dataset
  // We want to make sure that we can add it to a dataset now that we've fixed that issue.
  // Create a new datapoint from an inference
  const datasetName = await createDatapointFromInference(page, {
    inferenceId: "0196368f-1ae8-7551-b5df-9a61593eb307", // `extract_entities`
  });

  // Assert that the page URL starts with /datasets/test_json_dataset/datapoint/
  expect(page.url()).toMatch(
    new RegExp(`/datasets/${datasetName}/datapoint/.*`),
  );

  // Verify we can see the datapoint content
  await expect(page.getByText("Datapoint", { exact: true })).toBeVisible();

  // Clean up: delete the dataset by going to the list datasets page
  await page.goto("/datasets");
  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // Find the row containing our dataset
  const datasetRow = page.locator("tr").filter({ hasText: datasetName });

  // Hover over the row to make the delete button visible
  await datasetRow.hover();

  // Click on the delete button (trash icon)
  const deleteButton = datasetRow
    .locator("button")
    .filter({ has: page.locator("svg") });
  await deleteButton.click();

  // Wait for the shadcn dialog to appear and click the Delete button
  const dialog = page.locator('div[role="dialog"]');
  await dialog.waitFor({ state: "visible" });

  // Click the destructive "Delete" button in the dialog
  await dialog.getByRole("button", { name: /Delete/ }).click();

  // Wait for the deletion to complete and page to update
  await page.waitForTimeout(1000);

  // Assert that the dataset name is not in the list of datasets anymore
  await expect(
    page.locator("tr").filter({ hasText: datasetName }),
  ).not.toBeVisible();
});

test("should load an inference page with a tool call", async ({ page }) => {
  await page.goto(
    "/observability/inferences/0196a0ea-7f6e-7960-9087-9002150a46e6",
  );

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  await expect(
    page.getByText("0196a0ea-7f6e-7960-9087-9002150a46e6").first(),
  ).toBeVisible();
  await expect(page.getByText("multi_hop_rag_agent").first()).toBeVisible();
  await expect(page.getByText("Testerian catechisms").first()).toBeVisible();
});

test("should display inferences with thought block output", async ({
  page,
}) => {
  await page.goto(
    "/observability/inferences/019926fd-1a06-7fe2-b7f4-2318de2f2046",
  );

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // assert that there is a text element with specific styling class containing "Thought"
  await expect(
    page.locator(".text-fg-tertiary").getByText("Thought"),
  ).toBeVisible();
});

test("should handle model inference with null input and output tokens", async ({
  page,
}) => {
  await page.goto(
    "/observability/inferences/01954435-76a5-7331-8a3a-16296a0ba5b6",
  );

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // Verify the inference page loads
  await expect(
    page.getByText("01954435-76a5-7331-8a3a-16296a0ba5b6").first(),
  ).toBeVisible();

  // Click on the model inference ID to open the model inference detail sheet
  await page.getByText("01954435-76ab-78b1-a76e-d5676b0dd2f9").click();

  // Wait for the sheet/dialog to appear
  const sheet = page.locator('[role="dialog"]');
  await sheet.waitFor({ state: "visible" });

  // Verify we're on the model inference page (use exact match to avoid matching "Model Inferences")
  await expect(
    sheet.getByText("Model Inference", { exact: true }),
  ).toBeVisible();

  // Verify that null tokens are displayed in the sheet (they should show as "null tok")
  await expect(sheet.getByText("null tok")).toHaveCount(2);

  // Verify the crab description output is visible in the sheet
  await expect(sheet.getByText("cartoon-style crab").first()).toBeVisible();
});
