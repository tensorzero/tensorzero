import { test, expect } from "@playwright/test";

test.describe("Inference Preview Sheet from Episode Page", () => {
  const episodeId = "0196367a-842d-74c2-9e62-67f07369b6ad";
  const episodeUrl = `/observability/episodes/${episodeId}`;

  test("should open the inference preview sheet and load data", async ({
    page,
  }) => {
    await page.goto(episodeUrl);

    // Wait for the page to load
    await page.waitForLoadState("networkidle");

    // Find and click the preview button (eye icon) for the first inference
    const previewButton = page
      .getByRole("button", { name: "View inference details" })
      .first();
    await previewButton.waitFor({ state: "visible" });
    await previewButton.click();

    // Wait for the sheet to open
    const sheet = page.locator('[role="dialog"]');
    await sheet.waitFor({ state: "visible" });

    // Verify the sheet header shows "Inference" with a link
    await expect(sheet.getByText("Inference")).toBeVisible();

    // Wait for the inference data to load (BasicInfo should show the function name)
    await expect(
      sheet.getByText("tensorzero::llm_judge::haiku::topic_starts_with_f"),
    ).toBeVisible({ timeout: 10000 });

    // Verify key sections are present (use headings to be more specific)
    await expect(sheet.getByRole("heading", { name: "Input" })).toBeVisible();
    await expect(sheet.getByRole("heading", { name: "Output" })).toBeVisible();
    await expect(
      sheet.getByRole("heading", { name: "Feedback" }),
    ).toBeVisible();
    await expect(sheet.getByRole("heading", { name: "Tags" })).toBeVisible();
    await expect(
      sheet.getByRole("heading", { name: "Model Inferences" }),
    ).toBeVisible();
  });

  test("should close the sheet when clicking outside or pressing escape", async ({
    page,
  }) => {
    await page.goto(episodeUrl);
    await page.waitForLoadState("networkidle");

    // Open the preview sheet
    const previewButton = page
      .getByRole("button", { name: "View inference details" })
      .first();
    await previewButton.click();

    // Wait for sheet to open
    const sheet = page.locator('[role="dialog"]');
    await sheet.waitFor({ state: "visible" });

    // Press Escape to close
    await page.keyboard.press("Escape");

    // Verify sheet is closed
    await expect(sheet).not.toBeVisible();
  });

  test("should navigate to full inference page from sheet link", async ({
    page,
  }) => {
    await page.goto(episodeUrl);
    await page.waitForLoadState("networkidle");

    // Open the preview sheet
    const previewButton = page
      .getByRole("button", { name: "View inference details" })
      .first();
    await previewButton.click();

    // Wait for sheet to open and data to load
    const sheet = page.locator('[role="dialog"]');
    await sheet.waitFor({ state: "visible" });

    // Wait for the inference link to appear in the header
    const inferenceLink = sheet.locator(
      "a[href^='/observability/inferences/']",
    );
    await inferenceLink.waitFor({ state: "visible", timeout: 10000 });

    // Get the href to verify navigation later
    const href = await inferenceLink.getAttribute("href");

    // Click the link
    await inferenceLink.click();

    // Verify navigation to the inference page
    await page.waitForURL(href!, { timeout: 5000 });
    await expect(page.getByText("Inference", { exact: true })).toBeVisible();
  });

  test("should switch between different inferences in the sheet", async ({
    page,
  }) => {
    // Use an episode with multiple inferences
    const multiInferenceEpisodeUrl =
      "/observability/episodes/0aaedb7a-d19b-74f9-99fe-ebf8c288abaf";
    await page.goto(multiInferenceEpisodeUrl);
    await page.waitForLoadState("networkidle");

    // Get the inference IDs from the table rows to know what to expect
    const inferenceLinks = page.locator(
      "table a[href^='/observability/inferences/']",
    );
    const firstExpectedId = await inferenceLinks.nth(0).textContent();
    const secondExpectedId = await inferenceLinks.nth(1).textContent();

    // Get the preview buttons
    const previewButtons = page.getByRole("button", {
      name: "View inference details",
    });

    // Open the first inference
    await previewButtons.nth(0).click();

    const sheet = page.locator('[role="dialog"]');
    await sheet.waitFor({ state: "visible" });

    // Wait for the sheet to show the first inference ID
    const sheetInferenceLink = sheet.locator(
      "a[href^='/observability/inferences/']",
    );
    await expect(sheetInferenceLink).toHaveText(firstExpectedId!, {
      timeout: 10000,
    });
    const firstInferenceId = await sheetInferenceLink.textContent();

    // Close the sheet
    await page.keyboard.press("Escape");
    await expect(sheet).not.toBeVisible();

    // Open the second inference
    await previewButtons.nth(1).click();
    await sheet.waitFor({ state: "visible" });

    // Wait for the sheet to show the SECOND inference ID (different from the first)
    await expect(sheetInferenceLink).toHaveText(secondExpectedId!, {
      timeout: 10000,
    });
    const secondInferenceId = await sheetInferenceLink.textContent();

    // Verify the inference IDs are different
    expect(secondInferenceId).not.toBe(firstInferenceId);
  });

  test("should show action buttons in the sheet", async ({ page }) => {
    await page.goto(episodeUrl);
    await page.waitForLoadState("networkidle");

    // Open the preview sheet
    const previewButton = page
      .getByRole("button", { name: "View inference details" })
      .first();
    await previewButton.click();

    // Wait for sheet to open and data to load
    const sheet = page.locator('[role="dialog"]');
    await sheet.waitFor({ state: "visible" });

    // Wait for action buttons to appear
    await expect(sheet.getByRole("button", { name: /Try with/i })).toBeVisible({
      timeout: 10000,
    });
    // Add to dataset is a combobox button with "Add to dataset" text
    await expect(sheet.getByRole("combobox")).toBeVisible();
    await expect(sheet.getByText("Add to dataset")).toBeVisible();
    await expect(
      sheet.getByRole("button", { name: /Add feedback/i }),
    ).toBeVisible();
  });

  test("should be able to add feedback from the preview sheet", async ({
    page,
  }) => {
    // Use an episode that has inferences we can add feedback to
    await page.goto(
      "/observability/episodes/01963691-b93a-7973-bcf8-9688cc02a491",
    );
    await page.waitForLoadState("networkidle");

    // Open the preview sheet
    const previewButton = page
      .getByRole("button", { name: "View inference details" })
      .first();
    await previewButton.click();

    // Wait for sheet to open and data to load
    const sheet = page.locator('[role="dialog"]');
    await sheet.waitFor({ state: "visible" });

    // Wait for Add feedback button to be available
    const addFeedbackButton = sheet.getByRole("button", {
      name: /Add feedback/i,
    });
    await addFeedbackButton.waitFor({ state: "visible", timeout: 10000 });
    await addFeedbackButton.click();

    // Wait for the feedback modal to open (nested dialog)
    // The feedback modal has a specific title we can use to identify it
    const feedbackDialog = page.getByRole("dialog", { name: /add feedback/i });
    await feedbackDialog.waitFor({ state: "visible" });

    // Select the comment metric
    await feedbackDialog.getByRole("combobox", { name: "Metric" }).click();

    const metricItemLocator = page
      .locator('div[cmdk-item=""]')
      .filter({ hasText: "comment" });
    await metricItemLocator.waitFor({ state: "visible" });
    await metricItemLocator.click();

    // Fill in the comment value
    const randomValue = `Test comment from sheet ${Math.random().toString(36).substring(7)}`;
    await feedbackDialog.getByRole("textbox").fill(randomValue);

    // Submit the feedback
    await feedbackDialog.getByText("Submit Feedback").click();

    // Wait for the feedback modal to close (check the dialog with "Add feedback" title is gone)
    await expect(feedbackDialog).not.toBeVisible({ timeout: 10000 });

    // Verify the feedback appears in the sheet's feedback table
    // This confirms the feedback was added successfully and the sheet reloaded with new data
    await expect(sheet.getByText(randomValue)).toBeVisible({ timeout: 15000 });
  });
});
