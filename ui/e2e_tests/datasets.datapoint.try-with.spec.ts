import { test, expect } from "@playwright/test";
import { createDatapointFromInference } from "./helpers/datapoint-helpers";

test.describe("Try with on datapoint page", () => {
  const testData = [
    {
      buttonText: "Try with variant",
      inferenceId: "0196374b-0d7d-7422-b6dc-e94c572cc79b", // write_haiku function
      option: "initial_prompt_gpt4o_mini",
    },
    {
      buttonText: "Try with model",
      inferenceId: "019926fd-1a06-7fe2-b7f4-2318de2f2046", // default function
      option: "gpt-4o-mini-2024-07-18",
    },
  ];

  testData.forEach(({ buttonText, inferenceId, option }) => {
    test(buttonText, async ({ page }) => {
      // Create a datapoint from the inference
      const datasetName = await createDatapointFromInference(page, {
        inferenceId,
      });

      // Verify we're on the datapoint page
      await expect(page.getByText("Datapoint", { exact: true })).toBeVisible();

      // Click on the "Try with variant/model" button
      const button = page.getByText(buttonText);
      await button.waitFor({ state: "visible" });
      await expect(button).toBeEnabled();
      await button.click();

      // Wait for the dropdown menu to appear and select an option
      const menuOption = page.getByRole("menuitem").filter({
        has: page.locator(`text="${option}"`),
      });

      await menuOption.waitFor({ state: "visible" });
      await menuOption.click();

      // Wait for the response modal to open and show results
      await page.getByRole("dialog").waitFor({ state: "visible" });

      // Wait for the inference to complete - look for "Inference ID:" which indicates success
      const inferenceIdText = page
        .getByRole("dialog")
        .getByText(/Inference ID:/);
      await inferenceIdText.waitFor({ state: "visible", timeout: 15000 });

      // Verify the dialog is showing a successful response
      await expect(page.getByRole("dialog")).toBeVisible();

      // Close the modal
      await page.keyboard.press("Escape");

      // Clean up: delete the dataset
      await page.goto("/datasets");
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

      // Wait for the dialog to appear and click the Delete button
      const dialog = page.getByRole("alertdialog");
      await dialog.waitFor({ state: "visible" });
      await dialog.getByRole("button", { name: /Delete/ }).click();

      // Wait for the deletion to complete
      await page.waitForTimeout(1000);

      // Assert that the dataset is no longer visible
      await expect(
        page.locator("tr").filter({ hasText: datasetName }),
      ).not.toBeVisible();
    });
  });
});
