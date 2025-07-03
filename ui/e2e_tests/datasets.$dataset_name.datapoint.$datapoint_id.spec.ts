import { test, expect } from "@playwright/test";
import { v7 } from "uuid";

test("should show the datapoint detail page", async ({ page }) => {
  await page.goto(
    "/datasets/foo/datapoint/0196374b-d575-77b3-ac22-91806c67745c",
  );
  await expect(page.getByText("Input")).toBeVisible();

  // Clicking episode ID opens episode page
  await page
    .getByRole("link", { name: "0193da94-231b-72e0-bda1-dfd0f681462d" })
    .click();
  await expect(page).toHaveURL(
    /\/observability\/episodes\/0193da94-231b-72e0-bda1-dfd0f681462d/,
  );
  await page.goBack();

  // Clicking inference ID opens inference page
  await page
    .getByRole("link", { name: "019480f9-d420-73b1-9619-81d71adc18a5" })
    .click();
  await expect(page).toHaveURL(
    /\/observability\/inferences\/019480f9-d420-73b1-9619-81d71adc18a5/,
  );
  await page.goBack();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should be able to add, edit and save a datapoint", async ({ page }) => {
  await page.goto(
    "/observability/inferences/0197177a-7c00-70a2-82a6-741f60a03b2e",
  );
  await page.waitForLoadState("networkidle");

  await page.getByRole("button", { name: "Add to dataset" }).click();
  await page.getByRole("option", { name: "test_json_dataset" }).click();
  await page.getByRole("button", { name: "Inference Output" }).click();

  await page.waitForTimeout(5000);
  // Should then navigate to the datapoint page
  await expect(page.url()).toContain("/datasets/test_json_dataset/datapoint/");

  await expect(page.getByText("Custom")).not.toBeVisible();

  // Click the edit button
  await page.getByRole("button", { name: "Edit" }).click();

  // Edit the input
  const topic = v7();
  const input = `{"topic":"${topic}"}`;

  await page.locator("textarea.font-mono").first().fill(input);

  // Save the datapoint
  await page.getByRole("button", { name: "Save" }).click();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();

  // Assert that the input is updated
  await expect(page.getByText(input)).toBeVisible();

  // Should show "Custom" badge and link original inference
  await expect(page.getByText("Custom")).toBeVisible();
  await expect(page.getByText("Inference", { exact: true })).toBeVisible();
});
