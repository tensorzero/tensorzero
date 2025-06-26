import { test, expect } from "@playwright/test";
import { v7 } from "uuid";

test("should show the datapoint detail page", async ({ page }) => {
  await page.goto(
    "/datasets/foo/datapoint/0193930b-6da0-7fa2-be87-9603d2bde664",
  );
  await expect(page.getByText("Input")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should be able to edit and save a datapoint", async ({ page }) => {
  await page.goto("/datasets/test_json_dataset");
  // Click on the first ID in the first row
  await page.locator("table tbody tr:first-child td:first-child").click();
  // Wait for the page to load
  await page.waitForLoadState("networkidle");
  await expect(page.getByText("Input")).toBeVisible();

  // Click the edit button
  await page.locator("button svg.lucide-pencil").click();

  // Edit the input
  const topic = v7();
  const input = `{"topic":"${topic}"}`;

  await page.locator("textarea.font-mono").first().fill(input);

  // Save the datapoint
  await page.locator("button svg.lucide-save").click();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();

  // Assert that the input is updated
  await expect(page.getByText(input)).toBeVisible();
  // NOTE: there will now be a new datapoint ID for this datapoint as its input has been edited
});

test("should be able to upload files when editing datapoints", async ({ page }) => {
  await page.goto("/datasets/test_json_dataset");
  // Click on the first ID in the first row
  await page.locator("table tbody tr:first-child td:first-child").click();
  // Wait for the page to load
  await page.waitForLoadState("networkidle");
  await expect(page.getByText("Input")).toBeVisible();

  // Click the edit button
  await page.locator("button svg.lucide-pencil").click();

  // Click "Add File" button
  await page.locator("button").filter({ hasText: "+ Add File" }).first().click();

  // Verify that the file upload area is visible
  await expect(page.getByText("Click to upload a file")).toBeVisible();

  // Create a simple test image file
  const testImageData = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdj+O/P8B8ABe0CTsv8mHgAAAAASUVORK5CYII=";
  
  // Set up file input with test data
  await page.evaluate((imageData) => {
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    if (input) {
      // Create a file from the base64 data
      const byteString = atob(imageData.split(',')[1]);
      const ab = new ArrayBuffer(byteString.length);
      const ia = new Uint8Array(ab);
      for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
      }
      const file = new File([ab], 'test.png', { type: 'image/png' });
      
      // Create a DataTransfer object and set the file
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      
      // Trigger change event
      input.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }, testImageData);

  // Wait for the file to be processed
  await page.waitForTimeout(1000);

  // Verify that the file is displayed
  await expect(page.locator("img[src*='data:image/png']")).toBeVisible();

  // Save the datapoint
  await page.locator("button svg.lucide-save").click();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();

  // Verify that the uploaded file is still visible after saving
  await expect(page.locator("img[src*='data:image/png']")).toBeVisible();
});
