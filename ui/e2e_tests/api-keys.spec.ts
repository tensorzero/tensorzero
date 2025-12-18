import { test, expect } from "@playwright/test";
import { v7 as uuidv7 } from "uuid";

test("should create, display, delete, and persist API key states", async ({
  page,
}) => {
  // 1. Navigate to API Keys page
  await page.goto("/api-keys");
  await page.waitForLoadState("networkidle");

  // Verify page loaded
  await expect(
    page.getByRole("heading", { name: "TensorZero API Keys" }),
  ).toBeVisible();
  await expect(page.getByText("Public ID")).toBeVisible();

  // 2. Create API key without description
  await page.getByText("Generate API Key").click();
  await expect(
    page.getByRole("heading", { name: "Generate API Key" }),
  ).toBeVisible();

  // Submit without filling description
  const generateButton = page.getByRole("button", { name: "Generate Key" });
  await expect(generateButton).toBeVisible();
  await expect(generateButton).toBeEnabled();
  await page.waitForTimeout(300); // wait for modal animation to complete
  await generateButton.click();

  // Wait for success state
  await expect(
    page.getByRole("heading", { name: "API Key Generated" }),
  ).toBeVisible();

  // Extract the first API key
  const firstApiKeyPre = page.locator("pre");
  const firstApiKeyText = await firstApiKeyPre.textContent();
  expect(firstApiKeyText).toBeTruthy();

  // Extract a unique part of the public ID to identify this key later
  // API keys have format like "sk-t0-sGuiRyyK5H8Z-dE9kKkDGWoUdeSxB3ajakqVHQMWZMpwExBZwEbR3yRioQvUS"
  // Extract the first 12 characters after "sk-t0-" to uniquely identify this key
  const firstPublicIdMatch = firstApiKeyText!.match(/sk-t0-(\w{12})/);
  const firstPublicIdPart = firstPublicIdMatch ? firstPublicIdMatch[1] : null;
  expect(firstPublicIdPart).toBeTruthy();

  // Close modal (use first button which is the primary Close button in footer)
  await page.getByRole("button", { name: "Close" }).first().click();
  await expect(
    page.getByRole("heading", { name: "API Key Generated" }),
  ).not.toBeVisible();

  // Wait for page to reload
  await page.waitForLoadState("networkidle");

  // Verify first key appears in table without description
  const firstKeyRow = page.locator("tbody tr").filter({
    hasText: firstPublicIdPart!,
  });
  await expect(firstKeyRow).toBeVisible();

  // Verify description cell shows "—" (em dash for empty)
  const firstDescriptionCell = firstKeyRow.locator("td").nth(1);
  await expect(firstDescriptionCell).toContainText("—");

  // 3. Create API key with description
  await page.getByText("Generate API Key").click();
  await expect(
    page.getByRole("heading", { name: "Generate API Key" }),
  ).toBeVisible();

  // Fill description with a unique UUID to avoid conflicts
  const description = `Test key ${uuidv7()}`;
  await page.locator("#description").fill(description);

  // Verify character count updates
  await expect(
    page.getByText(`${description.length}/255 characters`),
  ).toBeVisible();

  // Submit form
  const secondGenerateButton = page.getByRole("button", {
    name: "Generate Key",
  });
  await expect(secondGenerateButton).toBeVisible();
  await expect(secondGenerateButton).toBeEnabled();
  await page.waitForTimeout(300); // wait for modal animation to complete
  await secondGenerateButton.click();

  // Wait for success state
  await expect(
    page.getByRole("heading", { name: "API Key Generated" }),
  ).toBeVisible();

  // Verify description is shown in success state within the dialog
  const dialog = page.locator('[role="dialog"]');
  await expect(dialog.getByText(description)).toBeVisible();

  // Extract the second API key
  const secondApiKeyPre = page.locator("pre");
  const secondApiKeyText = await secondApiKeyPre.textContent();
  expect(secondApiKeyText).toBeTruthy();

  // Extract unique part of second public ID (first 12 chars after "sk-t0-")
  const secondPublicIdMatch = secondApiKeyText!.match(/sk-t0-(\w{12})/);
  const secondPublicIdPart = secondPublicIdMatch
    ? secondPublicIdMatch[1]
    : null;
  expect(secondPublicIdPart).toBeTruthy();

  // Ensure we got two different keys
  expect(firstPublicIdPart).not.toBe(secondPublicIdPart);

  // Close modal (use first button which is the primary Close button in footer)
  await page.getByRole("button", { name: "Close" }).first().click();
  await expect(
    page.getByRole("heading", { name: "API Key Generated" }),
  ).not.toBeVisible();

  // Wait for page to reload
  await page.waitForLoadState("networkidle");

  // Verify second key appears in table with description
  const secondKeyRow = page.locator("tbody tr").filter({
    hasText: secondPublicIdPart!,
  });
  await expect(secondKeyRow).toBeVisible();

  // Verify description is shown
  const secondDescriptionCell = secondKeyRow.locator("td").nth(1);
  await expect(secondDescriptionCell).toContainText(description);

  // 4. Delete the first key (without description)
  const firstKeyRowToDelete = page.locator("tbody tr").filter({
    hasText: firstPublicIdPart!,
  });
  await expect(firstKeyRowToDelete).toBeVisible();

  // Click delete button (trash icon in last column)
  const deleteButton = firstKeyRowToDelete
    .locator("td")
    .last()
    .locator("button")
    .last();
  await expect(deleteButton).not.toBeDisabled(); // Should be enabled before deletion
  await deleteButton.click();

  // Verify confirmation dialog appears
  await expect(
    page.getByText("Are you sure you want to disable the API key"),
  ).toBeVisible();

  // Verify the public ID is shown in the dialog (should contain our identifying part)
  const dialogContent = page.locator("[role='dialog']");
  await expect(dialogContent).toContainText(firstPublicIdPart!);

  // Click "Disable" button
  await page.getByRole("button", { name: "Disable" }).click();

  // Wait for dialog to close
  await expect(
    page.getByText("Are you sure you want to disable the API key"),
  ).not.toBeVisible();

  // Wait for page to reload after deletion
  await page.waitForLoadState("networkidle");

  // Verify first key is now disabled
  const disabledKeyRow = page.locator("tbody tr").filter({
    hasText: firstPublicIdPart!,
  });
  await expect(disabledKeyRow).toBeVisible();
  await expect(disabledKeyRow).toHaveClass(/opacity-50/);

  // Verify public ID has line-through
  const disabledPublicIdCode = disabledKeyRow.locator("code");
  await expect(disabledPublicIdCode).toHaveClass(/line-through/);

  // Verify delete button is now disabled
  const disabledDeleteButton = disabledKeyRow
    .locator("td")
    .last()
    .locator("button")
    .last();
  await expect(disabledDeleteButton).toBeDisabled();

  // Verify second key is still active
  const activeKeyRow = page.locator("tbody tr").filter({
    hasText: secondPublicIdPart!,
  });
  await expect(activeKeyRow).toBeVisible();
  await expect(activeKeyRow).not.toHaveClass(/opacity-50/);

  const activeDeleteButton = activeKeyRow
    .locator("td")
    .last()
    .locator("button")
    .last();
  await expect(activeDeleteButton).not.toBeDisabled();

  // 5. Refresh the page
  await page.reload();
  await page.waitForLoadState("networkidle");

  // Verify page loaded after refresh
  await expect(
    page.getByRole("heading", { name: "TensorZero API Keys" }),
  ).toBeVisible();

  // 6. Verify final state persists after refresh
  // First key should still be disabled
  const refreshedDisabledRow = page.locator("tbody tr").filter({
    hasText: firstPublicIdPart!,
  });
  await expect(refreshedDisabledRow).toBeVisible();
  await expect(refreshedDisabledRow).toHaveClass(/opacity-50/);
  await expect(refreshedDisabledRow.locator("code")).toHaveClass(
    /line-through/,
  );

  const refreshedDisabledButton = refreshedDisabledRow
    .locator("td")
    .last()
    .locator("button")
    .last();
  await expect(refreshedDisabledButton).toBeDisabled();

  // Second key should still be active
  const refreshedActiveRow = page.locator("tbody tr").filter({
    hasText: secondPublicIdPart!,
  });
  await expect(refreshedActiveRow).toBeVisible();
  await expect(refreshedActiveRow).not.toHaveClass(/opacity-50/);
  await expect(refreshedActiveRow.locator("td").nth(1)).toContainText(
    description,
  );

  const refreshedActiveButton = refreshedActiveRow
    .locator("td")
    .last()
    .locator("button")
    .last();
  await expect(refreshedActiveButton).not.toBeDisabled();
});

test("should edit API key descriptions and persist changes", async ({
  page,
}) => {
  await page.goto("/api-keys");
  await page.waitForLoadState("networkidle");

  await expect(
    page.getByRole("heading", { name: "TensorZero API Keys" }),
  ).toBeVisible();

  await page.getByRole("button", { name: "Generate API Key" }).click();
  await expect(
    page.getByRole("heading", { name: "Generate API Key" }),
  ).toBeVisible();

  const originalDescription = `Editable key ${uuidv7()}`;
  const descriptionInput = page.locator("#description");
  await descriptionInput.fill(originalDescription);

  const generateButton = page.getByRole("button", { name: "Generate Key" });
  await expect(generateButton).toBeVisible();
  await expect(generateButton).toBeEnabled();
  await page.waitForTimeout(300);
  await generateButton.click();

  const successDialog = page.getByRole("dialog", {
    name: "API Key Generated",
  });
  await expect(successDialog).toBeVisible();
  const generatedKeyText = await successDialog.locator("pre").textContent();
  expect(generatedKeyText).toBeTruthy();

  const publicIdMatch = generatedKeyText!.match(/sk-t0-(\w{12})/);
  const publicIdPart = publicIdMatch ? publicIdMatch[1] : null;
  expect(publicIdPart).toBeTruthy();

  await successDialog.getByRole("button", { name: "Close" }).first().click();
  await page.waitForLoadState("networkidle");

  const apiKeyRow = page.locator("tbody tr").filter({
    hasText: publicIdPart!,
  });
  await expect(apiKeyRow).toBeVisible();
  await expect(apiKeyRow.locator("td").nth(1)).toContainText(
    originalDescription,
  );

  const editButton = apiKeyRow.locator("button").first();
  await expect(editButton).toBeVisible();
  await editButton.click();

  const editDialog = page.getByRole("dialog", {
    name: "Edit API key description",
  });
  await expect(editDialog).toBeVisible();

  const updatedDescription = `Updated description ${uuidv7()}`;
  const editInput = editDialog.getByPlaceholder("Optional description");
  await editInput.fill("");
  await editInput.fill(updatedDescription);

  const saveButton = editDialog.getByRole("button", { name: "Save" });
  await expect(saveButton).toBeEnabled();
  await saveButton.click();

  await expect(editDialog).not.toBeVisible();
  await page.waitForLoadState("networkidle");

  const updatedRow = page.locator("tbody tr").filter({
    hasText: publicIdPart!,
  });
  await expect(updatedRow.locator("td").nth(1)).toContainText(
    updatedDescription,
  );

  await page.reload();
  await page.waitForLoadState("networkidle");

  const refreshedRow = page.locator("tbody tr").filter({
    hasText: publicIdPart!,
  });
  await expect(refreshedRow).toBeVisible();
  await expect(refreshedRow.locator("td").nth(1)).toContainText(
    updatedDescription,
  );
});

test("should paginate API keys with custom limit", async ({ page }) => {
  // 1. Navigate to API Keys page with small limit
  await page.goto("/api-keys?limit=2");
  await page.waitForLoadState("networkidle");

  // Verify page loaded
  await expect(
    page.getByRole("heading", { name: "TensorZero API Keys" }),
  ).toBeVisible();

  // 2. Create 5 API keys (more than limit)
  for (let i = 0; i < 5; i++) {
    await page.getByRole("button", { name: "Generate API Key" }).click();
    await expect(
      page.getByRole("heading", { name: "Generate API Key" }),
    ).toBeVisible();
    await page.waitForLoadState("networkidle");

    await page.locator("#description").waitFor({ state: "visible" });
    await page.locator("#description").fill(`Pagination test ${uuidv7()}`);

    const generateButton = page.getByRole("button", { name: "Generate Key" });
    await expect(generateButton).toBeVisible();
    await expect(generateButton).toBeEnabled();
    await generateButton.click();
    await page.waitForLoadState("networkidle");

    await expect(
      page.getByRole("heading", { name: "API Key Generated" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Close" }).first().click();
    await page.waitForLoadState("networkidle");
  }

  // 3. Verify only 2 keys are visible on first page (limit=2)
  const tableRows = page.locator("tbody tr");
  await expect(tableRows).toHaveCount(2);

  // 4. Get pagination buttons
  const paginationContainer = page.locator(
    "div.mt-4.flex.items-center.justify-center.gap-2",
  );
  const prevButton = paginationContainer.locator("button").first();
  const nextButton = paginationContainer.locator("button").last();

  // 5. Verify initial pagination state (first page)
  await expect(prevButton).toBeDisabled(); // Can't go back from first page
  await expect(nextButton).not.toBeDisabled(); // Can go forward

  // 6. Click Next button to go to page 2
  await nextButton.click();

  // Wait for URL to change with offset parameter
  await page.waitForURL("**/api-keys?*offset=2*");
  await page.waitForLoadState("networkidle");

  // 7. Verify URL changed (offset should be 2)
  expect(page.url()).toContain("offset=2");
  expect(page.url()).toContain("limit=2");

  // 8. Verify still showing 2 rows on page 2
  await expect(tableRows).toHaveCount(2);

  // 9. Verify both buttons are now enabled (middle page)
  await expect(prevButton).not.toBeDisabled(); // Can go back
  await expect(nextButton).not.toBeDisabled(); // Can go forward

  // 10. Click Previous button to go back to page 1
  await prevButton.click();

  // Wait for URL to change back to offset=0
  await page.waitForURL("**/api-keys?*offset=0*");
  await page.waitForLoadState("networkidle");

  // 11. Verify we're back to first page (offset=0)
  expect(page.url()).toContain("offset=0");
  await expect(prevButton).toBeDisabled(); // Can't go back
  await expect(nextButton).not.toBeDisabled(); // Can go forward
});
