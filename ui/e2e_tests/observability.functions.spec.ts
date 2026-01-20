import { test, expect } from "@playwright/test";

test("should filter internal functions by default and toggle visibility", async ({
  page,
}) => {
  await page.goto("/observability/functions");

  const table = page.getByRole("table");
  await expect(table.getByText("Variants")).toBeVisible();

  const countDisplay = page.getByTestId("count-display").first();
  await expect(countDisplay).toBeVisible();
  const getCount = async () =>
    Number((await countDisplay.textContent())?.replace(/,/g, "").trim());

  const initialCount = await getCount();

  const showInternalToggle = page.getByLabel("Show internal functions");
  await expect(showInternalToggle).not.toBeChecked();

  const internalRows = table.locator("tbody tr").filter({
    hasText: "tensorzero::",
  });

  await expect(internalRows).toHaveCount(0);

  await showInternalToggle.check();

  await expect(internalRows).not.toHaveCount(0);

  const toggledCount = await getCount();
  expect(toggledCount).toBeGreaterThan(initialCount);

  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
