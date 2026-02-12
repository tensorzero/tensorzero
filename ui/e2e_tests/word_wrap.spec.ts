import { test, expect } from "@playwright/test";

test("ensure word wrap persists between pages", async ({ page }) => {
  await page.goto(
    "/datasets/foo/datapoint/0196374b-d575-77b3-ac22-91806c67745c",
  );

  // Clear localStorage to ensure clean state
  await page.evaluate(() => localStorage.removeItem("word-wrap"));

  // Reload the page to avoid any quirks
  await page.reload();

  const getWordWrapToggle = () =>
    page
      .locator("section")
      .filter({
        has: page.getByRole("heading", { name: "Input", exact: true }),
      })
      .getByTitle("Toggle word wrap")
      .first();
  const getWordWrap = async () =>
    await page.evaluate(() => localStorage.getItem("word-wrap"));
  await expect(getWordWrapToggle()).toBeVisible();

  // expect default value for word wrap to be true
  {
    const button = getWordWrapToggle();
    await expect(button).toBeVisible();
    await expect(button).toHaveAttribute("aria-pressed", "true");
    await expect.poll(getWordWrap).toBe("true");
  }

  // check that it gets updated when button toggle is clicked
  {
    const button = getWordWrapToggle();
    await button.click();
    await expect(button).toHaveAttribute("aria-pressed", "false");
    await expect.poll(getWordWrap).toBe("false");
  }

  // ensure that it is still set to false on page reload...
  {
    await page.reload();
    const button = getWordWrapToggle();
    await expect(button).toBeVisible();
    await expect(button).toHaveAttribute("aria-pressed", "false");
    await expect.poll(getWordWrap).toBe("false");
  }
});
