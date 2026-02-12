import { test, expect } from "@playwright/test";

test("ensure word wrap persists between pages", async ({ page }) => {
  await page.goto(
    "/datasets/foo/datapoint/0196374b-d575-77b3-ac22-91806c67745c",
  );

  // Clear localStorage to ensure clean state
  await page.evaluate(() => localStorage.removeItem("word-wrap"));

  // Reload the page to avoid any quirks
  await page.reload();

  const getWordWrapToggle = () => page.getByTitle("Toggle word wrap").first();
  const getWordWrap = async () =>
    await page.evaluate(() => localStorage.getItem("word-wrap"));
  await expect(getWordWrapToggle()).toBeVisible();

  // expect default value for word wrap to be true
  {
    const button = getWordWrapToggle();
    await expect(button).toBeVisible();
    expect(await button.getAttribute("aria-pressed")).toBe("true");

    // Wait for localStorage to be set by useEffect
    await page.waitForFunction(
      () => localStorage.getItem("word-wrap") !== null,
      { timeout: 5000 },
    );
    expect(await getWordWrap()).toEqual("true");
  }

  // check that it gets updated when button toggle is clicked
  {
    const button = getWordWrapToggle();
    await button.click();
    expect(await button.getAttribute("aria-pressed")).toBe("false");
    expect(await getWordWrap()).toEqual("false");
  }

  // ensure that it is still set to false on page reload...
  {
    await page.reload();
    const button = getWordWrapToggle();
    await expect(button).toBeVisible();
    await expect
      .poll(async () => await button.getAttribute("aria-pressed"))
      .toBe("false");
    await expect.poll(getWordWrap).toBe("false");
  }
});
