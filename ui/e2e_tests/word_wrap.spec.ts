import { test, expect } from "@playwright/test";

test("ensure word wrap persists between pages", async ({ page }) => {
  await page.goto(
    "/datasets/foo/datapoint/0196374b-d575-77b3-ac22-91806c67745c",
  );

  await expect(page.getByText("Input")).toBeVisible();

  const getWordWrapToggle = () => page.getByTitle("Toggle word wrap").first();
  const getWordWrap = async () =>
    await page.evaluate(() => localStorage.getItem("word-wrap"));

  // expect default value for word wrap to be true
  {
    const button = getWordWrapToggle();
    expect(await button.getAttribute("aria-pressed")).toBe("true");
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
    expect(await button.getAttribute("aria-pressed")).toBe("false");
    expect(await getWordWrap()).toEqual("false");
  }
});
