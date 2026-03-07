import { test, expect } from "@playwright/test";

// A fake snapshot hash that won't match the current config
const FAKE_SNAPSHOT_HASH = "abc123def456";

// The current config hash from the gateway (decimal string).
// When this exact value is passed as ?snapshot_hash, the server strips it via redirect.
const CURRENT_CONFIG_HASH =
  "46014305430377123138237384229499364238047438285692198569022910911979009324786";

test("function page with snapshot_hash shows banner", async ({ page }) => {
  await page.goto(
    `/observability/functions/write_haiku?snapshot_hash=${FAKE_SNAPSHOT_HASH}`,
  );
  await expect(
    page.getByText("Viewing historical configuration"),
  ).toBeVisible();
  // snapshot_hash should remain in the URL
  expect(page.url()).toContain(`snapshot_hash=${FAKE_SNAPSHOT_HASH}`);
});

test("function page without snapshot_hash shows no banner", async ({
  page,
}) => {
  await page.goto("/observability/functions/write_haiku");
  await expect(page.getByRole("heading", { name: "Variants" })).toBeVisible();
  await expect(
    page.getByText("Viewing historical configuration"),
  ).not.toBeVisible();
});

test("variant page with snapshot_hash shows banner and preserves hash in breadcrumb", async ({
  page,
}) => {
  await page.goto(
    `/observability/functions/write_haiku/variants/initial_prompt_gpt4o_mini?snapshot_hash=${FAKE_SNAPSHOT_HASH}`,
  );
  await expect(
    page.getByText("Viewing historical configuration"),
  ).toBeVisible();

  // The breadcrumb link back to the function should include the snapshot_hash
  const breadcrumbNav = page.getByRole("navigation", { name: "breadcrumb" });
  const functionLink = breadcrumbNav.getByRole("link", {
    name: "write_haiku",
  });
  await expect(functionLink).toBeVisible();
  const href = await functionLink.getAttribute("href");
  expect(href).toContain(`snapshot_hash=${FAKE_SNAPSHOT_HASH}`);
});

test("snapshot_hash matching current config is stripped via redirect", async ({
  page,
}) => {
  await page.goto(
    `/observability/functions/write_haiku?snapshot_hash=${CURRENT_CONFIG_HASH}`,
  );
  // Should redirect and strip the param — no banner
  await expect(page.getByRole("heading", { name: "Variants" })).toBeVisible();
  await expect(
    page.getByText("Viewing historical configuration"),
  ).not.toBeVisible();
  expect(page.url()).not.toContain("snapshot_hash");
});
