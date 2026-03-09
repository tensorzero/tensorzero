import { test, expect, type Page } from "@playwright/test";

const FUNCTION_NAME = "write_haiku";

// A non-matching hash to simulate historical config navigation.
// The server falls back to current config when it can't resolve this, but the
// URL and banner still reflect the "historical" state.
const HISTORICAL_HASH = "abc123historicalhash";

/**
 * Fetches the current config hash from the gateway's /status endpoint.
 * Always uses localhost:3000 because the Playwright test runner runs on the
 * host (not inside Docker), and the gateway container maps port 3000 to the host.
 */
async function getCurrentConfigHash(page: Page): Promise<string> {
  const response = await page.request.get("http://localhost:3000/status");
  const status = await response.json();
  return status.config_hash;
}

test("function page without snapshot_hash shows no banner", async ({
  page,
}) => {
  await page.goto(`/observability/functions/${FUNCTION_NAME}`);
  await expect(page.getByRole("heading", { name: "Variants" })).toBeVisible();
  await expect(
    page.getByText("Viewing historical configuration"),
  ).not.toBeVisible();
  expect(page.url()).not.toContain("snapshot_hash");
});

test("snapshot_hash shows banner and propagates to variant links", async ({
  page,
}) => {
  await page.goto(
    `/observability/functions/${FUNCTION_NAME}?snapshot_hash=${HISTORICAL_HASH}`,
  );
  await expect(page.getByRole("heading", { name: "Variants" })).toBeVisible();

  // Banner should appear
  await expect(
    page.getByText("Viewing historical configuration"),
  ).toBeVisible();
  expect(page.url()).toContain(`snapshot_hash=${HISTORICAL_HASH}`);

  // Variant links should carry the snapshot hash
  const variantLink = page.getByRole("link").filter({ hasText: /prompt/ });
  const firstLink = variantLink.first();
  await expect(firstLink).toBeVisible();
  const href = await firstLink.getAttribute("href");
  expect(href).toContain(`snapshot_hash=${HISTORICAL_HASH}`);
});

test("snapshot_hash matching current config is stripped via redirect", async ({
  page,
}) => {
  const currentHash = await getCurrentConfigHash(page);

  await page.goto(
    `/observability/functions/${FUNCTION_NAME}?snapshot_hash=${currentHash}`,
  );
  await expect(page.getByRole("heading", { name: "Variants" })).toBeVisible();
  await expect(
    page.getByText("Viewing historical configuration"),
  ).not.toBeVisible();
  expect(page.url()).not.toContain("snapshot_hash");
});
