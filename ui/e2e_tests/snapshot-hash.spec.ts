import { test, expect } from "@playwright/test";

// A snapshot hash present in fixture data (lowercase hex, as returned by the gateway).
// This is the "current" config hash in hex — but since the gateway's config_hash is
// a decimal string, this won't match and will be treated as historical.
const FIXTURE_SNAPSHOT_HASH =
  "65bb2c87383a9d781cff82fc7cfe114673c4facaf2b1a3024b8e9a1d7cb7b6f2";

// The current config hash from the gateway (decimal string).
// When this exact value is passed as ?snapshot_hash, the server strips it via redirect.
const CURRENT_CONFIG_HASH =
  "46014305430377123138237384229499364238047438285692198569022910911979009324786";

// A known inference with a snapshot_hash in the fixture data
const INFERENCE_ID = "019cba09-7f82-7233-bf78-6aa9bf21b666";
const INFERENCE_FUNCTION = "write_haiku";
const INFERENCE_VARIANT = "initial_prompt_gpt4o_mini";

test("inference detail page links to variant with snapshot_hash", async ({
  page,
}) => {
  await page.goto(`/observability/inferences/${INFERENCE_ID}`);
  await expect(page.getByText(INFERENCE_VARIANT).first()).toBeVisible();

  // The variant chip should link to the variant page with snapshot_hash
  const variantLink = page
    .getByRole("link", { name: INFERENCE_VARIANT })
    .first();
  const href = await variantLink.getAttribute("href");
  expect(href).toContain(
    `/observability/functions/${INFERENCE_FUNCTION}/variants/${INFERENCE_VARIANT}`,
  );
  expect(href).toContain(`snapshot_hash=${FIXTURE_SNAPSHOT_HASH}`);
});

test("navigating inference → function → variant propagates snapshot_hash", async ({
  page,
}) => {
  await page.goto(`/observability/inferences/${INFERENCE_ID}`);
  await expect(page.getByText(INFERENCE_FUNCTION).first()).toBeVisible();

  // Click the function link from the inference detail page
  await page.getByRole("link", { name: INFERENCE_FUNCTION }).first().click();
  await page.waitForURL(/\/observability\/functions\//);

  // Banner should appear on the function page
  await expect(
    page.getByText("Viewing historical configuration"),
  ).toBeVisible();
  expect(page.url()).toContain("snapshot_hash=");

  // Variant links on the function page should also carry the hash
  const variantLink = page
    .getByRole("link", { name: INFERENCE_VARIANT })
    .first();
  await expect(variantLink).toBeVisible();
  const variantHref = await variantLink.getAttribute("href");
  expect(variantHref).toContain("snapshot_hash=");
});

test("function page without snapshot_hash shows no banner", async ({
  page,
}) => {
  await page.goto(`/observability/functions/${INFERENCE_FUNCTION}`);
  await expect(page.getByRole("heading", { name: "Variants" })).toBeVisible();
  await expect(
    page.getByText("Viewing historical configuration"),
  ).not.toBeVisible();
  expect(page.url()).not.toContain("snapshot_hash");
});

test("snapshot_hash matching current config is stripped via redirect", async ({
  page,
}) => {
  await page.goto(
    `/observability/functions/${INFERENCE_FUNCTION}?snapshot_hash=${CURRENT_CONFIG_HASH}`,
  );
  await expect(page.getByRole("heading", { name: "Variants" })).toBeVisible();
  await expect(
    page.getByText("Viewing historical configuration"),
  ).not.toBeVisible();
  expect(page.url()).not.toContain("snapshot_hash");
});
