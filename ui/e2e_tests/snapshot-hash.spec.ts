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

// --- Path 1: Inference detail → Function → Variant (with snapshot hash) ---

test("inference detail page links to function with snapshot_hash", async ({
  page,
}) => {
  await page.goto(`/observability/inferences/${INFERENCE_ID}`);

  // Wait for the basic info to load
  await expect(page.getByText(INFERENCE_FUNCTION).first()).toBeVisible();

  // The function chip should link to the function page with snapshot_hash
  const functionLink = page
    .getByRole("link", { name: INFERENCE_FUNCTION })
    .first();
  const href = await functionLink.getAttribute("href");
  expect(href).toContain(`/observability/functions/${INFERENCE_FUNCTION}`);
  expect(href).toContain(`snapshot_hash=${FIXTURE_SNAPSHOT_HASH}`);
});

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

// --- Path 2: Navigate inference → function → variant, verify hash propagates ---

test("navigating from inference to function carries snapshot_hash and shows banner", async ({
  page,
}) => {
  await page.goto(`/observability/inferences/${INFERENCE_ID}`);
  await expect(page.getByText(INFERENCE_FUNCTION).first()).toBeVisible();

  // Click the function link
  await page.getByRole("link", { name: INFERENCE_FUNCTION }).first().click();
  await page.waitForURL(/\/observability\/functions\//);

  // Banner should appear on the function page
  await expect(
    page.getByText("Viewing historical configuration"),
  ).toBeVisible();
  expect(page.url()).toContain("snapshot_hash=");

  // Now click a variant on the function page to verify hash carries through
  const variantLink = page
    .getByRole("link", { name: INFERENCE_VARIANT })
    .first();
  await expect(variantLink).toBeVisible();
  const variantHref = await variantLink.getAttribute("href");
  expect(variantHref).toContain("snapshot_hash=");
});

// --- Path 3: Variant breadcrumb back to function preserves hash ---

test("variant page breadcrumb preserves snapshot_hash back to function", async ({
  page,
}) => {
  await page.goto(
    `/observability/functions/${INFERENCE_FUNCTION}/variants/${INFERENCE_VARIANT}?snapshot_hash=${FIXTURE_SNAPSHOT_HASH}`,
  );
  await expect(
    page.getByText("Viewing historical configuration"),
  ).toBeVisible();

  const breadcrumbNav = page.getByRole("navigation", { name: "breadcrumb" });
  const functionLink = breadcrumbNav.getByRole("link", {
    name: INFERENCE_FUNCTION,
  });
  await expect(functionLink).toBeVisible();
  const href = await functionLink.getAttribute("href");
  expect(href).toContain(
    `/observability/functions/${INFERENCE_FUNCTION}?snapshot_hash=${FIXTURE_SNAPSHOT_HASH}`,
  );
});

// --- Path 4: No snapshot hash — normal navigation ---

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

test("variant page without snapshot_hash shows no banner", async ({ page }) => {
  await page.goto(
    `/observability/functions/${INFERENCE_FUNCTION}/variants/${INFERENCE_VARIANT}`,
  );
  await expect(page.getByText(INFERENCE_VARIANT).first()).toBeVisible();
  await expect(
    page.getByText("Viewing historical configuration"),
  ).not.toBeVisible();
  expect(page.url()).not.toContain("snapshot_hash");
});

// --- Edge case: current config hash is stripped via redirect ---

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
