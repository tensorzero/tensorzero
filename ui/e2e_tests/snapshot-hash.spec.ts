import { test, expect, type Page } from "@playwright/test";
import { decimalToHex } from "~/utils/common";

const FUNCTION_NAME = "write_haiku";
const GATEWAY_URL = "http://localhost:3000";

async function getCurrentConfigHashAsHex(page: Page): Promise<string> {
  const response = await page.request.get(`${GATEWAY_URL}/status`);
  const status = await response.json();
  return decimalToHex(status.config_hash);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function stripNulls(obj: any): any {
  if (Array.isArray(obj)) return obj.map(stripNulls);
  if (obj !== null && typeof obj === "object") {
    return Object.fromEntries(
      Object.entries(obj)
        .filter(([, v]) => v !== null)
        .map(([k, v]) => [k, stripNulls(v)]),
    );
  }
  return obj;
}

/**
 * Writes a modified config to create a real historical snapshot.
 * Adds an extra template to produce a different hash from the current config.
 * Returns the hash as hex.
 */
async function createHistoricalSnapshot(page: Page): Promise<string> {
  const configResponse = await page.request.get(
    `${GATEWAY_URL}/internal/config`,
  );
  const { config, extra_templates, tags } = await configResponse.json();

  // Strip null values — GET serializes Option::None as null but POST rejects them.
  const writeResponse = await page.request.post(
    `${GATEWAY_URL}/internal/config`,
    {
      data: stripNulls({
        config,
        extra_templates: {
          ...extra_templates,
          "test_snapshot_marker.txt": "historical config for e2e test",
        },
        tags,
      }),
    },
  );
  if (!writeResponse.ok()) {
    const text = await writeResponse.text();
    throw new Error(
      `POST /internal/config failed (${writeResponse.status()}): ${text}`,
    );
  }
  const { hash } = await writeResponse.json();
  return decimalToHex(hash);
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

test("snapshot_hash with real historical config shows banner and propagates", async ({
  page,
}) => {
  const historicalHash = await createHistoricalSnapshot(page);

  await page.goto(
    `/observability/functions/${FUNCTION_NAME}?snapshot_hash=${historicalHash}`,
  );
  await expect(page.getByRole("heading", { name: "Variants" })).toBeVisible();

  await expect(
    page.getByText("Viewing historical configuration"),
  ).toBeVisible();
  expect(page.url()).toContain(`snapshot_hash=${historicalHash}`);

  // Variant links should carry the snapshot hash
  const variantLink = page.getByRole("link").filter({ hasText: /prompt/ });
  const firstLink = variantLink.first();
  await expect(firstLink).toBeVisible();
  const href = await firstLink.getAttribute("href");
  expect(href).toContain(`snapshot_hash=${historicalHash}`);
});

test("snapshot_hash matching current config is stripped via redirect", async ({
  page,
}) => {
  const currentHash = await getCurrentConfigHashAsHex(page);

  await page.goto(
    `/observability/functions/${FUNCTION_NAME}?snapshot_hash=${currentHash}`,
  );
  await expect(page.getByRole("heading", { name: "Variants" })).toBeVisible();
  await expect(
    page.getByText("Viewing historical configuration"),
  ).not.toBeVisible();
  expect(page.url()).not.toContain("snapshot_hash");
});
