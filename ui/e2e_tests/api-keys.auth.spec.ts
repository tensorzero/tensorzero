import { test, expect } from "@playwright/test";

/**
 * Tests for the API keys page auth guard.
 *
 * The page bypasses the gateway and talks straight to Postgres via the NAPI
 * binding, so it must validate the caller's API key itself
 * (`requireValidApiKeyIfEnabled` in `app/utils/auth.server.ts`).
 *
 * These run only in `gateway_auth_with_browser_key` mode — gateway auth is on
 * but the UI server has no `TENSORZERO_API_KEY`, so a fresh browser context
 * has no credentials and the guard must reject it.
 */

test.describe("@gateway-auth-with-browser-key API Keys page auth guard", () => {
  test("GET /api-keys without a cookie does not render the keys table", async ({
    browser,
  }) => {
    const context = await browser.newContext();
    try {
      const page = await context.newPage();
      await page.goto("/api-keys");
      await page.waitForLoadState("networkidle");

      // The keys table must not render — the route's loader threw before
      // `fetchApiKeys` could run.
      await expect(page.locator("tbody tr")).toHaveCount(0);

      // The auth dialog from the route's ErrorBoundary should be visible.
      await expect(page.getByText("TensorZero Gateway requires")).toBeVisible();
    } finally {
      await context.close();
    }
  });

  test("POST /api-keys without a cookie returns 401", async ({ request }) => {
    const response = await request.post("/api-keys", {
      form: { action: "generate" },
    });
    expect(response.status()).toBe(401);
  });

  test("GET /api-keys with a valid cookie renders the keys table", async ({
    browser,
  }) => {
    const apiKey = process.env.TENSORZERO_API_KEY_FOR_BROWSER_AUTH;
    expect(
      apiKey,
      "TENSORZERO_API_KEY_FOR_BROWSER_AUTH must be set for browser key tests",
    ).toBeTruthy();

    const context = await browser.newContext();
    try {
      const setKey = await context.request.post("/api/auth/set_gateway_key", {
        form: { apiKey: apiKey! },
      });
      expect(setKey.ok()).toBe(true);

      const page = await context.newPage();
      await page.goto("/api-keys");
      await page.waitForLoadState("networkidle");

      await expect(
        page.getByRole("heading", { name: "TensorZero API Keys" }),
      ).toBeVisible();
      await expect(
        page.getByText("TensorZero Gateway requires"),
      ).not.toBeVisible();
    } finally {
      await context.close();
    }
  });
});
