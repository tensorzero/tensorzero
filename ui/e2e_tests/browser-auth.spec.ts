import { test, expect } from "@playwright/test";

/**
 * Tests for the browser-based API key auth flow.
 *
 * These tests verify the /api/auth/set_gateway_key action endpoint and
 * cookie behavior. The current fixture gateway has no auth, so any key
 * passes validation — but that's fine: we're testing the UI's cookie
 * mechanics, not the gateway's auth logic (which has its own test suite).
 *
 * Full auth dialog flow tests (auth-enabled gateway) are covered in CI
 * via the auth_enabled matrix in ui-tests-e2e.yml.
 */

test.describe("Browser API Key - Action Endpoint", () => {
  test("POST with valid key sets HttpOnly cookie", async ({ request }) => {
    const response = await request.post("/api/auth/set_gateway_key", {
      form: { apiKey: "test-key-abc123" },
    });

    expect(response.ok()).toBe(true);
    const body = await response.json();
    expect(body.success).toBe(true);

    const setCookie = response.headers()["set-cookie"];
    expect(setCookie).toBeTruthy();
    expect(setCookie).toContain("tz_gateway_key");
    expect(setCookie).toContain("HttpOnly");
    expect(setCookie).toContain("SameSite=Strict");
    expect(setCookie).toContain("Path=/");
    expect(setCookie).toContain("Max-Age=2592000");
  });

  test("POST with empty key returns 400", async ({ request }) => {
    const response = await request.post("/api/auth/set_gateway_key", {
      form: { apiKey: "" },
    });

    expect(response.status()).toBe(400);
    const body = await response.json();
    expect(body.error).toBe("API key is required");
  });

  test("POST with whitespace-only key returns 400", async ({ request }) => {
    const response = await request.post("/api/auth/set_gateway_key", {
      form: { apiKey: "   " },
    });

    expect(response.status()).toBe(400);
    const body = await response.json();
    expect(body.error).toBe("API key is required");
  });

  test("POST with too-long key returns 400", async ({ request }) => {
    const longKey = "x".repeat(513);
    const response = await request.post("/api/auth/set_gateway_key", {
      form: { apiKey: longKey },
    });

    expect(response.status()).toBe(400);
    const body = await response.json();
    expect(body.error).toBe("API key is too long");
  });

  test("POST trims whitespace from key", async ({ request }) => {
    const response = await request.post("/api/auth/set_gateway_key", {
      form: { apiKey: "  trimmed-key  " },
    });

    expect(response.ok()).toBe(true);
    const body = await response.json();
    expect(body.success).toBe(true);
  });
});

test.describe("Browser API Key - Cookie Persistence", () => {
  test("cookie persists across page navigations", async ({ page }) => {
    // Set a cookie via the action endpoint
    const response = await page.request.post("/api/auth/set_gateway_key", {
      form: { apiKey: "persist-test-key" },
    });
    expect(response.ok()).toBe(true);

    // Navigate to multiple pages
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // Check cookie is still present
    const cookies = await page.context().cookies();
    const authCookie = cookies.find((c) => c.name === "tz_gateway_key");
    expect(authCookie).toBeTruthy();
    expect(authCookie!.httpOnly).toBe(true);
    expect(authCookie!.sameSite).toBe("Strict");

    // Navigate to another page — cookie should still be there
    await page.goto("/observability/inferences");
    await page.waitForLoadState("networkidle");
    const cookiesAfterNav = await page.context().cookies();
    const authCookieAfterNav = cookiesAfterNav.find(
      (c) => c.name === "tz_gateway_key",
    );
    expect(authCookieAfterNav).toBeTruthy();
  });

  test("different browser contexts have independent cookies", async ({
    browser,
  }) => {
    // Create two isolated browser contexts
    const contextA = await browser.newContext();
    const contextB = await browser.newContext();

    try {
      // Set cookie in context A only
      const requestA = contextA.request;
      const response = await requestA.post("/api/auth/set_gateway_key", {
        form: { apiKey: "context-a-key" },
      });
      expect(response.ok()).toBe(true);

      // Context A should have the cookie
      const pageA = await contextA.newPage();
      await pageA.goto("/");
      await pageA.waitForLoadState("networkidle");
      const cookiesA = await contextA.cookies();
      expect(cookiesA.find((c) => c.name === "tz_gateway_key")).toBeTruthy();

      // Context B should NOT have the cookie
      const pageB = await contextB.newPage();
      await pageB.goto("/");
      await pageB.waitForLoadState("networkidle");
      const cookiesB = await contextB.cookies();
      expect(cookiesB.find((c) => c.name === "tz_gateway_key")).toBeFalsy();
    } finally {
      await contextA.close();
      await contextB.close();
    }
  });
});

test.describe("Browser API Key - Auth Dialog Flow", () => {
  /**
   * This test requires an auth-enabled gateway. It's designed to run in CI
   * with the auth_enabled matrix, or locally when the gateway has
   * [gateway.auth] enabled and TENSORZERO_API_KEY is NOT set on the UI.
   *
   * To run locally:
   * 1. Add `[gateway.auth]\nenabled = true` to fixtures/config/tensorzero.toml
   * 2. Create an API key: docker exec fixtures-gateway-1 /app/tensorzero --create-api-key
   * 3. Remove TENSORZERO_API_KEY from ui/.env (or don't set it)
   * 4. Start the dev server and run: TENSORZERO_AUTH_TEST_KEY=<key> pnpm test-e2e --grep "Auth Dialog"
   */
  const authTestKey = process.env.TENSORZERO_AUTH_TEST_KEY;

  // Skip if no auth test key provided (gateway doesn't require auth)
  test.skip(
    !authTestKey,
    "Set TENSORZERO_AUTH_TEST_KEY to run auth dialog tests",
  );

  test("shows auth dialog and connects with valid key", async ({ page }) => {
    // Navigate to root — should see auth dialog
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // Auth dialog should be visible
    await expect(
      page.getByRole("heading", { name: "Connect to TensorZero" }),
    ).toBeVisible();
    await expect(page.getByText("Enter your TensorZero API key")).toBeVisible();

    // Enter the API key
    const input = page.getByLabel("API Key");
    await expect(input).toBeVisible();
    await input.fill(authTestKey!);

    // Click Connect
    const connectButton = page.getByRole("button", { name: "Connect" });
    await expect(connectButton).toBeEnabled();
    await connectButton.click();

    // Page should reload and show content (auth dialog gone)
    await page.waitForLoadState("networkidle");
    await expect(
      page.getByRole("heading", { name: "Connect to TensorZero" }),
    ).not.toBeVisible();

    // Sidebar should be functional
    await expect(page.getByRole("link", { name: "Overview" })).toBeVisible();
  });

  test("shows error for invalid key", async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // Enter an invalid key
    const input = page.getByLabel("API Key");
    await input.fill("invalid-key-that-will-fail");

    const connectButton = page.getByRole("button", { name: "Connect" });
    await connectButton.click();

    // Should show error message
    await expect(
      page.getByText("Invalid API key. The gateway rejected this key."),
    ).toBeVisible();

    // Dialog should still be visible
    await expect(
      page.getByRole("heading", { name: "Connect to TensorZero" }),
    ).toBeVisible();
  });

  test("cookie persists across page reload after auth", async ({ page }) => {
    // Enter valid key
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    await page.getByLabel("API Key").fill(authTestKey!);
    await page.getByRole("button", { name: "Connect" }).click();
    await page.waitForLoadState("networkidle");

    // Verify we're authenticated
    await expect(
      page.getByRole("heading", { name: "Connect to TensorZero" }),
    ).not.toBeVisible();

    // Reload page — should still be authenticated (cookie persists)
    await page.reload();
    await page.waitForLoadState("networkidle");
    await expect(
      page.getByRole("heading", { name: "Connect to TensorZero" }),
    ).not.toBeVisible();
    await expect(page.getByRole("link", { name: "Overview" })).toBeVisible();
  });

  test("separate browser context requires its own auth", async ({
    browser,
  }) => {
    // Auth in first context
    const context1 = await browser.newContext();
    const page1 = await context1.newPage();
    await page1.goto("/");
    await page1.waitForLoadState("networkidle");
    await page1.getByLabel("API Key").fill(authTestKey!);
    await page1.getByRole("button", { name: "Connect" }).click();
    await page1.waitForLoadState("networkidle");
    await expect(
      page1.getByRole("heading", { name: "Connect to TensorZero" }),
    ).not.toBeVisible();

    // Second context should still need auth (no cookie)
    const context2 = await browser.newContext();
    const page2 = await context2.newPage();
    await page2.goto("/");
    await page2.waitForLoadState("networkidle");
    await expect(
      page2.getByRole("heading", { name: "Connect to TensorZero" }),
    ).toBeVisible();

    await context1.close();
    await context2.close();
  });
});
