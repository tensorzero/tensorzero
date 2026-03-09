import { test, expect } from "@playwright/test";

/**
 * Tests for the browser-based API key auth flow.
 *
 * Verifies the /api/auth/set_gateway_key action endpoint and cookie behavior.
 * The fixture gateway has no auth so any key passes validation — we're testing
 * the UI's cookie mechanics, not the gateway's auth logic.
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
    expect(setCookie).toContain("t0_gateway_key");
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
    const authCookie = cookies.find((c) => c.name === "t0_gateway_key");
    expect(authCookie).toBeTruthy();
    expect(authCookie!.httpOnly).toBe(true);
    expect(authCookie!.sameSite).toBe("Strict");

    // Navigate to another page — cookie should still be there
    await page.goto("/observability/inferences");
    await page.waitForLoadState("networkidle");
    const cookiesAfterNav = await page.context().cookies();
    const authCookieAfterNav = cookiesAfterNav.find(
      (c) => c.name === "t0_gateway_key",
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
      expect(cookiesA.find((c) => c.name === "t0_gateway_key")).toBeTruthy();

      // Context B should NOT have the cookie
      const pageB = await contextB.newPage();
      await pageB.goto("/");
      await pageB.waitForLoadState("networkidle");
      const cookiesB = await contextB.cookies();
      expect(cookiesB.find((c) => c.name === "t0_gateway_key")).toBeFalsy();
    } finally {
      await contextA.close();
      await contextB.close();
    }
  });
});
