import { test, expect } from "@playwright/test";

/**
 * Tests for the browser-based API key auth flow.
 *
 * CI runs three auth modes via matrix (auth_mode):
 * - no_gateway_auth: gateway has no auth, these tests are skipped
 * - gateway_auth_with_ui_server_key: UI server has TENSORZERO_API_KEY in env → endpoint returns 409
 * - gateway_auth_with_browser_key: UI server does NOT have key in env → browser flow active
 *
 * Tags control which tests run in which mode:
 * - @gateway-auth-with-browser-key: runs only in gateway_auth_with_browser_key
 * - @gateway-auth-with-ui-server-key: runs only in gateway_auth_with_ui_server_key
 */

test.describe("@gateway-auth-with-browser-key Browser Auth Flow", () => {
  test("POST with valid key sets HttpOnly cookie", async ({ request }) => {
    const apiKey = process.env.TENSORZERO_API_KEY_FOR_BROWSER_AUTH;
    expect(
      apiKey,
      "TENSORZERO_API_KEY_FOR_BROWSER_AUTH must be set for browser key tests",
    ).toBeTruthy();

    const response = await request.post("/api/auth/set_gateway_key", {
      form: { apiKey: apiKey! },
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

  test("POST with invalid key returns 401", async ({ request }) => {
    const response = await request.post("/api/auth/set_gateway_key", {
      form: { apiKey: "definitely-not-a-valid-key" },
    });

    expect(response.status()).toBe(401);
    const body = await response.json();
    expect(body.error).toContain("Invalid API key");
  });

  test("cookie persists across page navigations", async ({ page }) => {
    const apiKey = process.env.TENSORZERO_API_KEY_FOR_BROWSER_AUTH;
    expect(apiKey).toBeTruthy();

    // Set a cookie via the action endpoint
    const response = await page.request.post("/api/auth/set_gateway_key", {
      form: { apiKey: apiKey! },
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

    // Navigate to another page - cookie should still be there
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
    const apiKey = process.env.TENSORZERO_API_KEY_FOR_BROWSER_AUTH;
    expect(apiKey).toBeTruthy();

    // Create two isolated browser contexts
    const contextA = await browser.newContext();
    const contextB = await browser.newContext();

    try {
      // Set cookie in context A only
      const requestA = contextA.request;
      const response = await requestA.post("/api/auth/set_gateway_key", {
        form: { apiKey: apiKey! },
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

test.describe("@gateway-auth-with-ui-server-key Browser Auth Blocked by UI Server Key", () => {
  test("POST returns 409 when TENSORZERO_API_KEY is set", async ({
    request,
  }) => {
    const response = await request.post("/api/auth/set_gateway_key", {
      form: { apiKey: "any-key" },
    });

    expect(response.status()).toBe(409);
    const body = await response.json();
    expect(body.error).toContain("TENSORZERO_API_KEY");
  });
});
