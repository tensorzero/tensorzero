import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  runWithRequest,
  getApiKeyFromRequest,
  getEffectiveApiKey,
  isSecureRequest,
  apiKeyCookie,
} from "./api-key-override.server";

const mockGetEnv = vi.fn(() => ({
  TENSORZERO_GATEWAY_URL: "http://localhost:3000",
  TENSORZERO_API_KEY: "",
}));

vi.mock("./env.server", () => ({
  getEnv: () => mockGetEnv(),
}));

async function makeRequest(apiKey?: string): Promise<Request> {
  const headers = new Headers();
  if (apiKey) {
    headers.set("cookie", await apiKeyCookie.serialize(apiKey));
  }
  return new Request("http://localhost:3000", { headers });
}

beforeEach(() => {
  mockGetEnv.mockReturnValue({
    TENSORZERO_GATEWAY_URL: "http://localhost:3000",
    TENSORZERO_API_KEY: "",
  });
});

describe("isSecureRequest", () => {
  it("returns false for http URLs", () => {
    expect(isSecureRequest(new Request("http://10.0.0.5:4000"))).toBe(false);
  });

  it("returns true for https URLs", () => {
    expect(isSecureRequest(new Request("https://example.com"))).toBe(true);
  });

  it("respects x-forwarded-proto header", () => {
    const request = new Request("http://localhost:3000", {
      headers: { "x-forwarded-proto": "https" },
    });
    expect(isSecureRequest(request)).toBe(true);
  });

  it("handles comma-separated x-forwarded-proto (multi-proxy)", () => {
    const request = new Request("http://localhost:3000", {
      headers: { "x-forwarded-proto": "https, http" },
    });
    expect(isSecureRequest(request)).toBe(true);
  });
});

describe("getApiKeyFromRequest", () => {
  it("parses cookie set by apiKeyCookie.serialize", async () => {
    const request = await makeRequest("my-key");
    expect(await getApiKeyFromRequest(request)).toBe("my-key");
  });
});

describe("getEffectiveApiKey", () => {
  it("prefers env var over cookie", async () => {
    mockGetEnv.mockReturnValue({
      TENSORZERO_GATEWAY_URL: "http://localhost:3000",
      TENSORZERO_API_KEY: "env-key",
    });
    const result = await runWithRequest(await makeRequest("cookie-key"), () =>
      getEffectiveApiKey(),
    );
    expect(result).toBe("env-key");
  });

  it("falls back to cookie when env var is empty", async () => {
    const result = await runWithRequest(await makeRequest("cookie-key"), () =>
      getEffectiveApiKey(),
    );
    expect(result).toBe("cookie-key");
  });

  it("isolates concurrent requests via AsyncLocalStorage", async () => {
    const results: (string | undefined)[] = [];
    await Promise.all([
      runWithRequest(await makeRequest("key-A"), () => {
        results[0] = getEffectiveApiKey();
      }),
      runWithRequest(await makeRequest("key-B"), () => {
        results[1] = getEffectiveApiKey();
      }),
    ]);
    expect(results[0]).toBe("key-A");
    expect(results[1]).toBe("key-B");
  });
});
