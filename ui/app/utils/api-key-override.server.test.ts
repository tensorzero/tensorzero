import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  runWithRequest,
  getApiKeyFromRequest,
  getEffectiveApiKey,
  buildApiKeyCookie,
} from "./api-key-override.server";

const mockGetEnv = vi.fn(() => ({
  TENSORZERO_GATEWAY_URL: "http://localhost:3000",
  TENSORZERO_API_KEY: "",
}));

vi.mock("./env.server", () => ({
  getEnv: () => mockGetEnv(),
}));

function makeRequest(cookie?: string): Request {
  const headers = new Headers();
  if (cookie) {
    headers.set("cookie", cookie);
  }
  return new Request("http://localhost:3000", { headers });
}

beforeEach(() => {
  mockGetEnv.mockReturnValue({
    TENSORZERO_GATEWAY_URL: "http://localhost:3000",
    TENSORZERO_API_KEY: "",
  });
});

describe("getApiKeyFromRequest", () => {
  it("parses among multiple cookies", () => {
    const request = makeRequest(
      "session=abc; tz_gateway_key=my-key; theme=dark",
    );
    expect(getApiKeyFromRequest(request)).toBe("my-key");
  });

  it("returns undefined for malformed percent-encoding instead of throwing", () => {
    const request = makeRequest("tz_gateway_key=%ZZ%invalid");
    expect(getApiKeyFromRequest(request)).toBeUndefined();
  });
});

describe("getEffectiveApiKey", () => {
  it("prefers env var over cookie", () => {
    mockGetEnv.mockReturnValue({
      TENSORZERO_GATEWAY_URL: "http://localhost:3000",
      TENSORZERO_API_KEY: "env-key",
    });
    const result = runWithRequest(
      makeRequest("tz_gateway_key=cookie-key"),
      () => getEffectiveApiKey(),
    );
    expect(result).toBe("env-key");
  });

  it("falls back to cookie when env var is empty", () => {
    const result = runWithRequest(
      makeRequest("tz_gateway_key=cookie-key"),
      () => getEffectiveApiKey(),
    );
    expect(result).toBe("cookie-key");
  });

  it("isolates concurrent requests via AsyncLocalStorage", async () => {
    const results: (string | undefined)[] = [];
    await Promise.all([
      new Promise<void>((resolve) => {
        runWithRequest(makeRequest("tz_gateway_key=key-A"), () => {
          results[0] = getEffectiveApiKey();
          resolve();
        });
      }),
      new Promise<void>((resolve) => {
        runWithRequest(makeRequest("tz_gateway_key=key-B"), () => {
          results[1] = getEffectiveApiKey();
          resolve();
        });
      }),
    ]);
    expect(results[0]).toBe("key-A");
    expect(results[1]).toBe("key-B");
  });
});

describe("buildApiKeyCookie", () => {
  it("omits Secure for IPv6 loopback", () => {
    mockGetEnv.mockReturnValue({
      TENSORZERO_GATEWAY_URL: "http://[::1]:3000",
      TENSORZERO_API_KEY: "",
    });
    expect(buildApiKeyCookie("key")).not.toContain("Secure");
  });

  it("includes Secure for non-localhost", () => {
    mockGetEnv.mockReturnValue({
      TENSORZERO_GATEWAY_URL: "https://gateway.example.com",
      TENSORZERO_API_KEY: "",
    });
    expect(buildApiKeyCookie("key")).toContain("Secure");
  });
});
