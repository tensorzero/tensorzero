import { describe, expect, test, beforeEach, vi, afterEach } from "vitest";
import type { UiConfig, StatusResponse } from "~/types/tensorzero";

// Mock the TensorZero client
const mockStatus = vi.fn<() => Promise<StatusResponse>>();
const mockGetUiConfig = vi.fn<() => Promise<UiConfig>>();

vi.mock("~/utils/get-tensorzero-client.server", () => ({
  getTensorZeroClient: vi.fn(() => ({
    status: mockStatus,
    getUiConfig: mockGetUiConfig,
  })),
}));

// Mutable env mock so individual tests can switch between env-var and
// cookie-auth modes. Default to env-var mode so the existing "should cache"
// tests continue to exercise the shared-cache path.
const mockGetEnv = vi.fn(() => ({
  TENSORZERO_GATEWAY_URL: "http://localhost:3000",
  TENSORZERO_API_KEY: "test-env-key" as string | undefined,
}));

vi.mock("../env.server", () => ({
  getEnv: () => mockGetEnv(),
}));

// Mock the logger to avoid console noise in tests
vi.mock("../logger", () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Import after mocks are set up
import {
  getConfig,
  _resetForTesting,
  _checkConfigHashForTesting,
  _getConfigCacheForTesting,
  _setConfigCacheForTesting,
} from "./index.server";
import { apiKeyCookie, runWithRequest } from "../api-key-override.server";
import {
  TensorZeroServerError,
  isAuthenticationError,
} from "../tensorzero/errors";

async function makeRequest(apiKey?: string): Promise<Request> {
  const headers = new Headers();
  if (apiKey) {
    headers.set("cookie", await apiKeyCookie.serialize(apiKey));
  }
  return new Request("http://localhost:3000", { headers });
}

function setCookieAuthMode(): void {
  mockGetEnv.mockReturnValue({
    TENSORZERO_GATEWAY_URL: "http://localhost:3000",
    TENSORZERO_API_KEY: undefined,
  });
}

function setEnvVarAuthMode(): void {
  mockGetEnv.mockReturnValue({
    TENSORZERO_GATEWAY_URL: "http://localhost:3000",
    TENSORZERO_API_KEY: "test-env-key",
  });
}

// Helper to create a mock UiConfig
function createMockConfig(hash: string): UiConfig {
  return {
    functions: {},
    metrics: {},
    tools: {},
    evaluations: {},
    model_names: [],
    config_hash: hash,
    config_in_database: false,
    auth_enabled: false,
  };
}

describe("config cache and hash polling", () => {
  beforeEach(() => {
    // Reset module state before each test
    _resetForTesting();
    // Use mockReset (not clearAllMocks) so mockResolvedValueOnce /
    // mockRejectedValueOnce queues don't leak across tests.
    mockGetUiConfig.mockReset();
    mockStatus.mockReset();

    // Default to env-var auth mode for the existing shared-cache tests.
    setEnvVarAuthMode();

    // Use fake timers to control setInterval
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe("getConfig", () => {
    test("should load config from gateway and cache it", async () => {
      const mockConfig = createMockConfig("hash123");
      mockGetUiConfig.mockResolvedValueOnce(mockConfig);

      const config = await getConfig();

      expect(mockGetUiConfig).toHaveBeenCalledTimes(1);
      expect(config.config_hash).toBe("hash123");

      // Second call should return cached config
      const config2 = await getConfig();
      expect(mockGetUiConfig).toHaveBeenCalledTimes(1); // Still 1, not 2
      expect(config2).toBe(config);
    });

    test("should add default function to config", async () => {
      const mockConfig = createMockConfig("hash123");
      mockGetUiConfig.mockResolvedValueOnce(mockConfig);

      const config = await getConfig();

      // eslint-disable-next-line no-restricted-syntax
      expect(config.functions["tensorzero::default"]).toBeDefined();
      // eslint-disable-next-line no-restricted-syntax
      expect(config.functions["tensorzero::default"]?.type).toBe("chat");
    });
  });

  describe("checkConfigHash", () => {
    test("should not invalidate cache when hash matches", async () => {
      const mockConfig = createMockConfig("hash123");
      _setConfigCacheForTesting(mockConfig);

      mockStatus.mockResolvedValueOnce({
        status: "ok",
        version: "1.0.0",
        config_hash: "hash123", // Same hash
      });

      await _checkConfigHashForTesting();

      // Cache should still exist
      expect(_getConfigCacheForTesting()).toBe(mockConfig);
    });

    test("should invalidate cache when hash changes", async () => {
      const mockConfig = createMockConfig("hash123");
      _setConfigCacheForTesting(mockConfig);

      mockStatus.mockResolvedValueOnce({
        status: "ok",
        version: "1.0.0",
        config_hash: "hash456", // Different hash
      });

      await _checkConfigHashForTesting();

      // Cache should be invalidated
      expect(_getConfigCacheForTesting()).toBeUndefined();
    });

    test("should skip check when no cached config", async () => {
      // No cached config
      _setConfigCacheForTesting(undefined);

      await _checkConfigHashForTesting();

      // status() should not be called
      expect(mockStatus).not.toHaveBeenCalled();
    });

    test("should skip check when config has empty hash (legacy disk mode)", async () => {
      const mockConfig = createMockConfig(""); // Empty hash
      _setConfigCacheForTesting(mockConfig);

      await _checkConfigHashForTesting();

      // status() should not be called
      expect(mockStatus).not.toHaveBeenCalled();
    });

    test("should not crash when status() throws", async () => {
      const mockConfig = createMockConfig("hash123");
      _setConfigCacheForTesting(mockConfig);

      mockStatus.mockRejectedValueOnce(new Error("Network error"));

      // Should not throw
      await expect(_checkConfigHashForTesting()).resolves.toBeUndefined();

      // Cache should still exist (not invalidated on error)
      expect(_getConfigCacheForTesting()).toBe(mockConfig);
    });
  });

  describe("polling integration", () => {
    test("should reload config after cache invalidation", async () => {
      // Initial load
      const initialConfig = createMockConfig("hash_v1");
      mockGetUiConfig.mockResolvedValueOnce(initialConfig);

      const config1 = await getConfig();
      expect(config1.config_hash).toBe("hash_v1");

      // Simulate hash change detected
      mockStatus.mockResolvedValueOnce({
        status: "ok",
        version: "1.0.0",
        config_hash: "hash_v2",
      });

      await _checkConfigHashForTesting();

      // Cache should be invalidated
      expect(_getConfigCacheForTesting()).toBeUndefined();

      // Next getConfig() should reload
      const updatedConfig = createMockConfig("hash_v2");
      mockGetUiConfig.mockResolvedValueOnce(updatedConfig);

      const config2 = await getConfig();
      expect(config2.config_hash).toBe("hash_v2");
      expect(mockGetUiConfig).toHaveBeenCalledTimes(2);
    });
  });

  describe("cookie-auth mode (no TENSORZERO_API_KEY)", () => {
    beforeEach(() => {
      setCookieAuthMode();
    });

    test("does not serve cached config across requests with different credentials", async () => {
      const mockConfig = createMockConfig("hash123");
      mockGetUiConfig.mockResolvedValueOnce(mockConfig);
      mockGetUiConfig.mockRejectedValueOnce(
        new TensorZeroServerError("Unauthorized", { status: 401 }),
      );

      // Authenticated request populates the response.
      await runWithRequest(await makeRequest("cookie-A"), () => getConfig());

      // Subsequent request with no cookie must trigger a fresh gateway call,
      // not return whatever was loaded under cookie-A.
      await expect(
        runWithRequest(await makeRequest(undefined), () => getConfig()),
      ).rejects.toThrow();

      expect(mockGetUiConfig).toHaveBeenCalledTimes(2);
    });

    test("propagates 401 on cookie-less request after a cached authenticated load", async () => {
      const mockConfig = createMockConfig("hash123");
      mockGetUiConfig.mockResolvedValueOnce(mockConfig);
      mockGetUiConfig.mockRejectedValueOnce(
        new TensorZeroServerError("Unauthorized", { status: 401 }),
      );

      await runWithRequest(await makeRequest("cookie-A"), () => getConfig());

      // The root loader uses isAuthenticationError() to decide whether to
      // render the API-key dialog. That signal must reach it even after the
      // first request "warmed" anything.
      let caught: unknown;
      try {
        await runWithRequest(await makeRequest(undefined), () => getConfig());
      } catch (e) {
        caught = e;
      }
      expect(isAuthenticationError(caught)).toBe(true);
    });

    test("does not start the hash-poll interval", async () => {
      mockGetUiConfig.mockResolvedValue(createMockConfig("hash123"));

      await runWithRequest(await makeRequest("cookie-A"), () => getConfig());

      // Polling has no request context inside its setInterval tick, so it
      // would 401 every 5s. It also has no shared cache to invalidate.
      // Advancing fake timers past the poll interval must not call status().
      await vi.advanceTimersByTimeAsync(30_000);
      expect(mockStatus).not.toHaveBeenCalled();
    });

    test("does not retain a process-global cache between requests", async () => {
      mockGetUiConfig.mockResolvedValue(createMockConfig("hash123"));

      await runWithRequest(await makeRequest("cookie-A"), () => getConfig());

      // The shared cache must remain empty in cookie-auth mode. If anything
      // gets stored here, a later request with no/different cookie would read
      // it without re-authenticating.
      expect(_getConfigCacheForTesting()).toBeUndefined();
    });
  });

  describe("env-var auth mode (TENSORZERO_API_KEY set) — regression guard", () => {
    test("caches config across requests", async () => {
      const mockConfig = createMockConfig("hash123");
      mockGetUiConfig.mockResolvedValueOnce(mockConfig);

      await getConfig();
      await getConfig();
      await getConfig();

      expect(mockGetUiConfig).toHaveBeenCalledTimes(1);
    });
  });
});
