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

// Mock the environment (only needed for gateway URL, config is always loaded from gateway)
vi.mock("../env.server", () => ({
  getEnv: vi.fn(() => ({
    TENSORZERO_GATEWAY_URL: "http://localhost:3000",
  })),
}));

// Mock the logger to avoid console noise in tests
vi.mock("../logger", () => ({
  logger: {
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

// Helper to create a mock UiConfig
function createMockConfig(hash: string): UiConfig {
  return {
    functions: {},
    metrics: {},
    tools: {},
    evaluations: {},
    model_names: [],
    config_hash: hash,
  };
}

describe("config cache and hash polling", () => {
  beforeEach(() => {
    // Reset module state before each test
    _resetForTesting();
    vi.clearAllMocks();

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
});
