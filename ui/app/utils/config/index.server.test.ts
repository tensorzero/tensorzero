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
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Import after mocks are set up
import {
  getConfig,
  resolveConfigEntry,
  resolveFunctionConfig,
  resolveEvaluationConfig,
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

  describe("resolveConfigEntry", () => {
    test("should return value on cache hit without refetching", async () => {
      const mockConfig: UiConfig = {
        ...createMockConfig("hash1"),
        functions: {
          "my-function": {
            type: "chat",
            variants: {},
            schemas: {},
            tools: [],
            tool_choice: "auto",
            parallel_tool_calls: null,
            description: "",
            experimentation: { base: { type: "uniform" }, namespaces: {} },
          },
        },
      };
      mockGetUiConfig.mockResolvedValueOnce(mockConfig);

      const result = await resolveConfigEntry(
        // eslint-disable-next-line no-restricted-syntax
        (cfg) => cfg.functions["my-function"],
      );

      expect(result).not.toBeNull();
      expect(result!.value.type).toBe("chat");
      expect(result!.config).toBe(await getConfig());
      // Only one fetch — no retry needed
      expect(mockGetUiConfig).toHaveBeenCalledTimes(1);
    });

    test("should retry and succeed when entry appears in fresh config", async () => {
      // First config: missing the function
      const staleConfig = createMockConfig("hash_old");
      mockGetUiConfig.mockResolvedValueOnce(staleConfig);

      // Pre-populate cache with stale config
      await getConfig();
      expect(mockGetUiConfig).toHaveBeenCalledTimes(1);

      // Second config: has the function (autopilot just created it)
      const freshConfig: UiConfig = {
        ...createMockConfig("hash_new"),
        functions: {
          "new-function": {
            type: "chat",
            variants: {},
            schemas: {},
            tools: [],
            tool_choice: "auto",
            parallel_tool_calls: null,
            description: "",
            experimentation: { base: { type: "uniform" }, namespaces: {} },
          },
        },
      };
      mockGetUiConfig.mockResolvedValueOnce(freshConfig);

      const result = await resolveConfigEntry(
        // eslint-disable-next-line no-restricted-syntax
        (cfg) => cfg.functions["new-function"],
      );

      expect(result).not.toBeNull();
      expect(result!.value.type).toBe("json");
      expect(result!.config.config_hash).toBe("hash_new");
      // Two fetches: initial + retry
      expect(mockGetUiConfig).toHaveBeenCalledTimes(2);
    });

    test("should return null when entry not found even after retry", async () => {
      const config1 = createMockConfig("hash1");
      const config2 = createMockConfig("hash2");
      mockGetUiConfig.mockResolvedValueOnce(config1);

      // Pre-populate cache
      await getConfig();

      mockGetUiConfig.mockResolvedValueOnce(config2);

      const result = await resolveConfigEntry(
        // eslint-disable-next-line no-restricted-syntax
        (cfg) => cfg.functions["nonexistent"],
      );

      expect(result).toBeNull();
      // Two fetches: initial cache hit (miss) + retry
      expect(mockGetUiConfig).toHaveBeenCalledTimes(2);
    });
  });

  describe("resolveFunctionConfig", () => {
    test("should resolve existing function config", async () => {
      const mockConfig: UiConfig = {
        ...createMockConfig("hash1"),
        functions: {
          "test-fn": {
            type: "chat",
            variants: {},
            schemas: {},
            tools: [],
            tool_choice: "auto",
            parallel_tool_calls: null,
            description: "",
            experimentation: { base: { type: "uniform" }, namespaces: {} },
          },
        },
      };
      mockGetUiConfig.mockResolvedValueOnce(mockConfig);

      const result = await resolveFunctionConfig("test-fn");

      expect(result).not.toBeNull();
      expect(result!.value.type).toBe("chat");
    });

    test("should return null for missing function after retry", async () => {
      const config1 = createMockConfig("hash1");
      const config2 = createMockConfig("hash2");
      mockGetUiConfig
        .mockResolvedValueOnce(config1)
        .mockResolvedValueOnce(config2);

      const result = await resolveFunctionConfig("nonexistent");

      expect(result).toBeNull();
    });
  });

  describe("resolveEvaluationConfig", () => {
    test("should resolve existing evaluation config", async () => {
      const mockConfig: UiConfig = {
        ...createMockConfig("hash1"),
        evaluations: {
          "test-eval": {
            type: "inference",
            function_name: "test-fn",
            evaluators: {},
          },
        },
      };
      mockGetUiConfig.mockResolvedValueOnce(mockConfig);

      const result = await resolveEvaluationConfig("test-eval");

      expect(result).not.toBeNull();
      expect(result!.value.function_name).toBe("test-fn");
    });

    test("should return null for missing evaluation after retry", async () => {
      const config1 = createMockConfig("hash1");
      const config2 = createMockConfig("hash2");
      mockGetUiConfig
        .mockResolvedValueOnce(config1)
        .mockResolvedValueOnce(config2);

      const result = await resolveEvaluationConfig("nonexistent");

      expect(result).toBeNull();
    });
  });
});
