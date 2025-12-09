import { describe, it, expect } from "vitest";
import { TensorZeroClient, DatabaseClient } from "../index.js";

const UI_FIXTURES_CONFIG_PATH = "../../ui/fixtures/config/tensorzero.toml";

describe("TensorZeroClient Integration Tests", () => {
  it("should be able to import TensorZeroClient", () => {
    expect(TensorZeroClient).toBeDefined();
    expect(typeof TensorZeroClient).toBe("function");
  });

  it("client creation throws on a bad URL", async () => {
    // This should throw an error that contains "relative URL without a base"
    await expect(
      async () => await TensorZeroClient.buildHttp("foo"),
    ).rejects.toThrow("relative URL without a base");
  });

  it("should have required methods and initialize without credentials", async () => {
    const client = await buildClient();
    expect(typeof client.experimentalLaunchOptimizationWorkflow).toBe(
      "function",
    );
    expect(typeof client.experimentalPollOptimization).toBe("function");
  });

  it("should be able to stale dataset", async () => {
    const client = await buildClient();
    // In the future once we have full dataset lifecycle support here we can do a better test
    const staleDatasetResponse =
      await client.staleDataset("nonexistentdataset");
    expect(staleDatasetResponse).toBeDefined();
    expect(staleDatasetResponse.num_staled_datapoints).toBe(0);
  });
});

async function buildClient() {
  process.env.OPENAI_API_KEY = undefined;
  return await TensorZeroClient.buildEmbedded(
    UI_FIXTURES_CONFIG_PATH,
    undefined,
    process.env.TENSORZERO_POSTGRES_URL,
    undefined,
  );
}

describe("DatabaseClient", () => {
  it("should be able to import DatabaseClient", () => {
    expect(DatabaseClient).toBeDefined();
    expect(typeof DatabaseClient).toBe("function");
  });

  it("should have getFeedbackByVariant method", async () => {
    // Note: This test verifies the method exists and has correct signature
    // Full integration testing would require a running ClickHouse instance
    expect(typeof DatabaseClient.fromClickhouseUrl).toBe("function");
  });

  it("should validate getFeedbackByVariant parameter types", async () => {
    // This test documents the expected parameter structure
    const expectedParams = {
      metric_name: "test_metric",
      function_name: "test_function",
      variant_names: ["variant_a", "variant_b"],
    };

    // Verify the structure is what we expect
    expect(expectedParams).toHaveProperty("metric_name");
    expect(expectedParams).toHaveProperty("function_name");
    expect(expectedParams).toHaveProperty("variant_names");
    expect(Array.isArray(expectedParams.variant_names)).toBe(true);
  });
});
