import { describe, it, expect } from "vitest";
import { TensorZeroClient } from "../index.js";

async function buildClient() {
  if (!process.env.TENSORZERO_GATEWAY_URL) {
    throw new Error("TENSORZERO_GATEWAY_URL is not set");
  }
  process.env.OPENAI_API_KEY = undefined;
  return await TensorZeroClient.buildHttp(process.env.TENSORZERO_GATEWAY_URL);
}

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
});
