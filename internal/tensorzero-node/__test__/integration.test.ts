import { describe, it, expect } from "vitest";
import { TensorZeroClient } from "../dist/index.js";

describe("TensorZeroClient Integration Tests", () => {
  it("should be able to import TensorZeroClient", () => {
    expect(TensorZeroClient).toBeDefined();
    expect(typeof TensorZeroClient).toBe("function");
  });

  it("client creation throws on a bad config path", async () => {
    // This should throw an error that contains "Failed to parse config: Internal TensorZero Error: Config file not found: "foo""
    expect(async () => await TensorZeroClient.build("foo")).rejects.toThrow(
      'Failed to parse config: Internal TensorZero Error: Config file not found: "foo"',
    );
  });

  it("should have required methods", async () => {
    const client = await TensorZeroClient.build(
      "../../examples/quickstart/config/tensorzero.toml",
    );
    expect(typeof client.experimentalStartOptimization).toBe("function");
    expect(typeof client.experimentalPollOptimization).toBe("function");
  });
});
