import { TensorZeroClient } from "tensorzero-node";
import { describe, it } from "vitest";
import { createFilters } from "./client";

const gatewayUrl = process.env.TENSORZERO_GATEWAY_URL;
if (!gatewayUrl) {
  throw new Error("TENSORZERO_GATEWAY_URL is not set");
}
const client = await TensorZeroClient.buildHttp(gatewayUrl);

// In CI, the gateway runs inside Docker and needs to reach the mock-inference-provider
// via Docker networking. Locally, the gateway can reach localhost:3030.
const openaiBaseUrl =
  process.env.OPENAI_BASE_URL ||
  (process.env.CI
    ? "http://mock-inference-provider:3030/openai"
    : "http://localhost:3030/openai");

describe("native sft", () => {
  // NOTE: This test hits a fake server so you can run it anytime without paying OpenAI
  it("should launch a job and poll it", async () => {
    const metric = "exact_match";
    const threshold = 0.9; // irrelevant
    const filters = await createFilters(metric, threshold);
    const job = await client.experimentalLaunchOptimizationWorkflow({
      function_name: "extract_entities",
      template_variant_name: "baseline",
      query_variant_name: null,
      filters: filters,
      output_source: "inference",
      limit: 100,
      offset: 0,
      val_fraction: 0.1,
      optimizer_config: {
        type: "openai_sft",
        model: "gpt-4o-mini",
        batch_size: 1,
        learning_rate_multiplier: 1,
        n_epochs: 1,
        credentials: null,
        api_base: openaiBaseUrl,
        seed: null,
        suffix: null,
      },
      order_by: null,
    });

    let status = await client.experimentalPollOptimization(job);
    while (status.status !== "completed") {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      status = await client.experimentalPollOptimization(job);
    }
  }, 10000); // timeout in ms
});
