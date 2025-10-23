import { TensorZeroClient } from "tensorzero-node";
import { describe, it } from "vitest";
import { createFilters } from "./client";

const configPath = process.env.TENSORZERO_UI_CONFIG_PATH;
if (!configPath) {
  throw new Error("TENSORZERO_UI_CONFIG_PATH is not set");
}
const clickhouseUrl = process.env.TENSORZERO_CLICKHOUSE_URL;
if (!clickhouseUrl) {
  throw new Error("TENSORZERO_CLICKHOUSE_URL is not set");
}
const postgresUrl = process.env.TENSORZERO_POSTGRES_URL;
if (!postgresUrl) {
  throw new Error("TENSORZERO_POSTGRES_URL is not set");
}
const client = await TensorZeroClient.buildEmbedded(
  configPath,
  clickhouseUrl,
  postgresUrl,
);

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
      output_source: "Inference",
      limit: BigInt(100),
      offset: BigInt(0),
      val_fraction: 0.1,
      format: "JsonEachRow",
      optimizer_config: {
        type: "openai_sft",
        model: "gpt-4o-mini",
        batch_size: 1,
        learning_rate_multiplier: 1,
        n_epochs: 1,
        credentials: null,
        api_base: process.env.OPENAI_BASE_URL || "http://localhost:3030/openai",
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
