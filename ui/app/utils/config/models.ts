import { stringify } from "smol-toml";
import type { OptimizerOutput } from "~/types/tensorzero";

export function dump_optimizer_output(optimizerOutput: OptimizerOutput) {
  if (optimizerOutput.type !== "model") {
    throw new Error(
      `Only model type is supported, got ${optimizerOutput.type}`,
    );
  }
  const modelConfig = optimizerOutput.content;
  if (modelConfig.routing.length !== 1) {
    throw new Error(
      `Expected 1 routing entry, got ${modelConfig.routing.length}`,
    );
  }
  const modelName = modelConfig.routing[0];
  const providerConfig = modelConfig.providers[modelName];
  // drop the timeout config
  // allow it to be unused
  if (!providerConfig) {
    throw new Error(
      `Provider config not found for model ${modelName} when dumping optimizer output.`,
    );
  }

  const { timeouts, ...restProviderConfig } = providerConfig;
  const fullyQualifiedProviderConfig = {
    models: {
      [modelName]: {
        routing: [modelName],
        providers: {
          [modelName]: restProviderConfig,
        },
      },
    },
  };
  const rawSerializedModelConfig = stringify(fullyQualifiedProviderConfig);
  // Remove the bare `[models]` section header if present (smol-toml <1.6.0),
  // and trim trailing whitespace.
  const lines = rawSerializedModelConfig.split("\n");
  const filtered = lines.filter((line) => line.trim() !== "[models]");
  // Remove leading/trailing empty lines
  while (filtered.length > 0 && filtered[0].trim() === "") filtered.shift();
  while (filtered.length > 0 && filtered[filtered.length - 1].trim() === "")
    filtered.pop();
  return filtered.join("\n");
}
