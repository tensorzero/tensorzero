import { stringify } from "smol-toml";
import type { OptimizerOutput } from "tensorzero-node";

export function dump_optimizer_output(optimizerOutput: OptimizerOutput) {
  /// Drop type key from the optimizer output
  const { type, ...rest } = optimizerOutput;
  if (type !== "model") {
    throw new Error(`Only model type is supported, got ${type}`);
  }
  if (rest.routing.length !== 1) {
    throw new Error(`Expected 1 routing entry, got ${rest.routing.length}`);
  }
  const modelName = rest.routing[0];
  const providerConfig = rest.providers[modelName];
  // drop the timeout config
  // allow it to be unused
  if (!providerConfig) {
    throw new Error(
      `Provider config not found for model ${modelName} when dumping optimizer output.`,
    );
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
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
  const lines = rawSerializedModelConfig.split("\n");
  const linesWithoutFirst = lines.slice(1);
  linesWithoutFirst.splice(3, 1);
  const trimmedSerializedModelConfig = linesWithoutFirst.join("\n");
  return trimmedSerializedModelConfig;
}
