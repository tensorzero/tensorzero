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
  console.log(JSON.stringify(providerConfig, null, 2));
  const fullyQualifiedProviderConfig = {
    models: {
      [modelName]: {
        routing: [modelName],
        providers: {
          [modelName]: providerConfig,
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
