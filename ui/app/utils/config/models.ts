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
