import { stringify } from "smol-toml";
import type { ProviderConfig } from "tensorzero-node";

type ProviderType = ProviderConfig["type"];

function createProviderConfig(
  type: ProviderType,
  model_name: string,
): ProviderConfig {
  switch (type) {
    case "fireworks":
      return { type, model_name, parse_think_blocks: false };
    case "openai":
      return { type, model_name, api_base: null };
    default:
      throw new Error(`Provider ${type} requires additional configuration`);
  }
}

export function get_fine_tuned_provider_config(
  model_name: string,
  model_provider_type: ProviderType,
) {
  const providerConfig = createProviderConfig(model_provider_type, model_name);
  return providerConfig;
}

export function dump_provider_config(
  modelName: string,
  providerConfig: ProviderConfig,
) {
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
