import {
  ModelConfig,
  ProviderConfig,
  ProviderType,
  createProviderConfig,
} from "../config/models";
import { stringify } from "smol-toml";
import { VariantConfig } from "../config/variant";

export async function get_fine_tuned_model_config(
  model_name: string,
  model_provider_type: ProviderType,
) {
  const providerConfig: ProviderConfig = createProviderConfig(
    model_provider_type,
    model_name,
  );
  const modelConfig: ModelConfig = {
    routing: [model_name],
    providers: {
      [model_name]: providerConfig,
    },
  };
  return modelConfig;
}

export function dump_model_config(model_config: ModelConfig) {
  return stringify(model_config);
}

export function create_dump_variant_config(
  oldVariant: VariantConfig,
  model_name: string,
  function_name: string,
) {
  const newVariantConfig = {
    ...oldVariant,
    weight: 0,
    model_name: model_name,
  };
  const fullNewVariantConfig = {
    functions: {
      [function_name]: {
        variants: {
          [model_name]: newVariantConfig,
        },
      },
    },
  };
  return stringify(fullNewVariantConfig);
}
