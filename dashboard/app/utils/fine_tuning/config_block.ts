import {
  type ModelConfig,
  type ProviderConfig,
  type ProviderType,
  createProviderConfig,
} from "../config/models";

export type FullyQualifiedModelConfig = {
  models: {
    [key: string]: ModelConfig;
  };
};

export function get_fine_tuned_model_config(
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
  const fullyQualifiedModelConfig: FullyQualifiedModelConfig = {
    models: {
      [model_name]: modelConfig,
    },
  };
  return fullyQualifiedModelConfig;
}
