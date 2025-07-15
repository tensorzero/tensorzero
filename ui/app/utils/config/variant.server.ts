import { stringify } from "smol-toml";
import type { ChatCompletionConfig } from "tensorzero-node";

export function create_dump_variant_config(
  oldVariant: ChatCompletionConfig,
  model_name: string,
  function_name: string,
) {
  // Convert back to ChatCompletionConfig
  const variantConfig = {
    ...oldVariant,
    weight: 0,
    model: model_name,
    system_template:
      typeof oldVariant.system_template === "string"
        ? oldVariant.system_template
        : oldVariant.system_template?.path,
    user_template:
      typeof oldVariant.user_template === "string"
        ? oldVariant.user_template
        : oldVariant.user_template?.path,
    assistant_template:
      typeof oldVariant.assistant_template === "string"
        ? oldVariant.assistant_template
        : oldVariant.assistant_template?.path,
  };

  const fullNewVariantConfig = {
    functions: {
      [function_name]: {
        variants: {
          [model_name]: variantConfig,
        },
      },
    },
  };

  return stringify(fullNewVariantConfig);
}
