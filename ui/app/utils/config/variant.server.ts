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
      typeof oldVariant.templates.system?.template === "string"
        ? oldVariant.templates.system?.template
        : oldVariant.templates.system?.template?.path,
    user_template:
      typeof oldVariant.templates.user?.template === "string"
        ? oldVariant.templates.user?.template
        : oldVariant.templates.user?.template?.path,
    assistant_template:
      typeof oldVariant.templates.assistant?.template === "string"
        ? oldVariant.templates.assistant?.template
        : oldVariant.templates.assistant?.template?.path,
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
