import { z } from "zod";
import { promises as fs } from "fs";
import path from "path";
import {
  BaseChatCompletionConfigSchema,
  BestOfNSamplingConfigSchema,
  DiclConfigSchema,
  MixtureOfNConfigSchema,
  ChainOfThoughtConfigSchema,
  type RawDiclConfig,
  type ChatCompletionConfig,
  type DiclConfig,
  type VariantConfig,
  type ChainOfThoughtConfig,
} from "./variant";
import { stringify } from "smol-toml";

export const RawChatCompletionConfigSchema =
  BaseChatCompletionConfigSchema.extend({
    type: z.literal("chat_completion"),
  }).partial({ retries: true });

export type RawChatCompletionConfig = z.infer<
  typeof RawChatCompletionConfigSchema
>;

export const RawChainOfThoughtConfigSchema =
  RawChatCompletionConfigSchema.extend({
    type: z.literal("experimental_chain_of_thought"),
  });

export type RawChainOfThoughtConfig = z.infer<
  typeof RawChainOfThoughtConfigSchema
>;

export const RawDiclConfigSchema = DiclConfigSchema.extend({
  type: z.literal("experimental_dynamic_in_context_learning"),
});

// Raw variant config using basic string templates
export const RawVariantConfigSchema = z
  .discriminatedUnion("type", [
    RawChatCompletionConfigSchema,
    BestOfNSamplingConfigSchema,
    DiclConfigSchema,
    MixtureOfNConfigSchema,
    RawChainOfThoughtConfigSchema,
  ])
  .transform((raw) => ({
    ...raw,
    load: async function (config_path: string): Promise<VariantConfig> {
      return await convertRawVariantConfig(raw, config_path);
    },
  }));

export type RawVariantConfig = z.infer<typeof RawVariantConfigSchema>;

export function create_dump_variant_config(
  oldVariant: ChatCompletionConfig,
  model_name: string,
  function_name: string,
) {
  // Convert back to ChatCompletionConfig
  const variantConfig: RawChatCompletionConfig = {
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
async function loadTemplateContent(
  templatePath: string,
  configPath: string,
): Promise<string> {
  // Dynamic import to prevent bundling fs module
  return await fs.readFile(
    path.join(path.dirname(configPath), templatePath),
    "utf-8",
  );
}

async function convertRawVariantConfig(
  variantConfig: Omit<RawVariantConfig, "load">,
  configPath: string,
): Promise<VariantConfig> {
  if (variantConfig.type === "chat_completion") {
    const rawChatCompletionConfig = variantConfig as RawChatCompletionConfig;

    return loadChatCompletionConfig(rawChatCompletionConfig, configPath);
  }
  if (variantConfig.type === "experimental_dynamic_in_context_learning") {
    const rawDiclConfig = variantConfig as RawDiclConfig;
    const diclConfig: DiclConfig = {
      ...rawDiclConfig,
      system_instructions: rawDiclConfig.system_instructions
        ? {
            path: rawDiclConfig.system_instructions,
            content: await loadTemplateContent(
              rawDiclConfig.system_instructions,
              configPath,
            ),
          }
        : undefined,
    };
    return diclConfig;
  }
  if (variantConfig.type === "experimental_chain_of_thought") {
    const rawChainOfThoughtConfig = variantConfig as RawChainOfThoughtConfig;
    const chatCompletionConfig = await loadChatCompletionConfig(
      rawChainOfThoughtConfig,
      configPath,
    );
    const chainOfThoughtConfig: ChainOfThoughtConfig =
      ChainOfThoughtConfigSchema.parse({
        ...chatCompletionConfig,
        type: "experimental_chain_of_thought",
      });
    return chainOfThoughtConfig;
  }

  // Other variant types don't need conversion
  return variantConfig as VariantConfig;
}

async function loadChatCompletionConfig(
  rawChatCompletionConfig: RawChatCompletionConfig | RawChainOfThoughtConfig,
  configPath: string,
): Promise<ChatCompletionConfig> {
  const templatedVariant: ChatCompletionConfig = {
    ...rawChatCompletionConfig,
    type: "chat_completion",
    system_template: rawChatCompletionConfig.system_template
      ? {
          path: rawChatCompletionConfig.system_template,
          content: await loadTemplateContent(
            rawChatCompletionConfig.system_template,
            configPath,
          ),
        }
      : undefined,
    user_template: rawChatCompletionConfig.user_template
      ? {
          path: rawChatCompletionConfig.user_template,
          content: await loadTemplateContent(
            rawChatCompletionConfig.user_template,
            configPath,
          ),
        }
      : undefined,
    assistant_template: rawChatCompletionConfig.assistant_template
      ? {
          path: rawChatCompletionConfig.assistant_template,
          content: await loadTemplateContent(
            rawChatCompletionConfig.assistant_template,
            configPath,
          ),
        }
      : undefined,
  };
  return templatedVariant;
}
