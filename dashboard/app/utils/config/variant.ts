import { z } from "zod";
import { jsonModeSchema, retryConfigSchema } from "./types";
import { create_env } from "../minijinja/pkg/minijinja_bindings";
import { stringify } from "smol-toml";
import { promises as fs } from "fs";
import path from "path";

// Runtime-only type for internal use
export interface TemplateWithContent {
  path: string;
  content?: string;
}

const BaseChatCompletionConfig = z.object({
  weight: z.number().default(0),
  model: z.string(),
  system_template: z.string().optional(),
  user_template: z.string().optional(),
  assistant_template: z.string().optional(),
  temperature: z.number().optional(),
  top_p: z.number().optional(),
  max_tokens: z.number().int().optional(),
  presence_penalty: z.number().optional(),
  frequency_penalty: z.number().optional(),
  seed: z.number().int().optional(),
  json_mode: jsonModeSchema.default("on"),
  retries: retryConfigSchema.default({ num_retries: 0, max_delay_s: 10 }),
});

export const RawChatCompletionConfig = BaseChatCompletionConfig.extend({
  type: z.literal("chat_completion"),
}).partial({ retries: true, weight: true });

export type RawChatCompletionConfig = z.infer<typeof RawChatCompletionConfig>;

export const EvaluatorConfig = z.object({
  ...BaseChatCompletionConfig.shape,
});

export type EvaluatorConfig = z.infer<typeof EvaluatorConfig>;

export const BestOfNSamplingConfig = z.object({
  type: z.literal("experimental_best_of_n_sampling"),
  weight: z.number().default(0),
  timeout_s: z.number().default(300),
  candidates: z.array(z.string()),
  evaluator: EvaluatorConfig,
});

export type BestOfNSamplingConfig = z.infer<typeof BestOfNSamplingConfig>;

export const DiclConfig = z.object({
  type: z.literal("experimental_dynamic_in_context_learning"),
  weight: z.number().default(0),
  embedding_model: z.string(),
  k: z.number().int(), // k as in k-nearest neighbors
  model: z.string(),
  system_instructions: z.string().optional(), // should be the path here not the actual system instructions
  temperature: z.number().optional(),
  top_p: z.number().optional(),
  presence_penalty: z.number().optional(),
  frequency_penalty: z.number().optional(),
  max_tokens: z.number().int().optional(),
  seed: z.number().int().optional(),
  json_mode: jsonModeSchema.default("on"),
  retries: retryConfigSchema
    .optional()
    .default({ num_retries: 0, max_delay_s: 10 }),
});

export type DiclConfig = z.infer<typeof DiclConfig>;

export const FuserConfig = z.object({
  ...BaseChatCompletionConfig.shape,
});

export type FuserConfig = z.infer<typeof FuserConfig>;

export const MixtureOfNConfig = z.object({
  type: z.literal("experimental_mixture_of_n"),
  weight: z.number().default(0),
  timeout_s: z.number().default(300),
  candidates: z.array(z.string()),
  fuser: FuserConfig,
});

export type MixtureOfNConfig = z.infer<typeof MixtureOfNConfig>;

// Raw variant config using basic string templates
export const RawVariantConfig = z
  .discriminatedUnion("type", [
    RawChatCompletionConfig,
    BestOfNSamplingConfig,
    DiclConfig,
    MixtureOfNConfig,
  ])
  .transform((raw) => ({
    ...raw,
    load: async function (config_path: string): Promise<VariantConfig> {
      return await convertRawVariantConfig(raw, config_path);
    },
  }));

export type RawVariantConfig = z.infer<typeof RawVariantConfig>;

// Extend the inferred type to include template content
export const ChatCompletionConfigSchema = BaseChatCompletionConfig.extend({
  type: z.literal("chat_completion"),
  system_template: z.custom<TemplateWithContent>().optional(),
  user_template: z.custom<TemplateWithContent>().optional(),
  assistant_template: z.custom<TemplateWithContent>().optional(),
}).partial({ retries: true, weight: true });

export type ChatCompletionConfig = z.infer<typeof ChatCompletionConfigSchema>;

// Variant config using template content
export const VariantConfig = z.discriminatedUnion("type", [
  ChatCompletionConfigSchema,
  BestOfNSamplingConfig,
  DiclConfig,
  MixtureOfNConfig,
]);

export type VariantConfig = z.infer<typeof VariantConfig>;

export async function get_template_env(variant: ChatCompletionConfig) {
  const env: {
    system?: string;
    user?: string;
    assistant?: string;
  } = {};

  if ("system_template" in variant && variant.system_template) {
    env.system =
      typeof variant.system_template === "string"
        ? variant.system_template
        : (variant.system_template.content ?? variant.system_template.path);
  }

  if ("user_template" in variant && variant.user_template) {
    env.user =
      typeof variant.user_template === "string"
        ? variant.user_template
        : (variant.user_template.content ?? variant.user_template.path);
  }

  if ("assistant_template" in variant && variant.assistant_template) {
    env.assistant =
      typeof variant.assistant_template === "string"
        ? variant.assistant_template
        : (variant.assistant_template.content ??
          variant.assistant_template.path);
  }

  return await create_env(env);
}

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

export async function convertRawVariantConfig(
  variantConfig: Omit<RawVariantConfig, "load">,
  configPath: string,
): Promise<VariantConfig> {
  if (variantConfig.type === "chat_completion") {
    const rawChatCompletionConfig = variantConfig as RawChatCompletionConfig;
    const templatedVariant: ChatCompletionConfig = {
      ...rawChatCompletionConfig,
      type: "chat_completion",
      system_template: rawChatCompletionConfig.system_template
        ? {
            path: rawChatCompletionConfig.system_template,
            content: await fs.readFile(
              path.join(
                path.dirname(configPath),
                rawChatCompletionConfig.system_template,
              ),
              "utf-8",
            ),
          }
        : undefined,
      user_template: rawChatCompletionConfig.user_template
        ? {
            path: rawChatCompletionConfig.user_template,
            content: await fs.readFile(
              path.join(
                path.dirname(configPath),
                rawChatCompletionConfig.user_template,
              ),
              "utf-8",
            ),
          }
        : undefined,
      assistant_template: rawChatCompletionConfig.assistant_template
        ? {
            path: rawChatCompletionConfig.assistant_template,
            content: await fs.readFile(
              path.join(
                path.dirname(configPath),
                rawChatCompletionConfig.assistant_template,
              ),
              "utf-8",
            ),
          }
        : undefined,
    };
    return templatedVariant;
  }

  // Other variant types don't need conversion
  return variantConfig as VariantConfig;
}
