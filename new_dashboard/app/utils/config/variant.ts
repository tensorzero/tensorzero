import { z } from "zod";
import { jsonModeSchema, RetryConfigSchema } from "./types";
import { create_env } from "../minijinja/pkg/minijinja_bindings";
import { stringify } from "smol-toml";

// Runtime-only type for internal use
export interface TemplateWithContent {
  path: string;
  content?: string;
}

const BaseChatCompletionConfigSchema = z.object({
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
  retries: RetryConfigSchema.default({ num_retries: 0, max_delay_s: 10 }),
});

export const RawChatCompletionConfigSchema =
  BaseChatCompletionConfigSchema.extend({
    type: z.literal("chat_completion"),
  }).partial({ retries: true, weight: true });

export type RawChatCompletionConfig = z.infer<
  typeof RawChatCompletionConfigSchema
>;

export const EvaluatorConfigSchema = z.object({
  ...BaseChatCompletionConfigSchema.shape,
});

export type EvaluatorConfig = z.infer<typeof EvaluatorConfigSchema>;

export const BestOfNSamplingConfigSchema = z.object({
  type: z.literal("experimental_best_of_n_sampling"),
  weight: z.number().default(0),
  timeout_s: z.number().default(300),
  candidates: z.array(z.string()),
  evaluator: EvaluatorConfigSchema,
});

export type BestOfNSamplingConfig = z.infer<typeof BestOfNSamplingConfigSchema>;

export const DiclConfigSchema = z.object({
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
  retries: RetryConfigSchema.optional().default({
    num_retries: 0,
    max_delay_s: 10,
  }),
});

export type DiclConfig = z.infer<typeof DiclConfigSchema>;

export const FuserConfigSchema = z.object({
  ...BaseChatCompletionConfigSchema.shape,
});

export type FuserConfig = z.infer<typeof FuserConfigSchema>;

export const MixtureOfNConfigSchema = z.object({
  type: z.literal("experimental_mixture_of_n"),
  weight: z.number().default(0),
  timeout_s: z.number().default(300),
  candidates: z.array(z.string()),
  fuser: FuserConfigSchema,
});

export type MixtureOfNConfig = z.infer<typeof MixtureOfNConfigSchema>;

// Raw variant config using basic string templates
export const RawVariantConfigSchema = z
  .discriminatedUnion("type", [
    RawChatCompletionConfigSchema,
    BestOfNSamplingConfigSchema,
    DiclConfigSchema,
    MixtureOfNConfigSchema,
  ])
  .transform((raw) => ({
    ...raw,
    load: async function (config_path: string): Promise<VariantConfig> {
      return await convertRawVariantConfig(raw, config_path);
    },
  }));

export type RawVariantConfig = z.infer<typeof RawVariantConfigSchema>;

// Extend the inferred type to include template content
export const ChatCompletionConfigSchema = BaseChatCompletionConfigSchema.extend(
  {
    type: z.literal("chat_completion"),
    system_template: z.custom<TemplateWithContent>().optional(),
    user_template: z.custom<TemplateWithContent>().optional(),
    assistant_template: z.custom<TemplateWithContent>().optional(),
  },
).partial({ retries: true, weight: true });

export type ChatCompletionConfig = z.infer<typeof ChatCompletionConfigSchema>;

// Variant config using template content
export const VariantConfigSchema = z.discriminatedUnion("type", [
  ChatCompletionConfigSchema,
  BestOfNSamplingConfigSchema,
  DiclConfigSchema,
  MixtureOfNConfigSchema,
]);

export type VariantConfig = z.infer<typeof VariantConfigSchema>;

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

const isServer = typeof window === "undefined";

async function loadTemplateContent(
  templatePath: string,
  configPath: string,
): Promise<string> {
  if (isServer) {
    // Dynamic import to prevent bundling fs module
    const { promises: fs } = await import("fs");
    const { default: path } = await import("path");
    return await fs.readFile(
      path.join(path.dirname(configPath), templatePath),
      "utf-8",
    );
  }
  return ""; // or return undefined/throw error for client
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

  // Other variant types don't need conversion
  return variantConfig as VariantConfig;
}
