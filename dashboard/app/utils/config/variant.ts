import { z } from "zod";
import { jsonModeSchema, retryConfigSchema } from "./types";
import { create_env } from "../minijinja/pkg/minijinja_bindings";

const BaseChatCompletionConfig = z.object({
  weight: z.number().default(0),
  model: z.string(),
  system_template: z.string().optional(), // should be the path here not the actual system template
  user_template: z.string().optional(), // should be the path here not the actual user template
  assistant_template: z.string().optional(), // should be the path here not the actual assistant template
  temperature: z.number().optional(),
  top_p: z.number().optional(),
  max_tokens: z.number().int().optional(),
  presence_penalty: z.number().optional(),
  frequency_penalty: z.number().optional(),
  seed: z.number().int().optional(),
  json_mode: jsonModeSchema.default("on"),
  retries: retryConfigSchema.default({ num_retries: 0, max_delay_s: 10 }),
});

export const ChatCompletionConfig = BaseChatCompletionConfig.extend({
  type: z.literal("chat_completion"),
}).partial({ retries: true, weight: true });

export type ChatCompletionConfig = z.infer<typeof ChatCompletionConfig>;

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

export const VariantConfig = z.discriminatedUnion("type", [
  ChatCompletionConfig,
  BestOfNSamplingConfig,
  DiclConfig,
  MixtureOfNConfig,
]);

export type VariantConfig = z.infer<typeof VariantConfig>;

export async function get_template_env(variant: VariantConfig) {
  const env: {
    system?: string;
    user?: string;
    assistant?: string;
  } = {};

  if ("system_template" in variant && variant.system_template) {
    env.system = variant.system_template;
  }

  if ("user_template" in variant && variant.user_template) {
    env.user = variant.user_template;
  }

  if ("assistant_template" in variant && variant.assistant_template) {
    env.assistant = variant.assistant_template;
  }

  return await create_env(env);
}
