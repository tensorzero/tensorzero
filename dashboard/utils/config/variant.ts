import { z } from "zod";
import { jsonModeSchema, retryConfigSchema } from "./types";

export const ChatCompletionConfig = z.object({
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

export type ChatCompletionConfig = z.infer<typeof ChatCompletionConfig>;

export const EvaluatorConfig = z.object({
  ...ChatCompletionConfig.shape,
});

export type EvaluatorConfig = z.infer<typeof EvaluatorConfig>;

export const BestOfNSamplingConfig = z.object({
  weight: z.number().default(0),
  timeout_s: z.number().default(300),
  candidates: z.array(z.string()),
  evaluator: EvaluatorConfig,
});

export type BestOfNSamplingConfig = z.infer<typeof BestOfNSamplingConfig>;

export const DiclConfig = z.object({
  weight: z.number().default(0),
  embedding_model: z.string(),
  k: z.number().int(), // k as in k-nearest neighbors
  model: z.string(),
  system_instructions: z.string(),
  temperature: z.number().optional(),
  top_p: z.number().optional(),
  presence_penalty: z.number().optional(),
  frequency_penalty: z.number().optional(),
  max_tokens: z.number().int().optional(),
  seed: z.number().int().optional(),
  json_mode: jsonModeSchema.default("on"),
  retries: retryConfigSchema.default({ num_retries: 0, max_delay_s: 10 }),
});

export type DiclConfig = z.infer<typeof DiclConfig>;

export const FuserConfig = z.object({
  ...ChatCompletionConfig.shape,
});

export type FuserConfig = z.infer<typeof FuserConfig>;

export const MixtureOfNConfig = z.object({
  weight: z.number().default(0),
  timeout_s: z.number().default(300),
  candidates: z.array(z.string()),
  fuser: FuserConfig,
});

export type MixtureOfNConfig = z.infer<typeof MixtureOfNConfig>;

export const VariantConfig = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("chat_completion"),
    config: ChatCompletionConfig,
  }),
  z.object({
    type: z.literal("experimental_best_of_n_sampling"),
    config: BestOfNSamplingConfig,
  }),
  z.object({
    type: z.literal("experimental_dynamic_in_context_learning"),
    config: DiclConfig,
  }),
  z.object({
    type: z.literal("experimental_mixture_of_n"),
    config: MixtureOfNConfig,
  }),
]);

export type VariantConfig = z.infer<typeof VariantConfig>;
