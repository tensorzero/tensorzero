import { z } from "zod";
import { jsonModeSchema, RetryConfigSchema } from "./types";

export const ExactMatchConfigSchema = z.object({
  cutoff: z.number().optional(),
});
export type ExactMatchConfig = z.infer<typeof ExactMatchConfigSchema>;

export const UninitializedLLMJudgeChatCompletionVariantConfigSchema = z.object({
  active: z.boolean().default(false),
  model: z.string(),
  system_instructions: z.string(), // Path to system instructions
  temperature: z.number().optional(),
  top_p: z.number().optional(),
  max_tokens: z.number().int().optional(),
  presence_penalty: z.number().optional(),
  frequency_penalty: z.number().optional(),
  seed: z.number().int().optional(),
  json_mode: jsonModeSchema,
  retries: RetryConfigSchema.default({ num_retries: 0, max_delay_s: 10 }),
});
export type UninitializedLLMJudgeChatCompletionVariantConfig = z.infer<
  typeof UninitializedLLMJudgeChatCompletionVariantConfigSchema
>;

export const UnintializedLLMJudgeVariantConfigSchema = z.discriminatedUnion(
  "type",
  [
    z.object({
      type: z.literal("chat_completion"),
      ...UninitializedLLMJudgeChatCompletionVariantConfigSchema.shape,
    }),
  ],
);
export type UninitializedLLMJudgeVariantConfig = z.infer<
  typeof UnintializedLLMJudgeVariantConfigSchema
>;

export const LLMJudgeIncludeConfigSchema = z
  .object({
    reference_output: z.boolean().default(false),
  })
  .default({ reference_output: false });
export type LLMJudgeIncludeConfig = z.infer<typeof LLMJudgeIncludeConfigSchema>;

export const UninitializedLLMJudgeConfigSchema = z.object({
  variants: z.record(z.string(), UnintializedLLMJudgeVariantConfigSchema),
  output_type: z.enum(["float", "boolean"]),
  optimize: z.enum(["min", "max"]),
  include: LLMJudgeIncludeConfigSchema,
  cutoff: z.number().optional(),
});
export type UninitializedLLMJudgeConfig = z.infer<
  typeof UninitializedLLMJudgeConfigSchema
>;
export const UninitializedEvaluatorConfigSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("exact_match"),
    ...ExactMatchConfigSchema.shape,
  }),
  z.object({
    type: z.literal("llm_judge"),
    ...UninitializedLLMJudgeConfigSchema.shape,
  }),
]);
export type UninitializedEvaluatorConfig = z.infer<
  typeof UninitializedEvaluatorConfigSchema
>;

export const UninitializedEvalConfigSchema = z.object({
  evaluators: z.record(z.string(), UninitializedEvaluatorConfigSchema),
  dataset_name: z.string(),
  function_name: z.string(),
});
export type UninitializedEvalConfig = z.infer<
  typeof UninitializedEvalConfigSchema
>;

export const EvaluatorConfigSchema = z.object({
  type: z.literal("chat"),
  model: z.string(),
  system_template: z.string().optional(),
  user_template: z.string().optional(),
});
export type EvaluatorConfig = z.infer<typeof EvaluatorConfigSchema>;

export const EvalConfigSchema = z.object({
  evaluators: z.record(z.string(), EvaluatorConfigSchema),
  dataset_name: z.string(),
  function_name: z.string(),
});
export type EvalConfig = z.infer<typeof EvalConfigSchema>;
