import { z } from "zod";

export const ExactMatchConfigSchema = z.object({
  cutoff: z.number().optional(),
});
export type ExactMatchConfig = z.infer<typeof ExactMatchConfigSchema>;

export const LLMJudgeIncludeConfigSchema = z
  .object({
    reference_output: z.boolean().default(false),
  })
  .default({ reference_output: false });
export type LLMJudgeIncludeConfig = z.infer<typeof LLMJudgeIncludeConfigSchema>;

export const LLMJudgeConfigSchema = z.object({
  output_type: z.enum(["float", "boolean"]),
  include: LLMJudgeIncludeConfigSchema,
  optimize: z.enum(["min", "max"]),
  cutoff: z.number().optional(),
});
export type LLMJudgeConfig = z.infer<typeof LLMJudgeConfigSchema>;

export const EvaluatorConfigSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("exact_match"),
    ...ExactMatchConfigSchema.shape,
  }),
  z.object({
    type: z.literal("llm_judge"),
    ...LLMJudgeConfigSchema.shape,
  }),
]);
export type EvaluatorConfig = z.infer<typeof EvaluatorConfigSchema>;

export const EvalConfigSchema = z.object({
  evaluators: z.record(z.string(), EvaluatorConfigSchema),
  dataset_name: z.string(),
  function_name: z.string(),
});
export type EvalConfig = z.infer<typeof EvalConfigSchema>;

export const getOptimize = (evaluatorConfig: EvaluatorConfig) => {
  switch (evaluatorConfig.type) {
    case "exact_match":
      return "max";
    case "llm_judge":
      return evaluatorConfig.optimize;
  }
};

export const getMetricType = (
  evaluatorConfig: EvaluatorConfig,
): "boolean" | "float" => {
  switch (evaluatorConfig.type) {
    case "exact_match":
      return "boolean";
    case "llm_judge":
      return evaluatorConfig.output_type;
  }
};
