import { z } from "zod";
import { contentBlockOutputSchema, jsonInferenceOutputSchema } from "./common";
import { inputSchema } from "./common";

export const EvaluationRunInfoSchema = z.object({
  eval_run_id: z.string(),
  variant_name: z.string(),
});

export type EvaluationRunInfo = z.infer<typeof EvaluationRunInfoSchema>;

export const EvaluationResultSchema = z.object({
  datapoint_id: z.string().uuid(),
  eval_run_id: z.string().uuid(),
  input: z.string(),
  generated_output: z.string(),
  reference_output: z.string(),
  metric_name: z.string(),
  metric_value: z.string(),
});

export type EvaluationResult = z.infer<typeof EvaluationResultSchema>;

export const JsonEvaluationResultSchema = z.object({
  datapoint_id: z.string().uuid(),
  eval_run_id: z.string().uuid(),
  input: inputSchema,
  generated_output: jsonInferenceOutputSchema,
  reference_output: jsonInferenceOutputSchema,
  metric_name: z.string(),
  metric_value: z.string(),
});

export type JsonEvaluationResult = z.infer<typeof JsonEvaluationResultSchema>;

export const ChatEvaluationResultSchema = z.object({
  datapoint_id: z.string().uuid(),
  eval_run_id: z.string().uuid(),
  input: inputSchema,
  generated_output: z.array(contentBlockOutputSchema),
  reference_output: z.array(contentBlockOutputSchema),
  metric_name: z.string(),
  metric_value: z.string(),
});

export type ChatEvaluationResult = z.infer<typeof ChatEvaluationResultSchema>;

export const ParsedEvaluationResultSchema = z.union([
  JsonEvaluationResultSchema,
  ChatEvaluationResultSchema,
]);

export type ParsedEvaluationResult = z.infer<
  typeof ParsedEvaluationResultSchema
>;

export function parseEvaluationResult(
  result: EvaluationResult,
): ParsedEvaluationResult {
  try {
    // Parse the input field
    const parsedInput = inputSchema.parse(JSON.parse(result.input));

    // Parse the outputs
    const generatedOutput = JSON.parse(result.generated_output);
    const referenceOutput = JSON.parse(result.reference_output);

    // Determine if this is a chat result by checking if generated_output is an array
    if (Array.isArray(generatedOutput)) {
      // This is likely a chat evaluation result
      return ChatEvaluationResultSchema.parse({
        ...result,
        input: parsedInput,
        generated_output: generatedOutput,
        reference_output: referenceOutput,
      });
    } else {
      // This is likely a JSON evaluation result
      return JsonEvaluationResultSchema.parse({
        ...result,
        input: parsedInput,
        generated_output: generatedOutput,
        reference_output: referenceOutput,
      });
    }
  } catch (error) {
    console.warn(
      "Failed to parse evaluation result using structure-based detection:",
      error,
    );
    // If structure-based detection fails, try the original parsing as fallback
    return ParsedEvaluationResultSchema.parse(result);
  }
}

export const EvaluationStatisticsSchema = z.object({
  eval_run_id: z.string(),
  metric_name: z.string(),
  datapoint_count: z.number(),
  mean_metric: z.number(),
  stderr_metric: z.number().nullable(),
});

export type EvaluationStatistics = z.infer<typeof EvaluationStatisticsSchema>;

export function getEvaluatorMetricName(
  evalName: string,
  evaluatorName: string,
): string {
  return `tensorzero::eval_name::${evalName}::evaluator_name::${evaluatorName}`;
}

export const evalInfoResultSchema = z.object({
  eval_run_id: z.string().uuid(),
  eval_name: z.string(),
  function_name: z.string(),
  variant_name: z.string(),
  last_inference_timestamp: z.string().datetime(),
});

export type EvalInfoResult = z.infer<typeof evalInfoResultSchema>;

export const EvalRunInfoSchema = z.object({
  eval_run_id: z.string().uuid(),
  eval_name: z.string(),
  function_name: z.string(),
  variant_name: z.string(),
  last_inference_timestamp: z.string().datetime(),
  dataset: z.string(),
});

export type EvalRunInfo = z.infer<typeof EvalRunInfoSchema>;
