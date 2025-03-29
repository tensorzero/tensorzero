import { z } from "zod";
import {
  contentBlockOutputSchema,
  jsonInferenceOutputSchema,
  resolvedInputSchema,
} from "./common";
import { inputSchema } from "./common";
import { resolveInput } from "../resolve.server";

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

export const EvaluationResultWithVariantSchema = EvaluationResultSchema.extend({
  variant_name: z.string(),
});

export type EvaluationResultWithVariant = z.infer<
  typeof EvaluationResultWithVariantSchema
>;

export const JsonEvaluationResultSchema = z.object({
  datapoint_id: z.string().uuid(),
  eval_run_id: z.string().uuid(),
  input: resolvedInputSchema,
  generated_output: jsonInferenceOutputSchema,
  reference_output: jsonInferenceOutputSchema,
  metric_name: z.string(),
  metric_value: z.string(),
});

export type JsonEvaluationResult = z.infer<typeof JsonEvaluationResultSchema>;

export const ChatEvaluationResultSchema = z.object({
  datapoint_id: z.string().uuid(),
  eval_run_id: z.string().uuid(),
  input: resolvedInputSchema,
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

export const JsonEvaluationResultWithVariantSchema =
  JsonEvaluationResultSchema.extend({
    variant_name: z.string(),
  });

export type JsonEvaluationResultWithVariant = z.infer<
  typeof JsonEvaluationResultWithVariantSchema
>;

export const ChatEvaluationResultWithVariantSchema =
  ChatEvaluationResultSchema.extend({
    variant_name: z.string(),
  });

export type ChatEvaluationResultWithVariant = z.infer<
  typeof ChatEvaluationResultWithVariantSchema
>;

export const ParsedEvaluationResultWithVariantSchema = z.union([
  JsonEvaluationResultWithVariantSchema,
  ChatEvaluationResultWithVariantSchema,
]);

export type ParsedEvaluationResultWithVariant = z.infer<
  typeof ParsedEvaluationResultWithVariantSchema
>;

export async function parseEvaluationResult(
  result: EvaluationResult,
): Promise<ParsedEvaluationResult> {
  try {
    // Parse the input field
    const parsedInput = inputSchema.parse(JSON.parse(result.input));
    const resolvedInput = await resolveInput(parsedInput);

    // Parse the outputs
    const generatedOutput = JSON.parse(result.generated_output);
    const referenceOutput = JSON.parse(result.reference_output);

    // Determine if this is a chat result by checking if generated_output is an array
    if (Array.isArray(generatedOutput)) {
      // This is likely a chat evaluation result
      return ChatEvaluationResultSchema.parse({
        ...result,
        input: resolvedInput,
        generated_output: generatedOutput,
        reference_output: referenceOutput,
      });
    } else {
      // This is likely a JSON evaluation result
      return JsonEvaluationResultSchema.parse({
        ...result,
        input: resolvedInput,
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

export async function parseEvaluationResultWithVariant(
  result: EvaluationResultWithVariant,
): Promise<ParsedEvaluationResultWithVariant> {
  try {
    // Parse using the same logic as parseEvaluationResult
    const parsedResult = await parseEvaluationResult(result);

    // Add the variant_name to the parsed result
    const parsedResultWithVariant = {
      ...parsedResult,
      variant_name: result.variant_name,
    };
    return ParsedEvaluationResultWithVariantSchema.parse(
      parsedResultWithVariant,
    );
  } catch (error) {
    console.warn(
      "Failed to parse evaluation result with variant using structure-based detection:",
      error,
    );
    // Fallback to direct parsing if needed
    return ParsedEvaluationResultWithVariantSchema.parse({
      ...result,
      input: result.input,
      generated_output: result.generated_output,
      reference_output: result.reference_output,
    });
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

function getEvaluatorNameFromMetricName(metricName: string): string {
  const parts = metricName.split("::");
  return parts[parts.length - 1];
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

// Define a type for consolidated metrics
export type ConsolidatedMetric = {
  metric_name: string;
  metric_value: string;
  evaluator_name: string;
};

// Define a type for consolidated evaluation results
export type ConsolidatedEvaluationResult = Omit<
  ParsedEvaluationResultWithVariant,
  "metric_name" | "metric_value"
> & {
  metrics: ConsolidatedMetric[];
};

export const consolidate_eval_results = (
  eval_results: ParsedEvaluationResultWithVariant[],
): ConsolidatedEvaluationResult[] => {
  // Create a map to store results by datapoint_id and eval_run_id
  const resultMap = new Map<string, ConsolidatedEvaluationResult>();

  // Process each evaluation result
  for (const result of eval_results) {
    const key = `${result.datapoint_id}:${result.eval_run_id}:${result.variant_name}`;

    if (!resultMap.has(key)) {
      // Create a new consolidated result without metric_name and metric_value
      const { metric_name, metric_value, ...baseResult } = result;

      resultMap.set(key, {
        ...baseResult,
        metrics: [
          {
            metric_name,
            metric_value,
            evaluator_name: getEvaluatorNameFromMetricName(metric_name),
          },
        ],
      });
    } else {
      // Add this metric to the existing result
      const existingResult = resultMap.get(key)!;
      existingResult.metrics.push({
        metric_name: result.metric_name,
        metric_value: result.metric_value,
        evaluator_name: getEvaluatorNameFromMetricName(result.metric_name),
      });
    }
  }

  // Convert the map values to an array and return
  return Array.from(resultMap.values());
};
