import { logger } from "~/utils/logger";
import type { EvaluationResultRow } from "~/types/tensorzero";

import { getConfig, getFunctionConfig } from "../config/index.server";
import { resolveInput } from "../resolve.server";
import { inputSchema } from "./common";
import {
  ChatEvaluationResultSchema,
  JsonEvaluationResultSchema,
  type ParsedEvaluationResult,
  type ParsedEvaluationResultWithVariant,
  ParsedEvaluationResultWithVariantSchema,
} from "./evaluations";
import { getTensorZeroClient } from "../tensorzero.server";

export async function parseEvaluationResult(
  result: EvaluationResultRow,
  function_name: string,
): Promise<ParsedEvaluationResult> {
  // Parse the input field
  const parsedInput = inputSchema.parse(JSON.parse(result.input));
  const config = await getConfig();
  const functionConfig = await getFunctionConfig(function_name, config);
  const resolvedInput = await resolveInput(parsedInput, functionConfig);

  // Parse the outputs
  const generatedOutput = JSON.parse(result.generated_output);
  const referenceOutput = result.reference_output
    ? JSON.parse(result.reference_output)
    : null;
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
}

async function parseEvaluationResultWithVariant(
  result: EvaluationResultRow,
  function_name: string,
): Promise<ParsedEvaluationResultWithVariant> {
  try {
    // Parse using the same logic as parseEvaluationResult
    const parsedResult = await parseEvaluationResult(result, function_name);

    // Add the variant_name to the parsed result
    const parsedResultWithVariant = {
      ...parsedResult,
      variant_name: result.variant_name,
    };
    return ParsedEvaluationResultWithVariantSchema.parse(
      parsedResultWithVariant,
    );
  } catch (error) {
    logger.warn(
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

/**
 * Gets paginated evaluation results using the TensorZero gateway API.
 *
 * @param evaluation_name The name of the evaluation.
 * @param function_name The name of the function being evaluated.
 * @param evaluation_run_ids Array of evaluation run IDs to query.
 * @param limit Maximum number of datapoints to return.
 * @param offset Offset for pagination.
 * @returns An array of parsed evaluation results.
 */
export async function getEvaluationResults(
  evaluation_name: string,
  function_name: string,
  evaluation_run_ids: string[],
  limit: number = 100,
  offset: number = 0,
): Promise<ParsedEvaluationResult[]> {
  const tensorZeroClient = getTensorZeroClient();

  const response = await tensorZeroClient.getEvaluationResults(
    evaluation_name,
    evaluation_run_ids,
    { limit, offset },
  );

  // Filter out results without metrics (from LEFT JOIN with feedback table)
  const resultsWithMetrics = response.results.filter(
    (row) => row.metric_name != null,
  );

  return Promise.all(
    resultsWithMetrics.map((row) => parseEvaluationResult(row, function_name)),
  );
}

export async function getEvaluationsForDatapoint(
  evaluation_name: string,
  datapoint_id: string,
  evaluation_run_ids: string[],
): Promise<ParsedEvaluationResultWithVariant[]> {
  const config = await getConfig();
  const evaluation_config = config.evaluations[evaluation_name];
  if (!evaluation_config) {
    throw new Error(`Evaluation ${evaluation_name} not found in config`);
  }
  const function_name = evaluation_config.function_name;
  if (!function_name) {
    throw new Error(`evaluation ${evaluation_name} not found in config`);
  }

  const tensorZeroClient = getTensorZeroClient();
  const response = await tensorZeroClient.getEvaluationResults(
    evaluation_name,
    evaluation_run_ids,
    {
      datapointId: datapoint_id,
      // Limit = u32::MAX to get all results (equivalent to before);
      // We should actually make this smaller but we will revisit these queries
      // as we migrate to Postgres.
      limit: 4294967295,
    },
  );

  // Filter out results without metrics (from LEFT JOIN with feedback table)
  const resultsWithMetrics = response.results.filter(
    (row) => row.metric_name != null,
  );

  const parsed_rows = await Promise.all(
    resultsWithMetrics.map((row) =>
      parseEvaluationResultWithVariant(row, function_name),
    ),
  );
  return parsed_rows;
}
/**
 * Polls for evaluations until a specific feedback item is found.
 * @param evaluation_name The name of the evaluation function.
 * @param datapoint_id The ID of the datapoint.
 * @param evaluation_run_ids Array of evaluation run IDs to query.
 * @param new_feedback_id The ID of the feedback item to find.
 * @param max_retries Maximum number of polling attempts.
 * @param retry_delay Delay between retries in milliseconds.
 * @returns An array of parsed evaluation results.
 */
export async function pollForEvaluations(
  evaluation_name: string,
  datapoint_id: string,
  evaluation_run_ids: string[],
  new_feedback_id: string,
  max_retries: number = 10,
  retry_delay: number = 200,
): Promise<ParsedEvaluationResultWithVariant[]> {
  let evaluations: ParsedEvaluationResultWithVariant[] = [];
  let found = false;

  for (let i = 0; i < max_retries; i++) {
    evaluations = await getEvaluationsForDatapoint(
      evaluation_name,
      datapoint_id,
      evaluation_run_ids,
    );

    if (
      evaluations.some(
        (evaluation) => evaluation.feedback_id === new_feedback_id,
      )
    ) {
      found = true;
      break;
    }

    if (i < max_retries - 1) {
      // Don't sleep after the last attempt
      await new Promise((resolve) => setTimeout(resolve, retry_delay));
    }
  }

  if (!found) {
    logger.warn(
      `Evaluation with feedback ${new_feedback_id} for datapoint ${datapoint_id} not found after ${max_retries} retries.`,
    );
  }

  return evaluations;
}

/**
 * Polls for evaluation results until they are available or max retries is reached.
 * This is useful when waiting for newly created evaluations to be available in ClickHouse.
 *
 * @param evaluation_name The name of the evaluation.
 * @param function_name The name of the function.
 * @param evaluation_run_ids Array of evaluation run IDs to query.
 * @param new_feedback_id The ID of the feedback item to find.
 * @param limit Maximum number of results to return.
 * @param offset Offset for pagination.
 * @param max_retries Maximum number of polling attempts.
 * @param retry_delay Delay between retries in milliseconds.
 * @returns An array of parsed evaluation results.
 */
export async function pollForEvaluationResults(
  evaluation_name: string,
  function_name: string,
  evaluation_run_ids: string[],
  new_feedback_id: string,
  limit: number = 100,
  offset: number = 0,
  max_retries: number = 10,
  retry_delay: number = 200,
): Promise<ParsedEvaluationResult[]> {
  let results: ParsedEvaluationResult[] = [];
  let found = false;

  for (let i = 0; i < max_retries; i++) {
    results = await getEvaluationResults(
      evaluation_name,
      function_name,
      evaluation_run_ids,
      limit,
      offset,
    );

    if (results.some((result) => result.feedback_id === new_feedback_id)) {
      found = true;
      break;
    }

    if (i < max_retries - 1) {
      // Don't sleep after the last attempt
      await new Promise((resolve) => setTimeout(resolve, retry_delay));
    }
  }

  if (!found) {
    logger.warn(
      `Evaluation result with feedback ${new_feedback_id} not found after ${max_retries} retries.`,
    );
  }

  return results;
}
