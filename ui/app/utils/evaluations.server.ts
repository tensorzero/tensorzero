import { z } from "zod";
import {
  EvaluationErrorSchema,
  type DisplayEvaluationError,
} from "./evaluations";
import { logger } from "~/utils/logger";
import { getEnv } from "./env.server";
import { runNativeEvaluationStreaming } from "./tensorzero/native_client.server";
import type {
  EvaluationRunEvent,
  FunctionConfig,
  EvaluationFunctionConfig,
} from "~/types/tensorzero";
import { getConfig } from "./config/index.server";

/**
 * Converts a FunctionConfig to the minimal EvaluationFunctionConfig format
 * required by the evaluation runner.
 */
function toEvaluationFunctionConfig(
  config: FunctionConfig,
): EvaluationFunctionConfig {
  if (config.type === "chat") {
    return { type: "chat" };
  }
  return { type: "json", output_schema: config.output_schema };
}

const INFERENCE_CACHE_SETTINGS = [
  "on",
  "off",
  "read_only",
  "write_only",
] as const;

export type InferenceCacheSetting = (typeof INFERENCE_CACHE_SETTINGS)[number];

interface RunningEvaluationInfo {
  errors: DisplayEvaluationError[];
  variantName: string;
  completed?: Date;
  started: Date;
}

// This is a map of evaluation run id to running evaluation info
const runningEvaluations = new Map<string, RunningEvaluationInfo>();

/**
 * Returns the running information for a specific evaluation run.
 * @param evaluationRunId The ID of the evaluation run to retrieve.
 * @returns The running information for the specified evaluation run, or undefined if not found.
 */
export function getRunningEvaluation(
  evaluationRunId: string,
): RunningEvaluationInfo | undefined {
  return runningEvaluations.get(evaluationRunId);
}

const evaluationFormDataSchema = z.object({
  evaluation_name: z.string().min(1),
  dataset_name: z.string().min(1),
  variant_name: z.string().min(1),
  concurrency_limit: z
    .string()
    .min(1)
    .transform((val) => Number.parseInt(val, 10))
    .refine((val) => !Number.isNaN(val) && val > 0, {
      message: "Concurrency limit must be a positive integer",
    }),
  inference_cache: z.enum(INFERENCE_CACHE_SETTINGS),
  max_datapoints: z
    .string()
    .optional()
    .transform((val) => (val ? Number.parseInt(val, 10) : undefined))
    .refine((val) => val === undefined || (!Number.isNaN(val) && val > 0), {
      message: "Max datapoints must be a positive integer",
    }),
  precision_targets: z
    .string()
    .optional()
    .transform((val) => {
      if (!val) return undefined;
      try {
        return JSON.parse(val) as Record<string, number>;
      } catch {
        return undefined;
      }
    })
    .refine(
      (val) =>
        val === undefined ||
        (typeof val === "object" &&
          Object.values(val).every((v) => typeof v === "number" && v >= 0)),
      {
        message:
          "Precision targets must be a JSON object mapping evaluator names to non-negative numbers",
      },
    ),
});
export type EvaluationFormData = z.infer<typeof evaluationFormDataSchema>;

export function parseEvaluationFormData(
  data: Record<keyof EvaluationFormData, FormDataEntryValue | null>,
): EvaluationFormData | null {
  const result = evaluationFormDataSchema.safeParse(data);
  return result.success ? result.data : null;
}

export async function runEvaluation(
  evaluationName: string,
  datasetName: string,
  variantName: string,
  concurrency: number,
  inferenceCache: InferenceCacheSetting,
  maxDatapoints?: number,
  precisionTargets?: Record<string, number>,
): Promise<EvaluationStartInfo> {
  const env = getEnv();
  const startTime = new Date();
  let evaluationRunId: string | null = null;
  let startResolved = false;

  // Get config and look up evaluation and function configs
  const config = await getConfig();
  const evaluationConfig = config.evaluations[evaluationName];
  if (!evaluationConfig) {
    throw new Error(`Evaluation '${evaluationName}' not found in config`);
  }
  // eslint-disable-next-line no-restricted-syntax
  const functionConfig = config.functions[evaluationConfig.function_name];
  if (!functionConfig) {
    throw new Error(
      `Function '${evaluationConfig.function_name}' not found in config`,
    );
  }

  // Convert to minimal EvaluationFunctionConfig and serialize
  const evaluationFunctionConfig = toEvaluationFunctionConfig(functionConfig);
  const serializedEvaluationConfig = JSON.stringify(evaluationConfig);
  const serializedFunctionConfig = JSON.stringify(evaluationFunctionConfig);

  let resolveStart: (value: EvaluationStartInfo) => void = () => {};
  let rejectStart: (reason?: unknown) => void = () => {};

  const startPromise = new Promise<EvaluationStartInfo>((resolve, reject) => {
    resolveStart = resolve;
    rejectStart = reject;
  });

  const handleEvent = (event: EvaluationRunEvent) => {
    switch (event.type) {
      case "start": {
        const startInfo = evaluationStartInfoSchema.safeParse({
          evaluation_run_id: event.evaluation_run_id,
          num_datapoints: event.num_datapoints,
        });
        if (!startInfo.success) {
          rejectStart(startInfo.error);
          return;
        }

        evaluationRunId = startInfo.data.evaluation_run_id;
        runningEvaluations.set(evaluationRunId, {
          variantName,
          errors: [],
          started: startTime,
        });
        startResolved = true;
        resolveStart(startInfo.data);
        break;
      }
      case "error": {
        if (!evaluationRunId) {
          return;
        }
        const parsedError = EvaluationErrorSchema.safeParse({
          datapoint_id: event.datapoint_id,
          message: event.message,
        });
        if (!parsedError.success) {
          logger.warn("Received malformed evaluation error", parsedError.error);
          return;
        }
        runningEvaluations
          .get(evaluationRunId)
          ?.errors.unshift(parsedError.data);
        break;
      }
      case "fatal_error": {
        if (!startResolved) {
          rejectStart(new Error(event.message));
          startResolved = true;
        }
        if (evaluationRunId) {
          const evaluation = runningEvaluations.get(evaluationRunId);
          if (evaluation) {
            evaluation.errors.unshift({ message: event.message });
            evaluation.completed = new Date();
          }
        }
        break;
      }
      case "complete": {
        if (evaluationRunId) {
          const evaluation = runningEvaluations.get(evaluationRunId);
          if (evaluation && !evaluation.completed) {
            evaluation.completed = new Date();
          }
        }
        break;
      }
      case "success": {
        // We don't currently surface success payloads in the UI.
        break;
      }
    }
  };

  const nativePromise = runNativeEvaluationStreaming({
    gatewayUrl: env.TENSORZERO_GATEWAY_URL,
    clickhouseUrl: env.TENSORZERO_CLICKHOUSE_URL,
    evaluationConfig: serializedEvaluationConfig,
    functionConfig: serializedFunctionConfig,
    evaluationName,
    datasetName,
    variantName,
    concurrency,
    inferenceCache,
    maxDatapoints,
    precisionTargets: precisionTargets
      ? JSON.stringify(precisionTargets)
      : undefined,
    onEvent: handleEvent,
  });

  void nativePromise
    .then(() => {
      if (evaluationRunId) {
        const evaluation = runningEvaluations.get(evaluationRunId);
        if (evaluation && !evaluation.completed) {
          evaluation.completed = new Date();
        }
      }
    })
    .catch((error) => {
      if (!startResolved) {
        rejectStart(error);
        startResolved = true;
        return;
      }
      if (evaluationRunId) {
        const evaluation = runningEvaluations.get(evaluationRunId);
        if (evaluation) {
          evaluation.errors.unshift({
            message:
              error instanceof Error
                ? error.message
                : String(error ?? "Unknown error"),
          });
          evaluation.completed = new Date();
        }
      }
    });

  return startPromise;
}

const evaluationStartInfoSchema = z.object({
  evaluation_run_id: z.string(),
  num_datapoints: z.number(),
});
export type EvaluationStartInfo = z.infer<typeof evaluationStartInfoSchema>;

const ONE_HOUR_MS = 60 * 60 * 1000;
const TWENTY_FOUR_HOURS_MS = 24 * 60 * 60 * 1000;
/**
 * Cleans up old evaluation entries from the runningEvaluations map:
 * - Removes completed evaluations older than 1 hour
 * - Removes stalled evaluations (started but not completed) older than 24 hours
 */
export function cleanupOldEvaluations(): void {
  if (runningEvaluations.size === 0) {
    return;
  }

  const now = new Date();
  for (const [evaluationRunId, evaluationInfo] of runningEvaluations) {
    const isCompleted =
      evaluationInfo.completed &&
      // Remove completed evaluations older than 1 hour
      now.getTime() - evaluationInfo.completed.getTime() > ONE_HOUR_MS;

    const isStalled =
      !evaluationInfo.completed &&
      // Remove stalled evaluations older than 24 hours
      now.getTime() - evaluationInfo.started.getTime() > TWENTY_FOUR_HOURS_MS;

    if (isCompleted || isStalled) {
      runningEvaluations.delete(evaluationRunId);
    }
  }
}

let cleanupIntervalId: NodeJS.Timeout | undefined;

/**
 * Starts a periodic cleanup of old evaluations.
 * @param intervalMs The interval in milliseconds between cleanups. Default: 1 hour.
 * @returns A function that can be called to stop the periodic cleanup.
 */
export function startPeriodicCleanup(intervalMs = ONE_HOUR_MS): () => void {
  const stopPeriodicCleanup = () => {
    if (cleanupIntervalId !== undefined) {
      clearInterval(cleanupIntervalId);
      cleanupIntervalId = undefined;
    }
  };

  if (cleanupIntervalId === undefined) {
    // Only start the interval if there are running evaluations to clean up
    if (runningEvaluations.size > 0) {
      // Run an initial cleanup immediately
      cleanupOldEvaluations();
      cleanupIntervalId = setInterval(cleanupOldEvaluations, intervalMs);
    }
  } else if (runningEvaluations.size === 0) {
    // Interval is still running but there are no running evaluations to clean
    // up, so we can stop it.
    stopPeriodicCleanup();
  }

  return stopPeriodicCleanup;
}
