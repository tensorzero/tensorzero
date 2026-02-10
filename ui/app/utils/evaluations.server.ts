import { z } from "zod";
import {
  EvaluationErrorSchema,
  type DisplayEvaluationError,
} from "./evaluations";
import { logger } from "~/utils/logger";
import type {
  EvaluationRunEvent,
  FunctionConfig,
  EvaluationFunctionConfig,
} from "~/types/tensorzero";
import { getConfig } from "./config/index.server";
import { getTensorZeroClient } from "./tensorzero.server";

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
  abortController: AbortController;
  errors: DisplayEvaluationError[];
  variantName: string;
  completed?: Date;
  cancelled?: boolean;
  started: Date;
}

// This is a map of evaluation run id to running evaluation info
const runningEvaluations = new Map<string, RunningEvaluationInfo>();

export type RunningEvaluationView = Omit<
  RunningEvaluationInfo,
  "abortController"
>;

/** @internal Exposed for testing only — injects an entry into the in-memory map. */
export function _test_registerRunningEvaluation(
  id: string,
  abortController: AbortController,
  variantName = "test-variant",
): void {
  runningEvaluations.set(id, {
    abortController,
    errors: [],
    variantName,
    started: new Date(),
  });
}

/**
 * Returns the running information for a specific evaluation run.
 * @param evaluationRunId The ID of the evaluation run to retrieve.
 * @returns The running information for the specified evaluation run, or undefined if not found.
 */
export function getRunningEvaluation(
  evaluationRunId: string,
): RunningEvaluationView | undefined {
  const info = runningEvaluations.get(evaluationRunId);
  if (!info) return undefined;
  const { abortController: _, ...view } = info;
  return view;
}

/**
 * Cancels a running evaluation by aborting its HTTP connection to the gateway.
 * This causes the gateway to cancel all in-flight evaluation tasks.
 * Partial results already written to ClickHouse are preserved.
 */
export function cancelEvaluation(evaluationRunId: string): {
  cancelled: boolean;
  already_completed: boolean;
} {
  const evaluation = runningEvaluations.get(evaluationRunId);
  if (!evaluation) {
    return { cancelled: false, already_completed: false };
  }
  if (evaluation.completed) {
    return { cancelled: false, already_completed: true };
  }
  evaluation.abortController.abort();
  evaluation.completed = new Date();
  evaluation.cancelled = true;
  return { cancelled: true, already_completed: false };
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
  const startTime = new Date();
  const abortController = new AbortController();
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

  // Convert to minimal EvaluationFunctionConfig
  const evaluationFunctionConfig = toEvaluationFunctionConfig(functionConfig);

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
          abortController,
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

  // Use the HTTP client instead of native bindings
  const client = getTensorZeroClient();
  const evaluationPromise = client.runEvaluationStreaming({
    evaluationConfig,
    functionConfig: evaluationFunctionConfig,
    evaluationName,
    datasetName,
    variantName,
    concurrency,
    inferenceCache,
    maxDatapoints,
    precisionTargets,
    onEvent: handleEvent,
    signal: abortController.signal,
  });

  void evaluationPromise
    .then(() => {
      if (evaluationRunId) {
        const evaluation = runningEvaluations.get(evaluationRunId);
        if (evaluation && !evaluation.completed) {
          evaluation.completed = new Date();
        }
      }
    })
    .catch((error) => {
      // Intentional cancellation via cancelEvaluation() — just mark completed
      if (error instanceof DOMException && error.name === "AbortError") {
        if (evaluationRunId) {
          const evaluation = runningEvaluations.get(evaluationRunId);
          if (evaluation && !evaluation.completed) {
            evaluation.completed = new Date();
          }
        }
        return;
      }
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
