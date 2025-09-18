import { spawn } from "node:child_process";
import { z } from "zod";
import {
  EvaluationErrorSchema,
  type DisplayEvaluationError,
} from "./evaluations";
import { logger } from "~/utils/logger";
import { getEnv } from "./env.server";
/**
 * Get the path to the evaluations binary from environment variables.
 * Defaults to 'evaluations' if not specified.
 */
function getEvaluationsPath(): string {
  return getEnv().TENSORZERO_EVALUATIONS_PATH || "evaluations";
}
function getConfigPath(): string {
  const configPath = getEnv().TENSORZERO_UI_CONFIG_PATH;
  if (!configPath) {
    throw new Error("TENSORZERO_UI_CONFIG_PATH is not set");
  }
  return configPath;
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
});
export type EvaluationFormData = z.infer<typeof evaluationFormDataSchema>;

export function parseEvaluationFormData(
  data: Record<keyof EvaluationFormData, FormDataEntryValue | null>,
): EvaluationFormData | null {
  const result = evaluationFormDataSchema.safeParse(data);
  return result.success ? result.data : null;
}

export function runEvaluation(
  evaluationName: string,
  datasetName: string,
  variantName: string,
  concurrency: number,
  inferenceCache: InferenceCacheSetting,
): Promise<EvaluationStartInfo> {
  const evaluationsPath = getEvaluationsPath();
  const gatewayURL = getEnv().TENSORZERO_GATEWAY_URL;
  // Construct the command to run the evaluations binary
  // Example: evaluations --gateway-url http://localhost:3000 --name entity_extraction --variant-name llama_8b_initial_prompt --concurrency 10 --format jsonl
  // We do not need special escaping for the evaluation name or variant name because
  // Node's spawn() does not use the shell to run the command.
  const command = [
    evaluationsPath,
    "--gateway-url",
    gatewayURL,
    "--config-file",
    getConfigPath(),
    "--evaluation-name",
    evaluationName,
    "--dataset-name",
    datasetName,
    "--variant-name",
    variantName,
    "--concurrency",
    concurrency.toString(),
    "--format",
    "jsonl",
    "--inference-cache",
    inferenceCache,
  ];

  return new Promise<EvaluationStartInfo>((resolve, reject) => {
    // Spawn a child process to run the evaluations command
    const child = spawn(command[0], command.slice(1));
    // We also want to forward stderr to the parent process
    // so it shows up in container logs.
    child.stderr.pipe(process.stderr);

    // Variables to track state
    let evaluationRunId: string | null = null;
    let initialDataReceived = false;

    // Buffer for incomplete lines
    let stdoutBuffer = "";
    let stderrBuffer = "";

    // Record the start time of the evaluation
    const startTime = new Date();

    // Process a complete line from stdout
    const processCompleteLine = (line: string) => {
      if (!line.trim()) return; // Skip empty lines

      try {
        // Try to parse the line as JSON
        const parsedLine = JSON.parse(line);

        // Check if this is the initial data containing evaluation_run_id
        if (
          !initialDataReceived &&
          parsedLine.evaluation_run_id &&
          parsedLine.num_datapoints
        ) {
          const evaluationStartInfo =
            evaluationStartInfoSchema.safeParse(parsedLine);
          if (!evaluationStartInfo.success) {
            return reject(evaluationStartInfo.error);
          }

          evaluationRunId = evaluationStartInfo.data.evaluation_run_id;

          // Initialize the tracking entry in our runningEvaluations map
          runningEvaluations.set(evaluationRunId, {
            variantName,
            errors: [],
            started: startTime,
          });

          // Mark that we've received the initial data
          initialDataReceived = true;

          // Resolve the promise with the evaluation start info
          return resolve(evaluationStartInfo.data);
        }
        // Check if this is an EvaluationError using the Zod schema
        else if (
          evaluationRunId &&
          parsedLine.datapoint_id &&
          parsedLine.message
        ) {
          // Parse using the Zod schema to validate
          const evaluationError = EvaluationErrorSchema.safeParse(parsedLine);
          if (evaluationError.success) {
            // Add to the beginning of the errors list
            runningEvaluations
              .get(evaluationRunId)
              ?.errors.unshift(evaluationError.data);
          }
        }
        // We're ignoring other types of output that don't match these patterns
      } catch {
        logger.warn(`Bad JSON line: ${line}`);
      }
    };

    // Handle stdout data from the process
    child.stdout.on("data", (data) => {
      // Add new data to our buffer
      stdoutBuffer += data.toString();

      // Process complete lines
      let newlineIndex;
      while ((newlineIndex = stdoutBuffer.indexOf("\n")) !== -1) {
        // Extract a complete line
        const line = stdoutBuffer.substring(0, newlineIndex);
        // Remove the processed line from the buffer
        stdoutBuffer = stdoutBuffer.substring(newlineIndex + 1);
        // Process the complete line
        processCompleteLine(line);
      }
      // stdoutBuffer now contains any incomplete line (or nothing)
    });

    // Handle stderr data and process it line-by-line
    child.stderr.on("data", (data) => {
      // Add new data to our buffer
      stderrBuffer += data.toString();
      const evaluation = evaluationRunId
        ? runningEvaluations.get(evaluationRunId)
        : undefined;

      // Process complete lines
      let newlineIndex;
      while ((newlineIndex = stderrBuffer.indexOf("\n")) !== -1) {
        // Extract a complete line
        const line = stderrBuffer.substring(0, newlineIndex).trim();
        // Remove the processed line from the buffer
        stderrBuffer = stderrBuffer.substring(newlineIndex + 1);
        // Accumulate the error
        if (evaluation) {
          evaluation.errors.unshift({
            message: `Process error: ${line}`,
          });
        }
      }
      // stderrBuffer now contains any incomplete line (or nothing)
    });

    // Handle process errors (e.g., if the process couldn't be spawned)
    child.on("error", (error) => {
      if (!initialDataReceived) {
        reject(error);
      }
    });

    // Handle process completion
    child.on("close", (code) => {
      // Process any remaining data in the stdout buffer
      if (stdoutBuffer.trim()) {
        processCompleteLine(stdoutBuffer.trim());
      }

      const evaluation = evaluationRunId
        ? runningEvaluations.get(evaluationRunId)
        : undefined;
      if (evaluation) {
        // Mark the evaluation as completed
        evaluation.completed = new Date();

        // Add exit code info if not successful
        if (code !== 0) {
          if (stderrBuffer.trim()) {
            evaluation.errors.push({
              message: `Error: ${stderrBuffer.trim()}`,
            });
          } else {
            evaluation.errors.push({
              message: `Process exited with code ${code}`,
            });
          }
        }
      }

      if (!initialDataReceived) {
        reject(
          new Error(
            stderrBuffer.trim() ||
              `Process exited with code ${code} without producing output`,
          ),
        );
      }
    });
  });
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
