import { spawn } from "node:child_process";
import { z } from "zod";
import { getConfigPath } from "./config/index.server";
import { EvalErrorSchema, type DisplayEvalError } from "./evaluations";
/**
 * Get the path to the evaluations binary from environment variables.
 * Defaults to 'evaluations' if not specified.
 */
function getEvaluationsPath(): string {
  return process.env.TENSORZERO_EVALUATIONS_PATH || "evaluations";
}

function getGatewayURL(): string {
  const gatewayURL = process.env.TENSORZERO_GATEWAY_URL;
  // This error is thrown on startup in tensorzero.server.ts
  if (!gatewayURL) {
    throw new Error("TENSORZERO_GATEWAY_URL environment variable is not set");
  }
  return gatewayURL;
}
interface RunningEvalInfo {
  errors: DisplayEvalError[];
  variantName: string;
  completed?: Date;
  started: Date;
}

// This is a map of evaluation run id to running evaluation info
const runningEvaluations: Record<string, RunningEvalInfo> = {};

/**
 * Returns the running information for a specific evaluation run.
 * @param evalRunId The ID of the evaluation run to retrieve.
 * @returns The running information for the specified evaluation run, or undefined if not found.
 */
export function getRunningEvaluation(
  evalRunId: string,
): RunningEvalInfo | undefined {
  return runningEvaluations[evalRunId];
}

export function runEvaluation(
  evalName: string,
  variantName: string,
  concurrency: number,
): Promise<EvaluationStartInfo> {
  const evaluationsPath = getEvaluationsPath();
  const gatewayURL = getGatewayURL();
  // Construct the command to run the evaluations binary
  // Example: evaluations --gateway-url http://localhost:3000 --name entity_extraction --variant llama_8b_initial_prompt --concurrency 10 --format jsonl
  const command = [
    evaluationsPath,
    "--gateway-url",
    gatewayURL,
    "--config-file",
    getConfigPath(),
    "--name",
    evalName,
    "--variant",
    variantName,
    "--concurrency",
    concurrency.toString(),
    "--format",
    "jsonl",
  ];

  return new Promise<EvaluationStartInfo>((resolve, reject) => {
    // Spawn a child process to run the evaluations command
    const child = spawn(command[0], command.slice(1));

    // Variables to track state
    let evalRunId: string | null = null;
    let initialDataReceived = false;

    // Buffer for incomplete lines
    let stdoutBuffer = "";

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
          try {
            const evaluationStartInfo =
              evaluationStartInfoSchema.parse(parsedLine);
            evalRunId = evaluationStartInfo.evaluation_run_id;

            // Initialize the tracking entry in our runningEvaluations map
            runningEvaluations[evalRunId] = {
              variantName,
              errors: [],
              started: startTime,
            };

            // Mark that we've received the initial data
            initialDataReceived = true;

            // Resolve the promise with the evaluation start info
            resolve(evaluationStartInfo);
          } catch (error) {
            reject(error);
          }
        }
        // Check if this is an EvalError using the Zod schema
        else if (evalRunId && parsedLine.datapoint_id && parsedLine.message) {
          try {
            // Parse using the Zod schema to validate
            const evalError = EvalErrorSchema.parse(parsedLine);

            // Add to the beginning of the errors list
            runningEvaluations[evalRunId].errors.unshift(evalError);
          } catch {
            // If it doesn't match our schema, just ignore it
          }
        }
        // We're ignoring other types of output that don't match these patterns
      } catch {
        console.warn(`Bad JSON line: ${line}`);
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

    // Ignore stderr completely

    // Handle process errors (e.g., if the process couldn't be spawned)
    child.on("error", (error) => {
      if (!initialDataReceived) {
        reject(error);
      }
    });

    // Handle process completion
    child.on("close", (code) => {
      // Process any remaining data in the buffer
      if (stdoutBuffer.trim()) {
        processCompleteLine(stdoutBuffer.trim());
      }

      if (evalRunId) {
        // Mark the evaluation as completed
        runningEvaluations[evalRunId].completed = new Date();

        // Add exit code info if not successful
        if (code !== 0) {
          runningEvaluations[evalRunId].errors.push({
            message: `Process exited with code ${code}`,
          });
        }
      }

      if (!initialDataReceived) {
        reject(
          new Error(
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
  const now = new Date();

  Object.keys(runningEvaluations).forEach((evalRunId) => {
    const evalInfo = runningEvaluations[evalRunId];

    // Remove completed evaluations older than 1 hour
    if (
      evalInfo.completed &&
      now.getTime() - evalInfo.completed.getTime() > ONE_HOUR_MS
    ) {
      delete runningEvaluations[evalRunId];
      return;
    }

    // Remove stalled evaluations older than 24 hours
    if (now.getTime() - evalInfo.started.getTime() > TWENTY_FOUR_HOURS_MS) {
      delete runningEvaluations[evalRunId];
    }
  });
}

/**
 * Starts a periodic cleanup of old evaluations.
 * @param intervalMs The interval in milliseconds between cleanups. Default: 1 hour.
 * @returns A function that can be called to stop the periodic cleanup.
 */
export function startPeriodicCleanup(intervalMs = ONE_HOUR_MS): () => void {
  const intervalId = setInterval(cleanupOldEvaluations, intervalMs);

  // Run an initial cleanup immediately
  cleanupOldEvaluations();

  // Return a function to stop the periodic cleanup
  return () => clearInterval(intervalId);
}
