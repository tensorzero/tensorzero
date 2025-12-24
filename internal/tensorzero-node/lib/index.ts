import { createRequire } from "module";
import type {
  CacheEnabledMode,
  EvaluationRunEvent,
  LaunchOptimizationWorkflowParams,
  OptimizationJobHandle,
  OptimizationJobInfo,
  KeyInfo,
} from "./bindings";
import type {
  TensorZeroClient as NativeTensorZeroClientType,
  PostgresClient as NativePostgresClientType,
} from "../index";
import { logger } from "./utils/logger";

// Re-export types from bindings
export type * from "./bindings";
export { createLogger } from "./utils/logger";

// Use createRequire to load CommonJS module
const require = createRequire(import.meta.url);

const {
  TensorZeroClient: NativeTensorZeroClient,
  PostgresClient: NativePostgresClient,
  getQuantiles,
  runEvaluationStreaming: nativeRunEvaluationStreaming,
} = require("../index.cjs") as typeof import("../index");

// Wrapper class for type safety and convenience
// since the interface is string in string out
// In each method we stringify the params and return the result as a string from the
// Rust codebase.
// However, since we generate types with TS-RS `pnpm build-bindings` we can
// just parse the JSON and it should be type safe to use the types we generated.
export class TensorZeroClient {
  private nativeClient: NativeTensorZeroClientType;

  constructor(client: NativeTensorZeroClientType) {
    this.nativeClient = client;
  }

  static async buildHttp(gatewayUrl: string): Promise<TensorZeroClient> {
    const nativeClient = await NativeTensorZeroClient.buildHttp(gatewayUrl);
    return new TensorZeroClient(nativeClient);
  }

  async experimentalLaunchOptimizationWorkflow(
    params: LaunchOptimizationWorkflowParams,
  ): Promise<OptimizationJobHandle> {
    const paramsString = safeStringify(params);
    const jobHandleString =
      await this.nativeClient.experimentalLaunchOptimizationWorkflow(
        paramsString,
      );
    return JSON.parse(jobHandleString) as OptimizationJobHandle;
  }

  async experimentalPollOptimization(
    jobHandle: OptimizationJobHandle,
  ): Promise<OptimizationJobInfo> {
    const jobHandleString = safeStringify(jobHandle);
    const statusString =
      await this.nativeClient.experimentalPollOptimization(jobHandleString);
    return JSON.parse(statusString) as OptimizationJobInfo;
  }
}

export default TensorZeroClient;

// Export quantiles array from migration_0035
export { getQuantiles };

interface RunEvaluationStreamingParams {
  gatewayUrl: string;
  clickhouseUrl: string;
  /** JSON-serialized EvaluationConfig */
  evaluationConfig: string;
  /** JSON-serialized EvaluationFunctionConfig */
  functionConfig: string;
  evaluationName: string;
  datasetName: string;
  /** Exactly one of variantName or internalDynamicVariantConfig must be provided */
  variantName?: string;
  /** JSON-serialized UninitializedVariantInfo */
  internalDynamicVariantConfig?: string;
  concurrency: number;
  inferenceCache: CacheEnabledMode;
  maxDatapoints?: number;
  precisionTargets?: string;
  onEvent: (event: EvaluationRunEvent) => void;
}

/**
 * Runs an evaluation asynchronously with streaming event updates.
 *
 * This function executes an evaluation by running inference on a dataset using a specified variant,
 * and streaming progress events back to the caller through a callback function. It bridges TypeScript
 * to native Rust code that performs the actual evaluation work.
 *
 * The `onEvent` callback will receive events in this typical sequence:
 * 1. `start`: Evaluation run has begun, includes the evaluation_run_id
 * 2. `success`: Each successful datapoint evaluation (may be many)
 * 3. `error`: Each failed datapoint evaluation (may be zero or many)
 * 4. `fatal_error`: Critical error that stops the evaluation (rare, may not occur)
 * 5. `complete`: Evaluation run has finished (always the last event)
 *
 */
export async function runEvaluationStreaming(
  params: RunEvaluationStreamingParams,
): Promise<void> {
  const { onEvent, ...nativeParams } = params;

  return nativeRunEvaluationStreaming(
    nativeParams,
    (err: Error | null, payload: string | null) => {
      // Handle errors from the native callback
      if (err) {
        logger.error("Native evaluation streaming error:", err);
        return;
      }

      // Check if payload is null or undefined
      if (!payload) {
        logger.error("Received null or undefined payload from native code");
        return;
      }

      try {
        const event = JSON.parse(payload) as EvaluationRunEvent;

        // Check if event is null or missing the required 'type' property
        if (!event || typeof event !== "object" || !("type" in event)) {
          logger.error(
            "Received invalid evaluation event from native code. Payload:",
            payload,
          );
          return;
        }

        onEvent(event);
      } catch (error) {
        logger.error(
          "Failed to parse evaluation event. Payload:",
          payload,
          "Error:",
          error,
        );
      }
    },
  );
}

function safeStringify(obj: unknown) {
  try {
    return JSON.stringify(obj, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value,
    );
  } catch {
    return "null";
  }
}

/**
 * Wrapper class for type safety and convenience
 * around the native PostgresClient
 */
export class PostgresClient {
  private nativePostgresClient: NativePostgresClientType;

  constructor(client: NativePostgresClientType) {
    this.nativePostgresClient = client;
  }

  static async fromPostgresUrl(url: string): Promise<PostgresClient> {
    return new PostgresClient(await NativePostgresClient.fromPostgresUrl(url));
  }

  async createApiKey(description?: string | null): Promise<string> {
    return this.nativePostgresClient.createApiKey(description);
  }

  async listApiKeys(limit?: number, offset?: number): Promise<KeyInfo[]> {
    const result = await this.nativePostgresClient.listApiKeys(limit, offset);
    return JSON.parse(result) as KeyInfo[];
  }

  async disableApiKey(publicId: string): Promise<string> {
    return this.nativePostgresClient.disableApiKey(publicId);
  }

  async updateApiKeyDescription(
    publicId: string,
    description?: string | null,
  ): Promise<KeyInfo> {
    const result = await this.nativePostgresClient.updateApiKeyDescription(
      publicId,
      description ?? null,
    );
    return JSON.parse(result) as KeyInfo;
  }
}
