import { createRequire } from "module";
import type {
  CacheEnabledMode,
  ClientInferenceParams,
  Config,
  CountDatapointsForDatasetFunctionParams,
  DatasetMetadata,
  DatasetQueryParams,
  EpisodeByIdRow,
  EvaluationRunEvent,
  CumulativeFeedbackTimeSeriesPoint,
  FeedbackByVariant,
  GetDatasetMetadataParams,
  GetFeedbackByVariantParams,
  InferenceResponse,
  LaunchOptimizationWorkflowParams,
  ModelLatencyDatapoint,
  ModelUsageTimePoint,
  OptimizationJobHandle,
  OptimizationJobInfo,
  StaleDatasetResponse,
  TableBoundsWithCount,
  FeedbackRow,
  FeedbackBounds,
  TimeWindow,
  QueryFeedbackBoundsByTargetIdParams,
  QueryFeedbackByTargetIdParams,
  CountFeedbackByTargetIdParams,
  QueryDemonstrationFeedbackByInferenceIdParams,
  DemonstrationFeedbackRow,
  GetCumulativeFeedbackTimeseriesParams,
  KeyInfo,
} from "./bindings";
import type {
  TensorZeroClient as NativeTensorZeroClientType,
  DatabaseClient as NativeDatabaseClientType,
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
  getConfig: nativeGetConfig,
  DatabaseClient: NativeDatabaseClient,
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

  static async buildEmbedded(
    configPath: string,
    clickhouseUrl?: string | undefined | null,
    postgresUrl?: string | undefined | null,
    timeout?: number | undefined | null,
  ): Promise<TensorZeroClient> {
    const nativeClient = await NativeTensorZeroClient.buildEmbedded(
      configPath,
      clickhouseUrl,
      postgresUrl,
      timeout,
    );
    return new TensorZeroClient(nativeClient);
  }

  static async buildHttp(gatewayUrl: string): Promise<TensorZeroClient> {
    const nativeClient = await NativeTensorZeroClient.buildHttp(gatewayUrl);
    return new TensorZeroClient(nativeClient);
  }

  async inference(params: ClientInferenceParams): Promise<InferenceResponse> {
    const paramsString = safeStringify(params);
    const responseString = await this.nativeClient.inference(paramsString);
    return JSON.parse(responseString) as InferenceResponse;
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

  async staleDataset(datasetName: string): Promise<StaleDatasetResponse> {
    const staleDatasetString =
      await this.nativeClient.staleDataset(datasetName);
    return JSON.parse(staleDatasetString) as StaleDatasetResponse;
  }

  async getVariantSamplingProbabilities(
    functionName: string,
  ): Promise<Record<string, number>> {
    const probabilitiesString =
      await this.nativeClient.getVariantSamplingProbabilities(functionName);
    return JSON.parse(probabilitiesString) as Record<string, number>;
  }
}

export default TensorZeroClient;

export async function getConfig(configPath: string | null): Promise<Config> {
  const configString = await nativeGetConfig(configPath);
  return JSON.parse(configString) as Config;
}

// Export quantiles array from migration_0035
export { getQuantiles };

interface RunEvaluationStreamingParams {
  gatewayUrl: string;
  clickhouseUrl: string;
  configPath: string;
  evaluationName: string;
  datasetName: string;
  variantName: string;
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

/// Wrapper class for type safety and convenience
/// around the native DatabaseClient
export class DatabaseClient {
  private nativeDatabaseClient: NativeDatabaseClientType;

  constructor(client: NativeDatabaseClientType) {
    this.nativeDatabaseClient = client;
  }

  static async fromClickhouseUrl(url: string): Promise<DatabaseClient> {
    return new DatabaseClient(
      await NativeDatabaseClient.fromClickhouseUrl(url),
    );
  }

  async getModelUsageTimeseries(
    timeWindow: TimeWindow,
    maxPeriods: number,
  ): Promise<ModelUsageTimePoint[]> {
    const params = safeStringify({
      time_window: timeWindow,
      max_periods: maxPeriods,
    });
    const modelUsageTimeseriesString =
      await this.nativeDatabaseClient.getModelUsageTimeseries(params);
    return JSON.parse(modelUsageTimeseriesString) as ModelUsageTimePoint[];
  }

  async getModelLatencyQuantiles(
    timeWindow: TimeWindow,
  ): Promise<ModelLatencyDatapoint[]> {
    const params = safeStringify({
      time_window: timeWindow,
    });
    const modelLatencyQuantilesString =
      await this.nativeDatabaseClient.getModelLatencyQuantiles(params);
    return JSON.parse(modelLatencyQuantilesString) as ModelLatencyDatapoint[];
  }

  async countDistinctModelsUsed(): Promise<number> {
    const response = await this.nativeDatabaseClient.countDistinctModelsUsed();
    return response;
  }

  async queryEpisodeTable(
    limit: number,
    before?: string,
    after?: string,
  ): Promise<EpisodeByIdRow[]> {
    const params = safeStringify({
      limit,
      before,
      after,
    });
    const episodeTableString =
      await this.nativeDatabaseClient.queryEpisodeTable(params);
    return JSON.parse(episodeTableString) as EpisodeByIdRow[];
  }

  async queryEpisodeTableBounds(): Promise<TableBoundsWithCount> {
    const bounds = await this.nativeDatabaseClient.queryEpisodeTableBounds();
    return JSON.parse(bounds) as TableBoundsWithCount;
  }

  async queryFeedbackByTargetId(
    params: QueryFeedbackByTargetIdParams,
  ): Promise<FeedbackRow[]> {
    const paramsString = safeStringify(params);
    const feedbackString =
      await this.nativeDatabaseClient.queryFeedbackByTargetId(paramsString);
    return JSON.parse(feedbackString) as FeedbackRow[];
  }

  async queryDemonstrationFeedbackByInferenceId(
    params: QueryDemonstrationFeedbackByInferenceIdParams,
  ): Promise<DemonstrationFeedbackRow[]> {
    const paramsString = safeStringify(params);
    const feedbackString =
      await this.nativeDatabaseClient.queryDemonstrationFeedbackByInferenceId(
        paramsString,
      );
    return JSON.parse(feedbackString) as DemonstrationFeedbackRow[];
  }

  async queryFeedbackBoundsByTargetId(
    params: QueryFeedbackBoundsByTargetIdParams,
  ): Promise<FeedbackBounds> {
    const paramsString = safeStringify(params);
    const boundsString =
      await this.nativeDatabaseClient.queryFeedbackBoundsByTargetId(
        paramsString,
      );
    return JSON.parse(boundsString) as FeedbackBounds;
  }

  async getCumulativeFeedbackTimeseries(
    params: GetCumulativeFeedbackTimeseriesParams,
  ): Promise<CumulativeFeedbackTimeSeriesPoint[]> {
    const paramsString = safeStringify(params);
    const feedbackTimeseriesString =
      await this.nativeDatabaseClient.getCumulativeFeedbackTimeseries(
        paramsString,
      );
    return JSON.parse(
      feedbackTimeseriesString,
    ) as CumulativeFeedbackTimeSeriesPoint[];
  }

  async countFeedbackByTargetId(
    params: CountFeedbackByTargetIdParams,
  ): Promise<number> {
    const paramsString = safeStringify(params);
    const countString =
      await this.nativeDatabaseClient.countFeedbackByTargetId(paramsString);
    return JSON.parse(countString) as number;
  }

  async countRowsForDataset(params: DatasetQueryParams): Promise<number> {
    const paramsString = safeStringify(params);
    const result =
      await this.nativeDatabaseClient.countRowsForDataset(paramsString);
    return result;
  }

  async insertRowsForDataset(params: DatasetQueryParams): Promise<number> {
    const paramsString = safeStringify(params);
    const result =
      await this.nativeDatabaseClient.insertRowsForDataset(paramsString);
    return result;
  }

  async getDatasetMetadata(
    params: GetDatasetMetadataParams,
  ): Promise<DatasetMetadata[]> {
    const paramsString = safeStringify(params);
    const result =
      await this.nativeDatabaseClient.getDatasetMetadata(paramsString);
    return JSON.parse(result) as DatasetMetadata[];
  }

  async countDatasets(): Promise<number> {
    return this.nativeDatabaseClient.countDatasets();
  }

  async countDatapointsForDatasetFunction(
    params: CountDatapointsForDatasetFunctionParams,
  ): Promise<number> {
    const paramsString = safeStringify(params);
    return this.nativeDatabaseClient.countDatapointsForDatasetFunction(
      paramsString,
    );
  }

  async getFeedbackByVariant(
    params: GetFeedbackByVariantParams,
  ): Promise<FeedbackByVariant[]> {
    const paramsString = safeStringify(params);
    const result =
      await this.nativeDatabaseClient.getFeedbackByVariant(paramsString);
    return JSON.parse(result) as FeedbackByVariant[];
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
}
