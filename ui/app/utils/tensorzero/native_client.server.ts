import {
  DatabaseClient,
  TensorZeroClient,
  runEvaluationStreaming,
} from "tensorzero-node";
import type { CacheEnabledMode, EvaluationRunEvent } from "~/types/tensorzero";
import { getEnv } from "../env.server";

let _tensorZeroClient: TensorZeroClient | undefined;
export async function getNativeTensorZeroClient(): Promise<TensorZeroClient> {
  if (_tensorZeroClient) {
    return _tensorZeroClient;
  }

  const env = getEnv();
  _tensorZeroClient = await TensorZeroClient.buildHttp(
    env.TENSORZERO_GATEWAY_URL,
  );
  return _tensorZeroClient;
}

let _databaseClient: DatabaseClient | undefined;
export async function getNativeDatabaseClient(): Promise<DatabaseClient> {
  if (_databaseClient) {
    return _databaseClient;
  }

  const env = getEnv();
  _databaseClient = await DatabaseClient.fromClickhouseUrl(
    env.TENSORZERO_CLICKHOUSE_URL,
  );
  return _databaseClient;
}

export function runNativeEvaluationStreaming(params: {
  gatewayUrl: string;
  clickhouseUrl: string;
  /** JSON-serialized EvaluationConfig */
  evaluationConfig: string;
  /** JSON-serialized EvaluationFunctionConfig */
  functionConfig: string;
  evaluationName: string;
  /** Exactly one of datasetName or datapointIds must be provided */
  datasetName?: string;
  /** Specific datapoint IDs to evaluate (alternative to datasetName) */
  datapointIds?: string[];
  /** Exactly one of variantName or internalDynamicVariantConfig must be provided */
  variantName?: string;
  /** JSON-serialized UninitializedVariantInfo */
  internalDynamicVariantConfig?: string;
  concurrency: number;
  inferenceCache: CacheEnabledMode;
  maxDatapoints?: number;
  precisionTargets?: string;
  onEvent: (event: EvaluationRunEvent) => void;
}): Promise<void> {
  return runEvaluationStreaming(params);
}
