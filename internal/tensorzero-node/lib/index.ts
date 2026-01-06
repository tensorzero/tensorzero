import { createRequire } from "module";
import type {
  ClientInferenceParams,
  InferenceResponse,
  LaunchOptimizationWorkflowParams,
  OptimizationJobHandle,
  OptimizationJobInfo,
  StaleDatasetResponse,
  KeyInfo,
} from "./bindings";
import type {
  TensorZeroClient as NativeTensorZeroClientType,
  PostgresClient as NativePostgresClientType,
} from "../index";

// Re-export types from bindings
export type * from "./bindings";
export { createLogger } from "./utils/logger";

// Use createRequire to load CommonJS module
const require = createRequire(import.meta.url);

const {
  TensorZeroClient: NativeTensorZeroClient,
  PostgresClient: NativePostgresClient,
  getQuantiles,
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

// Export quantiles array from migration_0035
export { getQuantiles };

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
