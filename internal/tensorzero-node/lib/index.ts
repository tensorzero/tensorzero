import { createRequire } from "module";
import {
  OptimizationJobHandle,
  OptimizationJobInfo,
  LaunchOptimizationWorkflowParams,
  StaleDatasetResponse,
  Config,
  ClientInferenceParams,
  InferenceResponse,
} from "./bindings";
import type { TensorZeroClient as NativeTensorZeroClientType } from "../index";

// Re-export types from bindings
export * from "./bindings";

// Use createRequire to load CommonJS module
const require = createRequire(import.meta.url);
const { TensorZeroClient: NativeTensorZeroClient, getConfig: nativeGetConfig } =
  require("../index.cjs") as typeof import("../index");

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
    timeout?: number | undefined | null
  ): Promise<TensorZeroClient> {
    const nativeClient = await NativeTensorZeroClient.buildEmbedded(
      configPath,
      clickhouseUrl,
      timeout
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
    params: LaunchOptimizationWorkflowParams
  ): Promise<OptimizationJobHandle> {
    const paramsString = safeStringify(params);
    const jobHandleString =
      await this.nativeClient.experimentalLaunchOptimizationWorkflow(
        paramsString
      );
    return JSON.parse(jobHandleString) as OptimizationJobHandle;
  }

  async experimentalPollOptimization(
    jobHandle: OptimizationJobHandle
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
}

export default TensorZeroClient;

export async function getConfig(configPath: string): Promise<Config> {
  const configString = await nativeGetConfig(configPath);
  return JSON.parse(configString) as Config;
}

function safeStringify(obj: unknown) {
  try {
    return JSON.stringify(obj, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value
    );
  } catch {
    return "null";
  }
}
