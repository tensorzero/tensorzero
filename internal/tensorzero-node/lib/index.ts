import { createRequire } from "module";
import {
  OptimizerJobHandle,
  OptimizerStatus,
  LaunchOptimizationWorkflowParams,
} from "./bindings";

// Re-export types from bindings
export * from "./bindings";

// Use createRequire to load CommonJS module
const require = createRequire(import.meta.url);
const { TensorZeroClient: NativeTensorZeroClient } = require("../index.cjs");
type NativeTensorZeroClient = typeof NativeTensorZeroClient;

// Wrapper class for type safety and convenience
// since the interface is string in string out
export class TensorZeroClient {
  private nativeClient!: NativeTensorZeroClient;

  constructor(client: NativeTensorZeroClient) {
    this.nativeClient = client;
  }

  static async build(
    configPath: string,
    clickhouseUrl?: string | undefined | null,
    timeout?: number | undefined | null,
  ): Promise<TensorZeroClient> {
    const nativeClient = await NativeTensorZeroClient.build(
      configPath,
      clickhouseUrl,
      timeout,
    );
    return new TensorZeroClient(nativeClient);
  }

  async experimentalLaunchOptimizationWorkflow(
    params: LaunchOptimizationWorkflowParams,
  ): Promise<OptimizerJobHandle> {
    const paramsString = JSON.stringify(params);
    const jobHandleString =
      await this.nativeClient.experimentalLaunchOptimizationWorkflow(
        paramsString,
      );
    return JSON.parse(jobHandleString) as OptimizerJobHandle;
  }

  async experimentalPollOptimization(
    jobHandle: OptimizerJobHandle,
  ): Promise<OptimizerStatus> {
    const jobHandleString = JSON.stringify(jobHandle);
    const statusString =
      await this.nativeClient.experimentalPollOptimization(jobHandleString);
    return JSON.parse(statusString) as OptimizerStatus;
  }
}

export default TensorZeroClient;
