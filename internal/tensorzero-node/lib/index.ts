// Import the NAPI-RS generated bindings
import { TensorZeroClient as NativeTensorZeroClient } from "../index";
import {
  OptimizerJobHandle,
  OptimizerStatus,
  StartOptimizationParams,
} from "./bindings";

// Re-export types from bindings
export * from "./bindings";

// Your TypeScript wrapper class
export class TensorZeroClient {
  private nativeClient: NativeTensorZeroClient;

  async new(
    configPath: string,
    clickhouseUrl?: string | undefined | null,
    timeout?: number | undefined | null
  ): Promise<TensorZeroClient> {
    this.nativeClient = await NativeTensorZeroClient.new(
      configPath,
      clickhouseUrl,
      timeout
    );
    return this;
  }

  async experimentalStartOptimization(
    params: StartOptimizationParams
  ): Promise<OptimizerJobHandle> {
    const paramsString = JSON.stringify(params);
    const jobHandleString =
      await this.nativeClient.experimentalStartOptimization(paramsString);
    return JSON.parse(jobHandleString) as OptimizerJobHandle;
  }

  async experimentalPollOptimization(
    jobHandle: OptimizerJobHandle
  ): Promise<OptimizerStatus> {
    const jobHandleString = JSON.stringify(jobHandle);
    const statusString = await this.nativeClient.experimentalPollOptimization(
      jobHandleString
    );
    return JSON.parse(statusString) as OptimizerStatus;
  }
}

export default TensorZeroClient;
