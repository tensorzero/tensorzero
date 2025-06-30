import { createRequire } from "module";
import {
  OptimizerJobHandle,
  OptimizerStatus,
  LaunchOptimizationWorkflowParams,
} from "./bindings";
import type { TensorZeroClient as NativeTensorZeroClientType } from "../index";

// Re-export types from bindings
export * from "./bindings";

// Use createRequire to load CommonJS module
const require = createRequire(import.meta.url);
const { TensorZeroClient: NativeTensorZeroClient } =
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
    const paramsString = JSON.stringify(params, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value,
    );
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
