import type { SFTFormValues } from "~/routes/optimization/supervised-fine-tuning/types";
import { OpenAISFTJob } from "./openai";
import { FireworksSFTJob } from "./fireworks";
import { SFTJob, type SFTJobStatus } from "./common";
import { TensorZeroClient } from "tensorzero-node";
import type {
  InferenceFilterTreeNode,
  InferenceOutputSource,
  OptimizerJobHandle,
  OptimizerStatus,
  UninitializedOptimizerInfo,
} from "tensorzero-node";
import { getConfig } from "~/utils/config/index.server";
import { getEnv } from "../env.server";

let _tensorZeroClient: TensorZeroClient | undefined;
export async function getNativeTensorZeroClient(): Promise<TensorZeroClient> {
  if (_tensorZeroClient) {
    return _tensorZeroClient;
  }

  const env = getEnv();
  _tensorZeroClient = await TensorZeroClient.build(
    env.TENSORZERO_UI_CONFIG_PATH,
    env.TENSORZERO_CLICKHOUSE_URL,
  );
  return _tensorZeroClient;
}

export function launch_sft_job(data: SFTFormValues): Promise<SFTJob> {
  const useNativeSFT = getEnv().TENSORZERO_UI_FF_USE_NATIVE_SFT === 1;
  if (useNativeSFT) {
    return launch_sft_job_native(data);
  } else {
    return launch_sft_job_ts(data);
  }
}

function launch_sft_job_ts(data: SFTFormValues): Promise<SFTJob> {
  switch (data.model.provider) {
    case "openai":
      return OpenAISFTJob.from_form_data(data);
    case "fireworks":
      return FireworksSFTJob.from_form_data(data);
    default:
      throw new Error("Invalid provider");
  }
}

class NativeSFTJob extends SFTJob {
  private jobStatus: OptimizerStatus | "created";
  private provider: "openai" | "fireworks" | "mistral";
  constructor(
    public jobHandle: OptimizerJobHandle,
    public formData: SFTFormValues,
  ) {
    super();
    this.jobHandle = jobHandle;
    this.formData = formData;
    this.jobStatus = "created";
    this.provider = formData.model.provider;
  }

  static from_job_handle_with_form_data(
    jobHandle: OptimizerJobHandle,
    formData: SFTFormValues,
  ): NativeSFTJob {
    return new NativeSFTJob(jobHandle, formData);
  }

  status(): SFTJobStatus {
    if (this.jobStatus === "created") {
      return {
        status: "idle",
      };
    }
    switch (this.jobStatus.status) {
      case "pending":
        return {
          status: "running",
          modelProvider: this.provider,
          jobUrl: this.jobHandle.job_url,
          formData: this.formData,
          rawData: {
            status: "ok",
            info: this.jobStatus,
          },
        };
      case "failed":
        return {
          status: "error",
          modelProvider: this.provider,
          formData: this.formData,
          jobUrl: this.jobHandle.job_url,
          rawData: {
            status: "error",
            message: "Job failed",
          },
          error: "Job failed",
        };
      case "completed": {
        // NOTE: the native SFT backend actually returns a model provider that is all we need
        // and guaranteed to match the Rust type.
        // For now we squeeze it through the existing interface.
        // In the future we should just rip all this code and render it directly
        const provider = Object.keys(this.jobStatus.output.providers)[0];
        if (!provider) {
          throw new Error("No provider found");
        }
        return {
          status: "completed",
          modelProvider: this.provider,
          formData: this.formData,
          jobUrl: this.jobHandle.job_url,
          rawData: {
            status: "ok",
            info: this.jobStatus,
          },
          result: provider,
        };
      }
    }
  }

  async poll(): Promise<SFTJob> {
    const client = await getNativeTensorZeroClient();
    const status = await client.experimentalPollOptimization(this.jobHandle);
    this.jobStatus = status;
    return this;
  }
}

async function launch_sft_job_native(data: SFTFormValues): Promise<SFTJob> {
  const openAINativeSFTBase = getEnv().OPENAI_BASE_URL;
  const fireworksNativeSFTBase = getEnv().FIREWORKS_BASE_URL;
  let filters: InferenceFilterTreeNode | null = null;
  let output_source: InferenceOutputSource = "Inference";
  if (data.metric === "demonstration") {
    output_source = "Demonstration";
  } else if (data.metric) {
    filters = await createFilters(data.metric, data.threshold);
  }
  const client = await getNativeTensorZeroClient();
  let optimizerConfig: UninitializedOptimizerInfo;
  if (data.model.provider == "openai") {
    optimizerConfig = {
      type: "openai_sft",
      model: data.model.name,
      batch_size: 1,
      learning_rate_multiplier: 1,
      n_epochs: 1,
      credentials: null,
      api_base: openAINativeSFTBase,
      seed: null,
      suffix: null,
    };
  } else if (data.model.provider == "fireworks") {
    const accountId = getEnv().FIREWORKS_ACCOUNT_ID;
    if (!accountId) {
      throw new Error("FIREWORKS_ACCOUNT_ID is not set");
    }
    optimizerConfig = {
      type: "fireworks_sft",
      model: data.model.name,
      credentials: null,
      api_base: fireworksNativeSFTBase,
      account_id: accountId,
    };
  } else {
    throw new Error(
      `Native SFT is not supported for provider ${data.model.provider}`,
    );
  }

  const job = await client.experimentalLaunchOptimizationWorkflow({
    function_name: data.function,
    template_variant_name: data.variant,
    query_variant_name: null,
    filters: filters,
    output_source: output_source,
    limit: data.maxSamples ? BigInt(data.maxSamples) : BigInt(0),
    offset: BigInt(0),
    val_fraction: data.validationSplitPercent / 100,
    format: "JsonEachRow",
    optimizer_config: optimizerConfig,
  });
  return NativeSFTJob.from_job_handle_with_form_data(job, data);
}

export async function createFilters(
  metric: string,
  threshold: number,
): Promise<InferenceFilterTreeNode> {
  const config = await getConfig();
  const metricConfig = config.metrics[metric];
  if (!metricConfig) {
    throw new Error(`Metric ${metric} not found`);
  }
  if (metricConfig.type === "float") {
    const comparison_operator = metricConfig.optimize === "max" ? ">=" : "<=";
    return {
      type: "float_metric",
      metric_name: metric,
      comparison_operator: comparison_operator,
      value: threshold,
    };
  } else if (metricConfig.type === "boolean") {
    const value = metricConfig.optimize === "max" ? true : false;
    return {
      type: "boolean_metric",
      metric_name: metric,
      value: value,
    };
  } else {
    throw new Error(`Unsupported metric type: ${metricConfig.type}`);
  }
}
