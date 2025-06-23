import type { SFTFormValues } from "~/routes/optimization/supervised-fine-tuning/types";
import { OpenAISFTJob } from "./openai";
import { FireworksSFTJob } from "./fireworks";
import { SFTJob, type SFTJobStatus } from "./common";
import {
  TensorZeroClient,
  type InferenceFilterTreeNode,
  type InferenceOutputSource,
  type OpenAISFTJobHandle,
  type OptimizerJobHandle,
  type OptimizerStatus,
} from "tensorzero-node";
import { getConfig } from "~/utils/config/index.server";

const configPath = process.env.TENSORZERO_UI_CONFIG_PATH;
if (!configPath) {
  throw new Error("TENSORZERO_UI_CONFIG_PATH is not set");
}
const clickhouseUrl = process.env.TENSORZERO_CLICKHOUSE_URL;
if (!clickhouseUrl) {
  throw new Error("TENSORZERO_CLICKHOUSE_URL is not set");
}
const client = await TensorZeroClient.build(configPath, clickhouseUrl);
const useNativeSFT = process.env.TENSORZERO_UI_FF_USE_NATIVE_SFT === "1";

export function launch_sft_job(data: SFTFormValues): Promise<SFTJob> {
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
  constructor(
    public jobHandle: OptimizerJobHandle,
    public formData: SFTFormValues,
  ) {
    super();
    this.jobHandle = jobHandle;
    this.formData = formData;
    this.jobStatus = "created";
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
    switch (this.jobStatus.type) {
      case "pending":
        return {
          status: "running",
          modelProvider: "openai",
          jobUrl: (this.jobHandle as OpenAISFTJobHandle).job_url,
          formData: this.formData,
          rawData: {
            status: "ok",
            info: this.jobStatus,
          },
        };
      case "failed":
        return {
          status: "error",
          modelProvider: "openai",
          formData: this.formData,
          jobUrl: (this.jobHandle as OpenAISFTJobHandle).job_url,
          rawData: {
            status: "error",
            message: "Job failed",
          },
          error: "Job failed",
        };
      case "completed":
        return {
          status: "completed",
          modelProvider: "openai",
          formData: this.formData,
          jobUrl: (this.jobHandle as OpenAISFTJobHandle).job_url,
          rawData: {
            status: "ok",
            info: this.jobStatus,
          },
          result: "Job completed",
        };
    }
  }

  async poll(): Promise<SFTJob> {
    const status = await client.experimentalPollOptimization(this.jobHandle);
    this.jobStatus = status;
    return this;
  }
}

async function launch_sft_job_native(data: SFTFormValues): Promise<SFTJob> {
  let filters: InferenceFilterTreeNode | null = null;
  let output_source: InferenceOutputSource = "Inference";
  if (data.metric === "demonstration") {
    output_source = "Demonstration";
  } else if (data.metric) {
    filters = await createFilters(data.metric, data.threshold);
  }
  const job = await client.experimentalStartOptimization({
    function_name: data.function,
    template_variant_name: data.variant,
    query_variant_name: null,
    filters: filters,
    output_source: output_source,
    limit: data.maxSamples ? BigInt(data.maxSamples) : BigInt(0),
    offset: BigInt(0),
    val_fraction: data.validationSplitPercent / 100,
    format: "JsonEachRow",
    optimizer_config: {
      type: "openai_sft",
      model: data.model.name,
      batch_size: 1,
      learning_rate_multiplier: 1,
      n_epochs: 1,
      credentials: null,
      api_base: null,
      seed: null,
      suffix: null,
    },
  });
  return NativeSFTJob.from_job_handle_with_form_data(job, data);
}

async function createFilters(
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
