import type { SFTFormValues } from "~/routes/optimization/supervised-fine-tuning/types";
import type {
  InferenceFilterTreeNode,
  InferenceOutputSource,
  OptimizationJobHandle,
  OptimizationJobInfo,
  UninitializedOptimizerInfo,
} from "tensorzero-node";
import { getConfig } from "~/utils/config/index.server";
import { getNativeTensorZeroClient } from "../tensorzero/native_client.server";
import { getEnv } from "../env.server";

export async function poll_sft_job(
  jobHandle: OptimizationJobHandle,
): Promise<OptimizationJobInfo> {
  const client = await getNativeTensorZeroClient();
  const status = await client.experimentalPollOptimization(jobHandle);
  if (status.status === "pending" && status.estimated_finish) {
    status.estimated_finish = new Date(status.estimated_finish);
  }
  return status;
}

export async function launch_sft_job(
  data: SFTFormValues,
): Promise<OptimizationJobHandle> {
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
    order_by: null,
  });
  return job;
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
