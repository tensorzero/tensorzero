import type { SFTFormValues } from "~/routes/optimization/supervised-fine-tuning/types";
import type {
  InferenceFilter,
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
  const togetherNativeSFTBase = getEnv().TOGETHER_BASE_URL;
  let filters: InferenceFilter | null = null;
  let output_source: InferenceOutputSource = "inference";
  if (data.metric === "demonstration") {
    output_source = "demonstration";
  } else if (data.metric) {
    const threshold =
      typeof data.threshold === "string"
        ? parseFloat(data.threshold)
        : data.threshold;
    filters = await createFilters(data.metric, threshold);
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
      early_stop: null,
      epochs: null,
      learning_rate: null,
      max_context_length: null,
      lora_rank: null,
      batch_size: null,
      display_name: null,
      output_model: null,
      warm_start_from: null,
      is_turbo: null,
      eval_auto_carveout: null,
      nodes: null,
      mtp_enabled: null,
      mtp_num_draft_tokens: null,
      mtp_freeze_base_model: null,
      credentials: null,
      api_base: fireworksNativeSFTBase,
      account_id: accountId,
    };
  } else if (data.model.provider == "together") {
    optimizerConfig = {
      type: "together_sft",
      model: data.model.name,
      credentials: null,
      api_base: togetherNativeSFTBase,
      n_epochs: 1,
      n_checkpoints: 1,
      n_evals: null,
      batch_size: "max",
      learning_rate: 0.00001,
      warmup_ratio: 0,
      max_grad_norm: 1,
      weight_decay: 0,
      suffix: null,
      lr_scheduler: {
        lr_scheduler_type: "linear",
        min_lr_ratio: 0,
      },
      wandb_api_key: null,
      wandb_base_url: null,
      wandb_project_name: null,
      wandb_name: null,
      training_method: {
        method: "sft",
      },
      training_type: {
        type: "Lora",
        lora_r: 8,
        lora_alpha: 16,
        lora_dropout: 0,
        lora_trainable_modules: "all-linear",
      },
      from_checkpoint: null,
      from_hf_model: null,
      hf_model_revision: null,
      hf_api_token: null,
      hf_output_repo_name: null,
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
    optimizer_config: optimizerConfig,
    order_by: null,
  });
  return job;
}

export async function createFilters(
  metric: string,
  threshold: number,
): Promise<InferenceFilter> {
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
