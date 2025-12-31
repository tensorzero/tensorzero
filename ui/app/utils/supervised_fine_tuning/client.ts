import type { SFTFormValues } from "~/routes/optimization/supervised-fine-tuning/types";
import type {
  InferenceFilter,
  InferenceOutputSource,
  OptimizationJobHandle,
  OptimizationJobInfo,
  UninitializedOptimizerInfo,
} from "~/types/tensorzero";
import { getConfig } from "~/utils/config/index.server";
import { getNativeTensorZeroClient } from "../tensorzero/native_client.server";

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
  switch (data.model.provider) {
    case "openai": {
      optimizerConfig = {
        type: "openai_sft",
        model: data.model.name,
        batch_size: 1,
        learning_rate_multiplier: 1,
        n_epochs: 1,
      };
      break;
    }
    case "fireworks": {
      optimizerConfig = {
        type: "fireworks_sft",
        model: data.model.name,
      };
      break;
    }
    case "together": {
      optimizerConfig = {
        type: "together_sft",
        model: data.model.name,
        n_epochs: 1,
        n_checkpoints: 1,
        batch_size: "max",
        learning_rate: 0.00001,
        warmup_ratio: 0,
        max_grad_norm: 1,
        weight_decay: 0,
        lr_scheduler: {
          lr_scheduler_type: "linear",
          min_lr_ratio: 0,
        },
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
      };
      break;
    }
    case "gcp_vertex_gemini": {
      // GCP Vertex Gemini SFT configuration (project_id, region, bucket_name, etc.)
      // comes from [provider_types.gcp_vertex_gemini.sft] in the gateway config
      optimizerConfig = {
        type: "gcp_vertex_gemini_sft",
        model: data.model.name,
      };
      break;
    }
  }

  const job = await client.experimentalLaunchOptimizationWorkflow({
    function_name: data.function,
    template_variant_name: data.variant,
    query_variant_name: null,
    filters: filters,
    output_source: output_source,
    limit: data.maxSamples ? data.maxSamples : 0,
    offset: 0,
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
