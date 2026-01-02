//! Tool for running evaluations on datasets.

use std::borrow::Cow;
use std::collections::HashMap;

use async_trait::async_trait;
use durable_tools::{
    CacheEnabledMode, RunEvaluationParams, RunEvaluationResponse, SimpleTool, SimpleToolContext,
    ToolError, ToolMetadata, ToolResult,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use autopilot_client::AutopilotSideInfo;

/// Parameters for the run_evaluation tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RunEvaluationToolParams {
    /// Name of the evaluation to run (must be defined in config).
    pub evaluation_name: String,
    /// Name of the dataset to evaluate on.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    #[serde(default)]
    pub dataset_name: Option<String>,
    /// Specific datapoint IDs to evaluate.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    #[serde(default)]
    pub datapoint_ids: Option<Vec<Uuid>>,
    /// Name of the variant to evaluate.
    pub variant_name: String,
    /// Number of concurrent inference requests to make.
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// Maximum number of datapoints to evaluate from the dataset.
    #[serde(default)]
    pub max_datapoints: Option<u32>,
    /// Precision targets for adaptive stopping.
    /// Maps evaluator names to target confidence interval half-widths.
    /// When the CI half-width for an evaluator falls below its target,
    /// evaluation may stop early for that evaluator.
    #[serde(default)]
    pub precision_targets: HashMap<String, f32>,
    /// Cache configuration for inference requests.
    /// Defaults to On (caching enabled) to match the evaluations CLI behavior.
    #[serde(default = "default_inference_cache")]
    pub inference_cache: CacheEnabledMode,
}

fn default_concurrency() -> usize {
    10
}

fn default_inference_cache() -> CacheEnabledMode {
    CacheEnabledMode::On
}

/// Tool for running evaluations on datasets.
///
/// This tool runs inference on each datapoint in a dataset using the specified variant,
/// then runs the configured evaluators on the results. It returns summary statistics
/// for each evaluator (mean, stderr, count).
#[derive(Default)]
pub struct RunEvaluationTool;

impl ToolMetadata for RunEvaluationTool {
    type SideInfo = AutopilotSideInfo;
    type Output = RunEvaluationResponse;
    type LlmParams = RunEvaluationToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("run_evaluation")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Run an evaluation on a dataset. This runs inference on each datapoint using the \
             specified variant, then runs the configured evaluators. Returns statistics \
             (mean, stderr, count) for each evaluator.",
        )
    }
}

#[async_trait]
impl SimpleTool for RunEvaluationTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        let params = RunEvaluationParams {
            evaluation_name: llm_params.evaluation_name,
            dataset_name: llm_params.dataset_name,
            datapoint_ids: llm_params.datapoint_ids,
            variant_name: llm_params.variant_name,
            concurrency: llm_params.concurrency,
            inference_cache: llm_params.inference_cache,
            max_datapoints: llm_params.max_datapoints,
            precision_targets: llm_params.precision_targets,
        };

        ctx.client()
            .run_evaluation(params)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
