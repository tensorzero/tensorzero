//! Tool for running evaluations on datasets.

use std::collections::HashMap;
use std::{borrow::Cow, time::Duration};

use async_trait::async_trait;
use durable_tools::{
    ActionInput, ActionResponse, CacheEnabledMode, NonControlToolError, RunEvaluationParams,
    RunEvaluationResponse, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult,
};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero_core::config::snapshot::SnapshotHash;
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
    /// Include per-datapoint results in the response.
    /// When true, the response will include individual results for each datapoint.
    /// Default is false to avoid response bloat for large evaluations.
    #[serde(default)]
    pub include_datapoint_results: bool,
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

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("run_evaluation")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Run an evaluation on a dataset. This runs inference on each datapoint using the \
             specified variant, then runs the configured evaluators. Returns statistics \
             (mean, stderr, count) for each evaluator.",
        )
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Run an evaluation on a dataset using configured evaluators.",
            "properties": {
                "evaluation_name": {
                    "type": "string",
                    "description": "Name of the evaluation to run (must be defined in config)."
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Name of the dataset to evaluate on. Either dataset_name or datapoint_ids must be provided, but not both."
                },
                "datapoint_ids": {
                    "type": "array",
                    "items": { "type": "string", "format": "uuid" },
                    "description": "Specific datapoint IDs to evaluate. Either dataset_name or datapoint_ids must be provided, but not both."
                },
                "variant_name": {
                    "type": "string",
                    "description": "Name of the variant to evaluate."
                },
                "concurrency": {
                    "type": "integer",
                    "description": "Number of concurrent inference requests (default: 10)."
                },
                "max_datapoints": {
                    "type": "integer",
                    "description": "Maximum number of datapoints to evaluate from the dataset (optional)."
                },
                "precision_targets": {
                    "type": "object",
                    "description": "Precision targets for adaptive stopping. Maps evaluator names to target confidence interval half-widths.",
                    "additionalProperties": { "type": "number" }
                },
                "inference_cache": {
                    "type": "string",
                    "enum": ["on", "off", "read_only"],
                    "description": "Cache configuration for inference requests (default: 'on')."
                },
                "include_datapoint_results": {
                    "type": "boolean",
                    "description": "Include per-datapoint results in the response (default: false)."
                }
            },
            "required": ["evaluation_name", "variant_name"]
        });

        serde_json::from_value(schema).map_err(|e| {
            NonControlToolError::SchemaGeneration {
                message: e.to_string(),
            }
            .into()
        })
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(30 * 60)
    }
}

#[async_trait]
impl SimpleTool for RunEvaluationTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
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
            include_datapoint_results: llm_params.include_datapoint_results,
            tags: side_info.to_tags(),
        };

        // Since autopilot sessions always have a config snapshot hash set, we use the action
        // endpoint to ensure evaluations run against the historical config snapshot.
        // If this assumption changes (e.g., we want to run against current config), use this instead:
        //
        // ctx.client()
        //     .run_evaluation(params)
        //     .await
        //     .map_err(|e| AutopilotToolError::client_error("run_evaluation", e).into())

        let snapshot_hash: SnapshotHash = side_info
            .config_snapshot_hash
            .parse()
            .map_err(|_| AutopilotToolError::validation("Invalid snapshot hash"))?;

        let response = ctx
            .client()
            .action(snapshot_hash, ActionInput::RunEvaluation(Box::new(params)))
            .await
            .map_err(|e| AutopilotToolError::client_error("run_evaluation", e))?;

        match response {
            ActionResponse::RunEvaluation(eval_response) => Ok(eval_response),
            _ => Err(AutopilotToolError::validation(
                "Unexpected response type from action endpoint",
            )
            .into()),
        }
    }
}
