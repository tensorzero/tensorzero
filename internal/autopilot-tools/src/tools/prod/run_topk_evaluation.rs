//! Tool for running top-k variant evaluation to identify best-performing variants.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{
    CacheEnabledMode, NonControlToolError, RunTopKEvaluationParams, RunTopKEvaluationResponse,
    ScoringFunctionType, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult,
};
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};

use crate::error::AutopilotToolError;
use autopilot_client::AutopilotSideInfo;

/// Parameters for the run_topk_evaluation tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RunTopKEvaluationToolParams {
    /// Name of the evaluation to run (must be defined in config).
    pub evaluation_name: String,
    /// Name of the dataset to evaluate on.
    pub dataset_name: String,
    /// List of variant names to compare.
    pub variant_names: Vec<String>,
    /// Minimum k for top-k identification.
    pub k_min: u32,
    /// Maximum k for top-k identification.
    pub k_max: u32,
    /// Tolerance for performance equivalence (epsilon).
    #[serde(default)]
    pub epsilon: Option<f64>,
    /// Maximum number of datapoints to process.
    #[serde(default)]
    pub max_datapoints: Option<usize>,
    /// Batch size for processing.
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// Failure rate threshold for variants (default: 0.05).
    /// Variants exceeding this threshold are marked as Failed.
    #[serde(default = "default_failure_threshold")]
    pub variant_failure_threshold: f64,
    /// Failure rate threshold for evaluators (default: 0.05).
    /// The run terminates if any evaluator exceeds this threshold.
    #[serde(default = "default_failure_threshold")]
    pub evaluator_failure_threshold: f64,
    /// Number of concurrent inference requests to make (default: 5).
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// Cache configuration for inference requests.
    /// Defaults to On (caching enabled).
    #[serde(default = "default_inference_cache")]
    pub inference_cache: CacheEnabledMode,
    /// Scoring function type for ranking variants (default: AverageEvaluatorScore).
    #[serde(default = "default_scoring_function")]
    #[schemars(skip)]
    pub scoring_function: ScoringFunctionType,
}

fn default_failure_threshold() -> f64 {
    0.05
}

fn default_concurrency() -> usize {
    5
}

fn default_inference_cache() -> CacheEnabledMode {
    CacheEnabledMode::On
}

fn default_scoring_function() -> ScoringFunctionType {
    ScoringFunctionType::AverageEvaluatorScore
}

/// Tool for running top-k variant evaluation.
///
/// This tool runs an adaptive evaluation algorithm that evaluates multiple variants
/// against a dataset, stopping when it can confidently identify the top-k variants
/// (for some k in [k_min, k_max]).
///
/// The evaluation uses betting confidence sequences for anytime-valid inference,
/// allowing early stopping when sufficient confidence is reached.
#[derive(Default)]
pub struct RunTopKEvaluationTool;

impl ToolMetadata for RunTopKEvaluationTool {
    type SideInfo = AutopilotSideInfo;
    type Output = RunTopKEvaluationResponse;
    type LlmParams = RunTopKEvaluationToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("run_topk_evaluation")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Run a top-k evaluation to identify the best-performing variants from a set. \
             This evaluates multiple variants against a dataset and stops when it can \
             confidently identify the top-k variants (for some k in [k_min, k_max]). \
             Returns the winning variants and confidence statistics.",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Run a top-k evaluation to identify the best-performing variants.",
            "properties": {
                "evaluation_name": {
                    "type": "string",
                    "description": "Name of the evaluation to run (must be defined in config)."
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Name of the dataset to evaluate on."
                },
                "variant_names": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of variant names to compare."
                },
                "k_min": {
                    "type": "integer",
                    "description": "Minimum k for top-k identification."
                },
                "k_max": {
                    "type": "integer",
                    "description": "Maximum k for top-k identification."
                },
                "epsilon": {
                    "type": "number",
                    "description": "Tolerance for performance equivalence (optional)."
                },
                "max_datapoints": {
                    "type": "integer",
                    "description": "Maximum number of datapoints to process (optional)."
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batch size for processing (optional)."
                },
                "variant_failure_threshold": {
                    "type": "number",
                    "description": "Failure rate threshold for variants (default: 0.05)."
                },
                "evaluator_failure_threshold": {
                    "type": "number",
                    "description": "Failure rate threshold for evaluators (default: 0.05)."
                },
                "concurrency": {
                    "type": "integer",
                    "description": "Number of concurrent inference requests (default: 5)."
                },
                "inference_cache": {
                    "type": "string",
                    "enum": ["on", "off", "read_only"],
                    "description": "Cache configuration for inference requests (default: 'on')."
                },
                "scoring_function": {
                    "type": "string",
                    "enum": ["AverageEvaluatorScore"],
                    "description": "Scoring function type for ranking variants (default: 'AverageEvaluatorScore')."
                }
            },
            "required": ["evaluation_name", "dataset_name", "variant_names", "k_min", "k_max"]
        });

        serde_json::from_value(schema).map_err(|e| {
            NonControlToolError::SchemaGeneration {
                message: e.to_string(),
            }
            .into()
        })
    }
}

#[async_trait]
impl SimpleTool for RunTopKEvaluationTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        let params = RunTopKEvaluationParams {
            evaluation_name: llm_params.evaluation_name,
            dataset_name: llm_params.dataset_name,
            variant_names: llm_params.variant_names,
            k_min: llm_params.k_min,
            k_max: llm_params.k_max,
            epsilon: llm_params.epsilon,
            max_datapoints: llm_params.max_datapoints,
            batch_size: llm_params.batch_size,
            variant_failure_threshold: llm_params.variant_failure_threshold,
            evaluator_failure_threshold: llm_params.evaluator_failure_threshold,
            concurrency: llm_params.concurrency,
            inference_cache: llm_params.inference_cache,
            scoring_function: llm_params.scoring_function,
        };

        ctx.client()
            .run_topk_evaluation(params)
            .await
            .map_err(|e| AutopilotToolError::client_error("run_topk_evaluation", e).into())
    }
}
