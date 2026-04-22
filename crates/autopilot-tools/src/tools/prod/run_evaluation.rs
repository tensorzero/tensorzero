//! Tool for running evaluations on datasets.

use std::collections::HashMap;
use std::{borrow::Cow, time::Duration};

use async_trait::async_trait;
use durable_tools::{
    ActionInput, ActionResponse, CacheEnabledMode, EvaluatorStats, NonControlToolError,
    RunEvaluationParams, RunEvaluationResponse, SimpleTool, SimpleToolContext, ToolMetadata,
    ToolResult,
};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero_core::config::snapshot::SnapshotHash;
use uuid::Uuid;

use autopilot_client::AutopilotSideInfo;

/// Output of the `run_evaluation` tool (visible to LLM).
///
/// Flattens `RunEvaluationResponse` to preserve the existing wire shape and
/// adds a `diagnostics` vec populated with heuristic warnings when the
/// response looks structurally fine but semantically null (e.g. `stats: {}`,
/// every evaluator uniformly 0.0, zero datapoints). Helps the LLM notice
/// unconfigured-eval paths without polling the tool repeatedly.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct RunEvaluationToolOutput {
    #[serde(flatten)]
    pub response: RunEvaluationResponse,
    /// Heuristic warnings about potentially meaningless results. Empty when
    /// no issues detected. These are hints, not hard errors — the LLM should
    /// still inspect `stats` and decide.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<EvaluationDiagnostic>,
}

/// Heuristic diagnostic for a `run_evaluation` response that looks degenerate.
///
/// Each variant is a pattern observed in agent sessions where workers burned
/// retries trying to figure out why an evaluation returned null/all-zero
/// numbers with no explicit error.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum EvaluationDiagnostic {
    /// `num_datapoints == 0`. The dataset was empty, every row was filtered
    /// out, or `datapoint_ids` resolved to nothing.
    EmptyDataset { message: String },
    /// `stats` is empty despite at least one successful inference. Typically
    /// means no evaluators ran — function/evaluator combo not configured.
    NoEvaluatorStats { message: String },
    /// A specific evaluator has `count == 0` even though other evaluators
    /// scored datapoints. The evaluator likely errored on every row.
    EvaluatorZeroCount { evaluator: String, message: String },
    /// Every inference errored. `num_successes == 0 && num_errors > 0`.
    AllDatapointsErrored { num_errors: usize, message: String },
    /// An evaluator has `count > 0` but both `mean` and `stderr` are exactly
    /// `0.0` — every single datapoint scored zero. Possible cause: the
    /// dataset lacks reference outputs for this evaluator (e.g.
    /// `exact_match` with no gold). Heuristic — a legitimate 100%-fail
    /// evaluation would also match.
    EvaluatorAllZero { evaluator: String, message: String },
}

/// Parameters for the run_evaluation tool (visible to LLM).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct RunEvaluationToolParams {
    /// Name of the evaluation to run (must be defined in config).
    /// Either `evaluation_name` or both (`function_name`, `evaluator_names`) must be provided.
    #[serde(default)]
    pub evaluation_name: Option<String>,
    /// Name of the function to evaluate when using `evaluator_names`.
    /// Either `evaluation_name` or both (`function_name`, `evaluator_names`) must be provided.
    #[serde(default)]
    pub function_name: Option<String>,
    /// Function-scoped evaluator names to run.
    /// Either `evaluation_name` or both (`function_name`, `evaluator_names`) must be provided.
    #[serde(default)]
    pub evaluator_names: Option<Vec<String>>,
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
    type Output = RunEvaluationToolOutput;
    type LlmParams = RunEvaluationToolParams;
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::RUN_EVALUATION_TOOL_PARAMS
    }
    fn llm_params_ts_bundle_type_name() -> String {
        "RunEvaluationToolParams".to_string()
    }
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::RUN_EVALUATION_TOOL_OUTPUT
    }
    fn output_ts_bundle_type_name() -> String {
        "RunEvaluationToolOutput".to_string()
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("run_evaluation")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Run an evaluation on a dataset. Supports two modes: (1) provide `evaluation_name` \
             to run a named evaluation, or (2) provide `function_name` and `evaluator_names` \
             to run specific evaluators for a function. This runs inference on each datapoint \
             using the specified variant, then runs the configured evaluators. Returns statistics \
             (mean, stderr, count) for each evaluator. If the results look degenerate (empty \
             stats, all zero, zero datapoints, all datapoints errored), the response includes \
             a non-empty `diagnostics` array with hints about the likely cause — read those \
             before retrying.",
        )
    }

    fn strict(&self) -> bool {
        false // precision_targets uses additionalProperties: {type: number} not supported in strict mode
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Run an evaluation on a dataset. Either provide `evaluation_name` to run a named evaluation, or provide `function_name` and `evaluator_names` to run specific evaluators for a function.",
            "properties": {
                "evaluation_name": {
                    "type": "string",
                    "description": "Name of the evaluation to run (must be defined in config). Use this OR (`function_name` + `evaluator_names`)."
                },
                "function_name": {
                    "type": "string",
                    "description": "Name of the function to evaluate. Must be provided together with `evaluator_names`. Use this OR `evaluation_name`."
                },
                "evaluator_names": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Function-scoped evaluator names to run. Must be provided together with `function_name`. Use this OR `evaluation_name`."
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
            "required": ["variant_name"],
            "additionalProperties": false
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
        let has_evaluation_name = llm_params.evaluation_name.is_some();
        let has_function_evaluators =
            llm_params.function_name.is_some() || llm_params.evaluator_names.is_some();

        if has_evaluation_name && has_function_evaluators {
            return Err(AutopilotToolError::validation(
                "Provide either `evaluation_name` or (`function_name` + `evaluator_names`), not both",
            )
            .into());
        }
        if !has_evaluation_name && !has_function_evaluators {
            return Err(AutopilotToolError::validation(
                "Must provide either `evaluation_name` or both `function_name` and `evaluator_names`",
            )
            .into());
        }
        if has_function_evaluators
            && (llm_params.function_name.is_none() || llm_params.evaluator_names.is_none())
        {
            return Err(AutopilotToolError::validation(
                "`function_name` and `evaluator_names` must both be provided together",
            )
            .into());
        }

        let params = RunEvaluationParams {
            evaluation_name: llm_params.evaluation_name,
            function_name: llm_params.function_name,
            evaluator_names: llm_params.evaluator_names,
            dataset_name: llm_params.dataset_name,
            datapoint_ids: llm_params.datapoint_ids,
            variant_name: llm_params.variant_name,
            concurrency: llm_params.concurrency,
            inference_cache: llm_params.inference_cache,
            max_datapoints: llm_params.max_datapoints,
            precision_targets: llm_params.precision_targets,
            include_datapoint_results: llm_params.include_datapoint_results,
            tags: side_info.to_tags(),
            internal_dynamic_variant_config: None,
            include_evaluation_infos: false,
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
            .action(
                snapshot_hash,
                ActionInput::RunEvaluation(Box::new(params)),
                ctx.heartbeater().clone(),
            )
            .await
            .map_err(|e| AutopilotToolError::client_error("run_evaluation", e))?;

        match response {
            ActionResponse::RunEvaluation(eval_response) => {
                let diagnostics = compute_diagnostics(&eval_response);
                Ok(RunEvaluationToolOutput {
                    response: eval_response,
                    diagnostics,
                })
            }
            _ => Err(AutopilotToolError::validation(
                "Unexpected response type from action endpoint",
            )
            .into()),
        }
    }
}

/// Inspect a `RunEvaluationResponse` for patterns that commonly indicate an
/// unconfigured or misaligned evaluation, and emit human-readable hints.
///
/// Called after a successful action-endpoint roundtrip, so we only see
/// aggregated results — sample error messages are not available here.
/// Diagnostics that point at likely data issues suggest re-running with
/// `include_datapoint_results=true` to get per-row error context.
fn compute_diagnostics(response: &RunEvaluationResponse) -> Vec<EvaluationDiagnostic> {
    let mut diagnostics = Vec::new();

    if response.num_datapoints == 0 {
        diagnostics.push(EvaluationDiagnostic::EmptyDataset {
            message: "No datapoints were evaluated. Verify `dataset_name` exists \
                      and contains rows, or that `datapoint_ids` resolves to \
                      known datapoints."
                .to_string(),
        });
        // Other diagnostics are meaningless when there's nothing to evaluate.
        return diagnostics;
    }

    if response.num_successes == 0 && response.num_errors > 0 {
        diagnostics.push(EvaluationDiagnostic::AllDatapointsErrored {
            num_errors: response.num_errors,
            message: format!(
                "All {} datapoints errored during inference. Re-run with \
                 `include_datapoint_results=true` to inspect per-datapoint error messages.",
                response.num_errors
            ),
        });
    }

    if response.stats.is_empty() && response.num_successes > 0 {
        diagnostics.push(EvaluationDiagnostic::NoEvaluatorStats {
            message: "Inference succeeded but no evaluator produced stats. \
                      The evaluator is likely not configured for this function — \
                      check the config or pick a different evaluator/function combo."
                .to_string(),
        });
    }

    let mut zero_count: Vec<&str> = response
        .stats
        .iter()
        .filter_map(|(name, s)| (s.count == 0).then_some(name.as_str()))
        .collect();
    zero_count.sort_unstable();
    for evaluator in zero_count {
        diagnostics.push(EvaluationDiagnostic::EvaluatorZeroCount {
            evaluator: evaluator.to_string(),
            message: format!(
                "Evaluator `{evaluator}` scored zero datapoints. It likely \
                 errored on every row — verify the datapoints have the fields \
                 this evaluator requires, or re-run with \
                 `include_datapoint_results=true` to see per-row errors."
            ),
        });
    }

    let mut all_zero: Vec<&str> = response
        .stats
        .iter()
        .filter_map(|(name, s)| is_uniformly_zero(s).then_some(name.as_str()))
        .collect();
    all_zero.sort_unstable();
    for evaluator in all_zero {
        diagnostics.push(EvaluationDiagnostic::EvaluatorAllZero {
            evaluator: evaluator.to_string(),
            message: format!(
                "Evaluator `{evaluator}` scored exactly 0 on every datapoint \
                 (mean and stderr both 0). Possible data issue — the dataset \
                 may lack reference outputs this evaluator compares against \
                 (e.g. `exact_match` with no gold). Re-run with \
                 `include_datapoint_results=true` to confirm."
            ),
        });
    }

    diagnostics
}

/// `count > 0 && mean == 0.0 && stderr == 0.0` — every scored datapoint
/// produced exactly 0. Heuristic: a legitimate 100%-fail evaluation would
/// also match, so callers should treat this as a possible-data-issue hint.
fn is_uniformly_zero(stats: &EvaluatorStats) -> bool {
    stats.count > 0 && stats.mean == 0.0 && stats.stderr == 0.0
}
