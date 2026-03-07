//! Tool for running evaluations on datasets.

use std::collections::HashMap;
use std::{borrow::Cow, time::Duration};

use async_trait::async_trait;
use durable_tools::{
    ActionInput, ActionResponse, CacheEnabledMode, NonControlToolError, RunEvaluationParams,
    RunEvaluationResponse, TaskTool, ToolContext, ToolMetadata, ToolResult,
};
use tensorzero::ListDatapointsRequest;

use crate::error::AutopilotToolError;
use durable_tools::EvaluatorStats;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero_core::config::snapshot::SnapshotHash;
use uuid::Uuid;

use autopilot_client::AutopilotSideInfo;

const DEFAULT_BATCH_SIZE: usize = 25;

/// Parameters for the run_evaluation tool (visible to LLM).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
    /// Number of datapoints per batch. Each batch is a separate checkpointed step.
    /// Default is 25.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

fn default_concurrency() -> usize {
    10
}

fn default_inference_cache() -> CacheEnabledMode {
    CacheEnabledMode::On
}

fn default_batch_size() -> usize {
    DEFAULT_BATCH_SIZE
}

/// Tool for running evaluations on datasets.
///
/// This tool runs inference on each datapoint in a dataset using the specified variant,
/// then runs the configured evaluators on the results. It returns summary statistics
/// for each evaluator (mean, stderr, count).
///
/// Evaluations are broken into batches, with each batch as a separate checkpointed step
/// to avoid exceeding durable task claim timeouts for large datasets.
#[derive(Default)]
pub struct RunEvaluationTool;

impl ToolMetadata for RunEvaluationTool {
    type SideInfo = AutopilotSideInfo;
    type Output = RunEvaluationResponse;
    type LlmParams = RunEvaluationToolParams;

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::RUN_EVALUATION_TOOL_PARAMS
    }

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle_type_name() -> String {
        "RunEvaluationToolParams".to_string()
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::RUN_EVALUATION_RESPONSE
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle_type_name() -> String {
        "RunEvaluationResponse".to_string()
    }

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

    fn strict(&self) -> bool {
        false // precision_targets uses additionalProperties: {type: number} not supported in strict mode
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
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Number of datapoints per batch. Each batch is a separate checkpointed step (default: 25)."
                }
            },
            "required": ["evaluation_name", "variant_name"],
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

/// Parameters for the fetch_datapoint_ids step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FetchDatapointIdsParams {
    dataset_name: String,
    limit: u32,
}

/// Parameters for a batch evaluation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchEvalParams {
    snapshot_hash: String,
    evaluation_name: String,
    datapoint_ids: Vec<Uuid>,
    variant_name: String,
    concurrency: usize,
    inference_cache: CacheEnabledMode,
    include_datapoint_results: bool,
    tags: HashMap<String, String>,
}

/// Running statistics for Welford's online algorithm.
#[derive(Debug, Clone)]
struct RunningEvaluatorStats {
    count: usize,
    mean: f64,
    m2: f64,
}

impl RunningEvaluatorStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Merge a batch's stats into running totals using parallel Welford's algorithm.
    fn update(&mut self, batch_stats: &EvaluatorStats) {
        if batch_stats.count == 0 {
            return;
        }

        let batch_count = batch_stats.count;
        let batch_mean = batch_stats.mean as f64;
        // Recover batch variance from stderr: var = stderr^2 * count
        let batch_variance = (batch_stats.stderr as f64).powi(2) * batch_count as f64;
        let batch_m2 = batch_variance * batch_count as f64;

        let new_count = self.count + batch_count;
        if new_count == 0 {
            return;
        }

        let delta = batch_mean - self.mean;
        let new_mean =
            (self.mean * self.count as f64 + batch_mean * batch_count as f64) / new_count as f64;

        // Parallel combination of M2 values
        let new_m2 = self.m2
            + batch_m2
            + delta * delta * (self.count as f64 * batch_count as f64) / new_count as f64;

        self.count = new_count;
        self.mean = new_mean;
        self.m2 = new_m2;
    }

    /// Compute current standard error from running stats.
    fn stderr(&self) -> f32 {
        if self.count <= 1 {
            return 0.0;
        }
        let variance = self.m2 / self.count as f64;
        (variance / self.count as f64).sqrt() as f32
    }

    /// Convert to final EvaluatorStats.
    fn to_evaluator_stats(&self) -> EvaluatorStats {
        EvaluatorStats {
            mean: self.mean as f32,
            stderr: self.stderr(),
            count: self.count,
        }
    }
}

/// Check if all targeted evaluators have converged (stderr < target).
fn check_precision_convergence(
    running: &HashMap<String, RunningEvaluatorStats>,
    targets: &HashMap<String, f32>,
) -> bool {
    if targets.is_empty() {
        return false;
    }
    targets.iter().all(|(name, &target)| {
        running
            .get(name)
            .is_some_and(|stats| stats.count > 1 && stats.stderr() < target)
    })
}

/// Aggregate multiple batch responses into a single response.
pub(crate) fn aggregate_responses(responses: Vec<RunEvaluationResponse>) -> RunEvaluationResponse {
    if responses.is_empty() {
        return RunEvaluationResponse {
            evaluation_run_id: Uuid::nil(),
            num_datapoints: 0,
            num_successes: 0,
            num_errors: 0,
            stats: HashMap::new(),
            datapoint_results: None,
        };
    }

    let evaluation_run_id = responses[0].evaluation_run_id;

    if responses.len() == 1 {
        return responses.into_iter().next().unwrap_or_else(|| {
            // This branch is unreachable since we checked len == 1
            RunEvaluationResponse {
                evaluation_run_id,
                num_datapoints: 0,
                num_successes: 0,
                num_errors: 0,
                stats: HashMap::new(),
                datapoint_results: None,
            }
        });
    }
    let num_datapoints: usize = responses.iter().map(|r| r.num_datapoints).sum();
    let num_successes: usize = responses.iter().map(|r| r.num_successes).sum();
    let num_errors: usize = responses.iter().map(|r| r.num_errors).sum();

    let stats = merge_evaluator_stats(&responses);

    // Concatenate datapoint_results if any batch has them
    let has_any_results = responses.iter().any(|r| r.datapoint_results.is_some());
    let datapoint_results = if has_any_results {
        let mut all_results = Vec::new();
        for r in &responses {
            if let Some(ref results) = r.datapoint_results {
                all_results.extend(results.iter().cloned());
            }
        }
        Some(all_results)
    } else {
        None
    };

    RunEvaluationResponse {
        evaluation_run_id,
        num_datapoints,
        num_successes,
        num_errors,
        stats,
        datapoint_results,
    }
}

/// Merge evaluator stats across multiple responses using combined statistics.
fn merge_evaluator_stats(responses: &[RunEvaluationResponse]) -> HashMap<String, EvaluatorStats> {
    let mut running: HashMap<String, RunningEvaluatorStats> = HashMap::new();

    for response in responses {
        for (name, batch_stats) in &response.stats {
            running
                .entry(name.clone())
                .or_insert_with(RunningEvaluatorStats::new)
                .update(batch_stats);
        }
    }

    running
        .into_iter()
        .map(|(name, stats)| (name, stats.to_evaluator_stats()))
        .collect()
}

#[async_trait]
impl TaskTool for RunEvaluationTool {
    type ExtraState = ();

    async fn execute(
        &self,
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: &mut ToolContext,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        let snapshot_hash: SnapshotHash = side_info
            .config_snapshot_hash
            .parse()
            .map_err(|_| AutopilotToolError::validation("Invalid snapshot hash"))?;

        let tags = side_info.to_tags();
        let batch_size = if llm_params.batch_size == 0 {
            DEFAULT_BATCH_SIZE
        } else {
            llm_params.batch_size
        };

        // Step 1: Resolve datapoint IDs
        let datapoint_ids: Vec<Uuid> = if let Some(ids) = llm_params.datapoint_ids {
            // Datapoint IDs provided directly — apply max_datapoints truncation
            match llm_params.max_datapoints {
                Some(max) => ids.into_iter().take(max as usize).collect(),
                None => ids,
            }
        } else if let Some(dataset_name) = &llm_params.dataset_name {
            let limit = llm_params.max_datapoints.unwrap_or(u32::MAX);
            let fetch_params = FetchDatapointIdsParams {
                dataset_name: dataset_name.clone(),
                limit,
            };

            let ids: Vec<Uuid> = ctx
                .step(
                    "fetch_datapoint_ids",
                    fetch_params,
                    |params, state| async move {
                        let response = state
                            .t0_client()
                            .list_datapoints(
                                params.dataset_name,
                                ListDatapointsRequest {
                                    limit: Some(params.limit),
                                    ..Default::default()
                                },
                            )
                            .await
                            .map_err(|e| {
                                anyhow::Error::msg(format!("Failed to list datapoints: {e}"))
                            })?;
                        Ok(response.datapoints.iter().map(|d| d.id()).collect())
                    },
                )
                .await?;
            ids
        } else {
            return Err(AutopilotToolError::validation(
                "Either `dataset_name` or `datapoint_ids` must be provided",
            )
            .into());
        };

        // Handle empty dataset
        if datapoint_ids.is_empty() {
            return Ok(RunEvaluationResponse {
                evaluation_run_id: Uuid::nil(),
                num_datapoints: 0,
                num_successes: 0,
                num_errors: 0,
                stats: HashMap::new(),
                datapoint_results: None,
            });
        }

        // Step 2: Run batches
        let chunks: Vec<Vec<Uuid>> = datapoint_ids
            .chunks(batch_size)
            .map(|c| c.to_vec())
            .collect();

        let mut responses: Vec<RunEvaluationResponse> = Vec::with_capacity(chunks.len());
        let mut running_stats: HashMap<String, RunningEvaluatorStats> = HashMap::new();

        for (i, chunk) in chunks.into_iter().enumerate() {
            let batch_params = BatchEvalParams {
                snapshot_hash: snapshot_hash.to_string(),
                evaluation_name: llm_params.evaluation_name.clone(),
                datapoint_ids: chunk,
                variant_name: llm_params.variant_name.clone(),
                concurrency: llm_params.concurrency,
                inference_cache: llm_params.inference_cache,
                include_datapoint_results: llm_params.include_datapoint_results,
                tags: tags.clone(),
            };

            let batch_response: RunEvaluationResponse = ctx
                .step(
                    &format!("batch_{i}"),
                    batch_params,
                    |params, state| async move {
                        let snapshot_hash: SnapshotHash =
                            params.snapshot_hash.parse().map_err(|_| {
                                anyhow::Error::msg("Invalid snapshot hash in batch params")
                            })?;

                        let eval_params = RunEvaluationParams {
                            evaluation_name: params.evaluation_name,
                            dataset_name: None,
                            datapoint_ids: Some(params.datapoint_ids),
                            variant_name: params.variant_name,
                            concurrency: params.concurrency,
                            inference_cache: params.inference_cache,
                            max_datapoints: None,
                            precision_targets: HashMap::new(),
                            include_datapoint_results: params.include_datapoint_results,
                            tags: params.tags,
                        };

                        let response = state
                            .t0_client()
                            .action(
                                snapshot_hash,
                                ActionInput::RunEvaluation(Box::new(eval_params)),
                            )
                            .await
                            .map_err(|e| anyhow::Error::msg(format!("{e}")))?;

                        match response {
                            ActionResponse::RunEvaluation(eval_response) => Ok(eval_response),
                            _ => Err(anyhow::Error::msg(
                                "Unexpected response type from action endpoint",
                            )),
                        }
                    },
                )
                .await?;

            // Update running stats for precision convergence check
            for (name, batch_stats) in &batch_response.stats {
                running_stats
                    .entry(name.clone())
                    .or_insert_with(RunningEvaluatorStats::new)
                    .update(batch_stats);
            }

            responses.push(batch_response);

            // Check precision convergence after each batch
            if !llm_params.precision_targets.is_empty()
                && check_precision_convergence(&running_stats, &llm_params.precision_targets)
            {
                break;
            }
        }

        Ok(aggregate_responses(responses))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_running_evaluator_stats_single_batch() {
        let mut running = RunningEvaluatorStats::new();
        running.update(&EvaluatorStats {
            mean: 0.85,
            stderr: 0.02,
            count: 100,
        });

        let result = running.to_evaluator_stats();
        assert_eq!(result.count, 100);
        assert!((result.mean - 0.85).abs() < 1e-5, "mean should be 0.85");
        assert!(
            (result.stderr - 0.02).abs() < 1e-4,
            "stderr should be ~0.02, got {}",
            result.stderr
        );
    }

    #[test]
    fn test_running_evaluator_stats_two_batches_equal() {
        // Two equal batches should produce the same mean
        let mut running = RunningEvaluatorStats::new();
        running.update(&EvaluatorStats {
            mean: 0.80,
            stderr: 0.02,
            count: 50,
        });
        running.update(&EvaluatorStats {
            mean: 0.90,
            stderr: 0.02,
            count: 50,
        });

        let result = running.to_evaluator_stats();
        assert_eq!(result.count, 100);
        assert!(
            (result.mean - 0.85).abs() < 1e-5,
            "combined mean should be 0.85"
        );
    }

    #[test]
    fn test_running_evaluator_stats_zero_count_batch() {
        let mut running = RunningEvaluatorStats::new();
        running.update(&EvaluatorStats {
            mean: 0.85,
            stderr: 0.02,
            count: 100,
        });
        running.update(&EvaluatorStats {
            mean: 0.0,
            stderr: 0.0,
            count: 0,
        });

        let result = running.to_evaluator_stats();
        assert_eq!(result.count, 100);
        assert!((result.mean - 0.85).abs() < 1e-5, "mean should stay 0.85");
    }

    #[test]
    fn test_aggregate_single_batch() {
        let response = RunEvaluationResponse {
            evaluation_run_id: Uuid::now_v7(),
            num_datapoints: 10,
            num_successes: 9,
            num_errors: 1,
            stats: HashMap::from([(
                "accuracy".to_string(),
                EvaluatorStats {
                    mean: 0.9,
                    stderr: 0.03,
                    count: 9,
                },
            )]),
            datapoint_results: None,
        };
        let expected_id = response.evaluation_run_id;

        let result = aggregate_responses(vec![response]);
        assert_eq!(result.evaluation_run_id, expected_id);
        assert_eq!(result.num_datapoints, 10);
        assert_eq!(result.num_successes, 9);
        assert_eq!(result.num_errors, 1);
        assert_eq!(result.stats["accuracy"].count, 9);
    }

    #[test]
    fn test_aggregate_multiple_batches() {
        let id1 = Uuid::now_v7();
        let responses = vec![
            RunEvaluationResponse {
                evaluation_run_id: id1,
                num_datapoints: 25,
                num_successes: 24,
                num_errors: 1,
                stats: HashMap::from([(
                    "accuracy".to_string(),
                    EvaluatorStats {
                        mean: 0.80,
                        stderr: 0.04,
                        count: 24,
                    },
                )]),
                datapoint_results: None,
            },
            RunEvaluationResponse {
                evaluation_run_id: Uuid::now_v7(),
                num_datapoints: 25,
                num_successes: 23,
                num_errors: 2,
                stats: HashMap::from([(
                    "accuracy".to_string(),
                    EvaluatorStats {
                        mean: 0.90,
                        stderr: 0.03,
                        count: 23,
                    },
                )]),
                datapoint_results: None,
            },
        ];

        let result = aggregate_responses(responses);
        assert_eq!(result.evaluation_run_id, id1, "should use first batch's ID");
        assert_eq!(result.num_datapoints, 50);
        assert_eq!(result.num_successes, 47);
        assert_eq!(result.num_errors, 3);

        let acc = &result.stats["accuracy"];
        assert_eq!(acc.count, 47);
        // Combined mean should be weighted average
        let expected_mean = (0.80 * 24.0 + 0.90 * 23.0) / 47.0;
        assert!(
            (acc.mean - expected_mean as f32).abs() < 1e-4,
            "mean should be ~{expected_mean}, got {}",
            acc.mean
        );
    }

    #[test]
    fn test_aggregate_empty_responses() {
        let result = aggregate_responses(vec![]);
        assert_eq!(result.num_datapoints, 0);
        assert_eq!(result.num_successes, 0);
        assert_eq!(result.num_errors, 0);
        assert!(result.stats.is_empty());
        assert!(result.datapoint_results.is_none());
    }

    #[test]
    fn test_aggregate_with_datapoint_results() {
        use durable_tools::DatapointResult;

        let dp1 = DatapointResult {
            datapoint_id: Uuid::now_v7(),
            success: true,
            evaluations: HashMap::new(),
            evaluator_errors: HashMap::new(),
            error: None,
        };
        let dp2 = DatapointResult {
            datapoint_id: Uuid::now_v7(),
            success: true,
            evaluations: HashMap::new(),
            evaluator_errors: HashMap::new(),
            error: None,
        };

        let responses = vec![
            RunEvaluationResponse {
                evaluation_run_id: Uuid::now_v7(),
                num_datapoints: 1,
                num_successes: 1,
                num_errors: 0,
                stats: HashMap::new(),
                datapoint_results: Some(vec![dp1.clone()]),
            },
            RunEvaluationResponse {
                evaluation_run_id: Uuid::now_v7(),
                num_datapoints: 1,
                num_successes: 1,
                num_errors: 0,
                stats: HashMap::new(),
                datapoint_results: Some(vec![dp2.clone()]),
            },
        ];

        let result = aggregate_responses(responses);
        let results = result
            .datapoint_results
            .expect("should have datapoint_results");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].datapoint_id, dp1.datapoint_id);
        assert_eq!(results[1].datapoint_id, dp2.datapoint_id);
    }

    #[test]
    fn test_merge_evaluator_stats_disjoint() {
        // One evaluator only in first batch, another only in second
        let responses = vec![
            RunEvaluationResponse {
                evaluation_run_id: Uuid::now_v7(),
                num_datapoints: 10,
                num_successes: 10,
                num_errors: 0,
                stats: HashMap::from([(
                    "accuracy".to_string(),
                    EvaluatorStats {
                        mean: 0.9,
                        stderr: 0.03,
                        count: 10,
                    },
                )]),
                datapoint_results: None,
            },
            RunEvaluationResponse {
                evaluation_run_id: Uuid::now_v7(),
                num_datapoints: 10,
                num_successes: 10,
                num_errors: 0,
                stats: HashMap::from([(
                    "quality".to_string(),
                    EvaluatorStats {
                        mean: 0.8,
                        stderr: 0.05,
                        count: 10,
                    },
                )]),
                datapoint_results: None,
            },
        ];

        let result = aggregate_responses(responses);
        assert!(result.stats.contains_key("accuracy"));
        assert!(result.stats.contains_key("quality"));
        assert_eq!(result.stats["accuracy"].count, 10);
        assert_eq!(result.stats["quality"].count, 10);
    }

    #[test]
    fn test_check_precision_convergence_empty_targets() {
        let running = HashMap::new();
        let targets = HashMap::new();
        assert!(
            !check_precision_convergence(&running, &targets),
            "empty targets should not converge"
        );
    }

    #[test]
    fn test_check_precision_convergence_met() {
        let mut running = HashMap::new();
        let mut stats = RunningEvaluatorStats::new();
        stats.update(&EvaluatorStats {
            mean: 0.85,
            stderr: 0.01,
            count: 100,
        });
        running.insert("accuracy".to_string(), stats);

        let targets = HashMap::from([("accuracy".to_string(), 0.02)]);
        assert!(
            check_precision_convergence(&running, &targets),
            "should converge when stderr < target"
        );
    }

    #[test]
    fn test_check_precision_convergence_not_met() {
        let mut running = HashMap::new();
        let mut stats = RunningEvaluatorStats::new();
        stats.update(&EvaluatorStats {
            mean: 0.85,
            stderr: 0.05,
            count: 10,
        });
        running.insert("accuracy".to_string(), stats);

        let targets = HashMap::from([("accuracy".to_string(), 0.01)]);
        assert!(
            !check_precision_convergence(&running, &targets),
            "should not converge when stderr > target"
        );
    }
}
