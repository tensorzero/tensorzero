//! Evaluation infrastructure for GEPA optimization
//!
//! This module provides functions for:
//! - Evaluating variants on datasets
//! - Consuming evaluation streams
//! - Creating evaluation datasets in ClickHouse
//! - Template-enriched evaluation configurations

use std::collections::HashMap;
use std::sync::Arc;

use serde::Serialize;
use tokio::sync::mpsc;
use uuid::Uuid;

use tensorzero_core::{
    cache::CacheEnabledMode,
    client::Client,
    config::{Config, MetricConfigOptimize, UninitializedVariantConfig, UninitializedVariantInfo},
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::datasets::v1::{
        create_datapoints, delete_dataset,
        types::{
            CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsRequest,
            CreateJsonDatapointRequest, JsonDatapointOutputUpdate,
        },
    },
    error::{Error, ErrorDetails},
    evaluations::{EvaluationConfig, EvaluatorConfig, ExactMatchConfig, LLMJudgeConfig},
    http::TensorzeroHttpClient,
    inference::types::Input,
    stored_inference::{RenderedSample, StoredOutput},
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use evaluations::{
    stats::EvaluationInfo, EvaluationCoreArgs, EvaluationStats, EvaluationUpdate,
    EvaluationVariant, EvaluatorStats, OutputFormat,
};

// Type aliases for cleaner score map signatures

/// Name of an evaluator/metric (e.g., "accuracy", "latency", "f1_score")
///
/// Used to identify specific evaluation metrics across the system.
pub type EvaluatorName = String;

/// Unique identifier for a datapoint/example in a dataset
///
/// Typically a UUID or other unique string identifying a specific test case.
pub type DatapointId = String;

/// Name of a variant being evaluated
///
/// Corresponds to variant names in the TensorZero configuration.
pub type VariantName = String;

/// Scores for all evaluators on a single datapoint
///
/// Maps each evaluator name to its score for this specific datapoint.
/// The score is `Option<f32>` because evaluation may fail (returns `None`).
///
/// # Structure
/// - Key: evaluator name (e.g., "accuracy")
/// - Value: score (e.g., `Some(0.85)`) or `None` if evaluation failed
pub type DatapointScores = HashMap<EvaluatorName, Option<f32>>;

/// Scores for all datapoints for a single variant
///
/// Maps each datapoint ID to the scores for all evaluators on that datapoint.
///
/// # Structure
/// - Key: datapoint ID (unique identifier for the test case)
/// - Value: `DatapointScores` (all evaluator scores for that datapoint)
pub type VariantScores = HashMap<DatapointId, DatapointScores>;

/// Scores for all variants on the validation set
///
/// Top-level structure containing complete evaluation results across all variants
/// and datapoints. This is the primary data structure used in GEPA's Pareto frontier
/// analysis.
///
/// # Structure
/// - Key: variant name (e.g., "baseline", "mutated_iter_3")
/// - Value: `VariantScores` (all datapoint scores for that variant)
///
/// # Example Access Pattern
/// ```ignore
/// let score = validation_scores_map["variant_a"]["datapoint_123"]["accuracy"];
/// ```
pub type ValidationScoresMap = HashMap<VariantName, VariantScores>;

/// Holds the results of evaluating variants on a dataset
#[derive(Clone, Debug)]
pub struct EvaluationResults {
    /// Full evaluation info for each datapoint
    /// Compatible with analyze_inferences(&[EvaluationInfo])
    pub evaluation_infos: Vec<EvaluationInfo>,

    /// Aggregated statistics across all datapoints
    /// Key: evaluator_name
    /// Value: EvaluatorStats with mean/stderr/count
    pub evaluation_stats: HashMap<String, EvaluatorStats>,
}

impl EvaluationResults {
    /// Extract per-datapoint scores for Pareto frontier analysis
    ///
    /// Returns a HashMap mapping datapoint_id to a HashMap of evaluator scores.
    /// Scores are extracted from evaluation_infos on-demand.
    pub fn per_datapoint_scores(&self) -> VariantScores {
        let mut score_map = HashMap::new();

        for info in &self.evaluation_infos {
            let datapoint_id = info.datapoint.id().to_string();
            let mut datapoint_scores = HashMap::new();

            for (evaluator_name, result_opt) in &info.evaluations {
                let score = result_opt.as_ref().and_then(|value| match value {
                    serde_json::Value::Number(n) => n.as_f64().map(|f| f as f32),
                    serde_json::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
                    _ => None,
                });
                datapoint_scores.insert(evaluator_name.clone(), score);
            }

            score_map.insert(datapoint_id, datapoint_scores);
        }

        score_map
    }
}

/// Parameters for evaluating a single variant
pub struct EvaluateVariantParams {
    pub gateway_client: Client,
    pub clickhouse_connection_info: ClickHouseConnectionInfo,
    pub tensorzero_config: Arc<Config>,
    pub evaluation_config: Arc<EvaluationConfigWithInstructions>,
    pub evaluation_name: String,
    pub variant_name: String,
    pub variant_config: UninitializedChatCompletionConfig,
    pub dataset_name: String,
    pub concurrency: usize,
}

/// Evaluate a single variant on a dataset
///
/// Returns the evaluation results or an error if evaluation fails.
/// This is a low-level function; for parallel evaluation of multiple variants,
/// see the orchestration logic in `run_gepa_optimization`.
pub async fn evaluate_variant(params: EvaluateVariantParams) -> Result<EvaluationResults, Error> {
    tracing::info!(
        "Evaluating variant '{}' on dataset '{}'",
        params.variant_name,
        params.dataset_name
    );

    let evaluation_run_id = Uuid::now_v7();

    // Create UninitializedVariantInfo from the chat config
    let dynamic_variant_config = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(params.variant_config),
        timeouts: None,
    };

    // Create EvaluationCoreArgs
    let core_args = EvaluationCoreArgs {
        tensorzero_client: params.gateway_client.clone(),
        clickhouse_client: params.clickhouse_connection_info.clone(),
        config: params.tensorzero_config,
        evaluation_name: params.evaluation_name,
        evaluation_run_id,
        dataset_name: params.dataset_name,
        variant: EvaluationVariant::Info(Box::new(dynamic_variant_config)),
        concurrency: params.concurrency,
        inference_cache: CacheEnabledMode::Off, // Disable caching for fair evaluation
    };

    // Call run_evaluation_core_streaming
    let stream_result = evaluations::run_evaluation_core_streaming(core_args)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to run evaluation: {e}"),
            })
        })?;

    // Consume the streaming channel and aggregate results
    let evaluation_results = consume_evaluation_stream(
        stream_result.receiver,
        &params.evaluation_config,
        stream_result.run_info.num_datapoints,
    )
    .await?;

    Ok(evaluation_results)
}

/// Consume the evaluation stream and aggregate results
///
/// Uses EvaluationStats infrastructure to handle stream consumption and statistics computation,
/// following the same pattern as the evaluations CLI for consistency and maintainability.
pub(crate) async fn consume_evaluation_stream(
    mut receiver: mpsc::Receiver<EvaluationUpdate>,
    evaluation_config: &std::sync::Arc<EvaluationConfigWithInstructions>,
    dataset_len: usize,
) -> Result<EvaluationResults, Error> {
    // Use EvaluationStats to track results (JSONL mode = no progress bar)
    let mut evaluation_stats = EvaluationStats::new(OutputFormat::Jsonl, dataset_len);
    let mut writer = std::io::sink(); // No-op writer for JSONL mode

    // Consume all updates - EvaluationStats handles everything!
    while let Some(update) = receiver.recv().await {
        evaluation_stats.push(update, &mut writer).map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to process evaluation update: {e}"),
            })
        })?;
    }

    // Compute aggregated statistics using EvaluationStats
    let evaluation_stats_map = {
        // Convert EvaluatorConfigWithInstructions back to EvaluatorConfig for compute_stats
        let evaluators: HashMap<String, EvaluatorConfig> = evaluation_config
            .evaluators
            .iter()
            .map(|(name, config_with_instructions)| {
                let config = match config_with_instructions {
                    EvaluatorConfigWithInstructions::ExactMatch(cfg) => {
                        EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: cfg.cutoff })
                    }
                    EvaluatorConfigWithInstructions::LLMJudge(cfg) => {
                        EvaluatorConfig::LLMJudge(LLMJudgeConfig {
                            input_format: cfg.config.input_format,
                            output_type: cfg.config.output_type,
                            include: cfg.config.include,
                            optimize: cfg.config.optimize,
                            cutoff: cfg.config.cutoff,
                        })
                    }
                };
                (name.clone(), config)
            })
            .collect();
        evaluation_stats.compute_stats(&evaluators)
    };

    Ok(EvaluationResults {
        evaluation_infos: evaluation_stats.evaluation_infos,
        evaluation_stats: evaluation_stats_map,
    })
}

/// Create an evaluation dataset from rendered samples
///
/// Uses the datasets v1 API to create datapoints in ClickHouse.
/// This approach provides type-safe validation and handles both Chat and JSON functions.
///
/// # Arguments
/// * `tensorzero_config` - The TensorZero configuration
/// * `http_client` - The HTTP client for fetching resources
/// * `clickhouse_connection_info` - The ClickHouse connection info
/// * `samples` - The rendered samples to convert into datapoints
/// * `dataset_name` - The name of the dataset to create
///
/// # Returns
/// * `()` - Returns success or error
pub async fn create_evaluation_dataset(
    tensorzero_config: &Config,
    http_client: &TensorzeroHttpClient,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    samples: &[RenderedSample],
    dataset_name: &str,
) -> Result<(), Error> {
    // Convert RenderedSamples to CreateDatapointRequest
    let datapoints: Result<Vec<CreateDatapointRequest>, Error> = samples
        .iter()
        .map(|sample| {
            // Convert StoredInput to Input via JSON round-trip
            let input: Input = serde_json::to_value(&sample.stored_input)
                .and_then(serde_json::from_value)
                .map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!("Failed to convert stored input to input: {e}"),
                    })
                })?;

            // Determine if this is a Chat or JSON function based on output type
            let request = if sample.output_schema.is_some()
                || matches!(sample.stored_output, Some(StoredOutput::Json(_)))
            {
                // JSON function
                let output = match &sample.stored_output {
                    Some(StoredOutput::Json(json_output)) => json_output
                        .raw
                        .as_ref()
                        .map(|raw| JsonDatapointOutputUpdate { raw: raw.clone() }),
                    _ => None,
                };

                CreateDatapointRequest::Json(CreateJsonDatapointRequest {
                    function_name: sample.function_name.clone(),
                    episode_id: sample.episode_id,
                    input,
                    output,
                    output_schema: sample.output_schema.clone(),
                    tags: Some(sample.tags.clone()),
                    name: None,
                })
            } else {
                // Chat function
                CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                    function_name: sample.function_name.clone(),
                    episode_id: sample.episode_id,
                    input,
                    output: sample.output.clone(),
                    dynamic_tool_params: sample.tool_params.clone(),
                    tags: Some(sample.tags.clone()),
                    name: None,
                })
            };

            Ok(request)
        })
        .collect();

    let request = CreateDatapointsRequest {
        datapoints: datapoints?,
    };

    // Call the datasets v1 create_datapoints function
    create_datapoints(
        tensorzero_config,
        http_client,
        clickhouse_connection_info,
        dataset_name,
        request,
    )
    .await?;

    Ok(())
}

/// Delete a temporary dataset created during GEPA optimization
///
/// Logs warnings on failure but does not return errors, since cleanup
/// is best-effort and shouldn't fail the optimization.
///
/// # Arguments
/// * `clickhouse_connection_info` - ClickHouse connection for dataset operations
/// * `dataset_name` - Name of the temporary dataset to delete
pub async fn cleanup_temporary_dataset(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    dataset_name: &str,
) {
    match delete_dataset(clickhouse_connection_info, dataset_name).await {
        Ok(response) => {
            tracing::debug!(
                "Cleaned up temporary dataset '{}': deleted {} datapoints",
                dataset_name,
                response.num_deleted_datapoints
            );
        }
        Err(e) => {
            tracing::warn!(
                "Failed to cleanup temporary dataset '{}': {}",
                dataset_name,
                e
            );
        }
    }
}

// ============================================================================
// Template-enriched evaluation configurations
// ============================================================================

/// Evaluation configuration with loaded system instructions for LLM Judge evaluators
///
/// This mirrors `InferenceEvaluationConfig` but includes the loaded system instructions
/// for all LLM Judge evaluators, enabling richer context in GEPA's analyze function.
#[derive(Debug, Clone, Serialize)]
pub struct EvaluationConfigWithInstructions {
    pub evaluators: HashMap<String, EvaluatorConfigWithInstructions>,
    pub function_name: String,
}

impl EvaluationConfigWithInstructions {
    /// Create an evaluation config enriched with system instructions from the TensorZero config
    ///
    /// This loads the system instructions for all LLM Judge evaluators using the
    /// template naming convention: `tensorzero::llm_judge::{evaluation_name}::{evaluator_name}::openai::system`
    ///
    /// # Arguments
    /// * `config` - TensorZero config containing evaluation and template configs
    /// * `evaluation_name` - Name of the evaluation to load
    ///
    /// # Returns
    /// * `Result<Self, Error>` - The enriched evaluation config, or error if template loading fails
    pub fn from_config(config: &Config, evaluation_name: &str) -> Result<Self, Error> {
        // Extract the evaluation config from the TensorZero config
        let evaluation_config = config.evaluations.get(evaluation_name).ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!("Evaluation '{evaluation_name}' not found in config"),
            })
        })?;

        let templates = &config.templates;
        // Extract InferenceEvaluationConfig from the enum
        let EvaluationConfig::Inference(inference_config) = &**evaluation_config;
        let mut enriched_evaluators = HashMap::new();

        for (evaluator_name, evaluator_config) in &inference_config.evaluators {
            let enriched_evaluator = match evaluator_config {
                EvaluatorConfig::ExactMatch(config) => {
                    // ExactMatch evaluators don't have instructions, manually construct
                    EvaluatorConfigWithInstructions::ExactMatch(ExactMatchConfig {
                        cutoff: config.cutoff,
                    })
                }
                EvaluatorConfig::LLMJudge(config) => {
                    // Load the system instructions for this LLM Judge evaluator
                    // Template key format: tensorzero::llm_judge::{evaluation_name}::{evaluator_name}::openai::system
                    let template_key =
                        format!("tensorzero::llm_judge::{evaluation_name}::{evaluator_name}::openai::system");

                    // Render the template with empty context (LLM Judge system instructions don't use context)
                    let system_instructions = templates
                        .template_message(&template_key, &serde_json::Value::Null)
                        .map_err(|e| {
                            Error::new(ErrorDetails::Config {
                                message: format!(
                                    "Failed to load system instructions for LLM Judge evaluator '{evaluator_name}' \
                                    in evaluation '{evaluation_name}': {e}"
                                ),
                            })
                        })?;

                    EvaluatorConfigWithInstructions::LLMJudge(LLMJudgeConfigWithInstructions {
                        config: LLMJudgeConfig {
                            input_format: config.input_format,
                            output_type: config.output_type,
                            include: config.include,
                            optimize: config.optimize,
                            cutoff: config.cutoff,
                        },
                        system_instructions,
                    })
                }
            };

            enriched_evaluators.insert(evaluator_name.clone(), enriched_evaluator);
        }

        Ok(Self {
            evaluators: enriched_evaluators,
            function_name: inference_config.function_name.clone(),
        })
    }
}

/// Evaluator configuration enriched with system instructions (for LLM Judge)
///
/// This mirrors `EvaluatorConfig` but includes loaded system instructions for LLM Judge evaluators.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvaluatorConfigWithInstructions {
    ExactMatch(ExactMatchConfig),
    #[serde(rename = "llm_judge")]
    LLMJudge(LLMJudgeConfigWithInstructions),
}

impl EvaluatorConfigWithInstructions {
    /// Get the optimization direction for this evaluator
    ///
    /// Returns whether this metric should be maximized or minimized.
    pub fn optimize(&self) -> MetricConfigOptimize {
        match self {
            EvaluatorConfigWithInstructions::ExactMatch(_) => MetricConfigOptimize::Max,
            EvaluatorConfigWithInstructions::LLMJudge(config) => config.config.optimize.into(),
        }
    }
}

/// LLM Judge configuration with loaded system instructions
///
/// This extends `LLMJudgeConfig` with the loaded system instructions string,
/// providing full context to the GEPA analyze function.
#[derive(Clone, Debug, Serialize)]
pub struct LLMJudgeConfigWithInstructions {
    #[serde(flatten)]
    pub config: LLMJudgeConfig,
    /// The loaded and rendered system instructions for this LLM Judge
    pub system_instructions: String,
}
