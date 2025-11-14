//! Evaluation infrastructure for GEPA optimization
//!
//! This module provides functions for:
//! - Evaluating variants on datasets
//! - Consuming evaluation streams
//! - Creating evaluation datasets in ClickHouse

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;
use uuid::Uuid;

use tensorzero_core::{
    cache::CacheEnabledMode,
    client::Client,
    config::{Config, UninitializedVariantConfig, UninitializedVariantInfo},
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::datasets::v1::{
        create_datapoints,
        types::{
            CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsRequest,
            CreateJsonDatapointRequest, JsonDatapointOutputUpdate,
        },
    },
    error::{Error, ErrorDetails},
    evaluations::EvaluationConfig,
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

/// An evaluator/metric name
pub type EvaluatorName = String;

/// A datapoint/example identifier
pub type DatapointId = String;

/// A variant name identifier
pub type VariantName = String;

/// Scores for all evaluators on a single datapoint
/// Key: evaluator_name, Value: score (None if evaluation failed)
pub type DatapointScores = HashMap<EvaluatorName, Option<f32>>;

/// Scores for all datapoints for a single variant
/// Key: datapoint_id, Value: scores for all evaluators on that datapoint
pub type VariantScores = HashMap<DatapointId, DatapointScores>;

/// Scores for all variants on the validation set
/// Key: variant_name, Value: scores for all datapoints for that variant
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
    pub evaluation_config: Arc<EvaluationConfig>,
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
pub async fn consume_evaluation_stream(
    mut receiver: mpsc::Receiver<EvaluationUpdate>,
    evaluation_config: &std::sync::Arc<EvaluationConfig>,
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
        let evaluators = match &**evaluation_config {
            EvaluationConfig::Inference(inference_config) => &inference_config.evaluators,
        };
        evaluation_stats.compute_stats(evaluators)
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
