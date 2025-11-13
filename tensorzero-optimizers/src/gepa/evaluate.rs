//! Evaluation infrastructure for GEPA optimization
//!
//! This module provides functions for:
//! - Evaluating variants on datasets
//! - Consuming evaluation streams
//! - Creating evaluation datasets in ClickHouse

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use tokio::sync::{mpsc, Semaphore};
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
    optimization::gepa::GEPAConfig,
    stored_inference::{RenderedSample, StoredOutput},
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use evaluations::{
    EvaluationCoreArgs, EvaluationStats, EvaluationUpdate, EvaluationVariant, EvaluatorStats,
    OutputFormat,
};

/// Holds the results of evaluating variants on a dataset
/// This matches the EvaluationResults structure from planning.md
#[derive(Clone, Debug)]
pub struct EvaluationResults {
    /// Per-datapoint evaluation results
    /// Outer key: datapoint_id (String)
    /// Inner key: evaluator_name (String)
    /// Value: Option<f32> - None if evaluation failed for that datapoint
    pub per_datapoint: HashMap<String, HashMap<String, Option<f32>>>,

    /// Aggregated statistics across all datapoints
    /// Key: evaluator_name
    /// Value: EvaluatorStats with mean/stderr/count
    pub metrics: HashMap<String, EvaluatorStats>,
}

/// Evaluate multiple variants on a dataset
/// Returns HashMap<variant_name, Option<evaluation_results>>
/// None indicates evaluation failure for that variant (graceful degradation)
pub async fn evaluate_variants(
    gateway_client: &Client,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    tensorzero_config: std::sync::Arc<Config>,
    config: &GEPAConfig,
    variant_configs: &HashMap<String, UninitializedChatCompletionConfig>,
    dataset_name: &str,
) -> Result<HashMap<String, Option<EvaluationResults>>, Error> {
    let concurrency = config.max_concurrency as usize;

    // Get evaluation config for later use
    let evaluation_config = tensorzero_config
        .evaluations
        .get(&config.evaluation_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "Evaluation '{}' not found in config",
                    config.evaluation_name
                ),
            })
        })?
        .clone();

    // Create semaphore for concurrency control
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let evaluation_name = config.evaluation_name.clone();

    // Create futures for parallel execution
    let evaluation_futures: Vec<_> = variant_configs
        .iter()
        .map(|(variant_name, chat_config)| {
            let semaphore = Arc::clone(&semaphore);
            let gateway_client = gateway_client.clone();
            let clickhouse_connection_info = clickhouse_connection_info.clone();
            let tensorzero_config = Arc::clone(&tensorzero_config);
            let evaluation_config = Arc::clone(&evaluation_config);
            let evaluation_name = evaluation_name.clone();
            let variant_name = variant_name.clone();
            let chat_config = chat_config.clone();
            let dataset_name = dataset_name.to_string();

            async move {
                // Acquire semaphore permit for concurrency control
                let _permit = semaphore.acquire().await.map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to acquire semaphore: {e}"),
                    })
                })?;

                tracing::info!(
                    "Evaluating variant '{}' on dataset '{}'",
                    variant_name,
                    dataset_name
                );

                let evaluation_run_id = Uuid::now_v7();

                // Create UninitializedVariantInfo from the chat config
                let dynamic_variant_config = UninitializedVariantInfo {
                    inner: UninitializedVariantConfig::ChatCompletion(chat_config.clone()),
                    timeouts: None,
                };

                // Create EvaluationCoreArgs
                let core_args = EvaluationCoreArgs {
                    tensorzero_client: gateway_client,
                    clickhouse_client: clickhouse_connection_info,
                    config: tensorzero_config,
                    evaluation_name,
                    evaluation_run_id,
                    dataset_name,
                    variant: EvaluationVariant::Info(Box::new(dynamic_variant_config)),
                    concurrency,
                    inference_cache: CacheEnabledMode::Off, // Disable caching for fair evaluation
                };

                // Call run_evaluation_core_streaming
                let stream_result =
                    match evaluations::run_evaluation_core_streaming(core_args).await {
                        Ok(result) => result,
                        Err(e) => {
                            tracing::warn!(
                                "Failed to start evaluation for variant '{}': {}",
                                variant_name,
                                e
                            );
                            return Ok::<_, Error>((variant_name, None));
                        }
                    };

                // Consume the streaming channel and aggregate results
                let evaluation_results = match consume_evaluation_stream(
                    stream_result.receiver,
                    &evaluation_config,
                    stream_result.run_info.num_datapoints,
                )
                .await
                {
                    Ok(results) => results,
                    Err(e) => {
                        tracing::warn!(
                            "Failed to complete evaluation for variant '{}': {}",
                            variant_name,
                            e
                        );
                        return Ok::<_, Error>((variant_name, None));
                    }
                };

                Ok::<_, Error>((variant_name, Some(evaluation_results)))
            }
        })
        .collect();

    // Execute all evaluations in parallel
    let results = join_all(evaluation_futures).await;

    // Collect results into HashMap
    let mut results_map = HashMap::new();
    for result in results {
        match result {
            Ok((variant_name, evaluation_results)) => {
                results_map.insert(variant_name, evaluation_results);
            }
            Err(e) => {
                // This shouldn't happen since we handle errors inside the futures,
                // but handle it gracefully just in case
                tracing::error!("Unexpected error in evaluation future: {}", e);
            }
        }
    }

    Ok(results_map)
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

    // Aggregate per-datapoint results (needed for Pareto frontier analysis)
    let mut per_datapoint: HashMap<String, HashMap<String, Option<f32>>> = HashMap::new();

    for info in &evaluation_stats.evaluation_infos {
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

        per_datapoint.insert(datapoint_id, datapoint_scores);
    }

    // Compute aggregated statistics using EvaluationStats
    let metrics = {
        let evaluators = match &**evaluation_config {
            EvaluationConfig::Inference(inference_config) => &inference_config.evaluators,
        };
        evaluation_stats.compute_stats(evaluators)
    };

    Ok(EvaluationResults {
        per_datapoint,
        metrics,
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
/// * `Vec<Uuid>` - The IDs of the created datapoints
pub async fn create_evaluation_dataset(
    tensorzero_config: &Config,
    http_client: &TensorzeroHttpClient,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    samples: &[RenderedSample],
    dataset_name: &str,
) -> Result<Vec<Uuid>, Error> {
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
    let response = create_datapoints(
        tensorzero_config,
        http_client,
        clickhouse_connection_info,
        dataset_name,
        request,
    )
    .await?;

    Ok(response.ids)
}
