use std::collections::HashMap;
use std::sync::Arc;

use uuid::Uuid;

use tensorzero_core::{
    cache::CacheEnabledMode,
    client::Client,
    config::{Config, UninitializedVariantConfig, UninitializedVariantInfo},
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::datasets::v1::{
        create_datapoints,
        types::{CreateDatapointRequest, CreateDatapointsRequest},
    },
    error::{Error, ErrorDetails},
    evaluations::EvaluationConfig,
    http::TensorzeroHttpClient,
    stored_inference::RenderedSample,
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use evaluations::{
    stats::EvaluationInfo, EvaluationCoreArgs, EvaluationStats, EvaluationVariant, EvaluatorStats,
    OutputFormat,
};

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

    let mut evaluation_stats =
        EvaluationStats::new(OutputFormat::Jsonl, stream_result.run_info.num_datapoints);
    let mut writer = std::io::sink(); // No-op writer for JSONL mode

    // Consume all updates - EvaluationStats handles everything!
    let mut receiver = stream_result.receiver;
    while let Some(update) = receiver.recv().await {
        evaluation_stats.push(update, &mut writer).map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to process evaluation update: {e}"),
            })
        })?;
    }

    // Compute aggregated statistics using EvaluationStats
    let evaluation_config = &params.evaluation_config;
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
/// Converts each RenderedSample to a CreateDatapointRequest using the
/// `into_create_datapoint_request()` method, which handles type discrimination
/// and validation for both Chat and JSON functions.
///
/// # Arguments
/// * `config` - The TensorZero configuration
/// * `http_client` - The HTTP client for fetching resources
/// * `clickhouse_connection_info` - The ClickHouse connection info
/// * `samples` - The rendered samples to convert into datapoints
/// * `dataset_name` - The name of the dataset to create
///
/// # Returns
/// * `()` - Returns success or error
pub async fn create_evaluation_dataset(
    config: &Config,
    http_client: &TensorzeroHttpClient,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    samples: &[RenderedSample],
    dataset_name: &str,
) -> Result<(), Error> {
    // Convert RenderedSamples to CreateDatapointRequest using the helper method
    let datapoints: Result<Vec<CreateDatapointRequest>, Error> = samples
        .iter()
        .map(|sample| sample.clone().into_create_datapoint_request())
        .collect();

    let request = CreateDatapointsRequest {
        datapoints: datapoints?,
    };

    // Call the datasets v1 create_datapoints function
    create_datapoints(
        config,
        http_client,
        clickhouse_connection_info,
        dataset_name,
        request,
    )
    .await?;

    Ok(())
}
