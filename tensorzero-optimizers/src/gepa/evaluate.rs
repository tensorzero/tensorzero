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
        types::{CreateDatapointRequest, CreateDatapointsRequest, CreateDatapointsResponse},
    },
    error::{Error, ErrorDetails},
    evaluations::EvaluationConfig,
    function::FunctionConfig,
    http::TensorzeroHttpClient,
    stored_inference::RenderedSample,
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use evaluations::{
    ClientInferenceExecutor, EvaluationCoreArgs, EvaluationFunctionConfig,
    EvaluationFunctionConfigTable, EvaluationStats, EvaluationVariant, EvaluatorStats,
    OutputFormat, stats::EvaluationInfo,
};

// Type aliases for score map signatures used for pareto filtering

/// Name of an evaluator/metric (e.g., "accuracy", "latency", "f1_score")
pub type EvaluatorName = String;
pub type VariantName = String;

/// Unique identifier for a datapoint/example in a dataset
pub type DatapointId = Uuid;

/// Scores for all evaluators on a single datapoint
pub type DatapointScores = HashMap<EvaluatorName, Option<f32>>;

/// Scores for all datapoints for a single variant
pub type VariantScores = HashMap<DatapointId, DatapointScores>;

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
/// * `CreateDatapointsResponse` - The IDs of the created datapoints
pub async fn create_evaluation_dataset(
    config: &Config,
    http_client: &TensorzeroHttpClient,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    samples: Vec<RenderedSample>,
    dataset_name: &str,
) -> Result<CreateDatapointsResponse, Error> {
    // Convert RenderedSamples to CreateDatapointRequest using the helper method
    let datapoints: Result<Vec<CreateDatapointRequest>, Error> = samples
        .into_iter()
        .map(|sample| sample.into_create_datapoint_request())
        .collect();

    let request = CreateDatapointsRequest {
        datapoints: datapoints?,
    };

    // Call the datasets v1 create_datapoints function
    let response = create_datapoints(
        config,
        http_client,
        clickhouse_connection_info,
        dataset_name,
        request,
    )
    .await?;

    Ok(response)
}

/// Holds the results of evaluating variants on a dataset
#[derive(Clone, Debug)]
pub struct EvaluationResults {
    /// Full evaluation info for each datapoint
    /// Compatible with analyze_inferences(&[EvaluationInfo])
    pub evaluation_infos: Vec<EvaluationInfo>,

    /// Aggregated statistics across all datapoints
    /// Key: evaluator_name
    /// Value: EvaluatorStats with mean/stderr/count
    pub evaluation_stats: HashMap<EvaluatorName, EvaluatorStats>,
}

impl EvaluationResults {
    /// Extract per-datapoint scores for Pareto frontier analysis
    ///
    /// Returns a HashMap mapping datapoint_id to a HashMap of evaluator scores.
    /// Scores are extracted from evaluation_infos on-demand.
    /// All other EvaluationInfo content can be dropped
    pub fn per_datapoint_scores(&self) -> VariantScores {
        let mut score_map = HashMap::new();

        for info in &self.evaluation_infos {
            let datapoint_id = info.datapoint.id();
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
    pub functions: HashMap<String, Arc<FunctionConfig>>,
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

    // Get function name from evaluation config and look up function
    // Build function configs table from all functions
    let function_configs: EvaluationFunctionConfigTable = params
        .functions
        .iter()
        .map(|(name, func)| (name.clone(), EvaluationFunctionConfig::from(func.as_ref())))
        .collect();
    let function_configs = Arc::new(function_configs);

    // Wrap the gateway client in ClientInferenceExecutor for use with evaluations
    let inference_executor = Arc::new(ClientInferenceExecutor::new(params.gateway_client));

    // Create EvaluationCoreArgs
    let core_args = EvaluationCoreArgs {
        inference_executor,
        clickhouse_client: params.clickhouse_connection_info.clone(),
        evaluation_config: params.evaluation_config.clone(),
        function_configs,
        evaluation_name: params.evaluation_name,
        evaluation_run_id,
        dataset_name: Some(params.dataset_name),
        datapoint_ids: None,
        variant: EvaluationVariant::Info(Box::new(dynamic_variant_config)),
        concurrency: params.concurrency,
        inference_cache: CacheEnabledMode::Off, // Disable caching for fair evaluation
        tags: HashMap::new(),                   // No external tags for optimizer evaluations
                                                // We may want to tag inferences made as part of GEPA later as well.
    };

    // Call run_evaluation_core_streaming
    let stream_result = evaluations::run_evaluation_core_streaming(core_args, None, HashMap::new())
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
