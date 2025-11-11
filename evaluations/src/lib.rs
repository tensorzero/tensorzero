#![recursion_limit = "256"]
use std::io::Write;
use std::sync::Arc;
use std::{collections::HashMap, path::PathBuf};

use anyhow::{anyhow, bail, Result};
use clap::Parser;
use dataset::query_dataset;
use evaluators::{evaluate_inference, EvaluateInferenceParams};
use helpers::{get_cache_options, get_tool_params_args};
use serde::{Deserialize, Serialize};

// Public re-exports for external consumers
pub use stats::{
    mean, std_deviation, EvaluationError, EvaluationInfo, EvaluationStats, EvaluationUpdate,
    EvaluatorStats,
};
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::client::{
    input_handling::resolved_input_to_client_input, Client, ClientBuilder, ClientBuilderMode,
    ClientInferenceParams, DynamicToolParams, FeedbackParams, InferenceOutput, InferenceParams,
    InferenceResponse,
};
use tensorzero_core::client::{ClientInput, StoragePath};
use tensorzero_core::config::{ConfigFileGlob, MetricConfigOptimize};
use tensorzero_core::error::Error;
use tensorzero_core::evaluations::{EvaluationConfig, EvaluatorConfig};
use tensorzero_core::inference::types::stored_input::StoragePathResolver;
use tensorzero_core::{
    config::Config, db::clickhouse::ClickHouseConnectionInfo, endpoints::datasets::StoredDatapoint,
    function::FunctionConfig,
};
use tokio::{
    sync::{mpsc, Semaphore},
    task::JoinSet,
};
use tracing::{debug, error, info, instrument};
use url::Url;
use uuid::Uuid;

pub mod dataset;
pub mod evaluators;
pub mod helpers;
pub mod stats;

/// Buffer size for the mpsc channel used to stream evaluation updates.
/// This provides backpressure if the consumer can't keep up with the producer.
const EVALUATION_CHANNEL_BUFFER_SIZE: usize = 128;

#[derive(clap::ValueEnum, Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[clap(rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    Jsonl,
    #[default]
    Pretty,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Path to tensorzero.toml.
    #[arg(long, default_value = "./config/tensorzero.toml")]
    pub config_file: PathBuf,

    /// URL of a running TensorZero HTTP gateway server to use for requests. This runs evaluations using that gateway.
    #[arg(long)]
    pub gateway_url: Option<Url>,

    /// Name of the evaluation to run.
    #[arg(short, long)]
    pub evaluation_name: String,

    /// Name of the dataset to run on.
    #[arg(short, long)]
    pub dataset_name: String,

    /// Name of the variant to run.
    #[arg(short, long)]
    pub variant_name: String,

    /// Number of concurrent requests to make.
    #[arg(short, long, default_value = "1")]
    pub concurrency: usize,

    #[arg(short, long, default_value = "pretty")]
    pub format: OutputFormat,

    #[arg(long, default_value = "on")]
    pub inference_cache: CacheEnabledMode,
}

pub struct Clients {
    pub tensorzero_client: ThrottledTensorZeroClient,
    pub clickhouse_client: ClickHouseConnectionInfo,
}

/// Parameters for running an evaluation using run_evaluation_core
/// This struct encapsulates all the necessary components for evaluation execution
pub struct EvaluationCoreArgs {
    /// TensorZero client for making inference requests
    pub tensorzero_client: Client,

    /// ClickHouse client for database operations
    pub clickhouse_client: ClickHouseConnectionInfo,

    /// Configuration containing function and evaluation definitions
    pub config: Arc<Config>,

    /// Name of the evaluation to run.
    pub evaluation_name: String,

    /// Unique identifier for this evaluation run
    pub evaluation_run_id: Uuid,

    /// Name of the dataset to run on.
    pub dataset_name: String,

    /// Name of the variant to run.
    pub variant_name: String,

    /// Number of concurrent requests to make.
    pub concurrency: usize,

    /// Cache configuration for inference requests
    pub inference_cache: CacheEnabledMode,
}

/// High-level wrapper function for running evaluations called from the CLI.
/// It handles all setup and teardown, then delegates to `run_evaluation_core_streaming`
/// for the actual evaluation logic.
///
/// ## What it does
///
/// 1. **Setup:**
///    - Loads environment variables (ClickHouse URL, optional Postgres URL)
///    - Loads the TensorZero configuration from the config file
///    - Initializes the TensorZero client (either HTTP gateway or embedded gateway)
///    - Initializes the ClickHouse client for storing evaluation results
///
/// 2. **Execution:**
///    - Calls `run_evaluation_core_streaming` with the initialized clients and config
///    - Receives a stream of `EvaluationUpdate` messages via a channel
///    - Writes each update to the output writer as it arrives
///
/// 3. **Results:**
///    - For `OutputFormat::Jsonl`: Writes each update as a JSON line
///    - For `OutputFormat::Pretty`: Shows a progress bar and computes statistics at the end
///    - Computes mean/stderr for each evaluator
///    - Checks if results meet the configured cutoff thresholds
///    - Fails if any evaluator's results are below its cutoff
///
/// 4. **Cleanup:**
///    - Waits for the ClickHouse batch writer to finish persisting all results
///    - This ensures all evaluation data is in the database before returning
///
/// ## Parameters
///
/// - `args`: CLI arguments containing evaluation name, dataset, variant, concurrency, etc.
/// - `evaluation_run_id`: Unique identifier for this evaluation run
/// - `writer`: Output writer (stdout, file, etc.) for writing results
///
/// ## Returns
///
/// - `Ok(())` if the evaluation completes successfully and meets all cutoffs
/// - `Err` if setup fails, evaluation fails, or results don't meet cutoffs
#[instrument(skip_all, fields(evaluation_run_id = %evaluation_run_id, evaluation_name = %args.evaluation_name, dataset_name = %args.dataset_name, variant_name = %args.variant_name, concurrency = %args.concurrency))]
pub async fn run_evaluation(
    args: Args,
    evaluation_run_id: Uuid,
    mut writer: impl Write,
) -> Result<()> {
    info!("Initializing evaluation environment");
    let clickhouse_url = std::env::var("TENSORZERO_CLICKHOUSE_URL")
        .map_err(|_| anyhow!("Missing ClickHouse URL at TENSORZERO_CLICKHOUSE_URL"))?;
    debug!(clickhouse_url = %clickhouse_url, "ClickHouse URL resolved");
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL").ok();
    if let Some(postgres_url) = postgres_url.as_ref() {
        debug!(postgres_url = %postgres_url, "PostgreSQL URL resolved");
    } else {
        debug!("PostgreSQL URL not provided");
    }

    // We do not validate credentials here since we just want the evaluator config
    // If we are using an embedded gateway, credentials are validated when that is initialized
    info!(config_file = ?args.config_file, "Loading configuration");
    let config = Arc::new(
        Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(&args.config_file)?,
            false,
        )
        .await?,
    );
    debug!("Configuration loaded successfully");
    let tensorzero_client = match args.gateway_url {
        Some(gateway_url) => {
            ClientBuilder::new(ClientBuilderMode::HTTPGateway { url: gateway_url })
        }
        None => ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(args.config_file),
            postgres_url,
            clickhouse_url: Some(clickhouse_url.clone()),
            timeout: None,
            verify_credentials: true,
            allow_batch_writes: true,
        }),
    }
    .build()
    .await
    .map_err(|e| anyhow!("Failed to build client: {e}"))?;

    let clickhouse_client = ClickHouseConnectionInfo::new(
        &clickhouse_url,
        config.gateway.observability.batch_writes.clone(),
    )
    .await?;

    let core_args = EvaluationCoreArgs {
        tensorzero_client,
        clickhouse_client: clickhouse_client.clone(),
        config,
        dataset_name: args.dataset_name,
        variant_name: args.variant_name,
        evaluation_name: args.evaluation_name,
        evaluation_run_id,
        inference_cache: args.inference_cache,
        concurrency: args.concurrency,
    };

    let output_format = args.format.clone();
    let result = run_evaluation_core_streaming(core_args).await?;

    let mut receiver = result.receiver;
    let dataset_len = result.run_info.num_datapoints;

    // Write the run info first
    write_run_info(&mut writer, &result.run_info, &output_format)?;

    // Collect results from the streaming channel
    let mut evaluation_stats = EvaluationStats::new(output_format.clone(), dataset_len);

    while let Some(update) = receiver.recv().await {
        match update {
            EvaluationUpdate::RunInfo(_) => {
                // Skip RunInfo as we already wrote it
                continue;
            }
            update => {
                evaluation_stats.push(update, &mut writer)?;
            }
        }
    }

    if let Some(progress_bar) = &evaluation_stats.progress_bar {
        progress_bar.finish_with_message("Done");
    }

    if evaluation_stats.output_format == OutputFormat::Pretty {
        let EvaluationConfig::Inference(inference_evaluation_config) = &*result.evaluation_config;
        let stats = evaluation_stats.compute_stats(&inference_evaluation_config.evaluators);

        // Print all stats
        for (evaluator_name, evaluator_stats) in &stats {
            writeln!(writer, "{evaluator_name}: {evaluator_stats}")?;
        }

        // Check cutoffs and handle failures
        let failures = check_evaluator_cutoffs(&stats, &inference_evaluation_config.evaluators)?;

        // Print failure messages
        for (name, cutoff, actual) in &failures {
            writeln!(
                writer,
                "Failed cutoff for evaluator {name} ({cutoff:.2}, got {actual:.2})"
            )?;
        }

        // If there are failures, return an error with all failures listed
        if !failures.is_empty() {
            let failure_messages = format_cutoff_failures(&failures);
            bail!("Failed cutoffs for evaluators: {failure_messages}");
        }
    }

    // Since we construct our own `ClickHouseConnectionInfo` outside of our `TensorZeroClient`,
    // we need to wait for the batch writer to finish.
    // This happens automatically when `run_evaluation` is called from the standalone `evaluations` binary
    // (since Tokio will wait for the `spawn_blocking` task to finish before shutting down the runtime).
    // We explicitly wait here for the batch writer to finish, so that `run_evaluation` can be called
    // from other places in the codebase (e.g. e2e tests), and subsequently query ClickHouse for the evaluation results.
    if let Some(handle) = clickhouse_client.batcher_join_handle() {
        drop(clickhouse_client);
        tracing::info!("Waiting for evaluations ClickHouse batch writer to finish");
        handle
            .await
            .map_err(|e| anyhow!("Error waiting for ClickHouse batch writer: {e}"))?;
        tracing::info!("Evaluations ClickHouse batch writer finished");
    }

    Ok(())
}

/// Core streaming evaluation function.
///
/// This function runs an evaluation and streams results as they complete via an mpsc channel.
///
/// ## How it works
///
/// 1. Creates an mpsc channel for streaming `EvaluationUpdate` messages
/// 2. Loads the evaluation and function configurations
/// 3. Queries the dataset to get all datapoints
/// 4. Sends `RunInfo` as the first message (evaluation_run_id, num_datapoints)
/// 5. Spawns a concurrent task for each datapoint that:
///    - Runs inference for the datapoint
///    - Evaluates the inference response
///    - Returns (Datapoint, InferenceResponse, EvaluationResult)
/// 6. Spawns a background collector task that:
///    - Collects results from the JoinSet as tasks complete
///    - Converts results to `EvaluationUpdate::Success` or `EvaluationUpdate::Error`
///    - Sends each update through the channel
///    - Waits for the ClickHouse batch writer to finish
///    - Closes the channel when done
/// 7. Returns immediately with the receiver, run_info, and evaluation_config
///
/// ## Return value
///
/// Returns `EvaluationStreamResult` containing:
/// - `receiver`: Channel receiver for consuming `EvaluationUpdate` messages
/// - `run_info`: Metadata (evaluation_run_id, num_datapoints)
/// - `evaluation_config`: The evaluation configuration (needed for computing statistics)
///
/// The caller receives updates by calling `receiver.recv().await` until the channel closes.
///
/// ## Error handling
///
/// Errors within evaluation tasks are caught and sent as `EvaluationUpdate::Error` messages
/// rather than failing the entire evaluation. Error messages include context:
/// - Inference errors: Include the datapoint_id
/// - Evaluation errors: Include both the inference_id and datapoint_id
#[instrument(skip_all, fields(evaluation_run_id = %args.evaluation_run_id, evaluation_name = %args.evaluation_name, dataset_name = %args.dataset_name, variant_name = %args.variant_name, concurrency = %args.concurrency))]
pub async fn run_evaluation_core_streaming(
    args: EvaluationCoreArgs,
) -> Result<EvaluationStreamResult> {
    let (sender, receiver) = mpsc::channel(EVALUATION_CHANNEL_BUFFER_SIZE);

    // Build the semaphore and clients
    let semaphore = Semaphore::new(args.concurrency);
    let clients = Arc::new(Clients {
        tensorzero_client: ThrottledTensorZeroClient::new(args.tensorzero_client, semaphore),
        clickhouse_client: args.clickhouse_client,
    });

    // Get evaluation configuration
    let evaluation_config = args
        .config
        .evaluations
        .get(&args.evaluation_name)
        .ok_or_else(|| anyhow!("evaluation '{}' not found", args.evaluation_name))?
        .clone();

    debug!(evaluation_name = %args.evaluation_name, "Evaluation config found");

    let EvaluationConfig::Inference(inference_evaluation_config) = &*evaluation_config;
    let function_config = args
        .config
        .get_function(&inference_evaluation_config.function_name)?
        .into_owned();

    info!(
        function_name = %inference_evaluation_config.function_name,
        evaluators = ?inference_evaluation_config.evaluators.keys().collect::<Vec<_>>(),
        "Function and evaluators configured"
    );

    let mut join_set = JoinSet::new();

    info!("Querying dataset");
    let dataset = query_dataset(
        &clients.clickhouse_client,
        &args.dataset_name,
        &inference_evaluation_config.function_name,
        &function_config,
    )
    .await?;
    info!(dataset_size = dataset.len(), "Dataset loaded successfully");
    let dataset_name = Arc::new(args.dataset_name);
    let variant_name = Arc::new(args.variant_name);
    let evaluation_name = Arc::new(args.evaluation_name);
    let dataset_len = dataset.len();
    let mut task_id_to_datapoint_id = HashMap::new();

    let run_info = RunInfo {
        evaluation_run_id: args.evaluation_run_id,
        num_datapoints: dataset_len,
    };

    // Send the run info as the first message
    if sender
        .send(EvaluationUpdate::RunInfo(run_info.clone()))
        .await
        .is_err()
    {
        tracing::warn!("Failed to send RunInfo: receiver dropped before evaluation started");
    }

    // Spawn concurrent tasks for each datapoint
    for datapoint in dataset {
        let clients_clone = clients.clone();
        let variant_name = variant_name.clone();
        let function_config = function_config.clone();
        let evaluation_config = evaluation_config.clone();
        let dataset_name = dataset_name.clone();
        let function_name = inference_evaluation_config.function_name.clone();
        let evaluation_name = evaluation_name.clone();
        let evaluation_run_id_clone = args.evaluation_run_id;
        let datapoint = Arc::new(datapoint);
        let datapoint_id = datapoint.id();
        let inference_cache = args.inference_cache;
        let abort_handle = join_set.spawn(async move {
            let input = Arc::new(resolved_input_to_client_input(datapoint.input().clone().reresolve(&clients_clone.tensorzero_client).await?)?);
            let inference_response = Arc::new(
                infer_datapoint(InferDatapointParams {
                    clients: &clients_clone,
                    function_name: &function_name,
                    variant_name: &variant_name,
                    evaluation_run_id: evaluation_run_id_clone,
                    dataset_name: &dataset_name,
                    datapoint: &datapoint,
                    evaluation_name: &evaluation_name,
                    function_config: &function_config,
                    input: &input,
                    inference_cache,
                })
                .await.map_err(|e| anyhow!("Error inferring for datapoint {datapoint_id}: {e}"))?,
            );

            let inference_id = inference_response.inference_id();
            let evaluation_result = evaluate_inference(
                EvaluateInferenceParams {
                    inference_response: inference_response.clone(),
                    datapoint: datapoint.clone(),
                    input,
                    evaluation_config,
                    evaluation_name,
                    clients: clients_clone.clone(),
                    evaluation_run_id: evaluation_run_id_clone,
                    inference_cache,
                })
                .await.map_err(|e| anyhow!("Error evaluating inference {inference_id} for datapoint {datapoint_id}: {e}"))?;
            debug!(datapoint_id = %datapoint.id(), evaluations_count = evaluation_result.len(), "Evaluations completed");

            Ok::<(StoredDatapoint, InferenceResponse, evaluators::EvaluationResult), anyhow::Error>((
                Arc::into_inner(datapoint).ok_or_else(|| anyhow!("Failed to get datapoint for datapoint. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/categories/bug-reports."))?,
                Arc::into_inner(inference_response).ok_or_else(|| anyhow!("Failed to get inference response for datapoint. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/categories/bug-reports."))?,
                evaluation_result,
            ))
        });
        task_id_to_datapoint_id.insert(abort_handle.id(), datapoint_id);
    }

    // Get a shared reference to the evaluation config (which includes evaluators)
    let evaluators = evaluation_config.clone();

    // Spawn a task to collect results and stream them
    let sender_clone = sender.clone();
    // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(async move {
        while let Some(result) = join_set.join_next_with_id().await {
            let update = match result {
                Ok((_, Ok((datapoint, inference_response, evaluation_result)))) => {
                    EvaluationUpdate::Success(EvaluationInfo::new(
                        datapoint,
                        inference_response,
                        evaluation_result,
                    ))
                }
                Ok((task_id, Err(e))) => {
                    tracing::warn!("Task error: {}", e);
                    EvaluationUpdate::Error(EvaluationError {
                        datapoint_id: task_id_to_datapoint_id[&task_id],
                        message: e.to_string(),
                    })
                }
                Err(e) => EvaluationUpdate::Error(EvaluationError {
                    datapoint_id: task_id_to_datapoint_id[&e.id()],
                    message: e.to_string(),
                }),
            };

            if sender_clone.send(update).await.is_err() {
                // Receiver dropped, stop sending
                break;
            }
        }
    });

    Ok(EvaluationStreamResult {
        receiver,
        run_info,
        evaluation_config: evaluators,
    })
}

/// Checks if evaluator results meet their cutoff thresholds
///
/// Returns a vector of failures with (evaluator_name, cutoff, actual_value)
pub fn check_evaluator_cutoffs(
    stats: &HashMap<String, stats::EvaluatorStats>,
    evaluator_configs: &HashMap<String, EvaluatorConfig>,
) -> Result<Vec<(String, f32, f32)>> {
    let mut failures = Vec::new();

    for (evaluator_name, evaluator_stats) in stats {
        let evaluator_config = evaluator_configs
            .get(evaluator_name)
            .ok_or_else(|| anyhow!("Evaluator not found for computing stats"))?;

        if let Some(cutoff) = evaluator_config.cutoff() {
            match evaluator_config.optimize() {
                MetricConfigOptimize::Max => {
                    if evaluator_stats.mean < cutoff {
                        failures.push((evaluator_name.clone(), cutoff, evaluator_stats.mean));
                    }
                }
                MetricConfigOptimize::Min => {
                    if evaluator_stats.mean > cutoff {
                        failures.push((evaluator_name.clone(), cutoff, evaluator_stats.mean));
                    }
                }
            }
        }
    }

    Ok(failures)
}

/// Formats a list of cutoff failures into a human-readable string
pub fn format_cutoff_failures(failures: &[(String, f32, f32)]) -> String {
    failures
        .iter()
        .map(|(name, cutoff, actual)| format!("{name} (cutoff: {cutoff:.2}, got: {actual:.2})"))
        .collect::<Vec<_>>()
        .join("\n")
}

struct InferDatapointParams<'a> {
    clients: &'a Clients,
    function_name: &'a str,
    variant_name: &'a str,
    evaluation_run_id: Uuid,
    dataset_name: &'a str,
    datapoint: &'a StoredDatapoint,
    input: &'a ClientInput,
    evaluation_name: &'a str,
    function_config: &'a FunctionConfig,
    inference_cache: CacheEnabledMode,
}

#[instrument(skip_all, fields(datapoint_id = %params.datapoint.id(), function_name = %params.function_name, variant_name = %params.variant_name))]
async fn infer_datapoint(params: InferDatapointParams<'_>) -> Result<InferenceResponse> {
    let InferDatapointParams {
        clients,
        function_name,
        variant_name,
        evaluation_run_id,
        dataset_name,
        datapoint,
        evaluation_name,
        function_config,
        input,
        inference_cache,
    } = params;

    debug!("Processing tool parameters");
    let dynamic_tool_params = match datapoint.tool_call_config() {
        Some(tool_params) => {
            debug!("Tool parameters found, processing");
            get_tool_params_args(tool_params, function_config).await
        }
        None => {
            debug!("No tool parameters found");
            DynamicToolParams::default()
        }
    };
    debug!("Processing output schema");
    let output_schema = match (datapoint.output_schema(), function_config) {
        // If the datapoint has an output schema, use it only in the case where it is not the same as the output schema of the function
        (Some(output_schema), FunctionConfig::Json(json_function_config)) => {
            if output_schema == &json_function_config.output_schema.value {
                debug!("Output schema matches function schema, using function default");
                None
            } else {
                debug!("Custom output schema provided");
                Some(output_schema)
            }
        }
        (Some(_), FunctionConfig::Chat(_)) => {
            return Err(anyhow!("Chat function does not support output schema"));
        }
        (None, _) => {
            debug!("No output schema specified");
            None
        }
    };
    let params = ClientInferenceParams {
        function_name: Some(function_name.to_string()),
        variant_name: Some(variant_name.to_string()),
        input: input.clone(),
        tags: HashMap::from([
            (
                "tensorzero::evaluation_run_id".to_string(),
                evaluation_run_id.to_string(),
            ),
            (
                "tensorzero::datapoint_id".to_string(),
                datapoint.id().to_string(),
            ),
            (
                "tensorzero::evaluation_name".to_string(),
                evaluation_name.to_string(),
            ),
            (
                "tensorzero::dataset_name".to_string(),
                dataset_name.to_string(),
            ),
        ]),
        dynamic_tool_params,
        output_schema: output_schema.cloned(),
        credentials: HashMap::new(),
        cache_options: get_cache_options(inference_cache),
        dryrun: Some(false),
        episode_id: None,
        model_name: None,
        stream: Some(false),
        params: InferenceParams::default(),
        include_original_response: false,
        internal: true,
        extra_body: Default::default(),
        extra_headers: Default::default(),
        internal_dynamic_variant_config: None,
        otlp_traces_extra_headers: HashMap::new(),
    };
    debug!("Making inference request");
    let inference_result = clients.tensorzero_client.inference(params).await?;
    match inference_result {
        InferenceOutput::NonStreaming(inference_response) => {
            debug!(inference_id = %inference_response.inference_id(), "Inference completed successfully");
            Ok(inference_response)
        }
        InferenceOutput::Streaming(_inference_stream) => {
            error!("Received streaming inference response when non-streaming was expected");
            bail!("Streaming inference should never happen in evaluations")
        }
    }
}

fn write_run_info(
    writer: &mut impl Write,
    run_info: &RunInfo,
    format: &OutputFormat,
) -> Result<()> {
    match format {
        OutputFormat::Jsonl => {
            writeln!(writer, "{}", serde_json::to_string(run_info)?)?;
        }
        OutputFormat::Pretty => {
            writeln!(writer, "Run ID: {}", run_info.evaluation_run_id)?;
            writeln!(writer, "Number of datapoints: {}", run_info.num_datapoints)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    pub evaluation_run_id: Uuid,
    pub num_datapoints: usize,
}

/// Result from running an evaluation that supports streaming
pub struct EvaluationStreamResult {
    pub receiver: mpsc::Receiver<EvaluationUpdate>,
    pub run_info: RunInfo,
    pub evaluation_config: Arc<EvaluationConfig>,
}

pub struct ThrottledTensorZeroClient {
    pub client: Client,
    semaphore: Semaphore,
}

impl ThrottledTensorZeroClient {
    pub fn new(client: Client, semaphore: Semaphore) -> Self {
        Self { client, semaphore }
    }

    async fn inference(&self, params: ClientInferenceParams) -> Result<InferenceOutput> {
        let _permit = self.semaphore.acquire().await?;
        let inference_output = self.client.inference(params).await?;
        Ok(inference_output)
    }

    async fn feedback(&self, params: FeedbackParams) -> Result<()> {
        self.client.feedback(params).await?;
        Ok(())
    }
}

impl StoragePathResolver for ThrottledTensorZeroClient {
    async fn resolve(&self, storage_path: StoragePath) -> Result<String, Error> {
        self.client.resolve(storage_path).await
    }
}

#[cfg(test)]
mod tests {
    use tensorzero_core::evaluations::ExactMatchConfig;

    use super::*;

    #[test]
    fn test_format_cutoff_failures() {
        let failures = vec![
            ("evaluator1".to_string(), 0.5, 0.4),
            ("evaluator2".to_string(), 0.6, 0.3),
        ];
        let formatted = format_cutoff_failures(&failures);
        assert_eq!(
            formatted,
            "evaluator1 (cutoff: 0.50, got: 0.40)\nevaluator2 (cutoff: 0.60, got: 0.30)"
        );
    }

    #[test]
    fn test_check_evaluator_cutoffs() {
        let stats = {
            let mut stats = HashMap::new();
            stats.insert(
                "evaluator1".to_string(),
                stats::EvaluatorStats {
                    mean: 0.4,
                    stderr: 0.1,
                    count: 10,
                },
            );
            stats.insert(
                "evaluator2".to_string(),
                stats::EvaluatorStats {
                    mean: 0.3,
                    stderr: 0.1,
                    count: 10,
                },
            );
            stats.insert(
                "evaluator3".to_string(),
                stats::EvaluatorStats {
                    mean: 0.1,
                    stderr: 0.05,
                    count: 10,
                },
            );
            stats
        };
        let evaluators = {
            let mut evaluators = HashMap::new();
            evaluators.insert(
                "evaluator1".to_string(),
                EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: Some(0.5) }),
            );
            evaluators.insert(
                "evaluator2".to_string(),
                EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: Some(0.6) }),
            );
            evaluators.insert(
                "evaluator3".to_string(),
                EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }),
            );
            evaluators
        };
        let failures = check_evaluator_cutoffs(&stats, &evaluators).unwrap();
        assert_eq!(failures.len(), 2);

        // Check that both expected failures are present, regardless of order
        assert!(failures.contains(&("evaluator1".to_string(), 0.5, 0.4)));
        assert!(failures.contains(&("evaluator2".to_string(), 0.6, 0.3)));

        // Check that evaluator3 is not in the failures list since it has no cutoff
        assert!(!failures.iter().any(|(name, _, _)| name == "evaluator3"));
    }
}
