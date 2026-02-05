#![recursion_limit = "256"]
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use evaluators::{EvaluateInferenceParams, evaluate_inference};
use helpers::get_cache_options;

// Public re-exports for external consumers
pub use cli::{Args, OutputFormat};
pub use stats::{
    EvaluationError, EvaluationInfo, EvaluationStats, EvaluationUpdate, EvaluatorStats,
    PerEvaluatorStats,
};
pub use tensorzero_core::evaluations::{EvaluationFunctionConfig, EvaluationFunctionConfigTable};
pub use tensorzero_core::statistics_util::{mean, std_deviation};
use tensorzero_core::utils::gateway::AppStateData;
pub use types::*;

use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::client::Input;
use tensorzero_core::client::{
    ClientBuilder, ClientBuilderMode, ClientInferenceParams, DynamicToolParams, InferenceOutput,
    InferenceParams, InferenceResponse, PostgresConfig,
    input_handling::resolved_input_to_client_input,
};
use tensorzero_core::config::{ConfigFileGlob, MetricConfigOptimize};
use tensorzero_core::endpoints::datasets::v1::{
    get_datapoints, list_datapoints,
    types::{GetDatapointsRequest, ListDatapointsRequest},
};
use tensorzero_core::evaluations::{EvaluationConfig, EvaluatorConfig};
use tensorzero_core::inference::types::InputExt;
use tensorzero_core::utils::spawn_ignoring_shutdown;
use tensorzero_core::{
    config::Config, db::clickhouse::ClickHouseConnectionInfo, endpoints::datasets::Datapoint,
};
use tokio::{
    sync::{Semaphore, mpsc},
    task::JoinSet,
};
use tracing::{debug, error, info, instrument};
use uuid::Uuid;

pub mod cli;
pub mod evaluators;
pub mod helpers;
pub mod stats;
pub mod stopping;
pub mod types;

/// Buffer size for the mpsc channel used to stream evaluation updates.
/// This provides backpressure if the consumer can't keep up with the producer.
const EVALUATION_CHANNEL_BUFFER_SIZE: usize = 128;

/// Merge external tags with internal tags, with internal tags taking precedence.
/// Returns an error if any external tags would be overridden by internal tags.
pub(crate) fn merge_tags(
    external_tags: &HashMap<String, String>,
    internal_tags: HashMap<String, String>,
) -> Result<HashMap<String, String>> {
    let mut merged = external_tags.clone();
    for (key, value) in internal_tags {
        if merged.contains_key(&key) {
            return Err(anyhow!(
                "Tag collision: external tag '{key}' conflicts with internal evaluation tag. \
                Reserved tag prefixes include 'tensorzero::evaluation', 'tensorzero::datapoint', \
                'tensorzero::dataset'"
            ));
        }
        merged.insert(key, value);
    }
    Ok(merged)
}

pub struct Clients {
    pub inference_executor: Arc<dyn EvaluationsInferenceExecutor>,
    pub clickhouse_client: ClickHouseConnectionInfo,
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
#[instrument(skip_all, fields(evaluation_run_id = %evaluation_run_id, evaluation_name = %args.evaluation_name, dataset_name = ?args.dataset_name, num_datapoint_ids = %args.datapoint_ids.as_deref().unwrap_or_default().len(), variant_name = %args.variant_name, concurrency = %args.concurrency))]
pub async fn run_evaluation(
    args: Args,
    evaluation_run_id: Uuid,
    mut writer: impl Write,
) -> Result<()> {
    // Convert Option<Vec<Uuid>> to Vec<Uuid> (None becomes empty vec)
    let datapoint_ids = args.datapoint_ids.unwrap_or_default();

    // Validate that exactly one of dataset_name or datapoint_ids is provided
    if args.dataset_name.is_some() && !datapoint_ids.is_empty() {
        bail!(
            "Cannot provide both dataset_name and datapoint_ids. Please specify one or the other."
        );
    }
    if args.dataset_name.is_none() && datapoint_ids.is_empty() {
        bail!("Must provide either dataset_name or datapoint_ids.");
    }

    // Validate that max_datapoints is not used with datapoint_ids
    if !datapoint_ids.is_empty() && args.max_datapoints.is_some() {
        bail!(
            "Cannot provide both datapoint_ids and max_datapoints. max_datapoints can only be used with dataset_name."
        );
    }

    // TODO(#5754): Extract environment variable reading to a centralized location
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
    let valkey_url = std::env::var("TENSORZERO_VALKEY_URL").ok();
    if let Some(valkey_url) = valkey_url.as_ref() {
        debug!(valkey_url = %valkey_url, "Valkey URL resolved");
    } else {
        debug!("Valkey URL not provided");
    }

    // We do not validate credentials here since we just want the evaluator config
    // If we are using an embedded gateway, credentials are validated when that is initialized
    info!(config_file = ?args.config_file, "Loading configuration");
    let unwritten_config = Config::load_from_path_optional_verify_credentials(
        &ConfigFileGlob::new_from_path(&args.config_file)?,
        false,
    )
    .await?;
    let clickhouse_client = ClickHouseConnectionInfo::new(
        &clickhouse_url,
        unwritten_config.gateway.observability.batch_writes.clone(),
    )
    .await?;
    let config = Box::pin(unwritten_config.into_config(&clickhouse_client)).await?;
    let config = Arc::new(config);
    debug!("Configuration loaded successfully");

    // Look up evaluation config from the loaded config
    let evaluation_config = config
        .evaluations
        .get(&args.evaluation_name)
        .ok_or_else(|| anyhow!("evaluation '{}' not found", args.evaluation_name))?
        .clone();

    // Build function configs table from all functions in config
    let function_configs: EvaluationFunctionConfigTable = config
        .functions
        .iter()
        .map(|(name, func)| (name.clone(), EvaluationFunctionConfig::from(func.as_ref())))
        .collect();
    let function_configs = Arc::new(function_configs);

    let tensorzero_client = match args.gateway_url {
        Some(gateway_url) => {
            ClientBuilder::new(ClientBuilderMode::HTTPGateway { url: gateway_url })
        }
        None => ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(args.config_file),
            postgres_config: postgres_url.map(PostgresConfig::Url),
            clickhouse_url: Some(clickhouse_url.clone()),
            valkey_url,
            timeout: None,
            verify_credentials: true,
            allow_batch_writes: true,
        }),
    }
    .build()
    .await
    .map_err(|e| anyhow!("Failed to build client: {e}"))?;

    // Wrap the client in ClientInferenceExecutor for use with evaluations
    let inference_executor = Arc::new(ClientInferenceExecutor::new(tensorzero_client));

    let core_args = EvaluationCoreArgs {
        inference_executor,
        clickhouse_client: clickhouse_client.clone(),
        evaluation_config,
        function_configs,
        dataset_name: args.dataset_name,
        datapoint_ids: Some(datapoint_ids),
        variant: EvaluationVariant::Name(args.variant_name),
        evaluation_name: args.evaluation_name,
        evaluation_run_id,
        inference_cache: args.inference_cache,
        concurrency: args.concurrency,
        tags: HashMap::new(), // CLI doesn't have autopilot context
    };

    // Convert Vec<(String, f32)> to HashMap<String, f32> for precision_targets
    let precision_targets: HashMap<String, f32> = args.precision_targets.into_iter().collect();

    let output_format = args.format.clone();
    let result =
        run_evaluation_core_streaming(core_args, args.max_datapoints, precision_targets).await?;

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

/// Run an evaluation using the gateway's `AppStateData` directly.
///
/// This is a higher-level function that sets up the evaluation infrastructure and
/// calls `run_evaluation_core_streaming`. It's used by:
/// - The gateway HTTP handler (`run_evaluation_handler`)
/// - Embedded mode in durable-tools
///
/// This function:
/// 1. Creates a fresh ClickHouse client for the evaluation (with independent batch writer)
/// 2. Creates an `AppStateInferenceExecutor` to call inference/feedback endpoints directly
/// 3. Builds the function configs table
/// 4. Generates a new evaluation run ID
/// 5. Calls `run_evaluation_core_streaming` with the prepared args
///
/// ## Returns
///
/// Returns `EvaluationStreamResult` containing:
/// - `receiver`: Channel receiver for consuming `EvaluationUpdate` messages
/// - `run_info`: Metadata (evaluation_run_id, num_datapoints)
/// - `evaluation_config`: The evaluation configuration
#[instrument(skip_all, fields(evaluation_name = %params.evaluation_name, dataset_name = ?params.dataset_name, variant = ?params.variant, concurrency = %params.concurrency))]
pub async fn run_evaluation_with_app_state(
    app_state: AppStateData,
    params: RunEvaluationWithAppStateParams,
) -> Result<EvaluationStreamResult> {
    // Create a fresh ClickHouse client for the evaluation (with independent batch writer)
    let clickhouse_client = app_state
        .clickhouse_connection_info
        .recreate()
        .await
        .map_err(|e| anyhow!("Failed to create ClickHouse client for evaluation: {e}"))?;

    // Create AppStateInferenceExecutor to call handlers directly without HTTP overhead
    let inference_executor = Arc::new(AppStateInferenceExecutor::new(app_state));

    // Extract function name from evaluation config
    let EvaluationConfig::Inference(ref inference_eval_config) = params.evaluation_config;
    let function_name = inference_eval_config.function_name.clone();

    // Build function configs table
    let mut function_configs = EvaluationFunctionConfigTable::new();
    function_configs.insert(function_name, params.function_config);
    let function_configs = Arc::new(function_configs);

    // Generate a new evaluation run ID
    let evaluation_run_id = Uuid::now_v7();

    // Build the core args
    let core_args = EvaluationCoreArgs {
        inference_executor,
        clickhouse_client,
        evaluation_config: Arc::new(params.evaluation_config),
        function_configs,
        dataset_name: params.dataset_name,
        datapoint_ids: params.datapoint_ids,
        variant: params.variant,
        evaluation_name: params.evaluation_name,
        evaluation_run_id,
        inference_cache: params.cache_mode,
        concurrency: params.concurrency,
        tags: params.tags,
    };

    // Run the evaluation
    run_evaluation_core_streaming(core_args, params.max_datapoints, params.precision_targets).await
}

/// Core streaming evaluation function with optional adaptive stopping.
///
/// This function runs an evaluation and streams results as they complete via an mpsc channel.
/// When `precision_targets` is provided, evaluators can stop independently once confidence
/// interval half-widths (the max of the distances from the CI endpoints to the point estimate)
/// are within the precision targets.
///
/// ## How it works
///
/// 1. Creates an mpsc channel for streaming `EvaluationUpdate` messages
/// 2. Loads the evaluation and function configurations
/// 3. Queries the dataset (limited by `max_datapoints` if specified)
/// 4. If `precision_targets` is provided, creates a `StoppingManager` which internally creates cancellation
///    tokens and tracks evaluator statistics
/// 5. Sends `RunInfo` as the first message (evaluation_run_id, num_datapoints)
/// 6. Spawns a concurrent task for each datapoint (up to `max_datapoints`) that:
///    - Acquires a semaphore permit (controls concurrency)
///    - Runs inference for the datapoint
///    - Evaluates the inference response (skipping cancelled evaluators via `StoppingManager::get_tokens()`)
///    - Returns (Datapoint, InferenceResponse, EvaluationResult)
/// 7. Spawns a background collector task that:
///    - Collects results from the JoinSet as tasks complete
///    - If `precision_targets` is provided:
///      - Updates per-evaluator statistics via `StoppingManager::update_stats()`
///      - Cancels converged evaluators via `StoppingManager::cancel_converged_evaluators()`
///      - Checks if all evaluators have stopped via `StoppingManager::all_evaluators_stopped()`
///      - Aborts remaining tasks when all evaluators have stopped
///    - Converts results to `EvaluationUpdate::Success` or `EvaluationUpdate::Error`
///    - Sends each update through the channel
///    - Closes the channel when all tasks complete or are aborted
/// 8. Returns immediately with the receiver, run_info, and evaluation_config
///
/// ## Parameters
///
/// **`max_datapoints`**: When `Some(max)`, limits dataset to at most `max` datapoints.
///
/// **`precision_targets`**: When non-empty, enables adaptive stopping:
/// - Per-evaluator CI half-width precision_targets (HashMap<String, f32>)
/// - Evaluator k stops when the larger of the two halves of the CI has width ≤ precision_target_k`
/// - Only checked after min_datapoints (hardcoded to 20) have been completed
/// - Evaluators not in the map run on all datapoints (up to max_datapoints)
/// - All datapoint tasks are spawned upfront for maximum concurrency
/// - When all evaluators have stopped, remaining tasks are aborted
///
/// When `precision_targets` is empty:
/// - All evaluators run on all datapoints (up to max_datapoints)
/// - Standard evaluation behavior
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
#[instrument(skip_all, fields(evaluation_run_id = %args.evaluation_run_id, evaluation_name = %args.evaluation_name, dataset_name = ?args.dataset_name, num_datapoint_ids = %args.datapoint_ids.as_deref().unwrap_or_default().len(), variant = ?args.variant, concurrency = %args.concurrency))]
pub async fn run_evaluation_core_streaming(
    args: EvaluationCoreArgs,
    max_datapoints: Option<u32>,
    precision_targets: HashMap<String, f32>,
) -> Result<EvaluationStreamResult> {
    // Convert Option<Vec<Uuid>> to Vec<Uuid> (None becomes empty vec)
    let datapoint_ids = args.datapoint_ids.unwrap_or_default();

    let (sender, receiver) = mpsc::channel(EVALUATION_CHANNEL_BUFFER_SIZE);

    // Build the semaphore and clients
    let semaphore = Arc::new(Semaphore::new(args.concurrency));
    let clients = Arc::new(Clients {
        inference_executor: args.inference_executor,
        clickhouse_client: args.clickhouse_client,
    });

    // Use the pre-resolved evaluation configuration
    let evaluation_config = args.evaluation_config.clone();

    debug!(evaluation_name = %args.evaluation_name, "Evaluation config found");

    let EvaluationConfig::Inference(inference_evaluation_config) = &*evaluation_config;

    info!(
        function_name = %inference_evaluation_config.function_name,
        evaluators = ?inference_evaluation_config.evaluators.keys().collect::<Vec<_>>(),
        "Function and evaluators configured"
    );

    // Validate that exactly one of dataset_name or datapoint_ids is provided
    if args.dataset_name.is_some() && !datapoint_ids.is_empty() {
        bail!(
            "Cannot provide both dataset_name and datapoint_ids. Please specify one or the other."
        );
    }
    if args.dataset_name.is_none() && datapoint_ids.is_empty() {
        bail!("Must provide either dataset_name or datapoint_ids.");
    }

    // Validate that max_datapoints is not used with datapoint_ids
    if !datapoint_ids.is_empty() && max_datapoints.is_some() {
        bail!(
            "Cannot provide both datapoint_ids and max_datapoints. max_datapoints can only be used with dataset_name."
        );
    }

    info!("Loading datapoints");
    let dataset = if let Some(dataset_name) = &args.dataset_name {
        // Load from dataset
        let request = ListDatapointsRequest {
            function_name: Some(inference_evaluation_config.function_name.clone()),
            limit: max_datapoints.or(Some(u32::MAX)), // Use u32::MAX when no limit specified, since otherwise ListDatapointsRequest defaults to 20
            offset: Some(0),
            ..Default::default()
        };
        list_datapoints(&clients.clickhouse_client, dataset_name.clone(), request)
            .await?
            .datapoints
    } else {
        // Load by IDs
        let request = GetDatapointsRequest {
            ids: datapoint_ids.clone(),
        };
        get_datapoints(
            &clients.clickhouse_client,
            /*dataset_name=*/ None,
            request,
        )
        .await?
        .datapoints
    };
    info!(
        dataset_size = dataset.len(),
        "Datapoints loaded successfully"
    );
    let dataset_name = Arc::new(
        args.dataset_name
            .clone()
            .unwrap_or_else(|| format!("datapoint_ids[{}]", datapoint_ids.len())),
    );
    let variant = Arc::new(args.variant);
    let variants = [variant]; // Single-element array for process_batch
    let evaluation_name = Arc::new(args.evaluation_name);
    let dataset_len = dataset.len();

    // Setup stopping manager for adaptive stopping
    let mut stopping_manager =
        stopping::StoppingManager::new(&inference_evaluation_config.evaluators, precision_targets);

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

    // Get cancellation tokens from stopping manager and wrap in Arc for cloning into tasks
    let cancellation_tokens_arc = Arc::new(stopping_manager.get_tokens().clone());

    // Save batcher_join_handle before moving clients into batch_params
    let batcher_join_handle = clients.clickhouse_client.batcher_join_handle();

    // Build batch processing params
    let batch_params = ProcessBatchParams {
        clients,
        function_configs: args.function_configs,
        evaluation_config: evaluation_config.clone(),
        evaluation_name,
        evaluation_run_id: args.evaluation_run_id,
        dataset_name,
        inference_cache: args.inference_cache,
        semaphore,
        cancellation_tokens: cancellation_tokens_arc,
        external_tags: Arc::new(args.tags),
    };

    // Process all datapoints across all variants
    let (mut join_set, task_id_map) = process_batch(&batch_params, dataset, &variants).await?;

    // Get a shared reference to the evaluation config (which includes evaluators)
    let evaluators = evaluation_config.clone();

    // Spawn a task to collect results and stream them
    let sender_clone = sender.clone();
    let mut num_completed_datapoints = 0;
    // We don't want to block (embedded) gateway shutdown on the evaluation finishing - the `receiver`
    // is responsible for determining if we're interested in the evaluation results.
    spawn_ignoring_shutdown(async move {
        while let Some(result) = join_set.join_next_with_id().await {
            let batch_result = collect_batch_result(result, &task_id_map);

            let update = match batch_result {
                BatchItemResult::Success(success) => {
                    num_completed_datapoints += 1;

                    // Update statistics and cancel any evaluators that have hit their precision target
                    stopping_manager.update_stats(&success.evaluation_result);
                    stopping_manager.cancel_converged_evaluators(num_completed_datapoints);

                    // If all evaluators have stopped, abort remaining tasks
                    if stopping_manager.all_evaluators_stopped() {
                        join_set.abort_all();
                    }

                    Some(EvaluationUpdate::Success(EvaluationInfo::new(
                        (*success.datapoint).clone(),
                        (*success.inference_response).clone(),
                        success.evaluation_result,
                    )))
                }
                BatchItemResult::Error(error) => {
                    tracing::warn!("Task error: {}", error.message);
                    Some(EvaluationUpdate::Error(EvaluationError {
                        datapoint_id: error.datapoint_id,
                        message: error.message,
                    }))
                }
                BatchItemResult::Cancelled => None,
            };

            // Check if update is Some; if so, unwrap and send inner value
            if let Some(update_value) = update
                && sender_clone.send(update_value).await.is_err()
            {
                // Receiver dropped, stop sending
                break;
            }
        }
    });

    Ok(EvaluationStreamResult {
        receiver,
        run_info,
        evaluation_config: evaluators,
        batcher_join_handle,
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
    variant: &'a EvaluationVariant,
    evaluation_run_id: Uuid,
    dataset_name: &'a str,
    datapoint: &'a Datapoint,
    input: &'a Input,
    evaluation_name: &'a str,
    function_config: &'a EvaluationFunctionConfig,
    inference_cache: CacheEnabledMode,
    external_tags: &'a HashMap<String, String>,
}

#[instrument(skip_all, fields(datapoint_id = %params.datapoint.id(), function_name = %params.function_name))]
async fn infer_datapoint(params: InferDatapointParams<'_>) -> Result<InferenceResponse> {
    let InferDatapointParams {
        clients,
        function_name,
        variant,
        evaluation_run_id,
        dataset_name,
        datapoint,
        evaluation_name,
        function_config,
        input,
        inference_cache,
        external_tags,
    } = params;

    // Extract variant_name, internal_dynamic_variant_config, and dryrun from the variant enum
    let (variant_name, internal_dynamic_variant_config, dryrun) = match variant {
        EvaluationVariant::Name(name) => (Some(name.clone()), None, false),
        EvaluationVariant::Info(info) => {
            // When using dynamic variant config, we must set dryrun=true to bypass
            // the safety check that prevents production use of unregistered variants.
            // For evaluations, this is safe because we're testing candidate variants.
            (None, Some((**info).clone()), true)
        }
    };

    debug!("Processing tool parameters");
    let dynamic_tool_params = match datapoint.tool_call_config() {
        Some(tool_params) => {
            debug!("Tool parameters found, processing");
            tool_params.clone()
        }
        None => {
            debug!("No tool parameters found");
            DynamicToolParams::default()
        }
    };
    debug!("Processing output schema");
    let output_schema = match (datapoint.output_schema(), function_config) {
        // If the datapoint has an output schema, use it only in the case where it is not the same as the output schema of the function
        (
            Some(output_schema),
            EvaluationFunctionConfig::Json {
                output_schema: fn_schema,
            },
        ) => {
            if output_schema == &fn_schema.value {
                debug!("Output schema matches function schema, using function default");
                None
            } else {
                debug!("Custom output schema provided");
                Some(output_schema)
            }
        }
        (Some(_), EvaluationFunctionConfig::Chat) => {
            return Err(anyhow!("Chat function does not support output schema"));
        }
        (None, _) => {
            debug!("No output schema specified");
            None
        }
    };
    // Create internal tags for this evaluation inference
    let internal_tags = HashMap::from([
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
    ]);

    // Merge external and internal tags, erroring on collision
    let tags = merge_tags(external_tags, internal_tags)?;

    let params = ClientInferenceParams {
        function_name: Some(function_name.to_string()),
        variant_name,
        input: input.clone(),
        tags,
        dynamic_tool_params,
        output_schema: output_schema.cloned(),
        credentials: HashMap::new(),
        cache_options: get_cache_options(inference_cache),
        dryrun: Some(dryrun),
        episode_id: None,
        model_name: None,
        stream: Some(false),
        params: InferenceParams::default(),
        include_original_response: false,
        include_raw_response: false,
        include_raw_usage: false,
        internal: true,
        extra_body: Default::default(),
        extra_headers: Default::default(),
        internal_dynamic_variant_config: internal_dynamic_variant_config.clone(),
        otlp_traces_extra_headers: HashMap::new(),
        otlp_traces_extra_attributes: HashMap::new(),
        otlp_traces_extra_resources: HashMap::new(),
        api_key: None,
    };
    debug!("Making inference request");
    let inference_result = clients.inference_executor.inference(params).await?;
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

// ============================================================================
// Shared Batch Processing Infrastructure
// ============================================================================

/// Parameters for processing a batch of datapoints across variants.
pub struct ProcessBatchParams {
    /// Shared clients for inference and database access
    pub clients: Arc<Clients>,
    /// Function configs table for looking up function definitions
    pub function_configs: Arc<EvaluationFunctionConfigTable>,
    /// Evaluation configuration
    pub evaluation_config: Arc<EvaluationConfig>,
    /// Name of the evaluation being run
    pub evaluation_name: Arc<String>,
    /// Unique ID for this evaluation run
    pub evaluation_run_id: Uuid,
    /// Name of the dataset (for tagging)
    pub dataset_name: Arc<String>,
    /// Cache mode for inference requests
    pub inference_cache: CacheEnabledMode,
    /// Semaphore for controlling concurrency
    pub semaphore: Arc<Semaphore>,
    /// Cancellation tokens for evaluators (empty if no adaptive stopping)
    pub cancellation_tokens: Arc<stopping::CancellationTokens>,
    /// External tags to apply to all inferences
    pub external_tags: Arc<HashMap<String, String>>,
}

/// Result of processing a single (datapoint, variant) pair.
///
/// Note: Fields are wrapped in `Arc` because they are shared across multiple concurrent tasks
/// during batch processing:
/// - `datapoint`: Shared across evaluator tasks for the same (datapoint, variant) pair
/// - `variant`: Shared across all datapoints being evaluated against this variant
/// - `inference_response`: Shared across evaluator tasks for the same (datapoint, variant) pair
pub struct DatapointVariantResult {
    /// The datapoint that was evaluated
    pub datapoint: Arc<Datapoint>,
    /// The variant that was used
    pub variant: Arc<EvaluationVariant>,
    /// The inference response
    pub inference_response: Arc<InferenceResponse>,
    /// Results from all evaluators (evaluator_name -> result)
    pub evaluation_result: evaluators::EvaluationResult,
}

/// Error from processing a single (datapoint, variant) pair.
pub struct DatapointVariantError {
    /// The datapoint ID that failed
    pub datapoint_id: Uuid,
    /// The variant that was used (if known)
    pub variant: Option<Arc<EvaluationVariant>>,
    /// Error message
    pub message: String,
}

/// Result from process_batch - either a successful result or an error.
pub enum BatchItemResult {
    Success(Box<DatapointVariantResult>),
    Error(DatapointVariantError),
    /// Task was cancelled (e.g., due to adaptive stopping)
    Cancelled,
}

/// Return type for process_batch: a JoinSet of results and a map from task ID to (datapoint_id, variant).
type ProcessBatchResult = Result<(
    JoinSet<Result<DatapointVariantResult>>,
    HashMap<tokio::task::Id, (Uuid, Arc<EvaluationVariant>)>,
)>;

/// Process a batch of datapoints across all variants.
///
/// This is the shared infrastructure for evaluation. It:
/// 1. Spawns concurrent tasks for each (datapoint, variant) pair
/// 2. Runs inference for each pair
/// 3. Runs evaluators on each inference result
/// 4. Returns results as they complete via a JoinSet
///
/// The caller is responsible for:
/// - Collecting results from the returned JoinSet
/// - Applying any stopping logic (evaluator CI convergence or variant elimination)
/// - Streaming results to consumers
///
/// # Arguments
/// * `params` - Shared parameters for the batch
/// * `datapoints` - The datapoints to process
/// * `variants` - The variants to evaluate (each datapoint is evaluated against all variants)
///
/// # Returns
/// A JoinSet that yields `DatapointVariantResult` as tasks complete, along with a mapping
/// from task ID to (datapoint_id, variant) for error reporting.
pub async fn process_batch(
    params: &ProcessBatchParams,
    datapoints: Vec<Datapoint>,
    variants: &[Arc<EvaluationVariant>],
) -> ProcessBatchResult {
    let mut join_set = JoinSet::new();
    let mut task_id_map = HashMap::new();

    let EvaluationConfig::Inference(inference_evaluation_config) = &*params.evaluation_config;
    let function_name = inference_evaluation_config.function_name.clone();

    // Pre-resolve all datapoint inputs before spawning tasks (avoids redundant work per variant)
    let mut datapoints_with_inputs: Vec<(Arc<Datapoint>, Arc<Input>)> =
        Vec::with_capacity(datapoints.len());
    for datapoint in datapoints {
        let stored_input = datapoint
            .input()
            .clone()
            .into_stored_input_without_file_handling()?;
        let resolver = ExecutorStorageResolver(params.clients.inference_executor.clone());
        let resolved_input = stored_input.reresolve(&resolver).await?;
        let input = Arc::new(resolved_input_to_client_input(resolved_input)?);
        datapoints_with_inputs.push((Arc::new(datapoint), input));
    }

    for (datapoint, input) in datapoints_with_inputs {
        let datapoint_id = datapoint.id();

        for variant in variants {
            let clients = params.clients.clone();
            let function_configs = params.function_configs.clone();
            let evaluation_config = params.evaluation_config.clone();
            let evaluation_name = params.evaluation_name.clone();
            let evaluation_run_id = params.evaluation_run_id;
            let dataset_name = params.dataset_name.clone();
            let inference_cache = params.inference_cache;
            let semaphore = params.semaphore.clone();
            let cancellation_tokens = params.cancellation_tokens.clone();
            let external_tags = params.external_tags.clone();
            let variant = variant.clone();
            let variant_for_map = variant.clone(); // Clone before moving into async block
            let datapoint = datapoint.clone();
            let function_name = function_name.clone();
            let input = input.clone();

            // Skip feedback for dynamic variants (they're not production-ready)
            let send_feedback = !matches!(variant.as_ref(), EvaluationVariant::Info(_));

            let abort_handle = join_set.spawn(async move {
                // Acquire semaphore permit for the entire task
                let _permit = semaphore.acquire().await?;

                // Look up function config
                let function_config = function_configs.get(&function_name).ok_or_else(|| {
                    anyhow!("Function '{function_name}' not found in function configs table")
                })?;

                // Run inference
                let inference_response = Arc::new(
                    infer_datapoint(InferDatapointParams {
                        clients: &clients,
                        function_name: &function_name,
                        variant: &variant,
                        evaluation_run_id,
                        dataset_name: &dataset_name,
                        datapoint: &datapoint,
                        evaluation_name: &evaluation_name,
                        function_config,
                        input: &input,
                        inference_cache,
                        external_tags: &external_tags,
                    })
                    .await
                    .map_err(|e| {
                        anyhow!("Error inferring for datapoint {}: {e}", datapoint.id())
                    })?,
                );

                // Run evaluators
                let evaluation_result = evaluate_inference(
                    EvaluateInferenceParams {
                        inference_response: inference_response.clone(),
                        datapoint: datapoint.clone(),
                        input,
                        evaluation_config,
                        evaluation_name,
                        clients: clients.clone(),
                        evaluation_run_id,
                        inference_cache,
                        external_tags: external_tags.clone(),
                        send_feedback,
                    },
                    cancellation_tokens.as_ref(),
                )
                .await
                .map_err(|e| {
                    anyhow!(
                        "Error evaluating inference {} for datapoint {}: {e}",
                        inference_response.inference_id(),
                        datapoint.id()
                    )
                })?;

                debug!(
                    datapoint_id = %datapoint.id(),
                    variant = ?variant,
                    evaluations_count = evaluation_result.len(),
                    "Evaluation completed"
                );

                Ok(DatapointVariantResult {
                    datapoint,
                    variant,
                    inference_response,
                    evaluation_result,
                })
            });

            task_id_map.insert(abort_handle.id(), (datapoint_id, variant_for_map));
        }
    }

    Ok((join_set, task_id_map))
}

/// Collect results from a JoinSet into BatchItemResults.
///
/// This helper converts JoinSet results into the appropriate BatchItemResult variant,
/// handling cancellation and errors appropriately.
pub fn collect_batch_result(
    result: Result<(tokio::task::Id, Result<DatapointVariantResult>), tokio::task::JoinError>,
    task_id_map: &HashMap<tokio::task::Id, (Uuid, Arc<EvaluationVariant>)>,
) -> BatchItemResult {
    match result {
        Ok((_, Ok(success))) => BatchItemResult::Success(Box::new(success)),
        Ok((task_id, Err(e))) => {
            let (datapoint_id, variant) = task_id_map
                .get(&task_id)
                .map(|(id, v)| (*id, Some(v.clone())))
                .unwrap_or((Uuid::nil(), None));
            BatchItemResult::Error(DatapointVariantError {
                datapoint_id,
                variant,
                message: e.to_string(),
            })
        }
        Err(join_error) => {
            if join_error.is_cancelled() {
                BatchItemResult::Cancelled
            } else {
                let (datapoint_id, variant) = task_id_map
                    .get(&join_error.id())
                    .map(|(id, v)| (*id, Some(v.clone())))
                    .unwrap_or((Uuid::nil(), None));
                BatchItemResult::Error(DatapointVariantError {
                    datapoint_id,
                    variant,
                    message: join_error.to_string(),
                })
            }
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

    #[test]
    fn test_merge_tags_no_collision() {
        let external_tags = HashMap::from([
            (
                "tensorzero::autopilot::session_id".to_string(),
                "session-123".to_string(),
            ),
            (
                "tensorzero::autopilot::tool_call_event_id".to_string(),
                "event-456".to_string(),
            ),
            ("custom_tag".to_string(), "custom_value".to_string()),
        ]);

        let internal_tags = HashMap::from([
            (
                "tensorzero::evaluation_run_id".to_string(),
                "eval-789".to_string(),
            ),
            ("tensorzero::datapoint_id".to_string(), "dp-001".to_string()),
        ]);

        let result = merge_tags(&external_tags, internal_tags);
        assert!(result.is_ok());

        let merged = result.unwrap();
        assert_eq!(merged.len(), 5);
        assert_eq!(
            merged.get("tensorzero::autopilot::session_id"),
            Some(&"session-123".to_string())
        );
        assert_eq!(
            merged.get("tensorzero::evaluation_run_id"),
            Some(&"eval-789".to_string())
        );
        assert_eq!(merged.get("custom_tag"), Some(&"custom_value".to_string()));
    }

    #[test]
    fn test_merge_tags_with_collision_evaluation_run_id() {
        let external_tags = HashMap::from([
            (
                "tensorzero::evaluation_run_id".to_string(),
                "external-eval-id".to_string(),
            ),
            ("custom_tag".to_string(), "custom_value".to_string()),
        ]);

        let internal_tags = HashMap::from([(
            "tensorzero::evaluation_run_id".to_string(),
            "internal-eval-id".to_string(),
        )]);

        let result = merge_tags(&external_tags, internal_tags);
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Tag collision"));
        assert!(error_msg.contains("tensorzero::evaluation_run_id"));
    }

    #[test]
    fn test_merge_tags_with_collision_datapoint_id() {
        let external_tags = HashMap::from([
            (
                "tensorzero::datapoint_id".to_string(),
                "external-dp-id".to_string(),
            ),
            ("tensorzero::autopilot".to_string(), "true".to_string()),
        ]);

        let internal_tags = HashMap::from([
            (
                "tensorzero::datapoint_id".to_string(),
                "internal-dp-id".to_string(),
            ),
            (
                "tensorzero::evaluation_name".to_string(),
                "test-eval".to_string(),
            ),
        ]);

        let result = merge_tags(&external_tags, internal_tags);
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Tag collision"));
        assert!(error_msg.contains("tensorzero::datapoint_id"));
        assert!(error_msg.contains("Reserved tag prefixes"));
    }

    #[test]
    fn test_merge_tags_empty_external() {
        let external_tags = HashMap::new();

        let internal_tags = HashMap::from([
            (
                "tensorzero::evaluation_run_id".to_string(),
                "eval-123".to_string(),
            ),
            ("tensorzero::datapoint_id".to_string(), "dp-456".to_string()),
        ]);

        let result = merge_tags(&external_tags, internal_tags);
        assert!(result.is_ok());

        let merged = result.unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(
            merged.get("tensorzero::evaluation_run_id"),
            Some(&"eval-123".to_string())
        );
    }

    #[test]
    fn test_merge_tags_empty_internal() {
        let external_tags = HashMap::from([
            (
                "tensorzero::autopilot::session_id".to_string(),
                "session-123".to_string(),
            ),
            ("custom_tag".to_string(), "value".to_string()),
        ]);

        let internal_tags = HashMap::new();

        let result = merge_tags(&external_tags, internal_tags);
        assert!(result.is_ok());

        let merged = result.unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(merged, external_tags);
    }

    #[test]
    fn test_merge_tags_multiple_collisions() {
        let external_tags = HashMap::from([
            (
                "tensorzero::evaluation_run_id".to_string(),
                "external-eval".to_string(),
            ),
            (
                "tensorzero::datapoint_id".to_string(),
                "external-dp".to_string(),
            ),
            ("safe_tag".to_string(), "safe_value".to_string()),
        ]);

        let internal_tags = HashMap::from([
            (
                "tensorzero::evaluation_run_id".to_string(),
                "internal-eval".to_string(),
            ),
            (
                "tensorzero::datapoint_id".to_string(),
                "internal-dp".to_string(),
            ),
        ]);

        let result = merge_tags(&external_tags, internal_tags);
        assert!(result.is_err());

        // It should error on the first collision it encounters
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Tag collision"));
    }
}
