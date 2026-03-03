//! GEPA optimizer implementation

use durable_tools_spawn::{SpawnClient, SpawnOptions, TaskStatus};
use futures::future::join_all;
use rand::{SeedableRng, rngs::StdRng, seq::IteratorRandom};
use std::{collections::HashMap, sync::Arc, time::Duration};
use uuid::Uuid;

use evaluations::{ClientInferenceExecutor, EvaluationsInferenceExecutor};
use tensorzero_core::{
    client::{ClientBuilder, ClientBuilderMode},
    config::{Config, UninitializedVariantConfig, provider_types::ProviderTypesConfig},
    db::{
        clickhouse::ClickHouseConnectionInfo, delegating_connection::DelegatingDatabaseQueries,
        postgres::PostgresConnectionInfo, valkey::ValkeyConnectionInfo,
    },
    endpoints::{datasets::v1::delete_dataset, inference::InferenceCredentials},
    error::{Error, ErrorDetails},
    evaluations::EvaluationConfig,
    http::TensorzeroHttpClient,
    model_table::ProviderTypeDefaultCredentials,
    optimization::{
        OptimizationJobInfo, OptimizerOutput,
        gepa::{GEPAConfig, GEPAJobHandle},
    },
    stored_inference::RenderedSample,
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use crate::{JobHandle, Optimizer};

pub mod analyze;
pub mod evaluate;
pub mod mutate;
pub mod pareto;
pub mod types;
pub mod validate;

use analyze::analyze_inferences;
use evaluate::{
    EvaluateVariantParams, VariantName, VariantScores, create_evaluation_dataset, evaluate_variant,
};
use mutate::mutate_variant;
use pareto::{Candidate, ParetoFrontier, is_improvement};
use validate::{get_uninitialized_variant_configs, validate_examples, validate_gepa_config};

pub use types::{
    GepaAnalyzeContinue, GepaAnalyzeParams, GepaAnalyzeResult, GepaCleanupParams,
    GepaEvalAndUpdateParams, GepaEvalParentContinue, GepaEvalParentParams, GepaEvalParentResult,
    GepaIterUpdateResult, GepaIterationParams, GepaIterationResult, GepaMutateContinue,
    GepaMutateParams, GepaMutateResult, GepaSampleContinue, GepaSampleParams, GepaSampleResult,
    GepaSetupParams, GepaSetupResult, GepaSideInfo, GepaSkipIteration, GepaToolOutput,
    GepaToolParams, ParetoCheckpoint, SelectedVariant,
};

/// A GEPA variant with its name and configuration
#[derive(Debug)]
pub struct GEPAVariant {
    pub name: VariantName,
    pub config: UninitializedChatCompletionConfig,
}

impl Optimizer for GEPAConfig {
    type Handle = GEPAJobHandle;

    async fn launch(
        &self,
        _client: &TensorzeroHttpClient,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        _credentials: &InferenceCredentials,
        _db: &Arc<dyn DelegatingDatabaseQueries + Send + Sync>,
        _config: std::sync::Arc<Config>,
        spawn_client: Option<&SpawnClient>,
    ) -> Result<Self::Handle, Error> {
        let spawn_client = spawn_client.ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: "GEPA optimization requires a durable task runtime (`spawn_client`). Ensure Postgres is configured.".to_string(),
            })
        })?;

        // Require validation examples for GEPA Pareto filtering
        let val_examples = val_examples.ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: "`val_examples` are required for GEPA optimization (used for Pareto frontier filtering)".to_string(),
            })
        })?;

        let llm_params = serde_json::to_value(GepaToolParams {
            gepa_config: self.as_uninitialized(),
        })
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: e.to_string(),
            })
        })?;

        let episode_id = Uuid::now_v7();
        let side_info = serde_json::to_value(GepaSideInfo {
            train_examples,
            val_examples,
        })
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: e.to_string(),
            })
        })?;

        let spawn_result = spawn_client
            .spawn_tool_by_name(
                "gepa_optimization",
                llm_params,
                side_info,
                episode_id,
                SpawnOptions::default(),
            )
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Failed to spawn durable GEPA task: {e}"),
                })
            })?;

        Ok(GEPAJobHandle {
            task_id: spawn_result.task_id,
        })
    }
}

impl JobHandle for GEPAJobHandle {
    async fn poll(
        &self,
        _client: &TensorzeroHttpClient,
        _credentials: &InferenceCredentials,
        _default_credentials: &ProviderTypeDefaultCredentials,
        _provider_types: &ProviderTypesConfig,
        spawn_client: Option<&SpawnClient>,
    ) -> Result<OptimizationJobInfo, Error> {
        let spawn_client = spawn_client.ok_or_else(|| {
            Error::new(ErrorDetails::InternalError {
                message: "GEPA poll requires a durable task runtime (`spawn_client`)".to_string(),
            })
        })?;

        let poll_result = spawn_client
            .get_task_result(self.task_id)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Failed to poll durable GEPA task {}: {e}", self.task_id),
                })
            })?;

        match poll_result.status {
            TaskStatus::Completed => {
                let result_value = poll_result.result.ok_or_else(|| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!(
                            "GEPA task {} completed but has no result payload",
                            self.task_id
                        ),
                    })
                })?;

                let output: GepaToolOutput = serde_json::from_value(result_value).map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!(
                            "Failed to deserialize GEPA task {} result: {e}",
                            self.task_id
                        ),
                    })
                })?;

                Ok(OptimizationJobInfo::Completed {
                    output: OptimizerOutput::Variants(
                        output
                            .variants
                            .into_iter()
                            .map(|(k, v)| {
                                (k, Box::new(UninitializedVariantConfig::ChatCompletion(v)))
                            })
                            .collect(),
                    ),
                })
            }
            TaskStatus::Failed => {
                let error_msg = poll_result
                    .error
                    .as_ref()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "Unknown error".to_string());

                Ok(OptimizationJobInfo::Failed {
                    message: error_msg,
                    error: poll_result.error,
                })
            }
            TaskStatus::Cancelled => Ok(OptimizationJobInfo::Failed {
                message: format!("GEPA task {} was cancelled", self.task_id),
                error: None,
            }),
            TaskStatus::Pending | TaskStatus::Running | TaskStatus::Sleeping => {
                Ok(OptimizationJobInfo::Pending {
                    message: format!("GEPA task {} is running", self.task_id),
                    estimated_finish: None,
                    trained_tokens: None,
                    error: None,
                })
            }
        }
    }
}

/// Build an inference executor from an HTTP client and config, suitable for GEPA inference calls.
pub async fn build_inference_executor(
    http_client: &TensorzeroHttpClient,
    config: &Arc<Config>,
    timeout_secs: u64,
) -> Result<Arc<dyn EvaluationsInferenceExecutor>, Error> {
    let gateway_clickhouse = match std::env::var("TENSORZERO_CLICKHOUSE_URL") {
        Ok(url) => {
            ClickHouseConnectionInfo::new(&url, config.gateway.observability.batch_writes.clone())
                .await?
        }
        Err(_) => ClickHouseConnectionInfo::new_disabled(),
    };
    let client = ClientBuilder::new(ClientBuilderMode::FromComponents {
        config: config.clone(),
        clickhouse_connection_info: gateway_clickhouse,
        postgres_connection_info: PostgresConnectionInfo::Disabled,
        valkey_connection_info: ValkeyConnectionInfo::Disabled,
        valkey_cache_connection_info: ValkeyConnectionInfo::Disabled,
        http_client: http_client.clone(),
        timeout: Some(Duration::from_secs(timeout_secs)),
    })
    .build()
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to build gateway client for GEPA optimization: {e}"),
        })
    })?;
    Ok(Arc::new(ClientInferenceExecutor::new(client)))
}

/// Convenience function that runs the full GEPA optimization pipeline.
///
/// Calls `run_gepa_setup`, then `run_gepa_iteration` in a loop, then `run_gepa_cleanup`.
/// Used by tests and the GepaTool for durable execution.
#[expect(clippy::too_many_arguments)]
pub async fn run_gepa(
    variant_executor: Arc<dyn EvaluationsInferenceExecutor>,
    judge_executor: Arc<dyn EvaluationsInferenceExecutor>,
    gepa_config: &GEPAConfig,
    train_examples: Vec<RenderedSample>,
    val_examples: Vec<RenderedSample>,
    db: &Arc<dyn DelegatingDatabaseQueries + Send + Sync>,
    config: &Arc<Config>,
    http_client: &TensorzeroHttpClient,
) -> Result<GepaToolOutput, Error> {
    let setup_result = run_gepa_setup(
        variant_executor.clone(),
        GepaSetupParams {
            gepa_config: gepa_config.as_uninitialized(),
            train_examples,
            val_examples,
        },
        db,
        config,
        http_client,
    )
    .await?;

    let max_iterations = setup_result.max_iterations;
    let mut checkpoint = setup_result.checkpoint;
    let mut temporary_datasets = setup_result.temporary_datasets;
    let train_examples = setup_result.train_examples;
    let gepa_config = setup_result.gepa_config;
    let val_dataset_name = setup_result.val_dataset_name;
    let per_variant_concurrency = setup_result.per_variant_concurrency;
    let run_id = setup_result.run_id;
    let original_variant_names = setup_result.original_variant_names;

    for iteration in 0..(max_iterations as usize) {
        let iteration_result = run_gepa_iteration(
            variant_executor.clone(),
            judge_executor.clone(),
            GepaIterationParams {
                checkpoint,
                train_examples: train_examples.clone(),
                iteration,
                gepa_config: gepa_config.clone(),
                val_dataset_name: val_dataset_name.clone(),
                temporary_datasets,
                per_variant_concurrency,
                run_id,
            },
            db,
            config,
            http_client,
        )
        .await?;

        checkpoint = iteration_result.checkpoint;
        temporary_datasets = iteration_result.temporary_datasets;
    }

    run_gepa_cleanup(
        GepaCleanupParams {
            checkpoint,
            temporary_datasets,
            original_variant_names,
        },
        db,
    )
    .await
}

/// GEPA setup step: validate config, create validation dataset, evaluate initial variants,
/// and build the initial Pareto frontier.
pub async fn run_gepa_setup(
    variant_executor: Arc<dyn EvaluationsInferenceExecutor>,
    params: GepaSetupParams,
    db: &Arc<dyn DelegatingDatabaseQueries + Send + Sync>,
    config: &Arc<Config>,
    http_client: &TensorzeroHttpClient,
) -> Result<GepaSetupResult, Error> {
    let GepaSetupParams {
        gepa_config: uninitialized_gepa_config,
        train_examples,
        val_examples,
    } = params;

    let gepa_config = uninitialized_gepa_config.clone().load();
    let function_context = validate_gepa_config(&gepa_config, config)?;

    let train_examples = validate_examples(train_examples)?;
    let val_examples = validate_examples(val_examples)?;

    tracing::info!(
        "Starting GEPA optimization for function '{}' with {} train examples and {} val examples",
        gepa_config.function_name,
        train_examples.len(),
        val_examples.len()
    );

    let original_variants = get_uninitialized_variant_configs(&gepa_config, &function_context)?;

    let original_variant_names: Vec<String> = original_variants.keys().cloned().collect();

    tracing::info!(
        "Initialized with {} baseline variants: {:?}",
        original_variant_names.len(),
        original_variant_names
    );

    let run_id = Uuid::now_v7();
    let val_dataset_name = format!("{}_gepa_val_{}", gepa_config.evaluation_name, run_id);

    let temporary_datasets = vec![val_dataset_name.clone()];

    tracing::info!(
        "Creating validation dataset '{}' with {} examples",
        val_dataset_name,
        val_examples.len()
    );

    let val_create_datapoints_response = create_evaluation_dataset(
        config,
        http_client,
        db.as_ref(),
        val_examples,
        &val_dataset_name,
    )
    .await?;

    tracing::info!("Validation dataset created successfully");

    let num_variants = original_variants.len();
    tracing::info!(
        "Evaluating {} initial variants on validation dataset",
        num_variants
    );

    let per_variant_concurrency = (gepa_config.max_concurrency as usize / num_variants).max(1);

    let evaluation_name = gepa_config.evaluation_name.clone();
    let evaluator_configs = match &*function_context.evaluation_config {
        EvaluationConfig::Inference(cfg) => &cfg.evaluators,
    };

    let evaluation_futures: Vec<_> = original_variants
        .iter()
        .map(|(variant_name, variant_config)| {
            let variant_executor = variant_executor.clone();
            let db = Arc::clone(db);
            let functions = config.functions.clone();
            let evaluation_config_param = Arc::clone(&function_context.evaluation_config);
            let evaluation_name = evaluation_name.clone();
            let variant_name = variant_name.clone();
            let variant_config = variant_config.clone();
            let val_dataset_name = val_dataset_name.clone();

            async move {
                match evaluate_variant(EvaluateVariantParams {
                    inference_executor: variant_executor,
                    db,
                    functions,
                    evaluation_config: evaluation_config_param,
                    evaluation_name,
                    variant_name: variant_name.clone(),
                    variant_config,
                    dataset_name: val_dataset_name,
                    concurrency: per_variant_concurrency,
                })
                .await
                {
                    Ok(results) => Ok::<_, Error>((variant_name, results)),
                    Err(e) => {
                        tracing::warn!("Evaluation failed for variant '{}': {}", variant_name, e);
                        Err(e)
                    }
                }
            }
        })
        .collect();

    let val_evaluation_results = join_all(evaluation_futures).await;

    let mut initial_scores: HashMap<VariantName, VariantScores> = HashMap::new();
    for result in val_evaluation_results {
        match result {
            Ok((variant_name, eval_results)) => {
                tracing::debug!(
                    "Initial evaluation stats for '{}': {:#?}",
                    variant_name,
                    eval_results.evaluation_stats
                );
                let scores = eval_results.per_datapoint_scores();
                if !scores.is_empty() {
                    initial_scores.insert(variant_name.clone(), scores);
                }
            }
            Err(e) => {
                tracing::error!("Unexpected error in evaluation: {}", e);
            }
        }
    }

    tracing::info!(
        "Initial evaluation complete: collected validation scores for {} variants",
        initial_scores.len()
    );

    if initial_scores.is_empty() {
        return Err(Error::new(ErrorDetails::InternalError {
            message: "No validation scores collected for initial variants".to_string(),
        }));
    }

    let mut pareto_frontier = ParetoFrontier::new(
        val_create_datapoints_response.ids,
        evaluator_configs,
        gepa_config.seed.map(|s| s as u64),
    );

    let mut initial_candidates: HashMap<VariantName, Candidate> = HashMap::new();
    for (variant_name, scores) in initial_scores {
        if let Some(variant_config) = original_variants.get(&variant_name) {
            initial_candidates.insert(
                variant_name.clone(),
                Candidate {
                    variant: variant_config.clone(),
                    scores,
                },
            );
        } else {
            tracing::warn!(
                "Validation scores found for unknown variant '{}'; skipping",
                variant_name
            );
        }
    }

    pareto_frontier.update(initial_candidates)?;

    let checkpoint = pareto_frontier.to_checkpoint();

    Ok(GepaSetupResult {
        checkpoint,
        train_examples,
        max_iterations: gepa_config.max_iterations,
        val_dataset_name,
        temporary_datasets,
        original_variant_names,
        gepa_config: uninitialized_gepa_config,
        per_variant_concurrency,
        run_id,
    })
}

/// Run a single GEPA iteration: sample → eval parent → analyze → mutate → eval & update.
///
/// This is a convenience wrapper that calls the 5 sub-step functions sequentially.
/// The sub-steps are also available individually for durable checkpointing.
pub async fn run_gepa_iteration(
    variant_executor: Arc<dyn EvaluationsInferenceExecutor>,
    judge_executor: Arc<dyn EvaluationsInferenceExecutor>,
    params: GepaIterationParams,
    db: &Arc<dyn DelegatingDatabaseQueries + Send + Sync>,
    config: &Arc<Config>,
    http_client: &TensorzeroHttpClient,
) -> Result<GepaIterationResult, Error> {
    let iteration = params.iteration;
    let gepa_config = params.gepa_config.clone();
    let val_dataset_name = params.val_dataset_name.clone();
    let per_variant_concurrency = params.per_variant_concurrency;

    // Sub-step 1: Sample parent and create minibatch dataset
    let sample_result = run_gepa_iter_sample(
        GepaSampleParams {
            checkpoint: params.checkpoint,
            train_examples: params.train_examples,
            iteration,
            gepa_config: gepa_config.clone(),
            val_dataset_name: val_dataset_name.clone(),
            temporary_datasets: params.temporary_datasets,
            per_variant_concurrency,
            run_id: params.run_id,
        },
        db,
        config,
        http_client,
    )
    .await?;

    let GepaSampleContinue {
        parent,
        mutation_dataset_name,
        temporary_datasets,
        checkpoint,
    } = match sample_result {
        GepaSampleResult::Continue(data) => *data,
        GepaSampleResult::SkipIteration(skip) => {
            return Ok(GepaIterationResult {
                checkpoint: skip.checkpoint,
                temporary_datasets: skip.temporary_datasets,
            });
        }
    };

    // Sub-step 2: Evaluate parent on minibatch
    let eval_parent_result = run_gepa_iter_eval_parent(
        variant_executor.clone(),
        GepaEvalParentParams {
            parent,
            mutation_dataset_name,
            gepa_config: gepa_config.clone(),
            iteration,
            checkpoint,
            temporary_datasets,
            val_dataset_name: val_dataset_name.clone(),
            per_variant_concurrency,
        },
        db,
        config,
    )
    .await?;

    let GepaEvalParentContinue {
        parent,
        parent_evaluation_infos,
        parent_evaluation_stats,
        mutation_dataset_name,
        checkpoint,
        temporary_datasets,
        ..
    } = match eval_parent_result {
        GepaEvalParentResult::Continue(data) => *data,
        GepaEvalParentResult::SkipIteration(skip) => {
            return Ok(GepaIterationResult {
                checkpoint: skip.checkpoint,
                temporary_datasets: skip.temporary_datasets,
            });
        }
    };

    // Sub-step 3: Analyze parent inferences
    let analyze_result = run_gepa_iter_analyze(
        judge_executor.clone(),
        GepaAnalyzeParams {
            parent,
            parent_evaluation_infos,
            parent_evaluation_stats,
            mutation_dataset_name,
            gepa_config: gepa_config.clone(),
            iteration,
            checkpoint,
            temporary_datasets,
            val_dataset_name: val_dataset_name.clone(),
            per_variant_concurrency,
        },
        config,
    )
    .await?;

    let GepaAnalyzeContinue {
        parent,
        parent_analyses,
        parent_evaluation_stats,
        mutation_dataset_name,
        checkpoint,
        temporary_datasets,
        ..
    } = match analyze_result {
        GepaAnalyzeResult::Continue(data) => *data,
        GepaAnalyzeResult::SkipIteration(skip) => {
            return Ok(GepaIterationResult {
                checkpoint: skip.checkpoint,
                temporary_datasets: skip.temporary_datasets,
            });
        }
    };

    // Sub-step 4: Mutate parent to produce child
    let mutate_result = run_gepa_iter_mutate(
        judge_executor,
        GepaMutateParams {
            parent,
            parent_analyses,
            parent_evaluation_stats,
            mutation_dataset_name,
            gepa_config: gepa_config.clone(),
            iteration,
            checkpoint,
            temporary_datasets,
            val_dataset_name: val_dataset_name.clone(),
            per_variant_concurrency,
        },
        config,
    )
    .await?;

    let GepaMutateContinue {
        child,
        parent_evaluation_stats,
        mutation_dataset_name,
        checkpoint,
        temporary_datasets,
        ..
    } = match mutate_result {
        GepaMutateResult::Continue(data) => *data,
        GepaMutateResult::SkipIteration(skip) => {
            return Ok(GepaIterationResult {
                checkpoint: skip.checkpoint,
                temporary_datasets: skip.temporary_datasets,
            });
        }
    };

    // Sub-step 5: Evaluate child, compare, conditional val eval, update frontier
    let update_result = run_gepa_iter_eval_and_update(
        variant_executor,
        GepaEvalAndUpdateParams {
            child,
            parent_evaluation_stats,
            mutation_dataset_name,
            gepa_config,
            iteration,
            checkpoint,
            temporary_datasets,
            val_dataset_name,
            per_variant_concurrency,
        },
        db,
        config,
    )
    .await?;

    Ok(GepaIterationResult {
        checkpoint: update_result.checkpoint,
        temporary_datasets: update_result.temporary_datasets,
    })
}

// ============================================================================
// Sub-step functions for checkpointable GEPA iteration
// ============================================================================

/// Sub-step 1: Sample a parent from the Pareto frontier and create a minibatch dataset.
///
/// No LLM calls — fast, local operations only.
pub async fn run_gepa_iter_sample(
    params: GepaSampleParams,
    db: &Arc<dyn DelegatingDatabaseQueries + Send + Sync>,
    config: &Arc<Config>,
    http_client: &TensorzeroHttpClient,
) -> Result<GepaSampleResult, Error> {
    let GepaSampleParams {
        checkpoint,
        train_examples,
        iteration,
        gepa_config: uninitialized_gepa_config,
        val_dataset_name: _,
        mut temporary_datasets,
        per_variant_concurrency: _,
        run_id,
    } = params;

    let gepa_config = uninitialized_gepa_config.load();
    let _function_context = validate_gepa_config(&gepa_config, config)?;

    let pareto_frontier = ParetoFrontier::from_checkpoint(checkpoint, iteration as u64);

    let parent = match pareto_frontier.sample_by_frequency() {
        Ok(variant) => variant,
        Err(err) => {
            tracing::warn!(
                "Skipping iteration {} because no candidates were available: {}",
                iteration,
                err
            );
            return Ok(GepaSampleResult::SkipIteration(Box::new(
                GepaSkipIteration {
                    checkpoint: pareto_frontier.to_checkpoint(),
                    temporary_datasets,
                },
            )));
        }
    };

    tracing::info!(
        "GEPA iteration {}: selected parent variant '{}'",
        iteration,
        parent.name
    );

    // Use a different seed derivation than from_checkpoint (which uses seed + iteration)
    // to avoid correlated sampling between parent selection and minibatch selection.
    let mut rng = match gepa_config.seed {
        Some(seed) => StdRng::seed_from_u64(
            (seed as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(iteration as u64),
        ),
        None => {
            let mut thread_rng = rand::rng();
            StdRng::from_rng(&mut thread_rng)
        }
    };

    let batch_size = gepa_config.batch_size.min(train_examples.len());
    let mutation_examples: Vec<RenderedSample> = train_examples
        .iter()
        .sample(&mut rng, batch_size)
        .into_iter()
        .cloned()
        .collect();

    let mutation_dataset_name = format!(
        "{}_gepa_mutation_{}_{}",
        gepa_config.evaluation_name, iteration, run_id,
    );
    temporary_datasets.push(mutation_dataset_name.clone());

    tracing::debug!(
        "GEPA iteration {}: creating minibatch dataset with {} examples",
        iteration,
        mutation_examples.len()
    );

    let _mutation_create_datapoints_response = create_evaluation_dataset(
        config,
        http_client,
        db.as_ref(),
        mutation_examples,
        &mutation_dataset_name,
    )
    .await?;

    Ok(GepaSampleResult::Continue(Box::new(GepaSampleContinue {
        parent: SelectedVariant {
            name: parent.name,
            config: parent.config,
        },
        mutation_dataset_name,
        temporary_datasets,
        checkpoint: pareto_frontier.to_checkpoint(),
    })))
}

/// Sub-step 2: Evaluate parent variant on the minibatch dataset.
///
/// This is SLOW — involves N LLM inference calls.
pub async fn run_gepa_iter_eval_parent(
    variant_executor: Arc<dyn EvaluationsInferenceExecutor>,
    params: GepaEvalParentParams,
    db: &Arc<dyn DelegatingDatabaseQueries + Send + Sync>,
    config: &Arc<Config>,
) -> Result<GepaEvalParentResult, Error> {
    let GepaEvalParentParams {
        parent,
        mutation_dataset_name,
        gepa_config: uninitialized_gepa_config,
        iteration,
        checkpoint,
        temporary_datasets,
        val_dataset_name,
        per_variant_concurrency,
    } = params;

    let gepa_config = uninitialized_gepa_config.load();
    let function_context = validate_gepa_config(&gepa_config, config)?;

    tracing::info!(
        "GEPA iteration {}: evaluating parent variant on minibatch",
        iteration
    );

    let parent_evaluation_results = match evaluate_variant(EvaluateVariantParams {
        inference_executor: variant_executor,
        db: Arc::clone(db),
        functions: config.functions.clone(),
        evaluation_config: Arc::clone(&function_context.evaluation_config),
        evaluation_name: gepa_config.evaluation_name.clone(),
        variant_name: parent.name.clone(),
        variant_config: parent.config.clone(),
        dataset_name: mutation_dataset_name.clone(),
        concurrency: gepa_config.max_concurrency as usize,
    })
    .await
    {
        Ok(evaluation_results) => evaluation_results,
        Err(e) => {
            tracing::warn!(
                "GEPA iteration {}: evaluation failed for parent '{}': {}",
                iteration,
                parent.name,
                e
            );
            return Ok(GepaEvalParentResult::SkipIteration(Box::new(
                GepaSkipIteration {
                    checkpoint,
                    temporary_datasets,
                },
            )));
        }
    };

    tracing::debug!(
        "GEPA iteration {}: parent '{}' minibatch evaluation stats: {:#?}",
        iteration,
        parent.name,
        parent_evaluation_results.evaluation_stats
    );

    Ok(GepaEvalParentResult::Continue(Box::new(
        GepaEvalParentContinue {
            parent,
            parent_evaluation_infos: parent_evaluation_results.evaluation_infos().to_vec(),
            parent_evaluation_stats: parent_evaluation_results.evaluation_stats,
            mutation_dataset_name,
            checkpoint,
            temporary_datasets,
            val_dataset_name,
            per_variant_concurrency,
        },
    )))
}

/// Sub-step 3: Analyze parent inferences using the judge model.
///
/// This is SLOW — involves LLM judge calls for each inference.
pub async fn run_gepa_iter_analyze(
    judge_executor: Arc<dyn EvaluationsInferenceExecutor>,
    params: GepaAnalyzeParams,
    config: &Arc<Config>,
) -> Result<GepaAnalyzeResult, Error> {
    let GepaAnalyzeParams {
        parent,
        parent_evaluation_infos,
        parent_evaluation_stats,
        mutation_dataset_name,
        gepa_config: uninitialized_gepa_config,
        iteration,
        checkpoint,
        temporary_datasets,
        val_dataset_name,
        per_variant_concurrency,
    } = params;

    let gepa_config = uninitialized_gepa_config.load();
    let function_context = validate_gepa_config(&gepa_config, config)?;

    tracing::info!(
        "GEPA iteration {}: analyzing {} parent inferences",
        iteration,
        parent_evaluation_infos.len()
    );

    let parent_analyses = match analyze_inferences(
        judge_executor.as_ref(),
        &parent_evaluation_infos,
        &function_context,
        &parent.config,
        &gepa_config,
    )
    .await
    {
        Ok(analyses) => analyses,
        Err(err) => {
            tracing::warn!(
                "GEPA iteration {}: analysis failed for parent '{}': {}",
                iteration,
                parent.name,
                err
            );
            return Ok(GepaAnalyzeResult::SkipIteration(Box::new(
                GepaSkipIteration {
                    checkpoint,
                    temporary_datasets,
                },
            )));
        }
    };

    tracing::info!(
        "GEPA iteration {}: completed {} analyses for parent '{}', generating child variant",
        iteration,
        parent_analyses.len(),
        parent.name
    );

    Ok(GepaAnalyzeResult::Continue(Box::new(GepaAnalyzeContinue {
        parent,
        parent_analyses,
        parent_evaluation_stats,
        mutation_dataset_name,
        checkpoint,
        temporary_datasets,
        val_dataset_name,
        per_variant_concurrency,
    })))
}

/// Sub-step 4: Mutate the parent variant to produce a child.
///
/// This is SLOW — involves an LLM call to generate new templates.
pub async fn run_gepa_iter_mutate(
    judge_executor: Arc<dyn EvaluationsInferenceExecutor>,
    params: GepaMutateParams,
    config: &Arc<Config>,
) -> Result<GepaMutateResult, Error> {
    let GepaMutateParams {
        parent,
        parent_analyses,
        parent_evaluation_stats,
        mutation_dataset_name,
        gepa_config: uninitialized_gepa_config,
        iteration,
        checkpoint,
        temporary_datasets,
        val_dataset_name,
        per_variant_concurrency,
    } = params;

    let gepa_config = uninitialized_gepa_config.load();
    let function_context = validate_gepa_config(&gepa_config, config)?;

    let parent_variant = GEPAVariant {
        name: parent.name.clone(),
        config: parent.config,
    };

    let child = match mutate_variant(
        judge_executor.as_ref(),
        &parent_analyses,
        &function_context,
        &parent_variant,
        &gepa_config,
        iteration,
    )
    .await
    {
        Ok(child_variant) => {
            tracing::info!(
                "GEPA iteration {}: mutation complete, evaluating child variant",
                iteration
            );
            child_variant
        }
        Err(err) => {
            tracing::warn!(
                "GEPA iteration {}: mutation failed for parent '{}': {}",
                iteration,
                parent_variant.name,
                err
            );
            return Ok(GepaMutateResult::SkipIteration(Box::new(
                GepaSkipIteration {
                    checkpoint,
                    temporary_datasets,
                },
            )));
        }
    };

    Ok(GepaMutateResult::Continue(Box::new(GepaMutateContinue {
        child: SelectedVariant {
            name: child.name,
            config: child.config,
        },
        parent_evaluation_stats,
        mutation_dataset_name,
        checkpoint,
        temporary_datasets,
        val_dataset_name,
        per_variant_concurrency,
    })))
}

/// Sub-step 5: Evaluate child on minibatch, compare with parent, conditionally evaluate
/// on validation dataset, and update the Pareto frontier.
///
/// This is SLOW — involves LLM inference calls for child evaluation.
pub async fn run_gepa_iter_eval_and_update(
    variant_executor: Arc<dyn EvaluationsInferenceExecutor>,
    params: GepaEvalAndUpdateParams,
    db: &Arc<dyn DelegatingDatabaseQueries + Send + Sync>,
    config: &Arc<Config>,
) -> Result<GepaIterUpdateResult, Error> {
    let GepaEvalAndUpdateParams {
        child,
        parent_evaluation_stats,
        mutation_dataset_name,
        gepa_config: uninitialized_gepa_config,
        iteration,
        checkpoint,
        temporary_datasets,
        val_dataset_name,
        per_variant_concurrency,
    } = params;

    let gepa_config = uninitialized_gepa_config.load();
    let function_context = validate_gepa_config(&gepa_config, config)?;

    let evaluator_configs = match &*function_context.evaluation_config {
        EvaluationConfig::Inference(cfg) => &cfg.evaluators,
    };

    let mut pareto_frontier = ParetoFrontier::from_checkpoint(checkpoint, iteration as u64);

    tracing::info!(
        "GEPA iteration {}: evaluating child variant '{}' on minibatch",
        iteration,
        child.name
    );

    let child_evaluation_results = match evaluate_variant(EvaluateVariantParams {
        inference_executor: variant_executor.clone(),
        db: Arc::clone(db),
        functions: config.functions.clone(),
        evaluation_config: Arc::clone(&function_context.evaluation_config),
        evaluation_name: gepa_config.evaluation_name.clone(),
        variant_name: child.name.clone(),
        variant_config: child.config.clone(),
        dataset_name: mutation_dataset_name,
        concurrency: gepa_config.max_concurrency as usize,
    })
    .await
    {
        Ok(evaluation_results) => evaluation_results,
        Err(e) => {
            tracing::warn!(
                "GEPA iteration {}: minibatch evaluation failed for child variant '{}': {}",
                iteration,
                child.name,
                e
            );
            return Ok(GepaIterUpdateResult {
                checkpoint: pareto_frontier.to_checkpoint(),
                temporary_datasets,
            });
        }
    };

    tracing::debug!(
        "GEPA iteration {}: child '{}' minibatch evaluation stats: {:#?}",
        iteration,
        child.name,
        child_evaluation_results.evaluation_stats
    );

    tracing::info!(
        "GEPA iteration {}: child variant '{}' minibatch evaluation complete",
        iteration,
        child.name
    );

    let child_improves = is_improvement(
        &parent_evaluation_stats,
        &child_evaluation_results.evaluation_stats,
        evaluator_configs,
    );

    if child_improves {
        tracing::info!(
            "GEPA iteration {}: evaluating child variant '{}' on validation dataset",
            iteration,
            child.name
        );

        match evaluate_variant(EvaluateVariantParams {
            inference_executor: variant_executor,
            db: Arc::clone(db),
            functions: config.functions.clone(),
            evaluation_config: Arc::clone(&function_context.evaluation_config),
            evaluation_name: gepa_config.evaluation_name.clone(),
            variant_name: child.name.clone(),
            variant_config: child.config.clone(),
            dataset_name: val_dataset_name,
            concurrency: per_variant_concurrency,
        })
        .await
        {
            Ok(val_mutation_evaluation_results) => {
                tracing::debug!(
                    "GEPA iteration {}: child '{}' validation evaluation stats: {:#?}",
                    iteration,
                    child.name,
                    val_mutation_evaluation_results.evaluation_stats
                );
                let child_val_scores = val_mutation_evaluation_results.per_datapoint_scores();
                let mut candidate = HashMap::new();
                candidate.insert(
                    child.name.clone(),
                    Candidate {
                        variant: child.config.clone(),
                        scores: child_val_scores,
                    },
                );
                tracing::info!(
                    "GEPA iteration {}: child variant '{}' validation scores collected ({} datapoints)",
                    iteration,
                    child.name,
                    val_mutation_evaluation_results.evaluation_infos().len()
                );
                match pareto_frontier.update(candidate) {
                    Ok(()) => {
                        tracing::info!(
                            "GEPA iteration {}: Pareto frontier updated; pool size: {}",
                            iteration,
                            pareto_frontier.variant_configs().len()
                        );
                    }
                    Err(err) => {
                        tracing::warn!(
                            "GEPA iteration {}: failed to update Pareto frontier with child '{}': {}",
                            iteration,
                            child.name,
                            err
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    "GEPA iteration {}: validation evaluation failed for child variant '{}': {}",
                    iteration,
                    child.name,
                    e
                );
            }
        }
    }

    Ok(GepaIterUpdateResult {
        checkpoint: pareto_frontier.to_checkpoint(),
        temporary_datasets,
    })
}

/// GEPA cleanup step: delete temporary datasets and extract final variant configs.
pub async fn run_gepa_cleanup(
    params: GepaCleanupParams,
    db: &Arc<dyn DelegatingDatabaseQueries + Send + Sync>,
) -> Result<GepaToolOutput, Error> {
    let GepaCleanupParams {
        checkpoint,
        temporary_datasets,
        original_variant_names,
    } = params;

    let original_set: std::collections::HashSet<String> =
        original_variant_names.into_iter().collect();

    let mut new_variants = checkpoint.variant_configs;
    new_variants.retain(|name, _| !original_set.contains(name));

    tracing::info!(
        "GEPA optimization complete: created {} new variant(s)",
        new_variants.len()
    );
    tracing::debug!("New variants: {:#?}", new_variants);

    for dataset_name in &temporary_datasets {
        if let Err(err) = delete_dataset(db.as_ref(), dataset_name).await {
            tracing::warn!(
                "Failed to delete temporary GEPA dataset '{}': {}",
                dataset_name,
                err
            );
        }
    }

    Ok(GepaToolOutput {
        variants: new_variants,
    })
}
