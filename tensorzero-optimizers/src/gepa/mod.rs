//! GEPA optimizer implementation

use futures::future::join_all;
use rand::{SeedableRng, rngs::StdRng, seq::IteratorRandom};
use std::{collections::HashMap, sync::Arc, time::Duration};
use uuid::Uuid;

use tensorzero_core::{
    client::{ClientBuilder, ClientBuilderMode},
    config::{Config, UninitializedVariantConfig, provider_types::ProviderTypesConfig},
    db::{
        clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo,
        valkey::ValkeyConnectionInfo,
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
pub mod validate;

use analyze::analyze_inferences;
use evaluate::{
    EvaluateVariantParams, VariantName, VariantScores, create_evaluation_dataset, evaluate_variant,
};
use mutate::mutate_variant;
use pareto::{Candidate, ParetoFrontier, is_improvement};
use validate::{get_uninitialized_variant_configs, validate_examples, validate_gepa_config};

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
        client: &TensorzeroHttpClient,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        _credentials: &InferenceCredentials,
        clickhouse_connection_info: &ClickHouseConnectionInfo,
        config: std::sync::Arc<Config>,
    ) -> Result<Self::Handle, Error> {
        // Validate configuration and examples, get the FunctionContext (function_config, static_tools, and evaluation_config)
        let function_context = validate_gepa_config(self, &config)?;

        // Require validation examples for GEPA Pareto filtering
        // TODO[#4772]: Random split from train_examples if None
        let val_examples = val_examples.ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: "`val_examples` are required for GEPA optimization (used for Pareto frontier filtering)".to_string(),
            })
        })?;

        // Validate both train and validation examples (this filters invalid examples)
        let train_examples = validate_examples(train_examples)?;
        let val_examples = validate_examples(val_examples)?;

        tracing::info!(
            "Starting GEPA optimization for function '{}' with {} train examples and {} val examples",
            self.function_name,
            train_examples.len(),
            val_examples.len()
        );

        // Build the gateway client once for the entire optimization run
        let gateway_client = ClientBuilder::new(ClientBuilderMode::FromComponents {
            config: config.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            valkey_connection_info: ValkeyConnectionInfo::Disabled,
            http_client: client.clone(),
            timeout: Some(Duration::from_secs(self.timeout)),
        })
        .build()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to build gateway client for GEPA optimization: {e}"),
            })
        })?;

        tracing::debug!("Gateway client built successfully for GEPA optimization");

        // Get uninitialized baseline variants for optimization
        // These will be used as the starting pool for GEPA iterations
        let original_variants = get_uninitialized_variant_configs(self, &function_context)?;

        // Track original variant names to filter them out at the end
        let original_variant_names: std::collections::HashSet<String> =
            original_variants.keys().cloned().collect();

        tracing::info!(
            "Initialized with {} baseline variants: {:?}",
            original_variant_names.len(),
            original_variant_names
        );

        // Create validation dataset for Pareto filtering
        let run_id = Uuid::now_v7();
        let val_dataset_name = format!("{}_gepa_val_{}", self.evaluation_name, run_id);

        // Track all temporary datasets for cleanup at the end
        let mut temporary_datasets = vec![val_dataset_name.clone()];

        tracing::info!(
            "Creating validation dataset '{}' with {} examples",
            val_dataset_name,
            val_examples.len()
        );

        let val_create_datapoints_response = create_evaluation_dataset(
            &config,
            client,
            clickhouse_connection_info,
            val_examples,
            &val_dataset_name,
        )
        .await?;

        tracing::info!("Validation dataset created successfully");

        // Evaluate initial variants on validation set
        let num_variants = original_variants.len();
        tracing::info!(
            "Evaluating {} initial variants on validation dataset",
            num_variants
        );

        // Divide concurrency among variants to avoid max_concurrency² explosion
        // Since each variant is evaluated on the same dataset, they'll take similar time
        // TODO[#4914] enable semaphore sharing with `run_evaluation_core_streaming`
        let per_variant_concurrency = (self.max_concurrency as usize / num_variants).max(1);

        let evaluation_name = self.evaluation_name.clone();
        let evaluator_configs = match &*function_context.evaluation_config {
            EvaluationConfig::Inference(cfg) => &cfg.evaluators,
        };

        // Evaluate all initial variants in parallel. Errors are returned from each future
        // and collected by join_all(), allowing us to proceed with any variants that succeed
        // rather than failing fast. The safety check below (lines 225-229) ensures at least
        // one variant succeeded before proceeding with optimization.
        let evaluation_futures: Vec<_> = original_variants
            .iter()
            .map(|(variant_name, variant_config)| {
                let gateway_client = gateway_client.clone();
                let clickhouse_connection_info = clickhouse_connection_info.clone();
                let functions = config.functions.clone();
                let evaluation_config_param = Arc::clone(&function_context.evaluation_config);
                let evaluation_name = evaluation_name.clone();
                let variant_name = variant_name.clone();
                let variant_config = variant_config.clone();
                let val_dataset_name = val_dataset_name.clone();

                async move {
                    match evaluate_variant(EvaluateVariantParams {
                        gateway_client,
                        clickhouse_connection_info,
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
                            tracing::warn!(
                                "Evaluation failed for variant '{}': {}",
                                variant_name,
                                e
                            );
                            Err(e)
                        }
                    }
                }
            })
            .collect();

        // Execute in parallel
        let val_evaluation_results = join_all(evaluation_futures).await;

        // Collect into validation maps for Pareto filtering and parent-child comparisons
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

        // Initialize Pareto frontier with validation datapoint IDs and evaluator configs
        let mut pareto_frontier = ParetoFrontier::new(
            val_create_datapoints_response.ids,
            evaluator_configs,
            self.seed.map(|s| s as u64),
        );

        // Seed the frontier with initial variants and their validation scores
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

        // Initialize RNG for sampling minibatches
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed as u64),
            None => {
                let mut thread_rng = rand::rng();
                StdRng::from_rng(&mut thread_rng)
            }
        };

        for iteration in 0..(self.max_iterations as usize) {
            let parent = match pareto_frontier.sample_by_frequency() {
                Ok(variant) => variant,
                Err(err) => {
                    tracing::warn!(
                        "Skipping iteration {} because no candidates were available: {}",
                        iteration,
                        err
                    );
                    continue;
                }
            };

            tracing::info!(
                "GEPA iteration {}: selected parent variant '{}'",
                iteration,
                parent.name
            );

            let batch_size = self.batch_size.min(train_examples.len());
            let mutation_examples: Vec<RenderedSample> = train_examples
                .iter()
                .choose_multiple(&mut rng, batch_size)
                .into_iter()
                .cloned()
                .collect();

            let mutation_dataset_name = format!(
                "{}_gepa_mutation_{}_{}",
                self.evaluation_name, iteration, run_id,
            );
            temporary_datasets.push(mutation_dataset_name.clone());

            tracing::debug!(
                "GEPA iteration {}: creating minibatch dataset with {} examples",
                iteration,
                mutation_examples.len()
            );

            let _mutation_create_datapoints_response = create_evaluation_dataset(
                &config,
                client,
                clickhouse_connection_info,
                mutation_examples,
                &mutation_dataset_name,
            )
            .await?;

            tracing::info!(
                "GEPA iteration {}: evaluating parent variant on minibatch",
                iteration
            );

            // Evaluate parent on minibatch. Unlike initial evaluation, this is a sequential
            // operation within the iteration loop. If it fails, we use 'continue' to skip
            // to the next iteration rather than stopping the entire optimization.
            let parent_evaluation_results = match evaluate_variant(EvaluateVariantParams {
                gateway_client: gateway_client.clone(),
                clickhouse_connection_info: clickhouse_connection_info.clone(),
                functions: config.functions.clone(),
                evaluation_config: Arc::clone(&function_context.evaluation_config),
                evaluation_name: self.evaluation_name.clone(),
                variant_name: parent.name.clone(),
                variant_config: parent.config.clone(),
                dataset_name: mutation_dataset_name.clone(),
                concurrency: self.max_concurrency as usize,
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
                    continue;
                }
            };

            tracing::debug!(
                "GEPA iteration {}: parent '{}' minibatch evaluation stats: {:#?}",
                iteration,
                parent.name,
                parent_evaluation_results.evaluation_stats
            );

            tracing::info!(
                "GEPA iteration {}: analyzing {} parent inferences",
                iteration,
                parent_evaluation_results.evaluation_infos.len()
            );

            let parent_analyses = match analyze_inferences(
                &gateway_client,
                &parent_evaluation_results.evaluation_infos,
                &function_context,
                &parent.config,
                self,
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
                    continue;
                }
            };

            tracing::info!(
                "GEPA iteration {}: completed {} analyses for parent '{}', generating child variant",
                iteration,
                parent_analyses.len(),
                parent.name
            );

            // Generate improved child variant using the mutate function
            let child = match mutate_variant(
                &gateway_client,
                &parent_analyses,
                &function_context,
                &parent,
                self,
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
                        parent.name,
                        err
                    );
                    continue;
                }
            };

            tracing::info!(
                "GEPA iteration {}: evaluating child variant '{}' on minibatch",
                iteration,
                child.name
            );

            // Evaluate child on minibatch. Use 'continue' to skip to next iteration if
            // evaluation fails, consistent with parent evaluation error handling.
            let child_evaluation_results = match evaluate_variant(EvaluateVariantParams {
                gateway_client: gateway_client.clone(),
                clickhouse_connection_info: clickhouse_connection_info.clone(),
                functions: config.functions.clone(),
                evaluation_config: Arc::clone(&function_context.evaluation_config),
                evaluation_name: self.evaluation_name.clone(),
                variant_name: child.name.clone(),
                variant_config: child.config.clone(),
                dataset_name: mutation_dataset_name,
                concurrency: self.max_concurrency as usize,
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
                    continue;
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

            // Check if the child Pareto-dominates the parent on the minibatch
            let child_improves = is_improvement(
                &parent_evaluation_results.evaluation_stats,
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
                    gateway_client: gateway_client.clone(),
                    clickhouse_connection_info: clickhouse_connection_info.clone(),
                    functions: config.functions.clone(),
                    evaluation_config: Arc::clone(&function_context.evaluation_config),
                    evaluation_name: self.evaluation_name.clone(),
                    variant_name: child.name.clone(),
                    variant_config: child.config.clone(),
                    dataset_name: val_dataset_name.clone(),
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
                        let child_val_scores =
                            val_mutation_evaluation_results.per_datapoint_scores();
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
                            val_mutation_evaluation_results.evaluation_infos.len()
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
        }

        // Filter out original variants to return only newly created variants
        let mut new_variants = pareto_frontier.variant_configs().clone();
        new_variants.retain(|name, _| !original_variant_names.contains(name));

        tracing::info!(
            "GEPA optimization complete: created {} new variant(s)",
            new_variants.len()
        );
        tracing::debug!("New variants: {:#?}", new_variants);

        // For synchronous optimizers, we store the result in the handle
        // rather than returning an error from launch()
        let handle = GEPAJobHandle {
            result: Ok(new_variants),
        };

        // Clean up temporary datasets
        for dataset_name in &temporary_datasets {
            if let Err(err) = delete_dataset(clickhouse_connection_info, dataset_name).await {
                tracing::warn!(
                    "Failed to delete temporary GEPA dataset '{}': {}",
                    dataset_name,
                    err
                );
            }
        }

        Ok(handle)
    }
}

impl JobHandle for GEPAJobHandle {
    async fn poll(
        &self,
        _client: &TensorzeroHttpClient,
        _credentials: &InferenceCredentials,
        _default_credentials: &ProviderTypeDefaultCredentials,
        _provider_types: &ProviderTypesConfig,
    ) -> Result<OptimizationJobInfo, Error> {
        // GEPA optimization is synchronous, so the result is available immediately
        // Check if optimization succeeded or failed
        match &self.result {
            Ok(variant_configs) => {
                // Return the Pareto frontier of variant configurations
                Ok(OptimizationJobInfo::Completed {
                    output: OptimizerOutput::Variants(
                        variant_configs
                            .iter()
                            .map(|(k, v)| {
                                (
                                    k.clone(),
                                    Box::new(UninitializedVariantConfig::ChatCompletion(v.clone())),
                                )
                            })
                            .collect(),
                    ),
                })
            }
            Err(error_message) => {
                // Return failure status with the error message
                Ok(OptimizationJobInfo::Failed {
                    message: error_message.clone(),
                    error: None,
                })
            }
        }
    }
}
