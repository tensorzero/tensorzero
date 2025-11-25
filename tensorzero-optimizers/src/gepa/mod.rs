//! GEPA optimizer implementation
//!
//! This module provides the trait implementations for the GEPA optimizer.
//! The actual GEPA algorithm will be implemented here.

use async_trait::async_trait;
use futures::future::join_all;
use rand::{rngs::StdRng, seq::IteratorRandom, Rng, SeedableRng};
use std::{collections::HashMap, sync::Arc, time::Duration};

use tensorzero_core::{
    client::{ClientBuilder, ClientBuilderMode},
    config::{Config, UninitializedVariantConfig},
    db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo},
    endpoints::{datasets::v1::delete_dataset, inference::InferenceCredentials},
    error::{Error, ErrorDetails},
    http::TensorzeroHttpClient,
    model_table::ProviderTypeDefaultCredentials,
    optimization::{
        gepa::{GEPAConfig, GEPAJobHandle},
        OptimizationJobInfo, OptimizerOutput,
    },
    stored_inference::RenderedSample,
};

use crate::{JobHandle, Optimizer};

pub mod analyze;
pub mod evaluate;
pub mod mutate;
pub mod validate;

use analyze::analyze_inferences;
use evaluate::{
    create_evaluation_dataset, evaluate_variant, EvaluateVariantParams, VariantName, VariantScores,
};
pub use mutate::mutate_variant;
use validate::{initialize_pareto_frontier, validate_examples, validate_gepa_config};

#[async_trait]
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

        // Initialize baseline variants for optimization
        // These will be used as the starting pool for GEPA iterations
        let mut initial_variants = initialize_pareto_frontier(self, &function_context)?;

        // Track original variant names to filter them out at the end
        let original_variant_names: std::collections::HashSet<String> =
            initial_variants.keys().cloned().collect();

        tracing::info!(
            "Initialized with {} baseline variants: {:?}",
            original_variant_names.len(),
            original_variant_names
        );

        // Create validation dataset for Pareto filtering
        let val_dataset_name =
            format!("{}_gepa_val_{}", self.evaluation_name, uuid::Uuid::now_v7());

        // Track all temporary datasets for cleanup at the end
        let mut temporary_datasets = vec![val_dataset_name.clone()];

        tracing::info!(
            "Creating validation dataset '{}' with {} examples",
            val_dataset_name,
            val_examples.len()
        );

        create_evaluation_dataset(
            &config,
            client,
            clickhouse_connection_info,
            val_examples,
            &val_dataset_name,
        )
        .await?;

        tracing::info!("Validation dataset created successfully");

        // Evaluate initial variants on validation set
        let num_variants = initial_variants.len();
        tracing::info!(
            "Evaluating {} initial variants on validation dataset",
            num_variants
        );

        // Divide concurrency among variants to avoid max_concurrency² explosion
        // Since each variant is evaluated on the same dataset, they'll take similar time
        let per_variant_concurrency = (self.max_concurrency as usize / num_variants).max(1);

        let evaluation_name = self.evaluation_name.clone();

        // Create parallel evaluation futures
        let evaluation_futures: Vec<_> = initial_variants
            .iter()
            .map(|(variant_name, variant_config)| {
                let gateway_client = gateway_client.clone();
                let clickhouse_connection_info = clickhouse_connection_info.clone();
                let tensorzero_config = Arc::clone(&config);
                let evaluation_config_param = Arc::clone(&function_context.evaluation_config);
                let evaluation_name = evaluation_name.clone();
                let variant_name = variant_name.clone();
                let variant_config = variant_config.clone();
                let val_dataset_name = val_dataset_name.clone();

                async move {
                    match evaluate_variant(EvaluateVariantParams {
                        gateway_client,
                        clickhouse_connection_info,
                        tensorzero_config,
                        evaluation_config: evaluation_config_param,
                        evaluation_name,
                        variant_name: variant_name.clone(),
                        variant_config,
                        dataset_name: val_dataset_name,
                        concurrency: per_variant_concurrency,
                    })
                    .await
                    {
                        Ok(results) => {
                            // Compute scores map inline and drop full EvaluationResults
                            let scores_map = results.per_datapoint_scores();
                            Ok::<_, Error>((variant_name, scores_map))
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Evaluation failed for variant '{}': {}",
                                variant_name,
                                e
                            );
                            Ok((variant_name, HashMap::new()))
                        }
                    }
                }
            })
            .collect();

        // Execute in parallel
        let results = join_all(evaluation_futures).await;

        // Collect into val_scores_map directly (only thing needed for Pareto filtering)
        let mut initial_scores: HashMap<VariantName, VariantScores> = HashMap::new();
        for result in results {
            match result {
                Ok((variant_name, scores)) => {
                    if !scores.is_empty() {
                        initial_scores.insert(variant_name, scores);
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

        // TODO[#4739]: Initialize Pareto frontier with both variant configs and their validation scores

        // Initialize RNG for sampling variants and minibatches
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed as u64),
            None => {
                let mut thread_rng = rand::rng();
                StdRng::from_rng(&mut thread_rng)
            }
        };

        for iteration in 0..self.max_iterations {
            // TODO[#4739]: sample from Pareto frontier
            let Some(parent_name) = initial_variants.keys().choose(&mut rng) else {
                tracing::warn!(
                    "Skipping iteration {} because no candidates were available (should be validated upstream)",
                    iteration
                );
                continue;
            };

            let Some(parent_config) = initial_variants.get(parent_name) else {
                tracing::warn!(
                    "Skipping iteration {} because selected candidate '{}' is missing",
                    iteration,
                    parent_name
                );
                continue;
            };

            tracing::info!(
                "GEPA iteration {}: selected parent variant '{}'",
                iteration,
                parent_name
            );

            let minibatch_size = self.batch_size.min(train_examples.len());
            let minibatch: Vec<RenderedSample> = train_examples
                .iter()
                .choose_multiple(&mut rng, minibatch_size)
                .into_iter()
                .cloned()
                .collect();

            let train_dataset_name = format!(
                "{}_gepa_train_{}_{}",
                self.evaluation_name,
                iteration,
                uuid::Uuid::now_v7()
            );
            temporary_datasets.push(train_dataset_name.clone());

            tracing::debug!(
                "GEPA iteration {}: creating minibatch dataset with {} examples",
                iteration,
                minibatch.len()
            );

            create_evaluation_dataset(
                &config,
                client,
                clickhouse_connection_info,
                minibatch,
                &train_dataset_name,
            )
            .await?;

            tracing::info!(
                "GEPA iteration {}: evaluating parent variant on minibatch",
                iteration
            );

            let parent_results = match evaluate_variant(EvaluateVariantParams {
                gateway_client: gateway_client.clone(),
                clickhouse_connection_info: clickhouse_connection_info.clone(),
                tensorzero_config: Arc::clone(&config),
                evaluation_config: Arc::clone(&function_context.evaluation_config),
                evaluation_name: self.evaluation_name.clone(),
                variant_name: (*parent_name).clone(),
                variant_config: parent_config.clone(),
                dataset_name: train_dataset_name.clone(),
                concurrency: self.max_concurrency as usize,
            })
            .await
            {
                Ok(results) => {
                    tracing::info!(
                        "GEPA iteration {}: analyzing {} parent inferences",
                        iteration,
                        results.evaluation_infos.len()
                    );
                    results
                }
                Err(e) => {
                    tracing::warn!(
                        "GEPA iteration {}: evaluation failed for parent '{}': {}",
                        iteration,
                        parent_name,
                        e
                    );
                    continue;
                }
            };

            let parent_analyses = match analyze_inferences(
                &gateway_client,
                &parent_results.evaluation_infos,
                &function_context,
                parent_config,
                self,
            )
            .await
            {
                Ok(analyses) => analyses,
                Err(err) => {
                    tracing::warn!(
                        "GEPA iteration {}: analysis failed for candidate '{}': {}",
                        iteration,
                        parent_name,
                        err
                    );
                    continue;
                }
            };

            tracing::info!(
                "GEPA iteration {}: completed {} analyses for parent '{}', generating child variant",
                iteration,
                parent_analyses.len(),
                parent_name
            );

            // Wrap parent in HashMap for mutate_variant API (expects single-entry map)
            let mut parent = HashMap::new();
            parent.insert(parent_name, parent_config);

            // Generate improved child variant using the mutate function
            let child = match mutate_variant(
                &gateway_client,
                &parent_analyses,
                &function_context,
                parent,
                self,
                iteration as usize,
            )
            .await
            {
                Ok(child_variants) => {
                    tracing::info!(
                        "GEPA iteration {}: mutation complete, evaluating child variant",
                        iteration
                    );
                    child_variants
                }
                Err(err) => {
                    tracing::warn!(
                        "GEPA iteration {}: mutation failed for parent '{}': {}",
                        iteration,
                        parent_name,
                        err
                    );
                    continue;
                }
            };

            // Extract the child variant name and config for minibatch evaluation
            // Clone to avoid borrowing child, so we can move it later
            let (child_name, child_config) = child
                .iter()
                .next()
                .map(|(k, v)| (k.clone(), v.clone()))
                .ok_or_else(|| {
                    Error::new(ErrorDetails::InternalError {
                        message: "child HashMap should contain exactly one entry".to_string(),
                    })
                })?;

            tracing::info!(
                "GEPA iteration {}: evaluating child variant '{}' on minibatch",
                iteration,
                child_name
            );

            let _child_results = match evaluate_variant(EvaluateVariantParams {
                gateway_client: gateway_client.clone(),
                clickhouse_connection_info: clickhouse_connection_info.clone(),
                tensorzero_config: Arc::clone(&config),
                evaluation_config: Arc::clone(&function_context.evaluation_config),
                evaluation_name: self.evaluation_name.clone(),
                variant_name: child_name.clone(),
                variant_config: child_config.clone(),
                dataset_name: train_dataset_name,
                concurrency: self.max_concurrency as usize,
            })
            .await
            {
                Ok(results) => {
                    tracing::info!(
                        "GEPA iteration {}: child variant '{}' minibatch evaluation complete",
                        iteration,
                        child_name
                    );
                    results
                }
                Err(e) => {
                    tracing::warn!(
                        "GEPA iteration {}: minibatch evaluation failed for child variant '{}': {}",
                        iteration,
                        child_name,
                        e
                    );
                    continue;
                }
            };

            // TODO[#4739]: Check for improvement against candidate variant
            // For now, randomly decide whether to evaluate on validation set
            let is_improvement = rng.random::<f64>() < 0.5;
            if is_improvement {
                tracing::info!(
                    "GEPA iteration {}: evaluating child variant '{}' on validation dataset",
                    iteration,
                    child_name
                );

                match evaluate_variant(EvaluateVariantParams {
                    gateway_client: gateway_client.clone(),
                    clickhouse_connection_info: clickhouse_connection_info.clone(),
                    tensorzero_config: Arc::clone(&config),
                    evaluation_config: Arc::clone(&function_context.evaluation_config),
                    evaluation_name: self.evaluation_name.clone(),
                    variant_name: child_name.clone(),
                    variant_config: child_config.clone(),
                    dataset_name: val_dataset_name.clone(),
                    concurrency: per_variant_concurrency,
                })
                .await
                {
                    Ok(results) => {
                        let child_val_scores = results.per_datapoint_scores();
                        tracing::info!(
                            "GEPA iteration {}: child variant '{}' validation scores collected ({} datapoints)",
                            iteration,
                            child_name,
                            child_val_scores.len()
                        );
                        // TODO: Update Pareto frontier with child variant and validation scores
                    }
                    Err(e) => {
                        tracing::warn!(
                            "GEPA iteration {}: validation evaluation failed for child variant '{}': {}",
                            iteration,
                            child_name,
                            e
                        );
                    }
                }
                // TODO[#4739]: Update Pareto frontier with new variant and scores
                // Add the child variant to the pool for future iterations
                initial_variants.extend(child);
                tracing::info!(
                    "GEPA iteration {}: successfully created child variant '{}' (pool size: {})",
                    iteration,
                    child_name,
                    initial_variants.len()
                );
            }
        }

        // Filter out original variants to return only newly created variants
        initial_variants.retain(|name, _| !original_variant_names.contains(name));

        tracing::info!(
            "GEPA optimization complete: created {} new variant(s)",
            initial_variants.len()
        );

        // For synchronous optimizers, we store the result in the handle
        // rather than returning an error from launch()
        let handle = GEPAJobHandle {
            result: Ok(initial_variants),
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

#[async_trait]
impl JobHandle for GEPAJobHandle {
    async fn poll(
        &self,
        _client: &TensorzeroHttpClient,
        _credentials: &InferenceCredentials,
        _default_credentials: &ProviderTypeDefaultCredentials,
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
