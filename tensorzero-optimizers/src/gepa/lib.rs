//! GEPA (Genetic Evolution with Pareto Analysis) optimizer implementation
//!
//! This module contains the core GEPA algorithm logic, separated from the trait
//! implementations in mod.rs for cleaner organization.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures::future::join_all;

use tensorzero_core::{
    client::{Client, ClientBuilder, ClientBuilderMode},
    config::Config,
    db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo},
    error::{Error, ErrorDetails},
    evaluations::EvaluationConfig,
    http::TensorzeroHttpClient,
    optimization::gepa::GEPAConfig,
    stored_inference::RenderedSample,
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use super::validate::FunctionConfigAndTools;

// Re-export public types and functions from sibling modules
#[expect(unused_imports)]
pub use super::analyze::{analyze_inferences, InferenceWithAnalysis};
#[expect(unused_imports)]
pub use super::evaluate::{
    cleanup_temporary_dataset, create_evaluation_dataset, evaluate_variant, EvaluateVariantParams,
    EvaluationResults, ValidationScoresMap,
};
#[expect(unused_imports)]
pub use super::mutate::{create_mutated_variant, mutate_templates, MutateOutput};
pub use super::pareto::{is_improvement, update_pareto_frontier};
pub use super::sample::{random_sample, sample_by_frequency};

// Import utils module for config extraction
use super::utils;

// Import validation functions
use super::validate::{validate_examples, validate_gepa_config};

/// Main GEPA optimization orchestration function
///
/// This function implements the GEPA algorithm:
/// 1. Initialize with baseline variants or provided initial variants
/// 2. For each iteration:
///    a. Analyze examples using the current variants
///    b. Identify Pareto-optimal variants based on multi-objective metrics
///    c. Generate new variants by mutating the Pareto frontier
///    d. Update the Pareto frontier with new variants
/// 3. Return the final Pareto frontier as variant configurations
pub async fn run_gepa_optimization(
    config: &GEPAConfig,
    client: &TensorzeroHttpClient,
    train_examples: Vec<RenderedSample>,
    val_examples: Option<Vec<RenderedSample>>,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    tensorzero_config: std::sync::Arc<Config>,
) -> Result<HashMap<String, UninitializedChatCompletionConfig>, Error> {
    // Validate configuration and examples, get the function config and tools
    let config_and_tools = validate_gepa_config(config, &tensorzero_config)?;

    // Require validation examples for GEPA
    let val_examples = val_examples.ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: "val_examples are required for GEPA optimization (used for Pareto frontier filtering)".to_string(),
        })
    })?;

    // Validate both train and validation examples (this filters invalid examples)
    let train_examples = validate_examples(&train_examples)?;
    let val_examples = validate_examples(&val_examples)?;

    tracing::info!(
        "Starting GEPA optimization for function '{}' with {} train examples and {} val examples",
        config.function_name,
        train_examples.len(),
        val_examples.len()
    );

    // Build the gateway client once for the entire optimization run
    let gateway_client = ClientBuilder::new(ClientBuilderMode::FromComponents {
        config: tensorzero_config.clone(),
        clickhouse_connection_info: clickhouse_connection_info.clone(),
        postgres_connection_info: PostgresConnectionInfo::Disabled,
        http_client: client.clone(),
        timeout: Some(Duration::from_secs(config.timeout)),
    })
    .build()
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to build gateway client for GEPA optimization: {e}"),
        })
    })?;

    tracing::info!("Gateway client built successfully for GEPA optimization");

    // Initialize the Pareto frontier with baseline or provided variants
    let pareto_frontier = initialize_pareto_frontier(config, &config_and_tools)?;

    // Track original variant names to filter them out at the end
    let original_variant_names: std::collections::HashSet<String> =
        pareto_frontier.keys().cloned().collect();

    tracing::info!(
        "Initialized with {} baseline variants: {:?}",
        original_variant_names.len(),
        original_variant_names
    );

    // Create validation dataset for Pareto filtering
    let val_dataset_name = format!(
        "{}_gepa_val_{}",
        config.evaluation_name,
        uuid::Uuid::now_v7()
    );

    // Track all temporary datasets for cleanup at the end
    let mut temporary_datasets = vec![val_dataset_name.clone()];

    tracing::info!(
        "Creating validation dataset '{}' with {} examples",
        val_dataset_name,
        val_examples.len()
    );

    create_evaluation_dataset(
        &tensorzero_config,
        client,
        clickhouse_connection_info,
        &val_examples,
        &val_dataset_name,
    )
    .await?;

    tracing::info!("Validation dataset created successfully");

    // Evaluate initial variants on validation set
    tracing::info!(
        "Evaluating {} initial variants on validation dataset",
        pareto_frontier.len()
    );

    // Extract evaluation config once for all variants
    let evaluation_config = tensorzero_config
        .evaluations
        .get(&config.evaluation_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!("Evaluation '{}' not found", config.evaluation_name),
            })
        })?
        .clone();

    // Divide concurrency among variants to avoid max_concurrency² explosion
    // Since each variant is evaluated on the same dataset, they'll take similar time
    let num_variants = pareto_frontier.len();
    let per_variant_concurrency = (config.max_concurrency as usize / num_variants).max(1);

    tracing::debug!(
        "Evaluating {} variants with {} concurrency each (total ≈ {})",
        num_variants,
        per_variant_concurrency,
        num_variants * per_variant_concurrency
    );

    let evaluation_name = config.evaluation_name.clone();

    // Create parallel evaluation futures
    let evaluation_futures: Vec<_> = pareto_frontier
        .iter()
        .map(|(variant_name, variant_config)| {
            let gateway_client = gateway_client.clone();
            let clickhouse_connection_info = clickhouse_connection_info.clone();
            let tensorzero_config = Arc::clone(&tensorzero_config);
            let evaluation_config = Arc::clone(&evaluation_config);
            let evaluation_name = evaluation_name.clone();
            let variant_name = variant_name.clone();
            let variant_config = variant_config.clone();
            let val_dataset_name = val_dataset_name.clone();

            async move {
                match evaluate_variant(EvaluateVariantParams {
                    gateway_client,
                    clickhouse_connection_info,
                    tensorzero_config,
                    evaluation_config,
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
                        tracing::warn!("Evaluation failed for variant '{}': {}", variant_name, e);
                        Ok((variant_name, HashMap::new()))
                    }
                }
            }
        })
        .collect();

    // Execute in parallel
    let results = join_all(evaluation_futures).await;

    // Collect into val_scores_map directly (only thing needed for Pareto filtering)
    let mut val_scores_map: ValidationScoresMap = HashMap::new();
    for result in results {
        match result {
            Ok((variant_name, scores)) => {
                if !scores.is_empty() {
                    val_scores_map.insert(variant_name, scores);
                }
            }
            Err(e) => {
                tracing::error!("Unexpected error in evaluation: {}", e);
            }
        }
    }

    tracing::info!("Initial evaluation complete");

    // Filter initial Pareto frontier using instance-wise dominance
    let mut frontier =
        update_pareto_frontier(pareto_frontier, &val_scores_map, config, &tensorzero_config)?;

    // Update scores map
    val_scores_map.retain(|variant_name, _| frontier.variants.contains_key(variant_name));

    tracing::info!(
        "Initial Pareto frontier filtered to {} variants",
        frontier.variants.len()
    );

    // Main GEPA loop
    for iteration in 0..config.max_iterations {
        tracing::info!("GEPA iteration {}/{}", iteration + 1, config.max_iterations);

        // Step 1: Sample mini-batch from train_examples
        let batch_examples = random_sample(&train_examples, config.batch_size);

        tracing::info!(
            "Sampled mini-batch of {} examples from {} training examples",
            batch_examples.len(),
            train_examples.len()
        );

        // Create batch dataset for this iteration
        let batch_dataset_name = format!(
            "{}_gepa_batch_{}_{}",
            config.evaluation_name,
            iteration,
            uuid::Uuid::now_v7()
        );

        // Track this dataset for cleanup
        temporary_datasets.push(batch_dataset_name.clone());

        if let Err(e) = create_evaluation_dataset(
            &tensorzero_config,
            client,
            clickhouse_connection_info,
            &batch_examples,
            &batch_dataset_name,
        )
        .await
        {
            tracing::warn!(
                "Failed to create batch dataset for iteration {}/{}: {}. Skipping iteration.",
                iteration + 1,
                config.max_iterations,
                e
            );
            continue;
        }

        tracing::debug!(
            "Created batch dataset '{}' with {} examples",
            batch_dataset_name,
            batch_examples.len()
        );

        // Step 2: Sample variant to mutate (proportional to frequency)
        let parent_variant_name = sample_by_frequency(&frontier.frequencies)?;

        let parent_variant_config =
            frontier.variants.get(&parent_variant_name).ok_or_else(|| {
                Error::new(ErrorDetails::InternalError {
                    message: format!(
                        "Sampled variant '{parent_variant_name}' not found in Pareto frontier"
                    ),
                })
            })?;

        tracing::info!(
            "Sampled variant '{}' for mutation (frequency: {}/{})",
            parent_variant_name,
            frontier.frequencies.get(&parent_variant_name).unwrap_or(&0),
            val_examples.len()
        );

        // Steps 3-6: Process iteration (evaluate, analyze, mutate, check improvement)
        let mutation_result = match gepa_step(GEPAStepParams {
            gateway_client: gateway_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            tensorzero_config: Arc::clone(&tensorzero_config),
            evaluation_config: Arc::clone(&evaluation_config),
            gepa_config: config,
            config_and_tools: &config_and_tools,
            parent_variant_name: &parent_variant_name,
            parent_variant_config,
            batch_dataset_name: &batch_dataset_name,
            iteration,
        })
        .await
        {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!(
                    "GEPA step failed for iteration {}/{}: {}. Skipping iteration.",
                    iteration + 1,
                    config.max_iterations,
                    e
                );
                continue;
            }
        };

        // If mutation improved over parent, evaluate on validation and update Pareto frontier
        if let Some((child_variant_name, child_variant_config)) = mutation_result {
            tracing::info!(
                "Mutation '{}' improved over parent, evaluating on validation set",
                child_variant_name
            );

            // Step 7a: Evaluate child on validation set
            let child_val_results = match evaluate_variant(EvaluateVariantParams {
                gateway_client: gateway_client.clone(),
                clickhouse_connection_info: clickhouse_connection_info.clone(),
                tensorzero_config: Arc::clone(&tensorzero_config),
                evaluation_config: Arc::clone(&evaluation_config),
                evaluation_name: config.evaluation_name.clone(),
                variant_name: child_variant_name.clone(),
                variant_config: child_variant_config.clone(),
                dataset_name: val_dataset_name.clone(),
                concurrency: config.max_concurrency as usize,
            })
            .await
            {
                Ok(result) => result,
                Err(e) => {
                    tracing::warn!(
                        "Failed to evaluate child variant '{}' on validation set for iteration {}/{}: {}. Skipping iteration.",
                        child_variant_name,
                        iteration + 1,
                        config.max_iterations,
                        e
                    );
                    continue;
                }
            };

            tracing::info!(
                "Child validation evaluation complete: {} datapoints",
                child_val_results.evaluation_infos.len()
            );

            // Step 7b: Add child scores to val_scores_map
            let child_scores = child_val_results.per_datapoint_scores();
            val_scores_map.insert(child_variant_name.clone(), child_scores);

            // Step 7c: Update Pareto frontier with new child variant
            // First add child to frontier HashMap
            frontier
                .variants
                .insert(child_variant_name.clone(), child_variant_config);

            // Re-filter Pareto frontier with updated val_scores_map
            // Note: We use `?` here instead of continue because we've already mutated state
            // (added child to val_scores_map and frontier.variants). If Pareto update fails,
            // we're in an inconsistent state and should fail rather than continue.
            frontier = update_pareto_frontier(
                frontier.variants,
                &val_scores_map,
                config,
                &tensorzero_config,
            )?;

            // Keep val_scores_map in sync with filtered frontier
            val_scores_map.retain(|variant_name, _| frontier.variants.contains_key(variant_name));

            tracing::info!(
                "Pareto frontier updated: {} variants (child {} retained: {})",
                frontier.variants.len(),
                child_variant_name,
                frontier.variants.contains_key(&child_variant_name)
            );
        } else {
            tracing::debug!("Mutation did not improve over parent, skipping validation");
        }

        tracing::info!(
            "GEPA iteration {} complete. Pareto frontier size: {}",
            iteration + 1,
            frontier.variants.len()
        );
    }

    // Filter out original baseline variants from final Pareto frontier
    tracing::info!(
        "GEPA optimization complete. Filtering out {} original baseline variants",
        original_variant_names.len()
    );

    frontier
        .variants
        .retain(|name, _| !original_variant_names.contains(name));

    // Sync val_scores_map after filtering
    val_scores_map.retain(|name, _| frontier.variants.contains_key(name));

    // Return error if no GEPA-generated variants survived
    if frontier.variants.is_empty() {
        // Clean up temporary datasets before returning error
        tracing::info!(
            "Cleaning up {} temporary datasets before returning error",
            temporary_datasets.len()
        );

        for dataset_name in temporary_datasets {
            cleanup_temporary_dataset(clickhouse_connection_info, &dataset_name).await;
        }

        return Err(Error::new(ErrorDetails::InternalError {
            message: format!(
                "GEPA optimization failed to produce any variants that survived Pareto filtering. \
                All {} generated variants were dominated by or equal to baseline variants.",
                config.max_iterations
            ),
        }));
    }

    tracing::info!(
        "Final Pareto frontier contains {} GEPA-evolved variants (originals filtered out)",
        frontier.variants.len()
    );

    // Clean up temporary datasets
    tracing::info!(
        "Cleaning up {} temporary datasets created during GEPA optimization",
        temporary_datasets.len()
    );

    for dataset_name in temporary_datasets {
        cleanup_temporary_dataset(clickhouse_connection_info, &dataset_name).await;
    }

    Ok(frontier.variants)
}

/// Parameters for a single GEPA iteration step
pub(crate) struct GEPAStepParams<'a> {
    pub gateway_client: Client,
    pub clickhouse_connection_info: ClickHouseConnectionInfo,
    pub tensorzero_config: Arc<Config>,
    pub evaluation_config: Arc<EvaluationConfig>,
    pub gepa_config: &'a GEPAConfig,
    pub config_and_tools: &'a FunctionConfigAndTools,
    pub parent_variant_name: &'a str,
    pub parent_variant_config: &'a UninitializedChatCompletionConfig,
    pub batch_dataset_name: &'a str,
    pub iteration: u32,
}

/// Process a single GEPA iteration: evaluate parent, analyze, mutate, check improvement
///
/// Returns Some((variant_name, variant_config)) if mutation improved over parent,
/// None otherwise (mutation failed, evaluation failed, or no improvement)
pub(crate) async fn gepa_step(
    params: GEPAStepParams<'_>,
) -> Result<Option<(String, UninitializedChatCompletionConfig)>, Error> {
    // Step 3: Evaluate parent variant on mini-batch
    tracing::debug!("Evaluating parent variant on batch dataset");

    let parent_batch_results = evaluate_variant(EvaluateVariantParams {
        gateway_client: params.gateway_client.clone(),
        clickhouse_connection_info: params.clickhouse_connection_info.clone(),
        tensorzero_config: Arc::clone(&params.tensorzero_config),
        evaluation_config: Arc::clone(&params.evaluation_config),
        evaluation_name: params.gepa_config.evaluation_name.clone(),
        variant_name: params.parent_variant_name.to_string(),
        variant_config: params.parent_variant_config.clone(),
        dataset_name: params.batch_dataset_name.to_string(),
        concurrency: params.gepa_config.max_concurrency as usize,
    })
    .await?;

    tracing::debug!(
        "Parent evaluation complete: {} datapoints evaluated",
        parent_batch_results.evaluation_infos.len()
    );

    // Step 4: Analyze parent inferences to generate improvement feedback
    tracing::debug!("Analyzing parent inferences");

    let analyses = analyze_inferences(
        &params.gateway_client,
        &parent_batch_results.evaluation_infos,
        params.config_and_tools,
        params.parent_variant_config,
        params.gepa_config,
    )
    .await?;

    tracing::info!(
        "Analysis complete: {}/{} inferences analyzed successfully",
        analyses.len(),
        parent_batch_results.evaluation_infos.len()
    );

    // Step 5: Generate mutation from analyses using mutate_templates
    tracing::debug!("Generating mutation from analyses");

    let mutate_result = mutate_templates(
        &params.gateway_client,
        &analyses,
        params.config_and_tools,
        params.parent_variant_config,
        params.gepa_config,
    )
    .await;

    let mutation_output = match mutate_result {
        Ok(output) => output,
        Err(e) => {
            tracing::warn!("Mutation generation failed: {}", e);
            return Ok(None);
        }
    };

    // Step 6: Create mutated variant config
    let variant_prefix = params
        .gepa_config
        .variant_prefix
        .as_deref()
        .unwrap_or("gepa");
    let (child_variant_name, child_variant_config) = create_mutated_variant(
        params.parent_variant_config,
        mutation_output,
        params.iteration as usize,
        variant_prefix,
        params.parent_variant_name,
        params.gepa_config.retries,
    );

    tracing::info!("Created mutation variant: {}", child_variant_name);

    // Step 7: Evaluate mutation on batch
    tracing::debug!("Evaluating mutation on batch dataset");

    let child_batch_results = evaluate_variant(EvaluateVariantParams {
        gateway_client: params.gateway_client.clone(),
        clickhouse_connection_info: params.clickhouse_connection_info,
        tensorzero_config: params.tensorzero_config,
        evaluation_config: params.evaluation_config.clone(),
        evaluation_name: params.gepa_config.evaluation_name.clone(),
        variant_name: child_variant_name.clone(),
        variant_config: child_variant_config.clone(),
        dataset_name: params.batch_dataset_name.to_string(),
        concurrency: params.gepa_config.max_concurrency as usize,
    })
    .await?;

    tracing::debug!(
        "Child evaluation complete: {} datapoints evaluated",
        child_batch_results.evaluation_infos.len()
    );

    if is_improvement(
        &parent_batch_results.evaluation_stats,
        &child_batch_results.evaluation_stats,
        &params.evaluation_config,
    ) {
        tracing::info!("Mutation shows Pareto improvement over parent on batch");
        Ok(Some((child_variant_name, child_variant_config)))
    } else {
        tracing::debug!("Mutation did not improve over parent on batch");
        Ok(None)
    }
}

/// Initializes the Pareto frontier with baseline variants or provided initial variants
fn initialize_pareto_frontier(
    config: &GEPAConfig,
    config_and_tools: &FunctionConfigAndTools,
) -> Result<HashMap<String, UninitializedChatCompletionConfig>, Error> {
    let variants = config_and_tools.function_config.variants();
    let mut frontier = HashMap::new();

    if let Some(initial_variant_names) = &config.initial_variants {
        // Use only the specified initial variants
        for variant_name in initial_variant_names {
            let variant_info = variants.get(variant_name).ok_or_else(|| {
                Error::new(ErrorDetails::Config {
                    message: format!(
                        "variant '{}' not found in function '{}'",
                        variant_name, config.function_name
                    ),
                })
            })?;

            if let Some(chat_config) = utils::extract_chat_completion_from_variant_info(
                variant_info,
                variant_name,
                config.retries,
            ) {
                frontier.insert(variant_name.clone(), chat_config);
                tracing::info!("Using initial variant: {}", variant_name);
            }
        }
    } else {
        // Use all ChatCompletion variants from the function
        for (variant_name, variant_info) in variants {
            if let Some(chat_config) = utils::extract_chat_completion_from_variant_info(
                variant_info,
                variant_name,
                config.retries,
            ) {
                frontier.insert(variant_name.clone(), chat_config);
            }
        }

        tracing::info!(
            "Initialized Pareto frontier with {} ChatCompletion variants from function '{}'",
            frontier.len(),
            config.function_name
        );
    }

    if frontier.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "No ChatCompletion variants found for function '{}'. GEPA requires at least one ChatCompletion variant.",
                config.function_name
            ),
        }));
    }

    Ok(frontier)
}
