//! GEPA (Genetic Evolution with Pareto Analysis) optimizer implementation
//!
//! This module contains the core GEPA algorithm logic, separated from the trait
//! implementations in mod.rs for cleaner organization.

use std::collections::HashMap;

use tensorzero_core::{
    client::{Client, ClientBuilder, ClientBuilderMode},
    config::Config,
    db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo},
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    http::TensorzeroHttpClient,
    optimization::gepa::GEPAConfig,
    stored_inference::RenderedSample,
    variant::chat_completion::UninitializedChatCompletionConfig,
};

// Re-export public types and functions from sibling modules
#[expect(unused_imports)]
pub use super::analyze::{analyze_inferences, InferenceWithAnalysis};
#[expect(unused_imports)]
pub use super::evaluate::{create_evaluation_dataset, evaluate_variants, EvaluationResults};
#[expect(unused_imports)]
pub use super::mutate::{create_mutated_variant, mutate_templates, MutateOutput};
#[expect(unused_imports)]
pub use super::pareto::{is_improvement, update_pareto_frontier};
#[expect(unused_imports)]
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
    _credentials: &InferenceCredentials,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    tensorzero_config: std::sync::Arc<Config>,
) -> Result<HashMap<String, UninitializedChatCompletionConfig>, Error> {
    // Validate configuration and examples, get the function config
    let function_config = validate_gepa_config(config, &tensorzero_config)?;

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

    // Build the gateway client ONCE for the entire optimization run
    // This avoids creating ~201 gateway instances (each with background tasks)
    let _gateway_client = ClientBuilder::new(ClientBuilderMode::FromComponents {
        config: tensorzero_config.clone(),
        clickhouse_connection_info: clickhouse_connection_info.clone(),
        postgres_connection_info: PostgresConnectionInfo::Disabled,
        http_client: client.clone(),
        timeout: None,
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
    let pareto_frontier = initialize_pareto_frontier(config, function_config)?;

    // Create validation dataset for Pareto filtering
    let val_dataset_name = format!(
        "{}_gepa_val_{}",
        config.evaluation_name,
        uuid::Uuid::now_v7()
    );
    tracing::info!(
        "Creating validation dataset '{}' with {} examples",
        val_dataset_name,
        val_examples.len()
    );

    let _val_datapoint_ids = create_evaluation_dataset(
        &tensorzero_config,
        client,
        clickhouse_connection_info,
        &val_examples,
        &val_dataset_name,
    )
    .await?;

    tracing::info!("Validation dataset created successfully");

    // TODO: Evaluate initial variants on validation set and get val_scores
    // let val_scores = evaluate_variants(..., &val_examples, ...)?;

    // TODO: Initial Pareto frontier filtering
    // let (mut pareto_frontier, mut frequencies) = update_pareto_frontier(
    //     pareto_frontier,
    //     &val_scores,
    //     config,
    //     tensorzero_config,
    // )?;

    // Main GEPA loop
    for iteration in 0..config.max_iterations {
        tracing::info!("GEPA iteration {}/{}", iteration + 1, config.max_iterations);

        // TODO: 1. Sample mini-batch from train_examples
        // TODO: 2. Sample variant to mutate (proportional to frequency)
        // TODO: 3. Generate mutation using mutate() function (which does analysis internally)
        // TODO: 4. Evaluate mutation on batch
        // TODO: 5. If improvement, evaluate on validation set and add to val_scores
        // TODO: 6. Update Pareto frontier using instance-wise dominance on val_scores

        tracing::info!(
            "GEPA iteration {} complete. Pareto frontier size: {}",
            iteration + 1,
            pareto_frontier.len()
        );
    }

    // Return the final Pareto frontier
    tracing::info!(
        "GEPA optimization complete. Final Pareto frontier size: {}",
        pareto_frontier.len()
    );

    Ok(pareto_frontier)
}

/// Initializes the Pareto frontier with baseline variants or provided initial variants
fn initialize_pareto_frontier(
    config: &GEPAConfig,
    function_config: &FunctionConfig,
) -> Result<HashMap<String, UninitializedChatCompletionConfig>, Error> {
    let variants = function_config.variants();
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

            if let Some(chat_config) =
                utils::extract_chat_completion_from_variant_info(variant_info, variant_name)
            {
                frontier.insert(variant_name.clone(), chat_config);
                tracing::info!("Using initial variant: {}", variant_name);
            }
        }
    } else {
        // Use all ChatCompletion variants from the function
        for (variant_name, variant_info) in variants {
            if let Some(chat_config) =
                utils::extract_chat_completion_from_variant_info(variant_info, variant_name)
            {
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

/// Generate mutation of a variant using GEPA analysis and mutation
/// Returns None if mutation generation fails
///
/// Process:
/// 1. Run inference on batch_examples using candidate variant (with internal_dynamic_variant_config)
/// 2. Analyze each inference using tensorzero::optimization::gepa::analyze built-in function
///    (with internal_dynamic_variant_config for the analysis model)
/// 3. Generate mutation using tensorzero::optimization::gepa::mutate built-in function
///    (with internal_dynamic_variant_config for the mutation model)
/// 4. Return new variant config with updated templates
#[expect(dead_code)]
#[expect(clippy::too_many_arguments)]
async fn mutate(
    _gateway_client: &Client,
    _credentials: &InferenceCredentials,
    _function_name: &str,
    _variant_config: &UninitializedChatCompletionConfig,
    _batch_examples: &[RenderedSample],
    _analysis_model: &str,
    _mutation_model: &str,
    _job_id: &uuid::Uuid,
    _max_concurrency: u32,
) -> Option<UninitializedChatCompletionConfig> {
    // TODO: Implement mutation generation
    // 1. Run inference on batch_examples using variant_config
    // 2. For each inference result, call tensorzero::optimization::gepa::analyze
    // 3. Aggregate analysis results
    // 4. Call tensorzero::optimization::gepa::mutate with analysis and variant config
    // 5. Parse mutation response and create new UninitializedChatCompletionConfig
    // 6. Return new config (or None if mutation failed)

    None
}

/// Convert candidate variants to output format for the job handle
/// This is the final step before returning the Pareto frontier
#[expect(dead_code)]
fn convert_variants_to_output(
    variants: &HashMap<String, UninitializedChatCompletionConfig>,
) -> HashMap<String, UninitializedChatCompletionConfig> {
    // TODO: Implement conversion if needed
    // For now, the variants are already in the correct format (UninitializedChatCompletionConfig)
    // This function exists as a placeholder in case we need to do any transformations
    // or validation before returning the final results

    variants.clone()
}
