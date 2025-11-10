//! GEPA (Genetic Evolution with Pareto Analysis) optimizer implementation
//!
//! This module contains the core GEPA algorithm logic, separated from the trait
//! implementations in mod.rs for cleaner organization.

use std::collections::{HashMap, HashSet};

use rand::Rng;
use tokio::sync::mpsc;
use uuid::Uuid;

use tensorzero_core::{
    cache::CacheEnabledMode,
    client::{Client, ClientBuilder, ClientBuilderMode, InferenceResponse},
    config::{
        path::ResolvedTomlPath, Config, MetricConfigOptimize, UninitializedVariantConfig,
        UninitializedVariantInfo,
    },
    db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo},
    endpoints::{
        datasets::v1::{
            create_datapoints,
            types::{
                CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsRequest,
                CreateJsonDatapointRequest, JsonDatapointOutputUpdate,
            },
        },
        inference::InferenceCredentials,
    },
    error::{Error, ErrorDetails},
    evaluations::EvaluationConfig,
    function::FunctionConfig,
    http::TensorzeroHttpClient,
    inference::types::Input,
    optimization::gepa::GEPAConfig,
    stored_inference::{RenderedSample, StoredOutput},
    variant::{chat_completion::UninitializedChatCompletionConfig, VariantConfig},
};

use evaluations::{
    stats::EvaluationInfo, EvaluationCoreArgs, EvaluationStats, EvaluationUpdate,
    EvaluationVariant, EvaluatorStats, OutputFormat,
};
use serde::{Deserialize, Serialize};

/// Analysis report from the GEPA analyze function
///
/// The analyze function uses one of three tools to report on inference quality:
/// - Error: Critical failures requiring correction
/// - Improvement: Suboptimal but acceptable outputs that could be better
/// - Optimal: High-quality exemplary outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnalysisReport {
    Error {
        reasoning: String,
        error_identification: String,
        root_cause_analysis: String,
        correct_output: String,
        key_insight: String,
    },
    Improvement {
        reasoning: String,
        suboptimality_description: String,
        better_output: String,
        key_insight: String,
    },
    Optimal {
        reasoning: String,
        key_strengths: Vec<String>,
    },
}

/// Represents an inference output paired with its analysis feedback
#[expect(dead_code)]
#[derive(Debug, Clone)]
pub struct InferenceWithAnalysis {
    pub inference_output: InferenceResponse,
    pub analysis: AnalysisReport,
}

/// Output from the GEPA mutate function
///
/// Contains improved prompt templates generated based on aggregated analysis feedback.
/// Templates are only present if they existed in the original variant configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutateOutput {
    pub system_template: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assistant_template: Option<String>,
}

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

    // Validate both train and validation examples
    validate_examples(&train_examples)?;
    validate_examples(&val_examples)?;

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

/// Validates the GEPA configuration and checks that required resources exist
/// Returns the FunctionConfig for the function being optimized
fn validate_gepa_config<'a>(
    config: &GEPAConfig,
    tensorzero_config: &'a Config,
) -> Result<&'a FunctionConfig, Error> {
    // Check that the function exists in the config
    let function_config = tensorzero_config
        .functions
        .get(&config.function_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "function '{}' not found in configuration",
                    config.function_name
                ),
            })
        })?;

    // Validate function configuration for GEPA
    validate_function_config(&config.function_name, function_config)?;

    // Check that the evaluation exists
    if !tensorzero_config
        .metrics
        .contains_key(&config.evaluation_name)
    {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "evaluation '{}' not found in configuration",
                config.evaluation_name
            ),
        }));
    }

    Ok(function_config)
}

/// Validates that the function configuration is compatible with GEPA
fn validate_function_config(
    function_name: &str,
    function_config: &FunctionConfig,
) -> Result<(), Error> {
    // GEPA currently only supports Chat functions (not JSON functions)
    match function_config {
        FunctionConfig::Chat(_) => Ok(()),
        FunctionConfig::Json(_) => Err(Error::new(ErrorDetails::Config {
            message: format!(
                "function '{function_name}' is a JSON function, but GEPA only supports Chat functions"
            ),
        })),
    }
}

/// Validates that examples are suitable for GEPA optimization
fn validate_examples(examples: &[RenderedSample]) -> Result<(), Error> {
    if examples.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: "Cannot run GEPA optimization with zero examples".to_string(),
        }));
    }

    // TODO: Add more validation:
    // - Check that examples have compatible input structures
    // - Validate that examples have the required fields for analysis

    Ok(())
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

            if let Some(chat_config) = extract_chat_completion_config(variant_info, variant_name)? {
                frontier.insert(variant_name.clone(), chat_config);
                tracing::info!("Using initial variant: {}", variant_name);
            }
        }
    } else {
        // Use all ChatCompletion variants from the function
        for (variant_name, variant_info) in variants {
            if let Some(chat_config) = extract_chat_completion_config(variant_info, variant_name)? {
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

/// Extracts a ChatCompletion config from a VariantInfo, or returns None if it's not a ChatCompletion variant
fn extract_chat_completion_config(
    variant_info: &tensorzero_core::variant::VariantInfo,
    variant_name: &str,
) -> Result<Option<UninitializedChatCompletionConfig>, Error> {
    match &variant_info.inner {
        VariantConfig::ChatCompletion(_chat_config) => {
            // TODO: Convert ChatCompletionConfig to UninitializedChatCompletionConfig
            // This will require extracting the model, system_template, user_template, etc.
            // For now, we'll need to understand the ChatCompletionConfig structure
            tracing::debug!("Found ChatCompletion variant: {}", variant_name);
            Err(Error::new(ErrorDetails::InternalError {
                message: format!(
                    "ChatCompletion config extraction not yet implemented for variant '{variant_name}'"
                ),
            }))
        }
        VariantConfig::BestOfNSampling(_)
        | VariantConfig::Dicl(_)
        | VariantConfig::MixtureOfN(_)
        | VariantConfig::ChainOfThought(_) => {
            tracing::warn!(
                "Skipping non-ChatCompletion variant '{}' (GEPA only supports ChatCompletion variants)",
                variant_name
            );
            Ok(None)
        }
    }
}

/// Updates the Pareto frontier based on instance-wise Pareto dominance
///
/// Filters candidates to only include Pareto-optimal variants based on validation scores.
/// Returns the filtered variants and their frequencies (count of instances dominated).
///
/// # Arguments
/// * `candidates` - Candidate variants to filter
/// * `val_scores` - Evaluation results on validation set for each candidate
/// * `config` - GEPA configuration (for evaluation_name)
/// * `tensorzero_config` - Full TensorZero config (for metric definitions)
///
/// # Returns
/// * Tuple of (filtered_variants, frequencies) where frequencies is used for sampling
#[expect(dead_code)]
/// Filter candidates using GEPA's instance-wise Pareto frontier algorithm
///
/// Implements the SELECTCANDIDATE algorithm from the GEPA paper:
/// 1. For each datapoint, find instance-wise Pareto-optimal variants (Step 1)
/// 2. Build candidate set C as union of all instance-wise Pareto sets (Step 2)
/// 3. Filter candidates globally to remove dominated variants (Step 3)
/// 4. Compute frequency of each variant's membership in instance-wise Pareto sets
///
/// Returns (filtered variants, frequency map) where frequencies are used for weighted sampling.
#[expect(clippy::type_complexity)]
fn update_pareto_frontier(
    candidates: HashMap<String, UninitializedChatCompletionConfig>,
    val_scores: &HashMap<String, Option<EvaluationResults>>,
    config: &GEPAConfig,
    tensorzero_config: &Config,
) -> Result<
    (
        HashMap<String, UninitializedChatCompletionConfig>,
        HashMap<String, usize>,
    ),
    Error,
> {
    tracing::info!(
        "Filtering Pareto frontier using 3-step GEPA algorithm ({} candidates)",
        candidates.len()
    );

    // Get the evaluation config to determine metric optimization directions
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
        })?;

    let evaluators = match &**evaluation_config {
        tensorzero_core::evaluations::EvaluationConfig::Inference(inference_config) => {
            &inference_config.evaluators
        }
    };

    // Filter out candidates that failed evaluation (None)
    let valid_scores: HashMap<&String, &EvaluationResults> = val_scores
        .iter()
        .filter_map(|(name, score_opt)| score_opt.as_ref().map(|score| (name, score)))
        .collect();

    if valid_scores.is_empty() {
        tracing::warn!("No valid scores found for any variant");
        return Err(Error::new(ErrorDetails::InternalError {
            message: "All variants failed evaluation".to_string(),
        }));
    }

    // Get all datapoint IDs from the first valid score
    let datapoint_ids: Vec<String> = valid_scores
        .values()
        .next()
        .map(|scores| scores.per_datapoint.keys().cloned().collect())
        .unwrap_or_default();

    tracing::debug!(
        "Step 1: Building instance-wise Pareto sets for {} datapoints",
        datapoint_ids.len()
    );

    // Step 1: Build instance-wise Pareto sets P*[i] for each datapoint
    let mut instance_pareto_sets: HashMap<String, HashSet<String>> = HashMap::new();

    for datapoint_id in &datapoint_ids {
        // Collect scores for this datapoint across all variants
        let mut instance_scores: Vec<(String, HashMap<String, Option<f32>>)> = Vec::new();

        for (variant_name, evaluation_results) in &valid_scores {
            if let Some(scores) = evaluation_results.per_datapoint.get(datapoint_id) {
                instance_scores.push(((*variant_name).clone(), scores.clone()));
            }
        }

        if instance_scores.is_empty() {
            instance_pareto_sets.insert(datapoint_id.clone(), HashSet::new());
            continue;
        }

        // Find non-dominated variants for this instance
        let non_dominated = find_non_dominated_variants(&instance_scores, evaluators);
        instance_pareto_sets.insert(datapoint_id.clone(), non_dominated.into_iter().collect());
    }

    // Step 2: Build candidate set C (union of all instance-wise Pareto sets)
    let mut candidate_set: HashSet<String> = HashSet::new();
    for pareto_set in instance_pareto_sets.values() {
        candidate_set.extend(pareto_set.iter().cloned());
    }

    tracing::debug!(
        "Step 2: Candidate set has {} variants (union of instance-wise Pareto sets)",
        candidate_set.len()
    );

    // Early exit if no candidates or only one
    if candidate_set.len() <= 1 {
        let frequencies: HashMap<String, usize> = candidate_set
            .iter()
            .map(|v| {
                let freq = instance_pareto_sets
                    .values()
                    .filter(|s| s.contains(v))
                    .count();
                (v.clone(), freq)
            })
            .collect();

        let filtered_candidates: HashMap<String, UninitializedChatCompletionConfig> = candidates
            .into_iter()
            .filter(|(name, _)| candidate_set.contains(name))
            .collect();

        tracing::info!(
            "Early exit: {} candidate(s) after instance-wise filtering",
            filtered_candidates.len()
        );
        return Ok((filtered_candidates, frequencies));
    }

    // Step 3: Global filtering - check dominance over all (D×E) objectives
    tracing::debug!("Step 3: Global Pareto filtering across all objectives");

    let mut non_dominated = candidate_set.clone();

    for variant_a in &candidate_set {
        for variant_b in &candidate_set {
            if variant_a != variant_b && non_dominated.contains(variant_b) {
                // Get evaluation results for both variants
                let (Some(results_a), Some(results_b)) =
                    (valid_scores.get(variant_a), valid_scores.get(variant_b))
                else {
                    continue;
                };

                if global_dominates(results_a, results_b, evaluators) {
                    non_dominated.remove(variant_b);
                }
            }
        }
    }

    tracing::debug!(
        "Global filtering: {} → {} variants",
        candidate_set.len(),
        non_dominated.len()
    );

    // Monitor missing data rates for final Pareto frontier variants
    let total_datapoints = datapoint_ids.len();
    let total_evaluators = evaluators.len();

    for variant_name in &non_dominated {
        if let Some(evaluation_results) = valid_scores.get(variant_name) {
            let total_possible = total_datapoints * total_evaluators;
            if total_possible > 0 {
                // Count non-None scores across all (datapoint, evaluator) pairs
                let mut non_none_count = 0;
                for datapoint_id in &datapoint_ids {
                    if let Some(scores) = evaluation_results.per_datapoint.get(datapoint_id) {
                        for evaluator_name in evaluators.keys() {
                            if let Some(Some(_)) = scores.get(evaluator_name) {
                                non_none_count += 1;
                            }
                        }
                    }
                }

                let missing_rate = 1.0 - (non_none_count as f32 / total_possible as f32);

                if missing_rate > 0.3 {
                    tracing::warn!(
                        "Variant '{}' has high missing score rate: {:.1}% ({}/{} evaluations succeeded)",
                        variant_name,
                        missing_rate * 100.0,
                        non_none_count,
                        total_possible
                    );
                }
            }
        }
    }

    // Compute frequency map for sampling (count instance-wise Pareto memberships)
    let mut frequencies: HashMap<String, usize> = HashMap::new();
    for variant in &non_dominated {
        let freq = instance_pareto_sets
            .values()
            .filter(|s| s.contains(variant))
            .count();
        frequencies.insert(variant.clone(), freq);
    }

    // Filter out variants not in final non-dominated set
    let filtered_candidates: HashMap<String, UninitializedChatCompletionConfig> = candidates
        .into_iter()
        .filter(|(name, _)| non_dominated.contains(name))
        .collect();

    tracing::info!(
        "Pareto frontier filtered: {} → {} variants",
        valid_scores.len(),
        filtered_candidates.len()
    );

    Ok((filtered_candidates, frequencies))
}

/// Impute missing score as worst-case value based on optimization direction
///
/// Missing data treatment: Aggressive imputation to penalize unreliable variants.
/// - For "max" optimization: missing scores treated as -inf
/// - For "min" optimization: missing scores treated as +inf
///
/// This ensures variants with missing evaluations are dominated unless they excel on other objectives.
fn impute_missing_score(score: Option<f32>, optimize: MetricConfigOptimize) -> f32 {
    match score {
        Some(s) => s,
        None => match optimize {
            MetricConfigOptimize::Max => f32::NEG_INFINITY,
            MetricConfigOptimize::Min => f32::INFINITY,
        },
    }
}

/// Check if variant A globally dominates variant B across all (datapoint, evaluator) objectives
///
/// A globally dominates B if:
/// - A is better than or equal to B on all (datapoint_id, evaluator_name) pairs, AND
/// - A is strictly better than B on at least one pair
///
/// This compares variants across the full (D×E)-dimensional space where D=datapoints, E=evaluators.
/// Missing scores are imputed as worst-case (-inf for max, +inf for min).
fn global_dominates(
    variant_a_results: &EvaluationResults,
    variant_b_results: &EvaluationResults,
    evaluators: &HashMap<String, tensorzero_core::evaluations::EvaluatorConfig>,
) -> bool {
    let mut better_or_equal_on_all = true;
    let mut strictly_better_on_at_least_one = false;

    // Get all (datapoint_id, evaluator_name) pairs from both variants
    let mut all_pairs = std::collections::HashSet::new();
    for datapoint_id in variant_a_results.per_datapoint.keys() {
        for evaluator_name in evaluators.keys() {
            all_pairs.insert((datapoint_id.clone(), evaluator_name.clone()));
        }
    }
    for datapoint_id in variant_b_results.per_datapoint.keys() {
        for evaluator_name in evaluators.keys() {
            all_pairs.insert((datapoint_id.clone(), evaluator_name.clone()));
        }
    }

    for (datapoint_id, evaluator_name) in all_pairs {
        let Some(evaluator_config) = evaluators.get(&evaluator_name) else {
            continue;
        };
        let optimize = evaluator_config.optimize();

        // Get scores for this (datapoint, evaluator) pair
        let score_a = variant_a_results
            .per_datapoint
            .get(&datapoint_id)
            .and_then(|scores| scores.get(&evaluator_name).and_then(|s| *s));

        let score_b = variant_b_results
            .per_datapoint
            .get(&datapoint_id)
            .and_then(|scores| scores.get(&evaluator_name).and_then(|s| *s));

        // Impute missing values as worst-case
        let a_val = impute_missing_score(score_a, optimize);
        let b_val = impute_missing_score(score_b, optimize);

        match optimize {
            MetricConfigOptimize::Max => {
                if a_val < b_val {
                    better_or_equal_on_all = false;
                    break;
                }
                if a_val > b_val {
                    strictly_better_on_at_least_one = true;
                }
            }
            MetricConfigOptimize::Min => {
                if a_val > b_val {
                    better_or_equal_on_all = false;
                    break;
                }
                if a_val < b_val {
                    strictly_better_on_at_least_one = true;
                }
            }
        }
    }

    better_or_equal_on_all && strictly_better_on_at_least_one
}

/// Check if variant A instance-dominates variant B on a specific datapoint
///
/// A dominates B if:
/// - A is better than or equal to B on all evaluators, AND
/// - A is strictly better than B on at least one evaluator
///
/// Missing scores are imputed as worst-case (-inf for max, +inf for min).
fn instance_dominates(
    a_scores: &HashMap<String, Option<f32>>,
    b_scores: &HashMap<String, Option<f32>>,
    evaluators: &HashMap<String, tensorzero_core::evaluations::EvaluatorConfig>,
) -> bool {
    let mut better_or_equal_on_all = true;
    let mut strictly_better_on_at_least_one = false;

    for (evaluator_name, evaluator_config) in evaluators {
        let score_a = a_scores.get(evaluator_name).and_then(|s| *s);
        let score_b = b_scores.get(evaluator_name).and_then(|s| *s);

        let optimize = evaluator_config.optimize();

        // Impute missing values as worst-case
        let a_val = impute_missing_score(score_a, optimize);
        let b_val = impute_missing_score(score_b, optimize);

        match optimize {
            MetricConfigOptimize::Max => {
                if a_val < b_val {
                    better_or_equal_on_all = false;
                    break;
                }
                if a_val > b_val {
                    strictly_better_on_at_least_one = true;
                }
            }
            MetricConfigOptimize::Min => {
                if a_val > b_val {
                    better_or_equal_on_all = false;
                    break;
                }
                if a_val < b_val {
                    strictly_better_on_at_least_one = true;
                }
            }
        }
    }

    better_or_equal_on_all && strictly_better_on_at_least_one
}

/// Find non-dominated variants for a single instance (datapoint) using instance-wise dominance
fn find_non_dominated_variants(
    instance_scores: &[(String, HashMap<String, Option<f32>>)],
    evaluators: &HashMap<String, tensorzero_core::evaluations::EvaluatorConfig>,
) -> Vec<String> {
    let mut non_dominated = Vec::new();

    for (variant_a_name, variant_a_scores) in instance_scores {
        let mut is_dominated = false;

        // Check if variant_a is dominated by any other variant
        for (variant_b_name, variant_b_scores) in instance_scores {
            if variant_a_name == variant_b_name {
                continue;
            }

            // Check if variant_b dominates variant_a
            if instance_dominates(variant_b_scores, variant_a_scores, evaluators) {
                is_dominated = true;
                break;
            }
        }

        if !is_dominated {
            non_dominated.push(variant_a_name.clone());
        }
    }

    non_dominated
}

/// Holds the results of evaluating variants on a dataset
/// This matches the EvaluationResults structure from planning.md
#[derive(Clone, Debug)]
struct EvaluationResults {
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
#[expect(dead_code)]
#[expect(clippy::too_many_arguments)]
async fn evaluate_variants(
    gateway_client: &Client,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    tensorzero_config: std::sync::Arc<Config>,
    _credentials: &InferenceCredentials,
    _function_name: &str,
    evaluation_name: &str,
    variant_configs: &HashMap<String, UninitializedChatCompletionConfig>,
    dataset_name: &str,
    max_concurrency: u32,
) -> Result<HashMap<String, Option<EvaluationResults>>, Error> {
    let mut results = HashMap::new();
    let concurrency = max_concurrency as usize;

    // Get evaluation config for later use
    let evaluation_config = tensorzero_config
        .evaluations
        .get(evaluation_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!("Evaluation '{evaluation_name}' not found in config"),
            })
        })?
        .clone();

    for (variant_name, chat_config) in variant_configs {
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
            tensorzero_client: gateway_client.clone(),
            clickhouse_client: clickhouse_connection_info.clone(),
            config: tensorzero_config.clone(),
            evaluation_name: evaluation_name.to_string(),
            evaluation_run_id,
            dataset_name: dataset_name.to_string(),
            variant: EvaluationVariant::Info(Box::new(dynamic_variant_config)),
            concurrency,
            inference_cache: CacheEnabledMode::Off, // Disable caching for fair evaluation
        };

        // Call run_evaluation_core_streaming
        let stream_result = match evaluations::run_evaluation_core_streaming(core_args).await {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!(
                    "Failed to start evaluation for variant '{}': {}",
                    variant_name,
                    e
                );
                results.insert(variant_name.clone(), None);
                continue;
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
                results.insert(variant_name.clone(), None);
                continue;
            }
        };

        results.insert(variant_name.clone(), Some(evaluation_results));
    }

    Ok(results)
}

/// Consume the evaluation stream and aggregate results
///
/// Uses EvaluationStats infrastructure to handle stream consumption and statistics computation,
/// following the same pattern as the evaluations CLI for consistency and maintainability.
async fn consume_evaluation_stream(
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

/// Sample a variant name proportional to its frequency
/// Uses cumulative distribution for weighted random sampling
#[expect(dead_code)]
fn sample_by_frequency(frequencies: &HashMap<String, usize>) -> Result<String, Error> {
    if frequencies.is_empty() {
        return Err(Error::new(ErrorDetails::InternalError {
            message: "Cannot sample from empty frequency map".to_string(),
        }));
    }

    // Calculate total frequency
    let total_frequency: usize = frequencies.values().sum();

    if total_frequency == 0 {
        return Err(Error::new(ErrorDetails::InternalError {
            message: "Cannot sample when all frequencies are zero".to_string(),
        }));
    }

    // Generate random number in [0, total_frequency)
    let mut rng = rand::rng();
    let mut random_value = rng.random_range(0..total_frequency);

    // Iterate through frequencies using cumulative distribution
    for (variant_name, &frequency) in frequencies {
        if random_value < frequency {
            return Ok(variant_name.clone());
        }
        random_value -= frequency;
    }

    // Fallback (should never happen due to total_frequency check)
    // Return the first variant as a safety measure
    Ok(frequencies
        .keys()
        .next()
        .ok_or_else(|| {
            Error::new(ErrorDetails::InternalError {
                message: "Frequency map unexpectedly empty".to_string(),
            })
        })?
        .clone())
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

/// Check if mutation improves over original variant (global Pareto dominance)
/// Returns false if either variant is missing from scores (evaluation failed)
///
/// Uses the evaluation config to determine objective directions (optimize: max/min)
#[expect(dead_code)]
fn is_improvement(
    scores: &HashMap<String, Option<EvaluationResults>>,
    original_variant: &str,
    mutation_variant: &str,
    config: &Config,
    evaluation_name: &str,
) -> Result<bool, Error> {
    // Get scores for both variants (return false if either is None)
    let original_scores = match scores.get(original_variant) {
        Some(Some(scores)) => scores,
        _ => return Ok(false), // Original variant evaluation failed
    };

    let mutation_scores = match scores.get(mutation_variant) {
        Some(Some(scores)) => scores,
        _ => return Ok(false), // Mutation variant evaluation failed
    };

    // Get the evaluation config to determine metric optimization directions
    let evaluation_config = config.evaluations.get(evaluation_name).ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: format!("Evaluation '{evaluation_name}' not found in config"),
        })
    })?;

    // Get evaluator names from the evaluation config
    let evaluators = match &**evaluation_config {
        tensorzero_core::evaluations::EvaluationConfig::Inference(inference_config) => {
            &inference_config.evaluators
        }
    };

    // Track whether mutation is better, worse, or equal on each metric
    let mut strictly_better_on_at_least_one = false;
    let mut worse_on_any = false;

    // Compare on each metric
    for (evaluator_name, evaluator_config) in evaluators {
        // Get metric stats for both variants
        let original_stats = original_scores.metrics.get(evaluator_name);
        let mutation_stats = mutation_scores.metrics.get(evaluator_name);

        // Skip if either variant doesn't have stats for this evaluator
        let (original_mean, mutation_mean) = match (original_stats, mutation_stats) {
            (Some(orig), Some(mut_stat)) => (orig.mean, mut_stat.mean),
            _ => continue, // Skip this metric if either variant failed
        };

        // Determine if mutation is better based on optimization direction
        let mutation_is_better = match evaluator_config.optimize() {
            MetricConfigOptimize::Max => mutation_mean > original_mean,
            MetricConfigOptimize::Min => mutation_mean < original_mean,
        };

        let mutation_is_worse = match evaluator_config.optimize() {
            MetricConfigOptimize::Max => mutation_mean < original_mean,
            MetricConfigOptimize::Min => mutation_mean > original_mean,
        };

        if mutation_is_better {
            strictly_better_on_at_least_one = true;
        }

        if mutation_is_worse {
            worse_on_any = true;
            break; // No need to check further if worse on any metric
        }
    }

    // Pareto dominance: better or equal on all metrics, strictly better on at least one
    Ok(strictly_better_on_at_least_one && !worse_on_any)
}

/// Create evaluation dataset from samples in ClickHouse
/// Uses the existing dataset storage infrastructure
///
/// NOTE: This function is currently a stub. Dataset creation requires:
/// 1. Converting RenderedSamples to DatapointInsert format (ChatInferenceDatapointInsert or JsonInferenceDatapointInsert)
/// 2. Inserting into ClickHouse using the DatasetQueries trait
/// 3. This is complex and will be implemented when needed for the full GEPA workflow
#[expect(dead_code)]
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
async fn create_evaluation_dataset(
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

/// Analyze inference outputs using the GEPA analyze function
///
/// Takes evaluation results (datapoint + inference pairs) and calls the built-in
/// `tensorzero::optimization::gepa::analyze` function to get structured feedback.
///
/// Returns a vector of InferenceWithAnalysis containing each inference paired with its analysis.
#[expect(dead_code)]
async fn analyze_inferences(
    _gateway_client: &Client,
    _evaluation_infos: &[EvaluationInfo],
    _function_config: &FunctionConfig,
    _analysis_model: &str,
) -> Result<Vec<InferenceWithAnalysis>, Error> {
    // TODO: Implement analysis phase
    // For each evaluation_info:
    // 1. Extract function metadata (templates, schemas, tools) from function_config
    // 2. Build input for tensorzero::optimization::gepa::analyze
    // 3. Call inference API with dynamic variant config for analysis model
    // 4. Parse tool call response into AnalysisReport enum

    tracing::warn!("analyze_inferences not yet implemented");
    Ok(Vec::new())
}

/// Generate improved templates using the GEPA mutate function
///
/// Takes aggregated analyses and calls the built-in `tensorzero::optimization::gepa::mutate`
/// function to synthesize improved prompt templates.
///
/// Returns MutateOutput with improved templates.
#[expect(dead_code)]
async fn mutate_templates(
    _gateway_client: &Client,
    _analyses: &[InferenceWithAnalysis],
    _function_config: &FunctionConfig,
    _mutation_model: &str,
) -> Result<MutateOutput, Error> {
    // TODO: Implement mutation phase
    // 1. Build analyses array for mutate function input
    // 2. Extract function metadata (templates, schemas, tools) from function_config
    // 3. Call tensorzero::optimization::gepa::mutate with dynamic variant config
    // 4. Parse JSON output into MutateOutput

    tracing::warn!("mutate_templates not yet implemented");
    Err(Error::new(ErrorDetails::InternalError {
        message: "mutate_templates not yet implemented".to_string(),
    }))
}

/// Create a new variant with mutated templates
///
/// Generates a new variant name and clones the parent variant config with updated templates.
#[expect(dead_code)]
fn create_mutated_variant(
    parent_config: &UninitializedChatCompletionConfig,
    mutated_templates: MutateOutput,
    iteration: usize,
    variant_prefix: &str,
    parent_name: &str,
) -> (String, UninitializedChatCompletionConfig) {
    // Generate variant name: {prefix}_iter{iteration}_{parent_name}
    let new_variant_name = format!("{variant_prefix}_iter{iteration}_{parent_name}");

    // Clone parent config
    let mut new_config = parent_config.clone();

    // Replace templates with mutated versions using fake paths
    // Fake paths are used because these templates are generated dynamically, not from files
    new_config.system_template = Some(ResolvedTomlPath::new_fake_path(
        format!("gepa_mutated/{new_variant_name}/system.minijinja"),
        mutated_templates.system_template,
    ));

    if let Some(user_template) = mutated_templates.user_template {
        new_config.user_template = Some(ResolvedTomlPath::new_fake_path(
            format!("gepa_mutated/{new_variant_name}/user.minijinja"),
            user_template,
        ));
    }

    if let Some(assistant_template) = mutated_templates.assistant_template {
        new_config.assistant_template = Some(ResolvedTomlPath::new_fake_path(
            format!("gepa_mutated/{new_variant_name}/assistant.minijinja"),
            assistant_template,
        ));
    }

    tracing::info!(
        "Created mutated variant '{}' from parent '{}'",
        new_variant_name,
        parent_name
    );

    (new_variant_name, new_config)
}

/// Sample a random subset of examples from a dataset without replacement
/// Uses Fisher-Yates shuffle algorithm for efficient sampling
#[expect(dead_code)]
fn random_sample(examples: &[RenderedSample], n: usize) -> Vec<RenderedSample> {
    // If n >= examples.len(), return all examples
    if n >= examples.len() {
        return examples.to_vec();
    }

    // Create a vector of indices and shuffle only the first n elements
    let mut rng = rand::rng();
    let mut indices: Vec<usize> = (0..examples.len()).collect();

    // Partial Fisher-Yates shuffle: only shuffle the first n elements
    for i in 0..n {
        let j = rng.random_range(i..indices.len());
        indices.swap(i, j);
    }

    // Collect the first n sampled examples
    indices[0..n].iter().map(|&i| examples[i].clone()).collect()
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
