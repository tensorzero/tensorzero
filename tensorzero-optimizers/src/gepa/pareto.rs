//! Pareto frontier analysis and dominance checking for GEPA
//!
//! This module implements the core Pareto optimality logic for multi-objective optimization:
//! - Instance-wise Pareto frontier computation
//! - Global Pareto dominance checking
//! - Missing data imputation

use std::collections::{HashMap, HashSet};

use tensorzero_core::{
    config::{Config, MetricConfigOptimize},
    error::{Error, ErrorDetails},
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use super::evaluate::EvaluationResults;

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
pub fn update_pareto_frontier(
    candidates: HashMap<String, UninitializedChatCompletionConfig>,
    val_scores: &HashMap<String, Option<EvaluationResults>>,
    config: &tensorzero_core::optimization::gepa::GEPAConfig,
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
pub fn impute_missing_score(score: Option<f32>, optimize: MetricConfigOptimize) -> f32 {
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
pub fn global_dominates(
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
pub fn instance_dominates(
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
pub fn find_non_dominated_variants(
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

/// Check if mutation improves over original variant (global Pareto dominance)
/// Returns false if either variant is missing from scores (evaluation failed)
///
/// Uses the evaluation config to determine objective directions (optimize: max/min)
#[expect(dead_code)]
pub fn is_improvement(
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
