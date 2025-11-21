//! Pareto frontier analysis and dominance checking for GEPA
//!
//! This module implements the core Pareto optimality logic for multi-objective optimization:
//! - Instance-wise Pareto frontier computation
//! - Global Pareto dominance checking
//! - Missing data imputation

use std::collections::{HashMap, HashSet};

use evaluations::EvaluatorStats;
use tensorzero_core::{
    config::MetricConfigOptimize,
    error::{Error, ErrorDetails},
    evaluations::{EvaluatorConfig, InferenceEvaluationConfig},
    variant::chat_completion::UninitializedChatCompletionConfig,
};

// Type aliases for cleaner score map signatures (TODO: will live next to evaluation code)
pub type EvaluatorName = String;

/// Unique identifier for a datapoint/example in a dataset
pub type DatapointId = String;

/// Threshold for warning about high missing score rates
///
/// Variants with more than 30% missing evaluations will trigger a warning.
/// This threshold helps identify variants that may be unreliable due to
/// systematic evaluation failures.
const HIGH_MISSING_RATE_THRESHOLD: f32 = 0.3;

/// Scores for all evaluators on a single datapoint
pub type DatapointScores = HashMap<EvaluatorName, Option<f32>>;

/// Scores for all datapoints for a single variant
pub type VariantScores = HashMap<DatapointId, DatapointScores>;

/// Result of Pareto frontier filtering with frequency-based sampling weights
#[derive(Debug, Clone)]
pub struct ParetoFrontier {
    /// Pareto-optimal variants (non-dominated after GEPA's 3-step filtering)
    pub variants: HashMap<String, UninitializedChatCompletionConfig>,

    /// Instance-wise membership frequencies for weighted sampling
    ///
    /// Maps each variant to the count of datapoints where it's Pareto-optimal.
    /// Higher values indicate the variant performs well across more instances.
    /// Used for proportional sampling in GEPA's mutation step.
    pub frequencies: HashMap<String, usize>,
}

/// Check if child improves over parent variant (summary statistic Pareto dominance)
///
/// Uses the evaluation config to determine objective directions (optimize: max/min).
///
/// # Arguments
/// * `parent_stats` - Summary statistics (mean, stderr, count) for parent variant's evaluations
/// * `child_stats` - Summary statistics (mean, stderr, count) for child variant's evaluations
/// * `evaluation_config` - Evaluation config with evaluator definitions and optimization directions
///
/// # Returns
/// * `bool` - True if child Pareto-dominates parent (better/equal on all, strictly better on ≥1 evaluator summary stats)
pub fn is_improvement(
    parent_stats: &HashMap<String, EvaluatorStats>,
    child_stats: &HashMap<String, EvaluatorStats>,
    evaluation_config: &InferenceEvaluationConfig,
) -> bool {
    // Get evaluators from the evaluation config
    let evaluators = &evaluation_config.evaluators;

    // Track whether child is better, worse, or equal on each metric
    let mut strictly_better_on_at_least_one = false;
    let mut worse_on_any = false;

    // Compare on each metric
    for (evaluator_name, evaluator_config) in evaluators {
        // Get metric stats for both variants
        let parent_stat = parent_stats.get(evaluator_name);
        let child_stat = child_stats.get(evaluator_name);

        // Skip if either variant doesn't have stats for this evaluator
        let (parent_mean, child_mean) = match (parent_stat, child_stat) {
            (Some(parent), Some(child)) => (parent.mean, child.mean),
            _ => continue, // Skip this metric if either variant failed
        };

        // Determine if child is better based on optimization direction
        let child_is_better = match evaluator_config.optimize() {
            MetricConfigOptimize::Max => child_mean > parent_mean,
            MetricConfigOptimize::Min => child_mean < parent_mean,
        };

        let child_is_worse = match evaluator_config.optimize() {
            MetricConfigOptimize::Max => child_mean < parent_mean,
            MetricConfigOptimize::Min => child_mean > parent_mean,
        };

        if child_is_better {
            strictly_better_on_at_least_one = true;
        }

        if child_is_worse {
            worse_on_any = true;
            break; // No need to check further if worse on any metric
        }
    }

    // Pareto dominance: better or equal on all metrics, strictly better on at least one
    strictly_better_on_at_least_one && !worse_on_any
}

/// Updates the Pareto frontier based on instance-wise Pareto dominance
///
/// Implements the SELECTCANDIDATE algorithm from the GEPA paper:
/// 1. For each datapoint, find instance-wise Pareto-optimal variants (Step 1)
/// 2. Build candidate set C as union of all instance-wise Pareto sets (Step 2)
/// 3. Filter candidates globally to remove dominated variants (Step 3)
/// 4. Compute frequency of each variant's membership in instance-wise Pareto sets
///
/// Reference: "GEPA: Improving Language Models through Genetic Evolutionary Prompt Adaptation"
/// <https://arxiv.org/abs/2507.19457>
///
/// Note: The original GEPA paper:
/// - Uses a single evaluator and selects variants achieving maximum score per instance
/// - In the online setting, also checks for optimality across functions (not applicable in our offline setting)
///
/// We extend this to multiple evaluators by selecting Pareto non-dominated variants per instance,
/// which naturally generalizes the single-objective case.
///
/// # Arguments
/// * `candidates` - Candidate variants to filter
/// * `val_scores_map` - Per-datapoint scores on validation set for each candidate
/// * `evaluation_config` - Evaluation config with evaluator definitions and optimization directions
///
/// # Returns
/// * `Result<ParetoFrontier, Error>` - ParetoFrontier containing filtered variants and their sampling frequencies,
///   or an error if validation scores are empty or all evaluations failed
pub fn update_pareto_frontier(
    candidates: HashMap<String, UninitializedChatCompletionConfig>,
    val_scores_map: &HashMap<String, VariantScores>,
    evaluation_config: &InferenceEvaluationConfig,
) -> Result<ParetoFrontier, Error> {
    tracing::info!(
        "Filtering Pareto frontier using 3-step GEPA algorithm ({} candidates)",
        candidates.len()
    );

    // Get evaluators from the evaluation config
    let evaluators = &evaluation_config.evaluators;

    // Check if we have any valid scores
    if val_scores_map.is_empty() {
        tracing::warn!("No validation scores provided for any variant");
        return Err(Error::new(ErrorDetails::InternalError {
            message: "No validation scores provided for any variant".to_string(),
        }));
    }

    // Check if there are any datapoints - if yes, ensure at least one valid score exists
    let has_any_datapoint = val_scores_map
        .values()
        .any(|per_datapoint| !per_datapoint.is_empty());

    if has_any_datapoint {
        // We have datapoints, check if all scores are None (all evaluations failed)
        let has_any_score = val_scores_map
            .values()
            .flat_map(|per_datapoint| per_datapoint.values())
            .flat_map(|scores| scores.values())
            .any(|score| score.is_some());

        if !has_any_score {
            tracing::warn!("All evaluation scores are None - no variant produced any valid scores");
            return Err(Error::new(ErrorDetails::InternalError {
                message: "All variants failed to produce valid scores".to_string(),
            }));
        }
    }

    // Get union of all datapoint IDs across all variants
    // This ensures we analyze ALL datapoints that ANY variant was evaluated on
    let datapoint_ids: Vec<String> = val_scores_map
        .values()
        .flat_map(|scores| scores.keys().cloned())
        .collect::<HashSet<_>>() // Deduplicate
        .into_iter()
        .collect();

    tracing::debug!(
        "Step 1: Building instance-wise Pareto sets for {} datapoints (union across all variants)",
        datapoint_ids.len()
    );

    // Step 1: Build instance-wise Pareto sets P*[i] for each datapoint
    // Original paper: P*[i] = {variants with max score on instance i} (single evaluator)
    // Our extension: P*[i] = {Pareto non-dominated variants on instance i} (multiple evaluators)
    let mut instance_pareto_sets: HashMap<String, HashSet<String>> = HashMap::new();

    for datapoint_id in &datapoint_ids {
        // Collect scores for this datapoint across all variants
        let mut instance_scores: Vec<(String, HashMap<String, Option<f32>>)> = Vec::new();

        for (variant_name, variant_scores) in val_scores_map {
            if let Some(scores) = variant_scores.get(datapoint_id) {
                instance_scores.push((variant_name.clone(), scores.clone()));
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
    let candidate_set: HashSet<String> = instance_pareto_sets
        .values()
        .flat_map(|set| set.iter().cloned())
        .collect();

    tracing::debug!(
        "Step 2: Candidate set has {} variants (union of instance-wise Pareto sets)",
        candidate_set.len()
    );

    // Early exit if no candidates or only one
    if candidate_set.len() <= 1 {
        let frequencies = calculate_frequencies(&candidate_set, &instance_pareto_sets);

        let filtered_candidates: HashMap<String, UninitializedChatCompletionConfig> = candidates
            .into_iter()
            .filter(|(name, _)| candidate_set.contains(name))
            .collect();

        tracing::info!(
            "Early exit: {} candidate(s) after instance-wise filtering",
            filtered_candidates.len()
        );
        return Ok(ParetoFrontier {
            variants: filtered_candidates,
            frequencies,
        });
    }

    // Step 3: Global filtering - check dominance over all (D×E) objectives
    tracing::debug!("Step 3: Global Pareto filtering across all objectives");

    // Note: This has O(n² × D × E) complexity where:
    // - n = size of candidate_set (number of variants)
    // - D = number of datapoints
    // - E = number of evaluators
    // The n² term dominates: acceptable for typical GEPA workloads (n < 100)
    // but could become a bottleneck for large variant populations (n > 100).
    let mut non_dominated = candidate_set.clone();

    for variant_a in &candidate_set {
        for variant_b in &candidate_set {
            if variant_a != variant_b
                && non_dominated.contains(variant_b)
                && global_dominates(variant_a, variant_b, val_scores_map, evaluators)
            {
                non_dominated.remove(variant_b);
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
        if let Some(per_datapoint) = val_scores_map.get(variant_name) {
            let total_possible = total_datapoints * total_evaluators;
            if total_possible > 0 {
                // Count non-None scores across all (datapoint, evaluator) pairs
                let non_none_count = datapoint_ids
                    .iter()
                    .filter_map(|datapoint_id| per_datapoint.get(datapoint_id))
                    .flat_map(|scores| {
                        evaluators.keys().filter_map(move |evaluator_name| {
                            scores.get(evaluator_name).and_then(|s| s.as_ref())
                        })
                    })
                    .count();

                let missing_rate = 1.0 - (non_none_count as f32 / total_possible as f32);

                if missing_rate > HIGH_MISSING_RATE_THRESHOLD {
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
    let frequencies = calculate_frequencies(&non_dominated, &instance_pareto_sets);

    // Filter out variants not in final non-dominated set
    let filtered_candidates: HashMap<String, UninitializedChatCompletionConfig> = candidates
        .into_iter()
        .filter(|(name, _)| non_dominated.contains(name))
        .collect();

    tracing::info!(
        "Pareto frontier filtered: {} → {} variants",
        val_scores_map.len(),
        filtered_candidates.len()
    );

    Ok(ParetoFrontier {
        variants: filtered_candidates,
        frequencies,
    })
}

/// Impute missing score as worst-case value based on optimization direction
///
/// Missing data treatment: Aggressive imputation to penalize unreliable variants.
/// - For "max" optimization: missing scores treated as -inf
/// - For "min" optimization: missing scores treated as +inf
///
/// Rationale: Aggressive imputation penalizes variants that systematically fail (e.g., causing
/// inference errors) while treating random failures (e.g., provider unavailable) equally across variants.
///
/// This ensures variants with missing evaluations are dominated unless they excel on other objectives.
///
/// # Arguments
/// * `score` - Optional evaluation score (None if evaluation failed or is missing)
/// * `optimize` - Optimization direction (Max or Min)
///
/// # Returns
/// * `f32` - The original score if present, or -inf for Max optimization / +inf for Min optimization if missing
fn impute_missing_score(score: Option<f32>, optimize: MetricConfigOptimize) -> f32 {
    match score {
        Some(s) => s,
        None => match optimize {
            MetricConfigOptimize::Max => f32::NEG_INFINITY,
            MetricConfigOptimize::Min => f32::INFINITY,
        },
    }
}

/// Compare two values according to optimization direction
///
/// # Arguments
/// * `a_val` - The first value to compare
/// * `b_val` - The second value to compare
/// * `optimize` - The optimization direction (Max or Min)
///
/// # Returns
/// A tuple `(is_worse, is_better)` where:
/// - `is_worse`: true if value `a` is worse than value `b` given the optimization direction
/// - `is_better`: true if value `a` is strictly better than value `b` given the optimization direction
fn compare_values(a_val: f32, b_val: f32, optimize: MetricConfigOptimize) -> (bool, bool) {
    match optimize {
        MetricConfigOptimize::Max => (a_val < b_val, a_val > b_val),
        MetricConfigOptimize::Min => (a_val > b_val, a_val < b_val),
    }
}

/// Calculate frequency map for variants based on instance-wise Pareto set memberships
///
/// For each variant, counts how many datapoint-level Pareto sets it appears in.
/// Higher frequency indicates the variant performs well on more instances.
///
/// # Arguments
/// * `variants` - Set of variant names to calculate frequencies for
/// * `instance_pareto_sets` - Map from datapoint ID to set of Pareto-optimal variant names for that instance
///
/// # Returns
/// HashMap mapping each variant name to its frequency (count of instance-wise Pareto set memberships)
fn calculate_frequencies(
    variants: &HashSet<String>,
    instance_pareto_sets: &HashMap<String, HashSet<String>>,
) -> HashMap<String, usize> {
    variants
        .iter()
        .map(|v| {
            let freq = instance_pareto_sets
                .values()
                .filter(|s| s.contains(v))
                .count();
            (v.clone(), freq)
        })
        .collect()
}

/// Check if variant A globally dominates variant B across all (datapoint, evaluator) objectives
///
/// A globally dominates B if:
/// - A is better than or equal to B on all (datapoint_id, evaluator_name) pairs, AND
/// - A is strictly better than B on at least one pair
///
/// This compares variants across the full (D×E)-dimensional space where D=datapoints, E=evaluators.
/// Missing scores are imputed as worst-case (-inf for max, +inf for min).
///
/// # Arguments
/// * `variant_a_name` - Name of the first variant to compare
/// * `variant_b_name` - Name of the second variant to compare
/// * `val_scores_map` - Map of variant names to their per-datapoint scores
/// * `evaluators` - Map of evaluator configurations containing optimization directions
///
/// # Returns
/// * `bool` - True if variant A globally dominates variant B
fn global_dominates(
    variant_a_name: &str,
    variant_b_name: &str,
    val_scores_map: &HashMap<String, VariantScores>,
    evaluators: &HashMap<String, EvaluatorConfig>,
) -> bool {
    let mut better_or_equal_on_all = true;
    let mut strictly_better_on_at_least_one = false;

    // Get scores for both variants
    let variant_a_scores = val_scores_map.get(variant_a_name);
    let variant_b_scores = val_scores_map.get(variant_b_name);

    // Get all (datapoint_id, evaluator_name) pairs from both variants
    let all_pairs: HashSet<_> = variant_a_scores
        .into_iter()
        .flat_map(|scores| scores.keys())
        .chain(
            variant_b_scores
                .into_iter()
                .flat_map(|scores| scores.keys()),
        )
        .flat_map(|datapoint_id| {
            evaluators
                .keys()
                .map(move |evaluator_name| (datapoint_id.clone(), evaluator_name.clone()))
        })
        .collect();

    for (datapoint_id, evaluator_name) in all_pairs {
        let Some(evaluator_config) = evaluators.get(&evaluator_name) else {
            continue;
        };
        let optimize = evaluator_config.optimize();

        // Get scores for this (datapoint, evaluator) pair
        let score_a = variant_a_scores
            .and_then(|scores| scores.get(&datapoint_id))
            .and_then(|scores| scores.get(&evaluator_name).and_then(|s| *s));

        let score_b = variant_b_scores
            .and_then(|scores| scores.get(&datapoint_id))
            .and_then(|scores| scores.get(&evaluator_name).and_then(|s| *s));

        // Impute missing values as worst-case
        let a_val = impute_missing_score(score_a, optimize);
        let b_val = impute_missing_score(score_b, optimize);

        let (is_worse, is_better) = compare_values(a_val, b_val, optimize);
        if is_worse {
            better_or_equal_on_all = false;
            break;
        }
        if is_better {
            strictly_better_on_at_least_one = true;
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
///
/// # Arguments
/// * `a_scores` - Scores for variant A on this datapoint, mapped by evaluator name
/// * `b_scores` - Scores for variant B on this datapoint, mapped by evaluator name
/// * `evaluators` - Map of evaluator configurations containing optimization directions
///
/// # Returns
/// * `bool` - True if variant A instance-dominates variant B on this datapoint
fn instance_dominates(
    a_scores: &HashMap<String, Option<f32>>,
    b_scores: &HashMap<String, Option<f32>>,
    evaluators: &HashMap<String, EvaluatorConfig>,
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

        let (is_worse, is_better) = compare_values(a_val, b_val, optimize);
        if is_worse {
            better_or_equal_on_all = false;
            break;
        }
        if is_better {
            strictly_better_on_at_least_one = true;
        }
    }

    better_or_equal_on_all && strictly_better_on_at_least_one
}

/// Find non-dominated variants for a single instance (datapoint) using instance-wise dominance
///
/// # Arguments
/// * `instance_scores` - Vector of (variant_name, scores) tuples where scores map evaluator names to optional values
/// * `evaluators` - Map of evaluator configurations containing optimization directions
///
/// # Returns
/// Vector of variant names that are not dominated by any other variant on this instance
fn find_non_dominated_variants(
    instance_scores: &[(String, HashMap<String, Option<f32>>)],
    evaluators: &HashMap<String, EvaluatorConfig>,
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

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Test Helper Functions
    // ============================================================================

    /// Create a HashMap of evaluator configs with specified optimize directions
    ///
    /// # Arguments
    /// * `evaluators` - Slice of (evaluator_name, optimize_direction) tuples
    ///   where optimize_direction is "max" or "min"
    fn create_test_evaluators(evaluators: &[(&str, &str)]) -> HashMap<String, EvaluatorConfig> {
        use tensorzero_core::evaluations::{
            ExactMatchConfig, LLMJudgeConfig, LLMJudgeIncludeConfig, LLMJudgeInputFormat,
            LLMJudgeOptimize, LLMJudgeOutputType,
        };

        evaluators
            .iter()
            .map(|(name, optimize)| {
                let evaluator_config = match *optimize {
                    "max" => {
                        // ExactMatch always maximizes
                        EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None })
                    }
                    "min" => {
                        // LLMJudge can be configured to minimize
                        EvaluatorConfig::LLMJudge(LLMJudgeConfig {
                            input_format: LLMJudgeInputFormat::Serialized,
                            output_type: LLMJudgeOutputType::Float,
                            include: LLMJudgeIncludeConfig {
                                reference_output: false,
                            },
                            optimize: LLMJudgeOptimize::Min,
                            cutoff: None,
                        })
                    }
                    _ => panic!("Invalid optimize direction: {optimize}"),
                };
                (name.to_string(), evaluator_config)
            })
            .collect()
    }

    /// Create a HashMap of test variants
    fn create_test_variants(names: &[&str]) -> HashMap<String, UninitializedChatCompletionConfig> {
        names
            .iter()
            .map(|&name| {
                (
                    name.to_string(),
                    UninitializedChatCompletionConfig {
                        weight: None,
                        model: "test-model".into(),
                        ..Default::default()
                    },
                )
            })
            .collect()
    }

    /// Create a HashMap of EvaluatorStats for testing is_improvement
    ///
    /// # Arguments
    /// * `stats` - Slice of (evaluator_name, mean, stderr, count) tuples
    fn create_evaluator_stats(
        stats: &[(&str, f32, f32, usize)],
    ) -> HashMap<String, EvaluatorStats> {
        stats
            .iter()
            .map(|(name, mean, stderr, count)| {
                (
                    name.to_string(),
                    EvaluatorStats {
                        mean: *mean,
                        stderr: *stderr,
                        count: *count,
                    },
                )
            })
            .collect()
    }

    /// Create an InferenceEvaluationConfig for testing
    fn create_evaluation_config(evaluators: &[(&str, &str)]) -> InferenceEvaluationConfig {
        let evaluator_configs = create_test_evaluators(evaluators);
        InferenceEvaluationConfig {
            evaluators: evaluator_configs,
            function_name: "test_function".to_string(),
        }
    }

    // ============================================================================
    // Unit Tests for Helper Functions
    // ============================================================================

    #[test]
    fn test_impute_missing_score_max() {
        // For maximize: missing score should be -infinity
        assert_eq!(
            impute_missing_score(None, MetricConfigOptimize::Max),
            f32::NEG_INFINITY
        );
        assert_eq!(
            impute_missing_score(Some(0.5), MetricConfigOptimize::Max),
            0.5
        );
    }

    #[test]
    fn test_impute_missing_score_min() {
        // For minimize: missing score should be +infinity
        assert_eq!(
            impute_missing_score(None, MetricConfigOptimize::Min),
            f32::INFINITY
        );
        assert_eq!(
            impute_missing_score(Some(0.5), MetricConfigOptimize::Min),
            0.5
        );
    }

    #[test]
    fn test_instance_dominates_basic() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        let a_scores = HashMap::from([("accuracy".to_string(), Some(0.9))]);
        let b_scores = HashMap::from([("accuracy".to_string(), Some(0.7))]);

        // A should dominate B (0.9 > 0.7)
        assert!(instance_dominates(&a_scores, &b_scores, &evaluators));
        // B should not dominate A
        assert!(!instance_dominates(&b_scores, &a_scores, &evaluators));
    }

    #[test]
    fn test_instance_dominates_equal() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        let a_scores = HashMap::from([("accuracy".to_string(), Some(0.8))]);
        let b_scores = HashMap::from([("accuracy".to_string(), Some(0.8))]);

        // Equal scores - neither dominates
        assert!(!instance_dominates(&a_scores, &b_scores, &evaluators));
        assert!(!instance_dominates(&b_scores, &a_scores, &evaluators));
    }

    #[test]
    fn test_instance_dominates_with_none() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        let a_scores = HashMap::from([("accuracy".to_string(), Some(0.7))]);
        let b_scores = HashMap::from([("accuracy".to_string(), None)]);

        // A (0.7) should dominate B (None = -inf)
        assert!(instance_dominates(&a_scores, &b_scores, &evaluators));
        // B should not dominate A
        assert!(!instance_dominates(&b_scores, &a_scores, &evaluators));
    }

    #[test]
    fn test_instance_dominates_mixed_directions() {
        let evaluators = create_test_evaluators(&[("accuracy", "max"), ("latency", "min")]);

        // A: high accuracy, low latency (good on both)
        let a_scores = HashMap::from([
            ("accuracy".to_string(), Some(0.9)),
            ("latency".to_string(), Some(0.1)),
        ]);
        // B: low accuracy, high latency (bad on both)
        let b_scores = HashMap::from([
            ("accuracy".to_string(), Some(0.7)),
            ("latency".to_string(), Some(0.3)),
        ]);

        // A should dominate B
        assert!(instance_dominates(&a_scores, &b_scores, &evaluators));
        // B should not dominate A
        assert!(!instance_dominates(&b_scores, &a_scores, &evaluators));
    }

    #[test]
    fn test_instance_dominates_tradeoff() {
        let evaluators = create_test_evaluators(&[("accuracy", "max"), ("latency", "min")]);

        // A: high accuracy, high latency
        let a_scores = HashMap::from([
            ("accuracy".to_string(), Some(0.9)),
            ("latency".to_string(), Some(0.5)),
        ]);
        // B: low accuracy, low latency
        let b_scores = HashMap::from([
            ("accuracy".to_string(), Some(0.7)),
            ("latency".to_string(), Some(0.1)),
        ]);

        // Neither should dominate (tradeoff)
        assert!(!instance_dominates(&a_scores, &b_scores, &evaluators));
        assert!(!instance_dominates(&b_scores, &a_scores, &evaluators));
    }

    #[test]
    fn test_global_dominates_basic() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // A dominates B on both datapoints
        let val_scores_map = HashMap::from([
            (
                "a".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.9))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.8))]),
                    ),
                ]),
            ),
            (
                "b".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.7))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.6))]),
                    ),
                ]),
            ),
        ]);

        assert!(global_dominates("a", "b", &val_scores_map, &evaluators));
        assert!(!global_dominates("b", "a", &val_scores_map, &evaluators));
    }

    #[test]
    fn test_global_dominates_no_domination() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // A better on dp1, B better on dp2 - no global dominance
        let val_scores_map = HashMap::from([
            (
                "a".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.9))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.6))]),
                    ),
                ]),
            ),
            (
                "b".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.7))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.8))]),
                    ),
                ]),
            ),
        ]);

        assert!(!global_dominates("a", "b", &val_scores_map, &evaluators));
        assert!(!global_dominates("b", "a", &val_scores_map, &evaluators));
    }

    #[test]
    fn test_find_non_dominated_variants_single() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        let instance_scores = vec![(
            "variant_a".to_string(),
            HashMap::from([("accuracy".to_string(), Some(0.9))]),
        )];

        let result = find_non_dominated_variants(&instance_scores, &evaluators);
        assert_eq!(result, vec!["variant_a"]);
    }

    #[test]
    fn test_find_non_dominated_variants_multiple() {
        let evaluators = create_test_evaluators(&[("accuracy", "max"), ("latency", "min")]);

        let instance_scores = vec![
            (
                "variant_a".to_string(),
                HashMap::from([
                    ("accuracy".to_string(), Some(0.9)),
                    ("latency".to_string(), Some(0.3)),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    ("accuracy".to_string(), Some(0.7)),
                    ("latency".to_string(), Some(0.1)),
                ]),
            ),
            (
                "variant_c".to_string(),
                HashMap::from([
                    ("accuracy".to_string(), Some(0.6)),
                    ("latency".to_string(), Some(0.5)),
                ]),
            ),
        ];

        let result = find_non_dominated_variants(&instance_scores, &evaluators);

        // A and B are non-dominated (tradeoff), C is dominated by both
        assert_eq!(result.len(), 2);
        assert!(result.contains(&"variant_a".to_string()));
        assert!(result.contains(&"variant_b".to_string()));
        assert!(!result.contains(&"variant_c".to_string()));
    }

    // ============================================================================
    // Unit Tests for is_improvement
    // ============================================================================

    #[test]
    fn test_is_improvement_basic() {
        // Child better on single metric (max)
        let parent_stats = create_evaluator_stats(&[("accuracy", 0.7, 0.1, 10)]);
        let child_stats = create_evaluator_stats(&[("accuracy", 0.9, 0.1, 10)]);
        let eval_config = create_evaluation_config(&[("accuracy", "max")]);

        assert!(is_improvement(&parent_stats, &child_stats, &eval_config)); // Child is better
    }

    #[test]
    fn test_is_improvement_worse() {
        // Child worse on single metric (max)
        let parent_stats = create_evaluator_stats(&[("accuracy", 0.9, 0.1, 10)]);
        let child_stats = create_evaluator_stats(&[("accuracy", 0.7, 0.1, 10)]);
        let eval_config = create_evaluation_config(&[("accuracy", "max")]);

        assert!(!is_improvement(&parent_stats, &child_stats, &eval_config)); // Child is worse
    }

    #[test]
    fn test_is_improvement_equal() {
        // Child equal on all metrics - no strict improvement
        let parent_stats = create_evaluator_stats(&[("accuracy", 0.8, 0.1, 10)]);
        let child_stats = create_evaluator_stats(&[("accuracy", 0.8, 0.1, 10)]);
        let eval_config = create_evaluation_config(&[("accuracy", "max")]);

        assert!(!is_improvement(&parent_stats, &child_stats, &eval_config)); // Equal is not improvement
    }

    #[test]
    fn test_is_improvement_pareto_dominant() {
        // Child better on one metric, equal on another - should be Pareto improvement
        let parent_stats =
            create_evaluator_stats(&[("accuracy", 0.8, 0.1, 10), ("f1", 0.7, 0.1, 10)]);
        let child_stats =
            create_evaluator_stats(&[("accuracy", 0.9, 0.1, 10), ("f1", 0.7, 0.1, 10)]);
        let eval_config = create_evaluation_config(&[("accuracy", "max"), ("f1", "max")]);

        assert!(is_improvement(&parent_stats, &child_stats, &eval_config)); // Better on one, equal on other = improvement
    }

    #[test]
    fn test_is_improvement_not_pareto_dominant() {
        // Child better on one metric, worse on another - not Pareto dominant
        let parent_stats =
            create_evaluator_stats(&[("accuracy", 0.9, 0.1, 10), ("f1", 0.7, 0.1, 10)]);
        let child_stats =
            create_evaluator_stats(&[("accuracy", 0.8, 0.1, 10), ("f1", 0.9, 0.1, 10)]);
        let eval_config = create_evaluation_config(&[("accuracy", "max"), ("f1", "max")]);

        assert!(!is_improvement(&parent_stats, &child_stats, &eval_config)); // Tradeoff - not an improvement
    }

    #[test]
    fn test_is_improvement_mixed_directions() {
        // Test with both max and min optimization directions
        // Child: higher accuracy (max, good), lower latency (min, good)
        let parent_stats =
            create_evaluator_stats(&[("accuracy", 0.7, 0.1, 10), ("latency", 0.5, 0.1, 10)]);
        let child_stats =
            create_evaluator_stats(&[("accuracy", 0.9, 0.1, 10), ("latency", 0.3, 0.1, 10)]);
        let eval_config = create_evaluation_config(&[("accuracy", "max"), ("latency", "min")]);

        assert!(is_improvement(&parent_stats, &child_stats, &eval_config)); // Better on both metrics
    }

    #[test]
    fn test_is_improvement_missing_stats() {
        // Child missing stats for one evaluator - should skip that metric
        let parent_stats = create_evaluator_stats(&[("accuracy", 0.8, 0.1, 10)]);
        let child_stats = create_evaluator_stats(&[("accuracy", 0.9, 0.1, 10)]);
        let eval_config = create_evaluation_config(&[("accuracy", "max"), ("f1", "max")]);

        assert!(is_improvement(&parent_stats, &child_stats, &eval_config)); // Better on accuracy, f1 skipped (missing)
    }

    #[test]
    fn test_is_improvement_multiple_evaluators() {
        // Child improves on 2/3 metrics, equal on 1
        let parent_stats = create_evaluator_stats(&[
            ("accuracy", 0.7, 0.1, 10),
            ("f1", 0.6, 0.1, 10),
            ("precision", 0.8, 0.1, 10),
        ]);
        let child_stats = create_evaluator_stats(&[
            ("accuracy", 0.9, 0.1, 10),
            ("f1", 0.8, 0.1, 10),
            ("precision", 0.8, 0.1, 10),
        ]);
        let eval_config =
            create_evaluation_config(&[("accuracy", "max"), ("f1", "max"), ("precision", "max")]);

        assert!(is_improvement(&parent_stats, &child_stats, &eval_config)); // Better on 2, equal on 1 = improvement
    }

    #[test]
    fn test_is_improvement_min_optimization() {
        // Test with minimize optimization - child has lower value (better)
        let parent_stats = create_evaluator_stats(&[("latency", 0.9, 0.1, 10)]);
        let child_stats = create_evaluator_stats(&[("latency", 0.3, 0.1, 10)]);
        let eval_config = create_evaluation_config(&[("latency", "min")]);

        assert!(is_improvement(&parent_stats, &child_stats, &eval_config)); // Lower latency = better for min
    }

    // ============================================================================
    // Integration Tests - Python Equivalents
    // ============================================================================

    #[test]
    fn test_basic_dominance() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        // Variant A dominates B: A has higher scores on all datapoints
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.9))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.8))]),
                    ),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.7))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.6))]),
                    ),
                ]),
            ),
        ]);

        let val_scores_map = variant_scores;
        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &val_scores_map, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // Only variant_a should remain
        assert_eq!(frontier.variants.len(), 1);
        assert!(frontier.variants.contains_key("variant_a"));
        assert!(!frontier.variants.contains_key("variant_b"));

        // variant_a should be in Pareto set for both datapoints
        assert_eq!(frontier.frequencies.get("variant_a"), Some(&2));
    }

    #[test]
    fn test_pareto_frontier() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        // A is better on dp1, B is better on dp2 - neither dominates
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.9))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.6))]),
                    ),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.7))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.8))]),
                    ),
                ]),
            ),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // Both should remain
        assert_eq!(frontier.variants.len(), 2);
        assert!(frontier.variants.contains_key("variant_a"));
        assert!(frontier.variants.contains_key("variant_b"));

        // Each variant is Pareto-optimal on one datapoint
        assert_eq!(frontier.frequencies.get("variant_a"), Some(&1));
        assert_eq!(frontier.frequencies.get("variant_b"), Some(&1));
    }

    #[test]
    fn test_instance_wise_vs_global() {
        let config = create_evaluation_config(&[("acc", "max"), ("f1", "max")]);

        // C is never instance-wise Pareto-optimal, so it should be filtered early
        // A and B are each optimal on different instances
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([
                            ("acc".to_string(), Some(0.9)),
                            ("f1".to_string(), Some(0.8)),
                        ]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([
                            ("acc".to_string(), Some(0.6)),
                            ("f1".to_string(), Some(0.7)),
                        ]),
                    ),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([
                            ("acc".to_string(), Some(0.7)),
                            ("f1".to_string(), Some(0.9)),
                        ]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([
                            ("acc".to_string(), Some(0.8)),
                            ("f1".to_string(), Some(0.6)),
                        ]),
                    ),
                ]),
            ),
            (
                "variant_c".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([
                            ("acc".to_string(), Some(0.5)),
                            ("f1".to_string(), Some(0.5)),
                        ]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([
                            ("acc".to_string(), Some(0.5)),
                            ("f1".to_string(), Some(0.5)),
                        ]),
                    ),
                ]),
            ),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b", "variant_c"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // C should be filtered out, A and B should remain
        assert!(!frontier.variants.contains_key("variant_c"));
        assert!(frontier.variants.contains_key("variant_a"));
        assert!(frontier.variants.contains_key("variant_b"));
    }

    #[test]
    fn test_mixed_optimize_directions() {
        let config = create_evaluation_config(&[("accuracy", "max"), ("latency", "min")]);

        // A has higher accuracy (max) and lower latency (min) - dominates B
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([(
                    "dp1".to_string(),
                    HashMap::from([
                        ("accuracy".to_string(), Some(0.9)),
                        ("latency".to_string(), Some(0.1)),
                    ]),
                )]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([(
                    "dp1".to_string(),
                    HashMap::from([
                        ("accuracy".to_string(), Some(0.7)),
                        ("latency".to_string(), Some(0.3)),
                    ]),
                )]),
            ),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // Only variant_a should remain
        assert_eq!(frontier.variants.len(), 1);
        assert!(frontier.variants.contains_key("variant_a"));
    }

    #[test]
    fn test_none_values() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        // A has None on dp1, B has None on dp2 - they're incomparable
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), None)]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.8))]),
                    ),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.7))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), None)]),
                    ),
                ]),
            ),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // Both should remain (incomparable due to None values)
        assert_eq!(frontier.variants.len(), 2);
        assert!(frontier.variants.contains_key("variant_a"));
        assert!(frontier.variants.contains_key("variant_b"));
    }

    #[test]
    fn test_single_variant() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        let variant_scores = HashMap::from([(
            "variant_a".to_string(),
            HashMap::from([(
                "dp1".to_string(),
                HashMap::from([("accuracy".to_string(), Some(0.9))]),
            )]),
        )]);

        let candidates = create_test_variants(&["variant_a"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // Single variant should be kept
        assert_eq!(frontier.variants.len(), 1);
        assert!(frontier.variants.contains_key("variant_a"));
        assert_eq!(frontier.frequencies.get("variant_a"), Some(&1));
    }

    #[test]
    fn test_error_results_ignored() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        // B has an error on dp1 (represented as missing datapoint)
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.9))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.8))]),
                    ),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([(
                    "dp2".to_string(),
                    HashMap::from([("accuracy".to_string(), Some(0.6))]),
                )]),
            ),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // A dominates B globally: Pareto-optimal on dp1 (only variant) and dp2 (0.8 > 0.6)
        assert_eq!(frontier.variants.len(), 1);
        assert!(frontier.variants.contains_key("variant_a"));
    }

    #[test]
    fn test_all_equal() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([(
                    "dp1".to_string(),
                    HashMap::from([("accuracy".to_string(), Some(0.8))]),
                )]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([(
                    "dp1".to_string(),
                    HashMap::from([("accuracy".to_string(), Some(0.8))]),
                )]),
            ),
            (
                "variant_c".to_string(),
                HashMap::from([(
                    "dp1".to_string(),
                    HashMap::from([("accuracy".to_string(), Some(0.8))]),
                )]),
            ),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b", "variant_c"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // All should remain (no one dominates)
        assert_eq!(frontier.variants.len(), 3);
        assert!(frontier.variants.contains_key("variant_a"));
        assert!(frontier.variants.contains_key("variant_b"));
        assert!(frontier.variants.contains_key("variant_c"));
    }

    #[test]
    fn test_incomparable_variants_different_datapoints() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        // Test scenario: variants evaluated on different datapoints
        // This tests the edge case where GEPA's datapoint collection from first variant matters
        // Both variants have data on both datapoints, but missing (None) on different ones
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.9))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), None)]),
                    ),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), None)]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.7))]),
                    ),
                ]),
            ),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // Both variants should remain because:
        // - On dp1: variant_a (0.9) dominates variant_b (None=-inf), so A is Pareto-optimal
        // - On dp2: variant_b (0.7) dominates variant_a (None=-inf), so B is Pareto-optimal
        // - Candidate set = {A, B}
        // - Global filtering: Neither globally dominates the other (A better on dp1, B better on dp2)
        assert_eq!(frontier.variants.len(), 2);
        assert!(frontier.variants.contains_key("variant_a"));
        assert!(frontier.variants.contains_key("variant_b"));
    }

    #[test]
    fn test_boolean_evaluator() {
        let config = create_evaluation_config(&[("exact_match", "max")]);

        // A has True (1.0), B has False (0.0) for bool evaluator
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([(
                    "dp1".to_string(),
                    HashMap::from([("exact_match".to_string(), Some(1.0))]),
                )]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([(
                    "dp1".to_string(),
                    HashMap::from([("exact_match".to_string(), Some(0.0))]),
                )]),
            ),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // A should dominate B (1.0 > 0.0)
        assert_eq!(frontier.variants.len(), 1);
        assert!(frontier.variants.contains_key("variant_a"));
    }

    #[test]
    fn test_three_way_dominance() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([(
                    "dp1".to_string(),
                    HashMap::from([("accuracy".to_string(), Some(0.9))]),
                )]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([(
                    "dp1".to_string(),
                    HashMap::from([("accuracy".to_string(), Some(0.7))]),
                )]),
            ),
            (
                "variant_c".to_string(),
                HashMap::from([(
                    "dp1".to_string(),
                    HashMap::from([("accuracy".to_string(), Some(0.5))]),
                )]),
            ),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b", "variant_c"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // Only A should remain
        assert_eq!(frontier.variants.len(), 1);
        assert!(frontier.variants.contains_key("variant_a"));
    }

    #[test]
    fn test_runtime_performance() {
        use std::time::Instant;

        let config = create_evaluation_config(&[("accuracy", "max")]);

        let num_variants = 10;
        let num_datapoints = 100;

        // Generate variant results with random scores
        let mut variant_scores = HashMap::new();
        for v in 0..num_variants {
            let variant_name = format!("variant_{v}");
            let mut datapoint_scores = HashMap::new();

            for d in 0..num_datapoints {
                let datapoint_id = format!("dp_{d}");
                // Use deterministic "random" values for reproducibility
                let score = ((v * 17 + d * 31) % 100) as f32 / 100.0;
                datapoint_scores.insert(
                    datapoint_id,
                    HashMap::from([("accuracy".to_string(), Some(score))]),
                );
            }

            variant_scores.insert(variant_name, datapoint_scores);
        }

        let val_scores_map = variant_scores;
        let variant_names: Vec<String> =
            (0..num_variants).map(|v| format!("variant_{v}")).collect();
        let variant_name_refs: Vec<&str> = variant_names.iter().map(|s| s.as_str()).collect();
        let candidates = create_test_variants(&variant_name_refs);

        // Measure execution time
        let start = Instant::now();
        let result = update_pareto_frontier(candidates, &val_scores_map, &config);
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        let frontier = result.unwrap();

        // Should complete in reasonable time (< 5 seconds)
        assert!(
            elapsed.as_secs() < 5,
            "Filtering took too long: {:.4}s",
            elapsed.as_secs_f64()
        );

        // Result should be non-empty
        assert!(
            !frontier.variants.is_empty(),
            "Result should contain at least one non-dominated variant"
        );
    }

    // ============================================================================
    // Rust-Specific Edge Cases
    // ============================================================================

    #[test]
    fn test_empty_candidates() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        let val_scores_map: HashMap<String, HashMap<String, HashMap<String, Option<f32>>>> =
            HashMap::new();
        let candidates: HashMap<String, UninitializedChatCompletionConfig> = HashMap::new();

        let result = update_pareto_frontier(candidates, &val_scores_map, &config);

        // Should return error for empty candidates
        assert!(result.is_err());
    }

    #[test]
    fn test_all_evaluations_failed() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        // All val_scores entries are None (evaluation failed)
        let val_scores_map: HashMap<String, HashMap<String, HashMap<String, Option<f32>>>> =
            HashMap::new();
        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &val_scores_map, &config);

        // Should return error when all evaluations failed
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error
            .to_string()
            .contains("No validation scores provided for any variant"));
    }

    #[test]
    fn test_all_scores_none() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        // Every per_datapoint score is None
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), None)]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), None)]),
                    ),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), None)]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), None)]),
                    ),
                ]),
            ),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);

        // Should return error when all scores are None
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error
            .to_string()
            .contains("All variants failed to produce valid scores"));
    }

    #[test]
    fn test_empty_datapoints() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        // Empty per_datapoint HashMap
        let variant_scores = HashMap::from([
            ("variant_a".to_string(), HashMap::new()),
            ("variant_b".to_string(), HashMap::new()),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // With no datapoints, no instance-wise Pareto sets can be formed
        // so candidate_set is empty and all variants are filtered out
        assert_eq!(frontier.variants.len(), 0);
        // No variants remain, so frequencies should be empty
        assert!(frontier.frequencies.is_empty());
    }

    #[test]
    fn test_frequency_calculation_detailed() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        // A wins on dp1 and dp2, B wins on dp3, C never wins
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.9))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.9))]),
                    ),
                    (
                        "dp3".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.5))]),
                    ),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.7))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.7))]),
                    ),
                    (
                        "dp3".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.9))]),
                    ),
                ]),
            ),
            (
                "variant_c".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.5))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.5))]),
                    ),
                    (
                        "dp3".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.5))]),
                    ),
                ]),
            ),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b", "variant_c"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();

        // C is dominated, A and B remain
        assert_eq!(frontier.variants.len(), 2);
        assert!(frontier.variants.contains_key("variant_a"));
        assert!(frontier.variants.contains_key("variant_b"));

        // A should be in Pareto set for dp1 and dp2 (freq=2)
        assert_eq!(frontier.frequencies.get("variant_a"), Some(&2));
        // B should be in Pareto set for dp3 (freq=1)
        assert_eq!(frontier.frequencies.get("variant_b"), Some(&1));
        // C should not be in frequencies (filtered out)
        assert!(!frontier.frequencies.contains_key("variant_c"));
    }

    #[test]
    fn test_high_missing_rate_warning() {
        // This test verifies that the warning log is triggered for high missing rates
        let config = create_evaluation_config(&[("accuracy", "max")]);

        // variant_a has 60% missing data (3 out of 5 datapoints have scores)
        let variant_scores = HashMap::from([(
            "variant_a".to_string(),
            HashMap::from([
                (
                    "dp1".to_string(),
                    HashMap::from([("accuracy".to_string(), Some(0.9))]),
                ),
                (
                    "dp2".to_string(),
                    HashMap::from([("accuracy".to_string(), Some(0.9))]),
                ),
                (
                    "dp3".to_string(),
                    HashMap::from([("accuracy".to_string(), None)]),
                ),
                (
                    "dp4".to_string(),
                    HashMap::from([("accuracy".to_string(), None)]),
                ),
                (
                    "dp5".to_string(),
                    HashMap::from([("accuracy".to_string(), None)]),
                ),
            ]),
        )]);

        let candidates = create_test_variants(&["variant_a"]);

        let result = update_pareto_frontier(candidates, &variant_scores, &config);
        assert!(result.is_ok());

        let frontier = result.unwrap();
        assert_eq!(frontier.variants.len(), 1);

        // Note: We can't directly assert on log output, but this test documents
        // the behavior and ensures the code path executes without panicking
        // The warning should be logged: "Variant 'variant_a' has high missing score rate: 60.0%"
    }
}
