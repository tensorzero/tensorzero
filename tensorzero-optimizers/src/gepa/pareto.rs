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
    evaluations::{EvaluationConfig, EvaluatorConfig},
    optimization::gepa::GEPAConfig,
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
/// Filter candidates using GEPA's instance-wise Pareto frontier algorithm
///
/// Implements the SELECTCANDIDATE algorithm from the GEPA paper:
/// 1. For each datapoint, find instance-wise Pareto-optimal variants (Step 1)
/// 2. Build candidate set C as union of all instance-wise Pareto sets (Step 2)
/// 3. Filter candidates globally to remove dominated variants (Step 3)
/// 4. Compute frequency of each variant's membership in instance-wise Pareto sets
///
/// Note: The original GEPA paper uses a single evaluator and selects variants achieving
/// maximum score per instance. We extend this to multiple evaluators by selecting Pareto
/// non-dominated variants per instance, which naturally generalizes the single-objective case.
///
/// Returns (filtered variants, frequency map) where frequencies are used for weighted sampling.
#[expect(clippy::type_complexity)]
pub fn update_pareto_frontier(
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
        EvaluationConfig::Inference(inference_config) => &inference_config.evaluators,
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
    // Original paper: P*[i] = {variants with max score on instance i} (single evaluator)
    // Our extension: P*[i] = {Pareto non-dominated variants on instance i} (multiple evaluators)
    let mut instance_pareto_sets: HashMap<String, HashSet<String>> = HashMap::new();

    for datapoint_id in &datapoint_ids {
        // Collect scores for this datapoint across all variants
        let mut instance_scores: Vec<(String, HashMap<String, Option<f32>>)> = Vec::new();

        for (variant_name, evaluation_results) in &valid_scores {
            if let Some(scores) = evaluation_results.per_datapoint.get(datapoint_id) {
                instance_scores.push((variant_name.to_string(), scores.clone()));
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
                let non_none_count = datapoint_ids
                    .iter()
                    .filter_map(|datapoint_id| evaluation_results.per_datapoint.get(datapoint_id))
                    .flat_map(|scores| {
                        evaluators.keys().filter_map(move |evaluator_name| {
                            scores.get(evaluator_name).and_then(|s| s.as_ref())
                        })
                    })
                    .count();

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
    let frequencies = calculate_frequencies(&non_dominated, &instance_pareto_sets);

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
/// Rationale: Aggressive imputation penalizes variants that systematically fail (e.g., causing
/// inference errors) while treating random failures (e.g., provider unavailable) equally across variants.
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

/// Compare two values according to optimization direction
///
/// Returns (is_worse, is_better) where:
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
pub fn global_dominates(
    variant_a_results: &EvaluationResults,
    variant_b_results: &EvaluationResults,
    evaluators: &HashMap<String, EvaluatorConfig>,
) -> bool {
    let mut better_or_equal_on_all = true;
    let mut strictly_better_on_at_least_one = false;

    // Get all (datapoint_id, evaluator_name) pairs from both variants
    let all_pairs: HashSet<_> = variant_a_results
        .per_datapoint
        .keys()
        .chain(variant_b_results.per_datapoint.keys())
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
pub fn instance_dominates(
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
pub fn find_non_dominated_variants(
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
        EvaluationConfig::Inference(inference_config) => &inference_config.evaluators,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::gepa::evaluate::EvaluationResults;
    use evaluations::EvaluatorStats;
    use tensorzero_core::evaluations::{
        EvaluationConfig, EvaluatorConfig, ExactMatchConfig, InferenceEvaluationConfig,
        LLMJudgeConfig, LLMJudgeIncludeConfig, LLMJudgeInputFormat, LLMJudgeOptimize,
        LLMJudgeOutputType,
    };

    // ============================================================================
    // Test Helper Functions
    // ============================================================================

    /// Create a HashMap of evaluator configs with specified optimize directions
    ///
    /// # Arguments
    /// * `evaluators` - Slice of (evaluator_name, optimize_direction) tuples
    ///   where optimize_direction is "max" or "min"
    fn create_test_evaluators(evaluators: &[(&str, &str)]) -> HashMap<String, EvaluatorConfig> {
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

    /// Create a Config with an evaluation containing the specified evaluators
    fn create_test_config(evaluation_name: &str, evaluators: &[(&str, &str)]) -> Config {
        let evaluator_configs = create_test_evaluators(evaluators);
        let mut config = Config::default();

        let eval_config = InferenceEvaluationConfig {
            evaluators: evaluator_configs,
            function_name: "test_function".to_string(),
        };

        config.evaluations.insert(
            evaluation_name.to_string(),
            Arc::new(EvaluationConfig::Inference(eval_config)),
        );

        config
    }

    /// Create a GEPAConfig for testing
    fn create_test_gepa_config(
        evaluation_name: &str,
    ) -> tensorzero_core::optimization::gepa::GEPAConfig {
        tensorzero_core::optimization::gepa::GEPAConfig {
            function_name: "test_function".to_string(),
            evaluation_name: evaluation_name.to_string(),
            initial_variants: None,
            variant_prefix: Some("test".to_string()),
            batch_size: 5,
            max_iterations: 1,
            max_concurrency: 10,
            analysis_model: "openai::gpt-5-mini".to_string(),
            mutation_model: "openai::gpt-5".to_string(),
            seed: Some(42),
            timeout: 300,
        }
    }

    /// Create EvaluationResults from per_datapoint scores
    ///
    /// # Arguments
    /// * `per_datapoint` - HashMap mapping datapoint_id -> (evaluator_name -> score)
    fn create_test_evaluation_results(
        per_datapoint: HashMap<String, HashMap<String, Option<f32>>>,
    ) -> EvaluationResults {
        // Calculate metrics stats from per_datapoint data
        let mut metrics: HashMap<String, EvaluatorStats> = HashMap::new();

        // Collect all evaluator names
        let evaluator_names: HashSet<String> = per_datapoint
            .values()
            .flat_map(|scores| scores.keys().cloned())
            .collect();

        for evaluator_name in evaluator_names {
            let values: Vec<f32> = per_datapoint
                .values()
                .filter_map(|scores| scores.get(&evaluator_name).and_then(|s| *s))
                .collect();

            if !values.is_empty() {
                let count = values.len();
                let mean = values.iter().sum::<f32>() / count as f32;

                // Calculate stderr (standard error of the mean)
                // This matches the calculation in evaluations/src/stats.rs:120-122
                let stderr = if count > 1 {
                    let sum_sq_diff: f32 = values.iter().map(|v| (v - mean).powi(2)).sum();
                    let variance = sum_sq_diff / (count - 1) as f32;
                    let std_dev = variance.sqrt();
                    std_dev / (count as f32).sqrt()
                } else {
                    0.0
                };

                metrics.insert(
                    evaluator_name,
                    EvaluatorStats {
                        mean,
                        stderr,
                        count,
                    },
                );
            }
        }

        EvaluationResults {
            per_datapoint,
            metrics,
        }
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

    /// Create val_scores HashMap from nested score data
    ///
    /// # Arguments
    /// * `variant_scores` - HashMap mapping variant_name -> datapoint_id -> (evaluator_name -> score)
    fn create_test_val_scores(
        variant_scores: HashMap<String, HashMap<String, HashMap<String, Option<f32>>>>,
    ) -> HashMap<String, Option<EvaluationResults>> {
        variant_scores
            .into_iter()
            .map(|(variant_name, per_datapoint)| {
                (
                    variant_name,
                    Some(create_test_evaluation_results(per_datapoint)),
                )
            })
            .collect()
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
        let a_results = create_test_evaluation_results(HashMap::from([
            (
                "dp1".to_string(),
                HashMap::from([("accuracy".to_string(), Some(0.9))]),
            ),
            (
                "dp2".to_string(),
                HashMap::from([("accuracy".to_string(), Some(0.8))]),
            ),
        ]));

        let b_results = create_test_evaluation_results(HashMap::from([
            (
                "dp1".to_string(),
                HashMap::from([("accuracy".to_string(), Some(0.7))]),
            ),
            (
                "dp2".to_string(),
                HashMap::from([("accuracy".to_string(), Some(0.6))]),
            ),
        ]));

        assert!(global_dominates(&a_results, &b_results, &evaluators));
        assert!(!global_dominates(&b_results, &a_results, &evaluators));
    }

    #[test]
    fn test_global_dominates_no_domination() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // A better on dp1, B better on dp2 - no global dominance
        let a_results = create_test_evaluation_results(HashMap::from([
            (
                "dp1".to_string(),
                HashMap::from([("accuracy".to_string(), Some(0.9))]),
            ),
            (
                "dp2".to_string(),
                HashMap::from([("accuracy".to_string(), Some(0.6))]),
            ),
        ]));

        let b_results = create_test_evaluation_results(HashMap::from([
            (
                "dp1".to_string(),
                HashMap::from([("accuracy".to_string(), Some(0.7))]),
            ),
            (
                "dp2".to_string(),
                HashMap::from([("accuracy".to_string(), Some(0.8))]),
            ),
        ]));

        assert!(!global_dominates(&a_results, &b_results, &evaluators));
        assert!(!global_dominates(&b_results, &a_results, &evaluators));
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
    // Integration Tests - Python Equivalents
    // ============================================================================

    #[test]
    fn test_basic_dominance() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // Variant A dominates B: A has higher scores on all datapoints
        let val_scores = create_test_val_scores(HashMap::from([
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
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, frequencies) = result.unwrap();

        // Only variant_a should remain
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("variant_a"));
        assert!(!filtered.contains_key("variant_b"));

        // variant_a should be in Pareto set for both datapoints
        assert_eq!(frequencies.get("variant_a"), Some(&2));
    }

    #[test]
    fn test_pareto_frontier() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // A is better on dp1, B is better on dp2 - neither dominates
        let val_scores = create_test_val_scores(HashMap::from([
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
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, frequencies) = result.unwrap();

        // Both should remain
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains_key("variant_a"));
        assert!(filtered.contains_key("variant_b"));

        // Each variant is Pareto-optimal on one datapoint
        assert_eq!(frequencies.get("variant_a"), Some(&1));
        assert_eq!(frequencies.get("variant_b"), Some(&1));
    }

    #[test]
    fn test_instance_wise_vs_global() {
        let config = create_test_config("test_eval", &[("acc", "max"), ("f1", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // C is never instance-wise Pareto-optimal, so it should be filtered early
        // A and B are each optimal on different instances
        let val_scores = create_test_val_scores(HashMap::from([
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
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b", "variant_c"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, _) = result.unwrap();

        // C should be filtered out, A and B should remain
        assert!(!filtered.contains_key("variant_c"));
        assert!(filtered.contains_key("variant_a"));
        assert!(filtered.contains_key("variant_b"));
    }

    #[test]
    fn test_mixed_optimize_directions() {
        let config = create_test_config("test_eval", &[("accuracy", "max"), ("latency", "min")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // A has higher accuracy (max) and lower latency (min) - dominates B
        let val_scores = create_test_val_scores(HashMap::from([
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
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, _) = result.unwrap();

        // Only variant_a should remain
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("variant_a"));
    }

    #[test]
    fn test_none_values() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // A has None on dp1, B has None on dp2 - they're incomparable
        let val_scores = create_test_val_scores(HashMap::from([
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
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, _) = result.unwrap();

        // Both should remain (incomparable due to None values)
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains_key("variant_a"));
        assert!(filtered.contains_key("variant_b"));
    }

    #[test]
    fn test_single_variant() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        let val_scores = create_test_val_scores(HashMap::from([(
            "variant_a".to_string(),
            HashMap::from([(
                "dp1".to_string(),
                HashMap::from([("accuracy".to_string(), Some(0.9))]),
            )]),
        )]));

        let candidates = create_test_variants(&["variant_a"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, frequencies) = result.unwrap();

        // Single variant should be kept
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("variant_a"));
        assert_eq!(frequencies.get("variant_a"), Some(&1));
    }

    #[test]
    fn test_error_results_ignored() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // B has an error on dp1 (represented as missing datapoint)
        let val_scores = create_test_val_scores(HashMap::from([
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
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, _) = result.unwrap();

        // A dominates B on dp2 (only comparable point)
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("variant_a"));
    }

    #[test]
    fn test_all_equal() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        let val_scores = create_test_val_scores(HashMap::from([
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
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b", "variant_c"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, _) = result.unwrap();

        // All should remain (no one dominates)
        assert_eq!(filtered.len(), 3);
        assert!(filtered.contains_key("variant_a"));
        assert!(filtered.contains_key("variant_b"));
        assert!(filtered.contains_key("variant_c"));
    }

    #[test]
    fn test_incomparable_variants_different_datapoints() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // Test scenario: variants evaluated on different datapoints
        // This tests the edge case where GEPA's datapoint collection from first variant matters
        // Both variants have data on both datapoints, but missing (None) on different ones
        let val_scores = create_test_val_scores(HashMap::from([
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
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, _) = result.unwrap();

        // Both variants should remain because:
        // - On dp1: variant_a (0.9) dominates variant_b (None=-inf), so A is Pareto-optimal
        // - On dp2: variant_b (0.7) dominates variant_a (None=-inf), so B is Pareto-optimal
        // - Candidate set = {A, B}
        // - Global filtering: Neither globally dominates the other (A better on dp1, B better on dp2)
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains_key("variant_a"));
        assert!(filtered.contains_key("variant_b"));
    }

    #[test]
    fn test_boolean_evaluator() {
        let config = create_test_config("test_eval", &[("exact_match", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // A has True (1.0), B has False (0.0) for bool evaluator
        let val_scores = create_test_val_scores(HashMap::from([
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
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, _) = result.unwrap();

        // A should dominate B (1.0 > 0.0)
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("variant_a"));
    }

    #[test]
    fn test_three_way_dominance() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        let val_scores = create_test_val_scores(HashMap::from([
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
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b", "variant_c"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, _) = result.unwrap();

        // Only A should remain
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("variant_a"));
    }

    #[test]
    fn test_runtime_performance() {
        use std::time::Instant;

        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

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

        let val_scores = create_test_val_scores(variant_scores);
        let variant_names: Vec<&str> = (0..num_variants)
            .map(|v| Box::leak(format!("variant_{v}").into_boxed_str()) as &str)
            .collect();
        let candidates = create_test_variants(&variant_names);

        // Measure execution time
        let start = Instant::now();
        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        let (filtered, _) = result.unwrap();

        // Should complete in reasonable time (< 5 seconds)
        assert!(
            elapsed.as_secs() < 5,
            "Filtering took too long: {:.4}s",
            elapsed.as_secs_f64()
        );

        // Result should be non-empty
        assert!(
            !filtered.is_empty(),
            "Result should contain at least one non-dominated variant"
        );
    }

    // ============================================================================
    // Rust-Specific Edge Cases
    // ============================================================================

    #[test]
    fn test_empty_candidates() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        let val_scores: HashMap<String, Option<EvaluationResults>> = HashMap::new();
        let candidates: HashMap<String, UninitializedChatCompletionConfig> = HashMap::new();

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);

        // Should return error for empty candidates
        assert!(result.is_err());
    }

    #[test]
    fn test_all_evaluations_failed() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // All val_scores entries are None (evaluation failed)
        let val_scores: HashMap<String, Option<EvaluationResults>> = HashMap::from([
            ("variant_a".to_string(), None),
            ("variant_b".to_string(), None),
        ]);

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);

        // Should return error when all evaluations failed
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("All variants failed evaluation"));
    }

    #[test]
    fn test_all_scores_none() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // Every per_datapoint score is None
        let val_scores = create_test_val_scores(HashMap::from([
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
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, _) = result.unwrap();

        // Both variants have all None - neither dominates
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_empty_datapoints() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // Empty per_datapoint HashMap
        let val_scores = create_test_val_scores(HashMap::from([
            ("variant_a".to_string(), HashMap::new()),
            ("variant_b".to_string(), HashMap::new()),
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, frequencies) = result.unwrap();

        // With no datapoints, no instance-wise Pareto sets can be formed
        // so candidate_set is empty and all variants are filtered out
        assert_eq!(filtered.len(), 0);
        // No variants remain, so frequencies should be empty
        assert!(frequencies.is_empty());
    }

    #[test]
    fn test_frequency_calculation_detailed() {
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // A wins on dp1 and dp2, B wins on dp3, C never wins
        let val_scores = create_test_val_scores(HashMap::from([
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
        ]));

        let candidates = create_test_variants(&["variant_a", "variant_b", "variant_c"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, frequencies) = result.unwrap();

        // C is dominated, A and B remain
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains_key("variant_a"));
        assert!(filtered.contains_key("variant_b"));

        // A should be in Pareto set for dp1 and dp2 (freq=2)
        assert_eq!(frequencies.get("variant_a"), Some(&2));
        // B should be in Pareto set for dp3 (freq=1)
        assert_eq!(frequencies.get("variant_b"), Some(&1));
        // C should not be in frequencies (filtered out)
        assert!(!frequencies.contains_key("variant_c"));
    }

    #[test]
    fn test_high_missing_rate_warning() {
        // This test verifies that the warning log is triggered for high missing rates
        let config = create_test_config("test_eval", &[("accuracy", "max")]);
        let gepa_config = create_test_gepa_config("test_eval");

        // variant_a has 60% missing data (3 out of 5 datapoints have scores)
        let val_scores = create_test_val_scores(HashMap::from([(
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
        )]));

        let candidates = create_test_variants(&["variant_a"]);

        let result = update_pareto_frontier(candidates, &val_scores, &gepa_config, &config);
        assert!(result.is_ok());

        let (filtered, _) = result.unwrap();
        assert_eq!(filtered.len(), 1);

        // Note: We can't directly assert on log output, but this test documents
        // the behavior and ensures the code path executes without panicking
        // The warning should be logged: "Variant 'variant_a' has high missing score rate: 60.0%"
    }
}
