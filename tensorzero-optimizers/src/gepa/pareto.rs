//! Pareto frontier analysis and dominance checking for GEPA
//!
//! This module implements the core Pareto optimality logic for multi-objective optimization:
//! - Instance-wise Pareto frontier computation
//! - Global Pareto dominance checking
//! - Missing data imputation

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use evaluations::EvaluatorStats;
use tensorzero_core::{
    config::MetricConfigOptimize,
    error::{Error, ErrorDetails},
    evaluations::{EvaluatorConfig, InferenceEvaluationConfig},
    variant::chat_completion::UninitializedChatCompletionConfig,
};

// Type aliases for cleaner score map signatures (TODO(#4669): move to evaluation module)
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
///
/// This struct maintains all state needed for iterative Pareto frontier updates,
/// including validation scores, cached objective vectors, and layout fingerprints
/// (datapoint IDs + metric optimization directions) for cache validation.
#[derive(Debug)]
pub struct ParetoFrontier {
    /// Pareto-optimal variants (non-dominated after GEPA's 3-step filtering)
    pub variants: HashMap<String, UninitializedChatCompletionConfig>,

    /// Instance-wise membership frequencies for weighted sampling
    ///
    /// Maps each variant to the count of datapoints where it's Pareto-optimal.
    /// Higher values indicate the variant performs well across more instances.
    /// Used for proportional sampling in GEPA's mutation step.
    pub frequencies: HashMap<String, usize>,

    /// Validation scores for current variants (pruned after each update)
    ///
    /// Maps variant names to their scores on each datapoint/evaluator pair.
    /// Only contains entries for variants in `self.variants`.
    /// Private - internal implementation detail.
    val_scores: HashMap<String, VariantScores>,

    /// Cached objective vectors for efficient dominance checking
    ///
    /// Maps variant names to pre-computed flattened objective vectors.
    /// Vectors are keyed by variant name and validated against the layout fingerprint.
    /// Private - internal implementation detail for performance optimization.
    objective_vector_cache: HashMap<String, Arc<Vec<f32>>>,

    /// Datapoint IDs defining the objective space (layout fingerprint)
    ///
    /// This ordering must remain constant for cache correctness.
    /// Private to enforce validation through update methods.
    datapoint_ids: Vec<String>,

    /// Optimization directions for each metric (min or max)
    ///
    /// Maps evaluator names to their optimization direction.
    /// Private to enforce validation through update methods.
    optimize_directions: HashMap<String, MetricConfigOptimize>,
}

impl ParetoFrontier {
    /// Create a new empty Pareto frontier with the specified layout
    ///
    /// # Arguments
    /// * `datapoint_ids` - Ordered list of datapoint IDs defining the objective space
    /// * `evaluators` - Borrowed evaluator configurations (optimization directions will be extracted)
    ///
    /// # Returns
    /// * Empty ParetoFrontier ready for iterative updates
    pub fn new(datapoint_ids: Vec<String>, evaluators: &HashMap<String, EvaluatorConfig>) -> Self {
        let optimize_directions = extract_optimize_directions(evaluators);
        Self {
            variants: HashMap::new(),
            frequencies: HashMap::new(),
            val_scores: HashMap::new(),
            objective_vector_cache: HashMap::new(),
            datapoint_ids,
            optimize_directions,
        }
    }

    /// Get expected objective vector length for current layout
    ///
    /// Used for cache validation
    fn expected_vector_length(&self) -> usize {
        self.datapoint_ids.len() * self.optimize_directions.len()
    }

    /// Update the Pareto frontier with new variants (incremental merge)
    ///
    /// This method incrementally adds new variants to the existing frontier:
    /// 1. Checks for name collisions between new and existing variants (returns error if found)
    /// 2. Merges new variants with existing frontier variants
    /// 3. Runs GEPA's SELECTCANDIDATE algorithm on the combined set:
    ///    a. For each datapoint, find instance-wise Pareto-optimal variants
    ///    b. Build candidate set C as union of all instance-wise Pareto sets
    ///    c. Filter candidates globally to remove dominated variants
    /// 4. Updates frontier with filtered results and computes frequency weights
    ///
    /// This method validates that the provided validation scores match the frontier's
    /// layout (datapoint_ids and evaluators), updates the frontier in-place, and prunes
    /// validation scores to only contain current variants.
    ///
    /// # Arguments
    /// * `new_variants` - Newly generated variants (mutations) to add to frontier
    /// * `new_variant_scores` - Validation scores for the new variants
    ///
    /// # Returns
    /// * `Result<(), Error>` - Ok if successful, Err if scores are invalid, collision, or layout mismatch
    ///
    /// # Errors
    /// * Returns error if any new variant name already exists in the frontier
    /// * Returns error if new_variant_scores is empty or all scores are None
    /// * Returns error if new_variant_scores contains datapoints not in self.datapoint_ids
    pub fn update(
        &mut self,
        new_variants: HashMap<String, UninitializedChatCompletionConfig>,
        new_variant_scores: HashMap<String, VariantScores>,
    ) -> Result<(), Error> {
        tracing::info!(
            "Updating Pareto frontier: merging {} new variants with {} existing variants",
            new_variants.len(),
            self.variants.len()
        );

        // Step 1: Check for name collisions
        let collisions: Vec<&String> = new_variants
            .keys()
            .filter(|name| self.variants.contains_key(*name))
            .collect();

        if !collisions.is_empty() {
            let collision_list = if collisions.len() <= 5 {
                collisions
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            } else {
                format!(
                    "{} and {} more",
                    collisions[..5]
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                    collisions.len() - 5
                )
            };
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Variant name collision(s) detected: {collision_list}. New variants cannot have the same names as existing frontier variants."
                ),
            }));
        }

        // Step 2: Merge new variants with existing frontier variants
        let mut all_candidates = self.variants.clone();
        all_candidates.extend(new_variants);

        // Check if we have any valid scores (before merging to avoid unnecessary clone)
        if new_variant_scores.is_empty() {
            tracing::warn!("No validation scores provided for any variant");
            return Err(Error::new(ErrorDetails::InternalError {
                message: "No validation scores provided for any variant".to_string(),
            }));
        }

        tracing::debug!(
            "Combined candidates: {} total ({} existing + {} new)",
            all_candidates.len(),
            self.variants.len(),
            new_variant_scores.len()
        );

        // Step 3: Merge new scores with existing scores (consuming new_variant_scores)
        let mut all_scores = self.val_scores.clone();
        all_scores.extend(new_variant_scores);

        // Check if there are any datapoints - if yes, ensure at least one valid score exists
        let has_any_datapoint = all_scores
            .values()
            .any(|per_datapoint| !per_datapoint.is_empty());

        if has_any_datapoint {
            // We have datapoints, check if all scores are None (all evaluations failed)
            let has_any_score = all_scores
                .values()
                .flat_map(|per_datapoint| per_datapoint.values())
                .flat_map(|scores| scores.values())
                .any(|score| score.is_some());

            if !has_any_score {
                tracing::warn!(
                    "All evaluation scores are None - no variant produced any valid scores"
                );
                return Err(Error::new(ErrorDetails::InternalError {
                    message: "All variants failed to produce valid scores".to_string(),
                }));
            }
        }

        // Validate that all_scores datapoints match self.datapoint_ids
        let actual_datapoints: HashSet<&String> = all_scores
            .values()
            .flat_map(|scores| scores.keys())
            .collect();
        let expected_datapoints: HashSet<&String> = self.datapoint_ids.iter().collect();

        if !actual_datapoints
            .iter()
            .all(|dp| expected_datapoints.contains(dp))
        {
            let unexpected: Vec<String> = actual_datapoints
                .difference(&expected_datapoints)
                .map(|s| (*s).clone())
                .collect();
            let unexpected_list = if unexpected.len() <= 5 {
                unexpected.join(", ")
            } else {
                format!(
                    "{} and {} more",
                    unexpected[..5].join(", "),
                    unexpected.len() - 5
                )
            };
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Validation scores contain {} datapoint(s) not in frontier's datapoint_ids: {unexpected_list}. \
                     Layout must remain constant across iterations.",
                    unexpected.len()
                ),
            }));
        }

        // Validate that all metric names are known in optimize_directions
        let unknown_metrics: HashSet<String> = all_scores
            .values()
            .flat_map(|datapoint_scores| datapoint_scores.values())
            .flat_map(|scores| scores.keys().cloned())
            .filter(|metric| !self.optimize_directions.contains_key(metric))
            .collect();

        if !unknown_metrics.is_empty() {
            let mut unknown_list: Vec<_> = unknown_metrics.iter().collect();
            unknown_list.sort();
            let formatted_list = if unknown_list.len() <= 5 {
                unknown_list
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            } else {
                format!(
                    "{} and {} more",
                    unknown_list[..5]
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                    unknown_list.len() - 5
                )
            };
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Validation scores contain {} evaluator(s) not in frontier's evaluator set: {formatted_list}. \
                     Evaluator layout must remain constant across iterations.",
                    unknown_metrics.len()
                ),
            }));
        }

        // Create sorted metric directions for deterministic objective vector ordering
        let mut sorted_directions: Vec<(&String, &MetricConfigOptimize)> =
            self.optimize_directions.iter().collect();
        sorted_directions.sort_by_key(|(name, _)| *name);

        tracing::debug!(
            "Step 1: Building instance-wise Pareto sets for {} datapoints (union across all variants)",
            self.datapoint_ids.len()
        );

        // Step 1: Build instance-wise Pareto sets P*[i] for each datapoint
        // Original paper: P*[i] = {variants with max score on instance i} (single evaluator)
        // Our extension: P*[i] = {Pareto non-dominated variants on instance i} (multiple evaluators)
        let mut instance_pareto_sets: HashMap<String, HashSet<String>> = HashMap::new();

        for datapoint_id in &self.datapoint_ids {
            // Collect scores for this datapoint across all variants
            let mut instance_scores: Vec<(String, HashMap<String, Option<f32>>)> = Vec::new();

            for (variant_name, variant_scores) in &all_scores {
                // Only consider variants that are in the combined candidates
                if !all_candidates.contains_key(variant_name) {
                    continue;
                }
                if let Some(scores) = variant_scores.get(datapoint_id) {
                    instance_scores.push((variant_name.clone(), scores.clone()));
                }
            }

            if instance_scores.is_empty() {
                instance_pareto_sets.insert(datapoint_id.clone(), HashSet::new());
                continue;
            }

            // Find non-dominated variants for this instance
            let non_dominated =
                find_non_dominated_variants(&instance_scores, &self.optimize_directions);
            instance_pareto_sets.insert(datapoint_id.clone(), non_dominated.into_iter().collect());
        }

        // Step 2: Build candidate set C (union of all instance-wise Pareto sets)
        let candidate_set: HashSet<String> = instance_pareto_sets
            .values()
            .flat_map(|set| set.iter().cloned())
            .collect();

        // Safety check: candidate_set should only contain variants from the combined candidates
        debug_assert!(
            candidate_set.iter().all(|v| all_candidates.contains_key(v)),
            "candidate_set contains variants not in combined candidates"
        );

        tracing::debug!(
            "Step 2: Candidate set has {} variants (union of instance-wise Pareto sets)",
            candidate_set.len()
        );

        // Early exit if no candidates or only one
        if candidate_set.len() <= 1 {
            let frequencies = calculate_frequencies(&candidate_set, &instance_pareto_sets);

            let filtered_candidates: HashMap<String, UninitializedChatCompletionConfig> =
                all_candidates
                    .into_iter()
                    .filter(|(name, _)| candidate_set.contains(name))
                    .collect();

            // Prune val_scores to only contain variants in filtered_candidates
            let filtered_val_scores: HashMap<String, VariantScores> = all_scores
                .into_iter()
                .filter(|(name, _)| filtered_candidates.contains_key(name))
                .collect();

            // Update self
            self.variants = filtered_candidates;
            self.frequencies = frequencies;
            self.val_scores = filtered_val_scores;
            self.objective_vector_cache
                .retain(|name, _| self.variants.contains_key(name));

            tracing::info!(
                "Early exit: {} candidate(s) after instance-wise filtering",
                self.variants.len()
            );
            return Ok(());
        }

        // Step 3: Global filtering - check dominance over all (D×E) objectives
        tracing::debug!("Step 3: Global Pareto filtering across all objectives");

        // Pre-compute objective vectors for efficient dominance checking
        let expected_vector_length = self.expected_vector_length();
        let mut objective_vectors: HashMap<String, Arc<Vec<f32>>> = HashMap::new();
        let mut new_variants_count = 0;
        let mut cached_variants_count = 0;

        for variant_name in &candidate_set {
            // Check if we have a cached vector
            let vec = if let Some(cached_vec) = self.objective_vector_cache.get(variant_name) {
                // Validate that cached vector has the expected length
                if cached_vec.len() == expected_vector_length {
                    cached_variants_count += 1;
                    Arc::clone(cached_vec)
                } else {
                    tracing::warn!(
                        "Cached vector for '{}' has wrong length (expected {}, got {}). \
                         Datapoint or evaluator layout may have changed. Recomputing.",
                        variant_name,
                        expected_vector_length,
                        cached_vec.len()
                    );
                    new_variants_count += 1;
                    Arc::new(compute_objective_vector(
                        variant_name,
                        &all_scores,
                        &self.datapoint_ids,
                        &sorted_directions,
                    ))
                }
            } else {
                new_variants_count += 1;
                Arc::new(compute_objective_vector(
                    variant_name,
                    &all_scores,
                    &self.datapoint_ids,
                    &sorted_directions,
                ))
            };

            objective_vectors.insert(variant_name.clone(), vec);
        }

        // Update cache with newly computed vectors
        for (variant_name, vec) in &objective_vectors {
            self.objective_vector_cache
                .insert(variant_name.clone(), Arc::clone(vec));
        }
        tracing::debug!(
            "Pre-computed vectors: {} cached, {} new (total: {})",
            cached_variants_count,
            new_variants_count,
            candidate_set.len()
        );

        // Global dominance filtering
        let mut non_dominated = candidate_set.clone();
        let num_datapoints = self.datapoint_ids.len();

        for variant_a in &candidate_set {
            let a_vec = &objective_vectors[variant_a];
            for variant_b in &candidate_set {
                if variant_a != variant_b && non_dominated.contains(variant_b) {
                    let b_vec = &objective_vectors[variant_b];
                    if vector_dominates(a_vec, b_vec, &sorted_directions, num_datapoints) {
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
        let total_datapoints = self.datapoint_ids.len();
        let total_evaluators = self.optimize_directions.len();

        for variant_name in &non_dominated {
            if let Some(per_datapoint) = all_scores.get(variant_name) {
                let total_possible = total_datapoints * total_evaluators;
                if total_possible > 0 {
                    // Count non-None scores across all (datapoint, evaluator) pairs
                    let non_none_count = self
                        .datapoint_ids
                        .iter()
                        .filter_map(|datapoint_id| per_datapoint.get(datapoint_id))
                        .flat_map(|scores| {
                            self.optimize_directions
                                .keys()
                                .filter_map(move |evaluator_name| {
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
        let filtered_candidates: HashMap<String, UninitializedChatCompletionConfig> =
            all_candidates
                .into_iter()
                .filter(|(name, _)| non_dominated.contains(name))
                .collect();

        // Prune val_scores to only contain variants in filtered_candidates
        let filtered_val_scores: HashMap<String, VariantScores> = all_scores
            .into_iter()
            .filter(|(name, _)| filtered_candidates.contains_key(name))
            .collect();

        tracing::info!(
            "Pareto frontier updated: {} variants in frontier",
            filtered_candidates.len()
        );

        // Update self
        self.variants = filtered_candidates;
        self.frequencies = frequencies;
        self.val_scores = filtered_val_scores;

        // Prune cache to only retain variants in the final frontier
        // This prevents memory leak from accumulating dominated variants across iterations
        self.objective_vector_cache
            .retain(|name, _| self.variants.contains_key(name));

        Ok(())
    }
}

/// Check if child improves over parent variant (summary statistic Pareto dominance)
///
/// Uses the evaluation config to determine objective directions (optimize: max/min).
///
/// **Missing Data Handling**: If either variant lacks stats for an evaluator, that metric is
/// skipped from comparison. This allows comparing partially-evaluated variants - a child can
/// be considered an improvement even if missing some metrics, as long as it's better on the
/// metrics it has. This is intentional since `is_improvement` is just a pre-check on the minibatch.
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

/// Extract optimization directions from evaluator configs
fn extract_optimize_directions(
    evaluators: &HashMap<String, EvaluatorConfig>,
) -> HashMap<String, MetricConfigOptimize> {
    evaluators
        .iter()
        .map(|(name, config)| (name.clone(), config.optimize()))
        .collect()
}

/// Compute flattened objective vector for a variant across all (datapoint × evaluator) pairs
///
/// Creates a single vector containing imputed scores for all combinations of datapoints and evaluators.
/// This pre-computation enables faster dominance checking by avoiding repeated lookups and hash operations.
///
/// # Arguments
/// * `variant_name` - Name of the variant to compute vector for
/// * `val_scores_map` - Map of variant names to their per-datapoint scores
/// * `datapoint_ids` - Ordered list of datapoint IDs (defines vector structure)
/// * `sorted_evaluators` - Sorted list of metric optimization directions (defines vector structure and optimization directions)
///
/// # Returns
/// * `Vec<f32>` - Flattened vector of length D×E where each element is an imputed score
fn compute_objective_vector(
    variant_name: &str,
    val_scores_map: &HashMap<String, VariantScores>,
    datapoint_ids: &[String],
    sorted_evaluators: &[(&String, &MetricConfigOptimize)],
) -> Vec<f32> {
    let mut vec = Vec::with_capacity(datapoint_ids.len() * sorted_evaluators.len());

    for datapoint_id in datapoint_ids {
        for (evaluator_name, optimize_direction) in sorted_evaluators {
            let score = val_scores_map
                .get(variant_name)
                .and_then(|variant_scores| variant_scores.get(datapoint_id))
                .and_then(|scores| scores.get(*evaluator_name).and_then(|s| *s));

            let imputed = impute_missing_score(score, **optimize_direction);
            vec.push(imputed);
        }
    }

    vec
}

/// Check if vector A dominates vector B using pre-computed objective vectors
///
/// More efficient than `global_dominates` because it avoids repeated hash lookups and
/// imputation. Vectors must be pre-computed using `compute_objective_vector` with the
/// same datapoint_ids and evaluators ordering.
///
/// # Arguments
/// * `a_vec` - Pre-computed objective vector for variant A
/// * `b_vec` - Pre-computed objective vector for variant B
/// * `sorted_directions` - Sorted list of metric optimization directions (must match vector structure)
/// * `num_datapoints` - Number of datapoints (must match vector structure)
///
/// # Returns
/// * `bool` - True if vector A dominates vector B
fn vector_dominates(
    a_vec: &[f32],
    b_vec: &[f32],
    sorted_directions: &[(&String, &MetricConfigOptimize)],
    num_datapoints: usize,
) -> bool {
    let mut better_or_equal_on_all = true;
    let mut strictly_better_on_at_least_one = false;

    // Vector is structured as: [dp0_eval0, dp0_eval1, ..., dp1_eval0, dp1_eval1, ...]
    let num_evaluators = sorted_directions.len();

    for idx in 0..(num_datapoints * num_evaluators) {
        let evaluator_idx = idx % num_evaluators;

        // Get the optimization direction for this metric
        let optimize_direction = sorted_directions[evaluator_idx].1;

        let a_val = a_vec[idx];
        let b_val = b_vec[idx];

        let (is_worse, is_better) = compare_values(a_val, b_val, *optimize_direction);

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
/// * `optimize_directions` - Map of metric names to their optimization directions
///
/// # Returns
/// * `bool` - True if variant A globally dominates variant B
#[cfg(test)]
fn global_dominates(
    variant_a_name: &str,
    variant_b_name: &str,
    val_scores_map: &HashMap<String, VariantScores>,
    optimize_directions: &HashMap<String, MetricConfigOptimize>,
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
            optimize_directions
                .keys()
                .map(move |evaluator_name| (datapoint_id.clone(), evaluator_name.clone()))
        })
        .collect();

    for (datapoint_id, evaluator_name) in all_pairs {
        let Some(optimize) = optimize_directions.get(&evaluator_name) else {
            continue;
        };

        // Get scores for this (datapoint, evaluator) pair
        let score_a = variant_a_scores
            .and_then(|scores| scores.get(&datapoint_id))
            .and_then(|scores| scores.get(&evaluator_name).and_then(|s| *s));

        let score_b = variant_b_scores
            .and_then(|scores| scores.get(&datapoint_id))
            .and_then(|scores| scores.get(&evaluator_name).and_then(|s| *s));

        // Impute missing values as worst-case
        let a_val = impute_missing_score(score_a, *optimize);
        let b_val = impute_missing_score(score_b, *optimize);

        let (is_worse, is_better) = compare_values(a_val, b_val, *optimize);
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
/// * `optimize_directions` - Map of metric names to their optimization directions
///
/// # Returns
/// * `bool` - True if variant A instance-dominates variant B on this datapoint
fn instance_dominates(
    a_scores: &HashMap<String, Option<f32>>,
    b_scores: &HashMap<String, Option<f32>>,
    optimize_directions: &HashMap<String, MetricConfigOptimize>,
) -> bool {
    let mut better_or_equal_on_all = true;
    let mut strictly_better_on_at_least_one = false;

    for (evaluator_name, optimize) in optimize_directions {
        let score_a = a_scores.get(evaluator_name).and_then(|s| *s);
        let score_b = b_scores.get(evaluator_name).and_then(|s| *s);

        // Impute missing values as worst-case
        let a_val = impute_missing_score(score_a, *optimize);
        let b_val = impute_missing_score(score_b, *optimize);

        let (is_worse, is_better) = compare_values(a_val, b_val, *optimize);
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
/// * `optimize_directions` - Map of metric names to their optimization directions
///
/// # Returns
/// Vector of variant names that are not dominated by any other variant on this instance
fn find_non_dominated_variants(
    instance_scores: &[(String, HashMap<String, Option<f32>>)],
    optimize_directions: &HashMap<String, MetricConfigOptimize>,
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
            if instance_dominates(variant_b_scores, variant_a_scores, optimize_directions) {
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
    use tensorzero_core::evaluations::EvaluatorConfig;

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
                            description: Some("test_llm_judge_evaluator".to_string()),
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

    /// Extract and sort datapoint IDs from validation scores map
    ///
    /// Collects all unique datapoint IDs across all variants, deduplicates,
    /// and sorts them alphabetically for deterministic ordering.
    ///
    /// # Arguments
    /// * `val_scores_map` - Map of variant names to their per-datapoint scores
    ///
    /// # Returns
    /// * `Vec<String>` - Sorted list of unique datapoint IDs
    fn extract_sorted_datapoint_ids(
        val_scores_map: &HashMap<String, VariantScores>,
    ) -> Vec<String> {
        let mut datapoint_ids: Vec<String> = val_scores_map
            .values()
            .flat_map(|scores| scores.keys().cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        datapoint_ids.sort();
        datapoint_ids
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
            description: Some("test_evaluation".to_string()),
        }
    }

    /// Global filtering using the old hash-based approach (for performance comparison)
    ///
    /// This implements the original algorithm that repeatedly calls `global_dominates`,
    /// which performs hash lookups and imputation on every comparison.
    fn global_filter_old_approach(
        candidate_set: &HashSet<String>,
        val_scores_map: &HashMap<String, VariantScores>,
        optimize_directions: &HashMap<String, MetricConfigOptimize>,
    ) -> HashSet<String> {
        let mut non_dominated = candidate_set.clone();
        for variant_a in candidate_set {
            for variant_b in candidate_set {
                if variant_a != variant_b
                    && non_dominated.contains(variant_b)
                    && global_dominates(variant_a, variant_b, val_scores_map, optimize_directions)
                {
                    non_dominated.remove(variant_b);
                }
            }
        }
        non_dominated
    }

    /// Global filtering using the new vectorized approach (for performance comparison)
    ///
    /// This implements the optimized algorithm that pre-computes objective vectors
    /// once and then performs fast vector comparisons.
    fn global_filter_new_approach(
        candidate_set: &HashSet<String>,
        val_scores_map: &HashMap<String, VariantScores>,
        datapoint_ids: &[String],
        optimize_directions: &HashMap<String, MetricConfigOptimize>,
    ) -> HashSet<String> {
        // Create sorted evaluator list for deterministic ordering
        let mut sorted_directions: Vec<(&String, &MetricConfigOptimize)> =
            optimize_directions.iter().collect();
        sorted_directions.sort_by_key(|(name, _)| *name);

        // Pre-compute objective vectors
        let mut objective_vectors: HashMap<String, Vec<f32>> = HashMap::new();
        for variant_name in candidate_set {
            let vec = compute_objective_vector(
                variant_name,
                val_scores_map,
                datapoint_ids,
                &sorted_directions,
            );
            objective_vectors.insert(variant_name.clone(), vec);
        }

        let mut non_dominated = candidate_set.clone();
        let num_datapoints = datapoint_ids.len();

        for variant_a in candidate_set {
            let a_vec = &objective_vectors[variant_a];
            for variant_b in candidate_set {
                if variant_a != variant_b && non_dominated.contains(variant_b) {
                    let b_vec = &objective_vectors[variant_b];
                    if vector_dominates(a_vec, b_vec, &sorted_directions, num_datapoints) {
                        non_dominated.remove(variant_b);
                    }
                }
            }
        }

        non_dominated
    }

    /// Create large test dataset for performance testing
    ///
    /// # Returns
    /// * Tuple of (candidates, val_scores_map, datapoint_ids)
    fn create_large_test_data(
        num_variants: usize,
        num_datapoints: usize,
    ) -> (
        HashMap<String, UninitializedChatCompletionConfig>,
        HashMap<String, VariantScores>,
        Vec<String>,
    ) {
        let variant_names: Vec<String> =
            (0..num_variants).map(|v| format!("variant_{v}")).collect();
        let variant_name_refs: Vec<&str> = variant_names.iter().map(|s| s.as_str()).collect();
        let candidates = create_test_variants(&variant_name_refs);

        let datapoint_ids: Vec<String> = (0..num_datapoints).map(|d| format!("dp_{d}")).collect();

        let mut val_scores_map = HashMap::new();
        for (v, variant_name) in variant_names.iter().enumerate() {
            let mut datapoint_scores = HashMap::new();
            for (d, datapoint_id) in datapoint_ids.iter().enumerate() {
                // Generate pseudo-random scores for reproducibility
                let accuracy = ((v * 17 + d * 31) % 100) as f32 / 100.0;
                let latency = ((v * 23 + d * 41) % 100) as f32 / 100.0;
                let f1 = ((v * 29 + d * 37) % 100) as f32 / 100.0;

                datapoint_scores.insert(
                    datapoint_id.clone(),
                    HashMap::from([
                        ("accuracy".to_string(), Some(accuracy)),
                        ("latency".to_string(), Some(latency)),
                        ("f1".to_string(), Some(f1)),
                    ]),
                );
            }
            val_scores_map.insert(variant_name.clone(), datapoint_scores);
        }

        (candidates, val_scores_map, datapoint_ids)
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
        let optimize_directions = extract_optimize_directions(&evaluators);

        let a_scores = HashMap::from([("accuracy".to_string(), Some(0.9))]);
        let b_scores = HashMap::from([("accuracy".to_string(), Some(0.7))]);

        // A should dominate B (0.9 > 0.7)
        assert!(instance_dominates(
            &a_scores,
            &b_scores,
            &optimize_directions
        ));
        // B should not dominate A
        assert!(!instance_dominates(
            &b_scores,
            &a_scores,
            &optimize_directions
        ));
    }

    #[test]
    fn test_instance_dominates_equal() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);
        let optimize_directions = extract_optimize_directions(&evaluators);

        let a_scores = HashMap::from([("accuracy".to_string(), Some(0.8))]);
        let b_scores = HashMap::from([("accuracy".to_string(), Some(0.8))]);

        // Equal scores - neither dominates
        assert!(!instance_dominates(
            &a_scores,
            &b_scores,
            &optimize_directions
        ));
        assert!(!instance_dominates(
            &b_scores,
            &a_scores,
            &optimize_directions
        ));
    }

    #[test]
    fn test_instance_dominates_with_none() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);
        let optimize_directions = extract_optimize_directions(&evaluators);

        let a_scores = HashMap::from([("accuracy".to_string(), Some(0.7))]);
        let b_scores = HashMap::from([("accuracy".to_string(), None)]);

        // A (0.7) should dominate B (None = -inf)
        assert!(instance_dominates(
            &a_scores,
            &b_scores,
            &optimize_directions
        ));
        // B should not dominate A
        assert!(!instance_dominates(
            &b_scores,
            &a_scores,
            &optimize_directions
        ));
    }

    #[test]
    fn test_instance_dominates_mixed_directions() {
        let evaluators = create_test_evaluators(&[("accuracy", "max"), ("latency", "min")]);
        let optimize_directions = extract_optimize_directions(&evaluators);

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
        assert!(instance_dominates(
            &a_scores,
            &b_scores,
            &optimize_directions
        ));
        // B should not dominate A
        assert!(!instance_dominates(
            &b_scores,
            &a_scores,
            &optimize_directions
        ));
    }

    #[test]
    fn test_instance_dominates_tradeoff() {
        let evaluators = create_test_evaluators(&[("accuracy", "max"), ("latency", "min")]);
        let optimize_directions = extract_optimize_directions(&evaluators);

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
        assert!(!instance_dominates(
            &a_scores,
            &b_scores,
            &optimize_directions
        ));
        assert!(!instance_dominates(
            &b_scores,
            &a_scores,
            &optimize_directions
        ));
    }

    #[test]
    fn test_global_dominates_basic() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);
        let optimize_directions = extract_optimize_directions(&evaluators);

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

        assert!(global_dominates(
            "a",
            "b",
            &val_scores_map,
            &optimize_directions
        ));
        assert!(!global_dominates(
            "b",
            "a",
            &val_scores_map,
            &optimize_directions
        ));
    }

    #[test]
    fn test_global_dominates_no_domination() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);
        let optimize_directions = extract_optimize_directions(&evaluators);

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

        assert!(!global_dominates(
            "a",
            "b",
            &val_scores_map,
            &optimize_directions
        ));
        assert!(!global_dominates(
            "b",
            "a",
            &val_scores_map,
            &optimize_directions
        ));
    }

    #[test]
    fn test_find_non_dominated_variants_single() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);
        let optimize_directions = extract_optimize_directions(&evaluators);

        let instance_scores = vec![(
            "variant_a".to_string(),
            HashMap::from([("accuracy".to_string(), Some(0.9))]),
        )];

        let result = find_non_dominated_variants(&instance_scores, &optimize_directions);
        assert_eq!(result, vec!["variant_a"]);
    }

    #[test]
    fn test_find_non_dominated_variants_multiple() {
        let evaluators = create_test_evaluators(&[("accuracy", "max"), ("latency", "min")]);
        let optimize_directions = extract_optimize_directions(&evaluators);

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

        let result = find_non_dominated_variants(&instance_scores, &optimize_directions);

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
    #[expect(clippy::print_stderr)]
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

        let datapoint_ids = extract_sorted_datapoint_ids(&val_scores_map);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, val_scores_map);
        if let Err(ref e) = result {
            eprintln!("Error: {e:?}");
        }
        assert!(result.is_ok());

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

        // A should dominate B (1.0 > 0.0)
        assert_eq!(frontier.variants.len(), 1);
        assert!(frontier.variants.contains_key("variant_a"));
    }

    #[test]
    fn test_unknown_metric_rejected() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        // Include an unexpected evaluator key ("weird_metric") not present in config
        let variant_scores = HashMap::from([(
            "variant_a".to_string(),
            HashMap::from([(
                "dp1".to_string(),
                HashMap::from([
                    ("accuracy".to_string(), Some(0.9)),
                    ("weird_metric".to_string(), Some(0.1)),
                ]),
            )]),
        )]);

        let candidates = create_test_variants(&["variant_a"]);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not in frontier's evaluator set"),
            "Expected evaluator-set error, got: {err}"
        );
    }

    #[test]
    fn test_cache_pruning_removes_dominated_vectors() {
        let config = create_evaluation_config(&[("accuracy", "max")]);

        // dp1: tie, dp2: variant_a better => candidate_set has {a,b}, global filtering keeps only A
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.8))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.9))]),
                    ),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.8))]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([("accuracy".to_string(), Some(0.7))]),
                    ),
                ]),
            ),
        ]);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let candidates = create_test_variants(&["variant_a", "variant_b"]);

        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

        // Only variant_a should remain in frontier and cache
        assert_eq!(frontier.variants.len(), 1);
        assert!(frontier.variants.contains_key("variant_a"));
        assert_eq!(
            frontier
                .objective_vector_cache
                .keys()
                .cloned()
                .collect::<HashSet<_>>(),
            HashSet::from(["variant_a".to_string()])
        );
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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

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
        let datapoint_ids = extract_sorted_datapoint_ids(&val_scores_map);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, val_scores_map);
        let elapsed = start.elapsed();

        assert!(result.is_ok());

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

        let datapoint_ids = extract_sorted_datapoint_ids(&val_scores_map);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, val_scores_map);

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

        let datapoint_ids = extract_sorted_datapoint_ids(&val_scores_map);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, val_scores_map);

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

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

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result = frontier.update(candidates, variant_scores);
        assert!(result.is_ok());

        assert_eq!(frontier.variants.len(), 1);

        // Note: We can't directly assert on log output, but this test documents
        // the behavior and ensures the code path executes without panicking
        // The warning should be logged: "Variant 'variant_a' has high missing score rate: 60.0%"
    }

    #[test]
    #[expect(clippy::print_stdout)]
    fn test_optimization_speedup() {
        use std::time::Instant;

        // Test parameters (reasonable size to see measurable speedup)
        let num_variants = 50;
        let num_datapoints = 1_000;

        // Generate test data
        let config =
            create_evaluation_config(&[("accuracy", "max"), ("latency", "min"), ("f1", "max")]);

        let (_candidates, val_scores_map, datapoint_ids) =
            create_large_test_data(num_variants, num_datapoints);

        let candidate_set: HashSet<_> = val_scores_map.keys().cloned().collect();
        let optimize_directions = extract_optimize_directions(&config.evaluators);

        // Benchmark old approach (hash-based, repeated lookups)
        let start_old = Instant::now();
        let result_old =
            global_filter_old_approach(&candidate_set, &val_scores_map, &optimize_directions);
        let duration_old = start_old.elapsed();

        // Benchmark new approach (pre-computed vectors)
        let start_new = Instant::now();
        let result_new = global_filter_new_approach(
            &candidate_set,
            &val_scores_map,
            &datapoint_ids,
            &optimize_directions,
        );
        let duration_new = start_new.elapsed();

        // Assert correctness: both approaches must produce identical results
        assert_eq!(
            result_old, result_new,
            "Old and new approaches must produce identical results"
        );

        // Calculate and log speedup
        let speedup = duration_old.as_secs_f64() / duration_new.as_secs_f64();

        println!("\n========================================================");
        println!("          Performance Comparison Results");
        println!("========================================================");
        println!(
            "Dataset: {} variants x {} datapoints x {} evaluators",
            num_variants,
            num_datapoints,
            optimize_directions.len()
        );
        println!(
            "Total objectives: {} (D x E)",
            num_datapoints * optimize_directions.len()
        );
        println!("--------------------------------------------------------");
        println!("Old approach (hash lookups):  {duration_old:?}");
        println!("New approach (pre-computed):  {duration_new:?}");
        println!("--------------------------------------------------------");
        println!("Speedup: {speedup:.2}x faster");
        println!("========================================================\n");

        // Assert speedup: new approach should be meaningfully faster
        // We expect at least 1.5x speedup, but typically see 10-50x
        assert!(
            speedup > 1.5,
            "New approach should be at least 1.5x faster, got {speedup:.2}x. \
             This may indicate a performance regression."
        );

        // Document expected speedup range
        if speedup < 5.0 {
            println!("Warning: Speedup ({speedup:.2}x) is lower than expected (typically 10-50x)");
        }
    }

    #[test]
    #[expect(clippy::print_stdout)]
    fn test_cache_correctness() {
        // This test validates that the cache produces correct results in a typical GEPA workflow:
        // - Iteration 1: Evaluate initial population
        // - Iteration 2: Keep some variants from frontier + add new mutations
        // The cache should reuse vectors for persisting variants correctly.

        let config = create_evaluation_config(&[("accuracy", "max"), ("latency", "min")]);

        // Simulate Iteration 1: Initial population [A, B, C]
        let iteration1_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (
                        "dp1".to_string(),
                        HashMap::from([
                            ("accuracy".to_string(), Some(0.9)),
                            ("latency".to_string(), Some(0.2)),
                        ]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([
                            ("accuracy".to_string(), Some(0.8)),
                            ("latency".to_string(), Some(0.3)),
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
                            ("accuracy".to_string(), Some(0.7)),
                            ("latency".to_string(), Some(0.1)),
                        ]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([
                            ("accuracy".to_string(), Some(0.9)),
                            ("latency".to_string(), Some(0.2)),
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
                            ("accuracy".to_string(), Some(0.6)),
                            ("latency".to_string(), Some(0.4)),
                        ]),
                    ),
                    (
                        "dp2".to_string(),
                        HashMap::from([
                            ("accuracy".to_string(), Some(0.7)),
                            ("latency".to_string(), Some(0.5)),
                        ]),
                    ),
                ]),
            ),
        ]);

        let candidates1 = create_test_variants(&["variant_a", "variant_b", "variant_c"]);

        // Run iteration 1 with cache (frontier instance will be reused for iteration 2)
        let datapoint_ids = extract_sorted_datapoint_ids(&iteration1_scores);
        let mut frontier1_cached = ParetoFrontier::new(datapoint_ids.clone(), &config.evaluators);
        let result1_cached =
            frontier1_cached.update(candidates1.clone(), iteration1_scores.clone());
        assert!(result1_cached.is_ok());

        // Run iteration 1 without cache (for comparison)
        let mut frontier1_no_cache = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result1_no_cache = frontier1_no_cache.update(candidates1, iteration1_scores.clone());
        assert!(result1_no_cache.is_ok());

        // Iteration 1 results should be identical
        assert_eq!(
            frontier1_cached.variants.keys().collect::<HashSet<_>>(),
            frontier1_no_cache.variants.keys().collect::<HashSet<_>>(),
            "Iteration 1: cached and non-cached should produce same variants"
        );

        // Simulate Iteration 2: Keep A and B (from frontier), add new variant D
        // variant_d is NEW (mutation)
        let variant_d_scores = HashMap::from([(
            "variant_d".to_string(),
            HashMap::from([
                (
                    "dp1".to_string(),
                    HashMap::from([
                        ("accuracy".to_string(), Some(0.85)),
                        ("latency".to_string(), Some(0.15)),
                    ]),
                ),
                (
                    "dp2".to_string(),
                    HashMap::from([
                        ("accuracy".to_string(), Some(0.75)),
                        ("latency".to_string(), Some(0.25)),
                    ]),
                ),
            ]),
        )]);

        // For cached path: only add the NEW variant D (A and B already in frontier from iteration 1)
        let candidates2_new = create_test_variants(&["variant_d"]);

        // Run iteration 2 WITH cache (reuse same frontier1_cached instance, so cache is preserved)
        // Only pass the new variant D since A and B are already in the frontier
        let result2_cached = frontier1_cached.update(candidates2_new, variant_d_scores.clone());
        assert!(result2_cached.is_ok());

        // For non-cached path: all variants A, B, D together (fresh frontier)
        // Include variants that survived iteration 1 (A, B) plus new variant D
        let iteration2_all_scores = HashMap::from([
            // variant_a and variant_b have SAME scores as iteration 1 (immutable validation scores)
            (
                "variant_a".to_string(),
                iteration1_scores.get("variant_a").unwrap().clone(),
            ),
            (
                "variant_b".to_string(),
                iteration1_scores.get("variant_b").unwrap().clone(),
            ),
            (
                "variant_d".to_string(),
                variant_d_scores.get("variant_d").unwrap().clone(),
            ),
        ]);

        let candidates2_all = create_test_variants(&["variant_a", "variant_b", "variant_d"]);

        // Run iteration 2 WITHOUT cache (create fresh frontier, compute all from scratch)
        let datapoint_ids = extract_sorted_datapoint_ids(&iteration2_all_scores);
        let mut frontier2_no_cache = ParetoFrontier::new(datapoint_ids, &config.evaluators);
        let result2_no_cache = frontier2_no_cache.update(candidates2_all, iteration2_all_scores);
        assert!(result2_no_cache.is_ok());

        // Critical assertion: cached and non-cached should produce identical results
        assert_eq!(
            frontier1_cached.variants.keys().collect::<HashSet<_>>(),
            frontier2_no_cache.variants.keys().collect::<HashSet<_>>(),
            "Iteration 2: cached and non-cached should produce same variants"
        );

        assert_eq!(
            frontier1_cached.frequencies, frontier2_no_cache.frequencies,
            "Iteration 2: cached and non-cached should produce same frequencies"
        );

        println!(
            "\nCache correctness validated: {} cached entries in frontier",
            frontier1_cached.objective_vector_cache.len()
        );
    }
}
