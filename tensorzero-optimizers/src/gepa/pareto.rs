//! Pareto frontier analysis and dominance checking for GEPA
//!
//! This module implements the core Pareto optimality logic for multi-objective optimization:
//! - Instance-wise Pareto frontier computation
//! - Global Pareto dominance checking
//! - Missing data imputation (scores are per-datapoint × per-evaluator as `Option<f32>`)

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use evaluations::EvaluatorStats;
use rand::{SeedableRng, prelude::IndexedRandom, rngs::StdRng};
use tensorzero_core::{
    config::MetricConfigOptimize,
    error::{Error, ErrorDetails},
    evaluations::EvaluatorConfig,
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use crate::gepa::{
    GEPAVariant,
    evaluate::{DatapointId, EvaluatorName, VariantName, VariantScores},
};

/// Threshold for warning about high missing score rates
///
/// Variants with more than 10% missing evaluations will trigger a warning.
/// Threshold is intentionally strict to flag variants that may be unreliable
/// due to systematic evaluation failures.
const HIGH_MISSING_RATE_THRESHOLD: f32 = 0.1;

/// Scores map keyed by variant name
pub type VariantScoresMap = HashMap<VariantName, VariantScores>;

/// Candidate input bundle: variant config + evaluation scores
#[derive(Clone, Debug)]
pub struct Candidate {
    pub variant: UninitializedChatCompletionConfig,
    pub scores: VariantScores,
}

/// Result of Pareto frontier filtering with frequency-based sampling weights
///
/// This struct maintains all state needed for iterative Pareto frontier updates,
/// including validation scores, cached objective vectors, and layout fingerprints
/// (datapoint IDs + metric optimization directions) for cache validation.
#[derive(Debug)]
pub struct ParetoFrontier {
    /// Pareto-optimal variants (non-dominated after GEPA's 3-step filtering)
    variant_configs: HashMap<VariantName, UninitializedChatCompletionConfig>,

    /// Evaluator scores for current variants (pruned after each update)
    ///
    /// Maps variant names to their scores on each datapoint/evaluator pair.
    /// Only contains entries for variants in `self.variant_configs`.
    /// Private - internal implementation detail.
    variant_scores_map: VariantScoresMap,

    /// Instance-wise membership frequencies for weighted sampling
    ///
    /// Maps each variant to the count of datapoints where it's Pareto-optimal.
    /// Higher values indicate the variant performs well across more instances.
    /// Used for proportional sampling in GEPA's mutation step.
    variant_frequencies: HashMap<VariantName, usize>,

    /// Cached objective vectors for efficient dominance checking
    ///
    /// Maps variant names to pre-computed flattened objective vectors.
    /// Vectors are keyed by variant name and validated against the layout fingerprint.
    /// Private - internal implementation detail for performance optimization.
    objective_vector_cache: HashMap<VariantName, Arc<Vec<f32>>>,

    /// Datapoint IDs defining the objective space (layout fingerprint)
    ///
    /// This ordering must remain constant for cache correctness.
    /// Private to enforce validation through update methods.
    datapoint_ids: Vec<DatapointId>,

    /// Optimization directions for each metric (min or max)
    ///
    /// Maps evaluator names to their optimization direction.
    /// BTreeMap ensures deterministic iteration order (sorted by evaluator name).
    /// Private to enforce validation through update methods.
    optimize_directions: BTreeMap<EvaluatorName, MetricConfigOptimize>,

    /// RNG for deterministic frequency sampling (seeded in constructor)
    rng: RefCell<StdRng>,
}

impl ParetoFrontier {
    /// Create a new empty Pareto frontier with the specified layout
    ///
    /// # Arguments
    /// * `datapoint_ids` - List of datapoint IDs defining the objective space
    /// * `evaluators` - Map of evaluator configurations from which optimization directions (min/max) will be extracted
    /// * `rng_seed` - Optional seed for deterministic sampling; None uses a random seed
    ///
    /// # Returns
    /// * Empty ParetoFrontier ready for iterative updates
    pub fn new(
        datapoint_ids: Vec<DatapointId>,
        evaluators: &HashMap<EvaluatorName, EvaluatorConfig>,
        rng_seed: Option<u64>,
    ) -> Self {
        let optimize_directions = extract_optimize_directions(evaluators);
        let rng = match rng_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut thread_rng = rand::rng();
                StdRng::from_rng(&mut thread_rng)
            }
        };
        Self {
            variant_configs: HashMap::new(),
            variant_frequencies: HashMap::new(),
            variant_scores_map: HashMap::new(),
            objective_vector_cache: HashMap::new(),
            datapoint_ids,
            optimize_directions,
            rng: RefCell::new(rng),
        }
    }

    /// Returns a reference to the variant configurations in the current Pareto frontier for read-only access (e.g., diagnostics)
    pub fn variant_configs(&self) -> &HashMap<VariantName, UninitializedChatCompletionConfig> {
        &self.variant_configs
    }

    /// Access variant scores for read-only callers (e.g., diagnostics)
    pub fn variant_scores_map(&self) -> &VariantScoresMap {
        &self.variant_scores_map
    }

    /// Returns a reference to the frequency map (instance-wise Pareto set membership counts) for read-only access (e.g., diagnostics)
    pub fn variant_frequencies(&self) -> &HashMap<VariantName, usize> {
        &self.variant_frequencies
    }

    /// Update the Pareto frontier with new candidates (variant config + scores)
    ///
    /// Flow:
    /// 1. Validate names/scores and normalize the incoming score maps to the frontier layout
    ///    (drop unknown datapoints/metrics, add empty maps for missing datapoints).
    /// 2. Merge new candidates with the existing frontier state.
    /// 3. Run GEPA SELECTCANDIDATE: instance-wise Pareto sets → union → global Pareto filter.
    /// 4. Replace configs, scores, frequencies, and cache with the surviving frontier variants.
    ///
    /// # Arguments
    /// * `candidates` - Map of variant name → `Candidate { variant config, scores }`
    ///
    /// # Returns
    /// * `Result<(), Error>` - Ok if successful, Err if scores are invalid or names collide
    ///
    /// # Errors
    /// * Returns error if any new variant name already exists in the frontier
    /// * Returns error if no validation scores are provided or all scores are None
    pub fn update(&mut self, candidates: HashMap<VariantName, Candidate>) -> Result<(), Error> {
        tracing::info!(
            "Updating Pareto frontier: assessing {} new variants with {} existing variants",
            candidates.len(),
            self.variant_configs.len()
        );
        // Step 1: Validate new candidates
        // 1a: Check for name collisions and extract variant configurations
        self.validate_no_name_collisions(&candidates)?;
        let new_variant_configs: HashMap<VariantName, UninitializedChatCompletionConfig> =
            candidates
                .iter()
                .map(|(name, candidate)| (name.clone(), candidate.variant.clone()))
                .collect();

        // 1b: Extract and validate scores (performs normalization and validation)
        let new_variant_scores = self.extract_and_validate_new_scores(candidates)?;

        // Step 2: Merge
        // Merge new variant configs with existing frontier variant configs
        let mut all_variant_configs: HashMap<VariantName, UninitializedChatCompletionConfig> =
            self.variant_configs.clone();
        all_variant_configs.extend(new_variant_configs.clone());

        // Merge new scores with existing scores
        let mut all_variant_scores: VariantScoresMap = self.variant_scores_map.clone();
        all_variant_scores.extend(new_variant_scores);

        tracing::debug!(
            "Step 2: Combined candidates: {} total ({} existing + {} new)",
            all_variant_configs.len(),
            self.variant_configs.len(),
            new_variant_configs.len()
        );

        // Step 3: Run modified GEPA SELECTCANDIDATE algorithm
        // 3a: Build instance-wise Pareto sets P*[i] for each datapoint
        // Original paper: P*[i] = {variants with max score on instance i} (single evaluator)
        // Our extension: P*[i] = {Pareto non-dominated variants on instance i} (multiple evaluators)
        tracing::debug!(
            "Step 3a: Building instance-wise Pareto sets for {} datapoints (union across all variants)",
            self.datapoint_ids.len()
        );
        let instance_optimal_variant_names =
            self.build_instance_pareto_sets(&all_variant_scores, &all_variant_configs);

        // 3b: Build candidate set C (union of all instance-wise Pareto sets)
        let candidate_optimal_variant_names: HashSet<VariantName> = instance_optimal_variant_names
            .values()
            .flat_map(|set| set.iter().cloned())
            .collect();

        tracing::debug!(
            "Step 3b: Candidate set has {} variants (union of instance-wise Pareto sets)",
            candidate_optimal_variant_names.len()
        );

        // Early exit if no candidates or only one
        if candidate_optimal_variant_names.len() <= 1 {
            self.update_frontier_variables(
                candidate_optimal_variant_names.clone(),
                &instance_optimal_variant_names,
                all_variant_configs,
                all_variant_scores,
            );
            tracing::info!(
                "Early exit: {} candidate(s) after instance-wise filtering",
                candidate_optimal_variant_names.len()
            );
            return Ok(());
        }

        // 3c: Global filtering - check dominance over all objectives
        tracing::debug!("Step 3c: Global Pareto filtering across all objectives");

        // Pre-compute objective vectors and update cache
        self.update_objective_vector_cache(&candidate_optimal_variant_names, &all_variant_scores);

        // Global dominance filtering (reads vectors from cache)
        let optimal_variant_names =
            self.filter_by_global_dominance(candidate_optimal_variant_names);

        // Monitor missing data rates for final Pareto frontier variants
        self.log_missing_data_warnings(&optimal_variant_names, &all_variant_scores);

        // Step 4: Update frontier
        self.update_frontier_variables(
            optimal_variant_names,
            &instance_optimal_variant_names,
            all_variant_configs,
            all_variant_scores,
        );

        tracing::info!(
            "Pareto frontier updated: {} variants in frontier",
            self.variant_configs.len()
        );

        Ok(())
    }

    /// Sample a single variant using frequency weights
    ///
    /// Frequencies must be non-empty and contain at least one non-zero count. Unknown
    /// variant names (not present in `self.variant_configs`) are ignored. Errors if:
    /// - the frequency map is empty,
    /// - all weights sum to zero, or
    /// - no weighted entries correspond to existing variants.
    /// Uses the frontier's internally maintained RNG (seedable via `new`) to allow deterministic
    /// sampling in tests.
    pub fn sample_by_frequency(&self) -> Result<GEPAVariant, Error> {
        if self.variant_frequencies.is_empty() {
            return Err(Error::new(ErrorDetails::InternalError {
                message: "Cannot sample from empty frequency map".to_string(),
            }));
        }

        let total_frequency: usize = self.variant_frequencies.values().sum();
        if total_frequency == 0 {
            return Err(Error::new(ErrorDetails::InternalError {
                message: "Cannot sample when all frequencies are zero".to_string(),
            }));
        }

        let items: Vec<_> = self
            .variant_frequencies
            .iter()
            .filter(|(name, _)| self.variant_configs.contains_key(*name))
            .collect();

        if items.is_empty() {
            return Err(Error::new(ErrorDetails::InternalError {
                message: "Cannot sample because no frequencies correspond to existing variants"
                    .to_string(),
            }));
        }

        let mut rng = self.rng.borrow_mut();
        let sampled_name = items
            .choose_weighted(&mut *rng, |&(_, &count)| count)
            .map(|(name, _)| (*name).clone())
            .map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Weighted sampling failed: {e}"),
                })
            })?;

        let config = self
            .variant_configs
            .get(&sampled_name)
            .cloned()
            .ok_or_else(|| {
                Error::new(ErrorDetails::InternalError {
                    message: format!(
                        "Sampled variant '{sampled_name}' not found in Pareto frontier"
                    ),
                })
            })?;

        Ok(GEPAVariant {
            name: sampled_name,
            config,
        })
    }

    /// Validate that incoming candidates do not collide with existing frontier variant names
    fn validate_no_name_collisions(
        &self,
        candidates: &HashMap<VariantName, Candidate>,
    ) -> Result<(), Error> {
        let collisions: Vec<&VariantName> = candidates
            .keys()
            .filter(|name| self.variant_configs.contains_key(*name))
            .collect();

        if collisions.is_empty() {
            return Ok(());
        }

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

        Err(Error::new(ErrorDetails::InternalError {
            message: format!(
                "Variant name collision(s) detected: {collision_list}. \
                 New variants cannot have the same names as existing frontier variants."
            ),
        }))
    }

    /// Extract and validate new scores from candidates
    ///
    /// Performs score extraction, normalization, and validation:
    /// 1. Normalizes datapoint coverage to match the frontier layout: drops scores for unknown
    ///    datapoints and inserts empty maps for missing expected datapoints (imputed later as
    ///    worst case).
    /// 2. Drops unknown evaluator metrics with a warning so mis-specified candidates cannot
    ///    pollute the frontier.
    /// 3. Validates that at least one expected datapoint is present and at least one concrete
    ///    score (Some) exists across all candidates.
    ///
    /// # Returns
    /// * `Ok(VariantScoresMap)` - Validated and normalized scores
    /// * `Err` - If candidates are empty, have no datapoints, or all scores are None
    fn extract_and_validate_new_scores(
        &self,
        candidates: HashMap<VariantName, Candidate>,
    ) -> Result<VariantScoresMap, Error> {
        // Check if we have any new variants
        if candidates.is_empty() {
            tracing::warn!("No validation scores provided for any variant");
            return Err(Error::new(ErrorDetails::InternalError {
                message: "No validation scores provided for any variant".to_string(),
            }));
        }

        // Extract and normalize scores
        let mut new_variant_scores: VariantScoresMap = HashMap::with_capacity(candidates.len());
        let expected_datapoints: HashSet<DatapointId> =
            self.datapoint_ids.iter().copied().collect();
        let mut dropped_metrics: HashSet<String> = HashSet::new();

        for (name, candidate) in candidates {
            let mut scores = candidate.scores;

            // Normalize datapoint coverage:
            // - Drop scores for datapoints outside the frontier layout
            // - Insert empty score maps for missing expected datapoints
            scores.retain(|dp, _| expected_datapoints.contains(dp));
            for dp in &self.datapoint_ids {
                let per_eval = scores.entry(*dp).or_insert_with(HashMap::new);
                // Retain only known evaluator metrics; drop unknowns
                per_eval.retain(|metric, _| {
                    let keep = self.optimize_directions.contains_key(metric);
                    if !keep {
                        dropped_metrics.insert(metric.clone());
                    }
                    keep
                });
            }

            new_variant_scores.insert(name, scores);
        }

        // Log warning for dropped metrics
        if !dropped_metrics.is_empty() {
            let mut dropped_list: Vec<_> = dropped_metrics.iter().collect();
            dropped_list.sort();
            tracing::warn!(
                "Dropped {} evaluator(s) not in frontier's evaluator set: {}",
                dropped_list.len(),
                dropped_list
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        // Validation 1: Check if all new variants have empty datapoint maps
        let has_any_datapoint = new_variant_scores
            .values()
            .any(|per_datapoint| !per_datapoint.is_empty());

        if !has_any_datapoint {
            tracing::warn!("All new variants have empty datapoint maps");
            return Err(Error::new(ErrorDetails::InternalError {
                message: "All new variants have empty datapoint maps - no evaluations provided"
                    .to_string(),
            }));
        }

        // Validation 2: Check if all new scores are None (all new evaluations failed)
        let has_any_score = new_variant_scores
            .values()
            .flat_map(|per_datapoint| per_datapoint.values())
            .flat_map(|scores| scores.values())
            .any(|score| score.is_some());

        if !has_any_score {
            tracing::warn!(
                "All evaluation scores are None - no new variant produced any valid scores"
            );
            return Err(Error::new(ErrorDetails::InternalError {
                message: "All new variants failed to produce valid scores".to_string(),
            }));
        }

        Ok(new_variant_scores)
    }

    /// Build instance-wise Pareto sets for each datapoint
    ///
    /// For each datapoint, finds the non-dominated variants when considering only
    /// that single instance. This is Step 1 of GEPA's 3-step algorithm.
    ///
    /// # Arguments
    /// * `variants_scores` - Scores for all variants (existing + new)
    /// * `variant_configs` - Variant configs for all variants (existing + new)
    ///
    /// # Returns
    /// Map from datapoint ID to set of variant names that are Pareto-optimal on that datapoint
    fn build_instance_pareto_sets(
        &self,
        variants_scores: &VariantScoresMap,
        variant_configs: &HashMap<VariantName, UninitializedChatCompletionConfig>,
    ) -> HashMap<DatapointId, HashSet<VariantName>> {
        let mut instance_pareto_sets: HashMap<DatapointId, HashSet<VariantName>> = HashMap::new();

        for datapoint_id in &self.datapoint_ids {
            // Collect scores for this datapoint across all variants
            let mut instance_scores: Vec<(VariantName, HashMap<EvaluatorName, Option<f32>>)> =
                Vec::new();

            for (variant_name, variant_scores) in variants_scores {
                // Only consider variants that are in the combined candidates
                if !variant_configs.contains_key(variant_name) {
                    continue;
                }
                if let Some(scores) = variant_scores.get(datapoint_id) {
                    instance_scores.push((variant_name.clone(), scores.clone()));
                }
            }

            if instance_scores.is_empty() {
                instance_pareto_sets.insert(*datapoint_id, HashSet::new());
                continue;
            }

            // Find non-dominated variants for this instance
            let non_dominated =
                find_non_dominated_variants(&instance_scores, &self.optimize_directions);
            instance_pareto_sets.insert(*datapoint_id, non_dominated.into_iter().collect());
        }

        instance_pareto_sets
    }

    /// Update frontier internal state with optimal variants
    ///
    /// Updates variant frequencies, configs, scores, and prunes the objective vector cache
    /// to only contain the optimal variants.
    ///
    /// # Arguments
    /// * `optimal_variant_names` - Set of variant names to keep in the frontier
    /// * `instance_optimal_variant_names` - Instance-wise Pareto sets for frequency calculation
    /// * `variant_configs` - All variant configs (consumed)
    /// * `variants_scores` - All variant scores (consumed)
    fn update_frontier_variables(
        &mut self,
        optimal_variant_names: HashSet<VariantName>,
        instance_optimal_variant_names: &HashMap<DatapointId, HashSet<VariantName>>,
        variant_configs: HashMap<VariantName, UninitializedChatCompletionConfig>,
        variants_scores: VariantScoresMap,
    ) {
        // Compute frequency map for sampling (count instance-wise Pareto memberships)
        self.variant_frequencies =
            calculate_frequencies(&optimal_variant_names, instance_optimal_variant_names);

        // Filter out variants not in final non-dominated set
        self.variant_configs = variant_configs
            .into_iter()
            .filter(|(name, _)| optimal_variant_names.contains(name))
            .collect();

        // Prune variant_scores_map to only contain variants in optimal_variant_names
        self.variant_scores_map = variants_scores
            .into_iter()
            .filter(|(name, _)| optimal_variant_names.contains(name))
            .collect();

        // Prune cache to only retain variants in the final frontier
        // This prevents memory leak from accumulating dominated variants across iterations
        self.objective_vector_cache
            .retain(|name, _| optimal_variant_names.contains(name));
    }

    /// Update objective vector cache for candidate variants
    ///
    /// For each variant in the candidate set:
    /// - If already cached: skip (cache hit)
    /// - If not cached: compute objective vector and insert into cache
    ///
    /// Logs cache statistics (hits/misses).
    /// Uses the BTreeMap ordering of evaluator directions to keep vector layout stable.
    ///
    /// # Arguments
    /// * `variant_names` - Set of variant names to ensure are cached
    /// * `variants_scores` - Score map containing all variants (existing + new)
    fn update_objective_vector_cache(
        &mut self,
        variant_names: &HashSet<VariantName>,
        variants_scores: &VariantScoresMap,
    ) {
        // BTreeMap iteration is already sorted by key
        let sorted_directions: Vec<(&String, &MetricConfigOptimize)> =
            self.optimize_directions.iter().collect();

        let mut new_variants_count = 0;
        let mut cached_variants_count = 0;

        for variant_name in variant_names {
            if self.objective_vector_cache.contains_key(variant_name) {
                // Already cached
                cached_variants_count += 1;
            } else {
                // Compute and cache
                new_variants_count += 1;
                let vec = Arc::new(compute_objective_vector(
                    variant_name,
                    variants_scores,
                    &self.datapoint_ids,
                    &sorted_directions,
                ));
                self.objective_vector_cache
                    .insert(variant_name.clone(), vec);
            }
        }

        tracing::debug!(
            "Updated vector cache: {} cached, {} new (total: {})",
            cached_variants_count,
            new_variants_count,
            variant_names.len()
        );
    }

    /// Filter candidate variants by global Pareto dominance
    ///
    /// Performs pairwise dominance checking across all (datapoint × evaluator) objectives
    /// to identify and remove globally dominated variants. A variant is dominated if
    /// another variant is better-or-equal on all objectives and strictly better on at least one.
    ///
    /// This implements Step 3c of the GEPA algorithm: global filtering after instance-wise
    /// filtering has produced the candidate set.
    ///
    /// Reads pre-computed objective vectors from the cache (populated by
    /// `update_objective_vector_cache`).
    ///
    /// # Arguments
    /// * `variant_names` - Set of candidate variant names to filter (consumed)
    ///
    /// # Returns
    /// Set of non-dominated variant names (the final Pareto frontier)
    fn filter_by_global_dominance(
        &self,
        variant_names: HashSet<VariantName>,
    ) -> HashSet<VariantName> {
        // BTreeMap iteration is already sorted by key
        let sorted_directions: Vec<(&String, &MetricConfigOptimize)> =
            self.optimize_directions.iter().collect();

        let mut non_dominated = variant_names.clone();

        for variant_a in &variant_names {
            let a_vec = &self.objective_vector_cache[variant_a];
            for variant_b in &variant_names {
                if variant_a != variant_b && non_dominated.contains(variant_b) {
                    let b_vec = &self.objective_vector_cache[variant_b];
                    if vector_dominates(a_vec, b_vec, &sorted_directions) {
                        non_dominated.remove(variant_b);
                    }
                }
            }
        }

        tracing::debug!(
            "Global filtering: {} → {} variants",
            variant_names.len(),
            non_dominated.len()
        );

        non_dominated
    }

    /// Log warnings for variants with high missing evaluation rates
    ///
    /// Monitors each variant in the final Pareto frontier and logs warnings when the
    /// missing score rate exceeds the threshold (defined by HIGH_MISSING_RATE_THRESHOLD).
    ///
    /// Missing scores can indicate:
    /// - Systematic evaluation failures (e.g., variant causes errors)
    /// - Random failures (e.g., external evaluator unavailable)
    /// - Incomplete evaluation coverage
    ///
    /// This monitoring uses the frontier layout as the denominator (datapoints × evaluators).
    /// Missing covers both absent datapoint/evaluator entries and explicit `None` scores.
    /// It helps surface variants that reached the frontier despite sparse evaluation data.
    ///
    /// # Arguments
    /// * `variant_names` - Set of variant names in the final Pareto frontier
    /// * `variants_scores` - Score map containing all variants
    fn log_missing_data_warnings(
        &self,
        variant_names: &HashSet<VariantName>,
        variants_scores: &VariantScoresMap,
    ) {
        let total_datapoints = self.datapoint_ids.len();
        let total_evaluators = self.optimize_directions.len();

        for variant_name in variant_names {
            if let Some(variant_scores) = variants_scores.get(variant_name) {
                let total_possible = total_datapoints * total_evaluators;
                if total_possible > 0 {
                    // Count non-None scores across all (datapoint, evaluator) pairs
                    let non_none_count = self
                        .datapoint_ids
                        .iter()
                        .filter_map(|datapoint_id| variant_scores.get(datapoint_id))
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
    }

    #[cfg(test)]
    pub fn variant_frequencies_mut(&mut self) -> &mut HashMap<VariantName, usize> {
        &mut self.variant_frequencies
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
/// * `evaluators` - Map of evaluator configs with optimization directions (max/min) for each metric
///
/// # Returns
/// * `bool` - True if child Pareto-dominates parent (better/equal on all, strictly better on ≥1 evaluator summary stats)
pub fn is_improvement(
    parent_stats: &HashMap<EvaluatorName, EvaluatorStats>,
    child_stats: &HashMap<EvaluatorName, EvaluatorStats>,
    evaluators: &HashMap<EvaluatorName, EvaluatorConfig>,
) -> bool {
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
///
/// Returns a BTreeMap to ensure deterministic iteration order (sorted by evaluator name).
fn extract_optimize_directions(
    evaluators: &HashMap<EvaluatorName, EvaluatorConfig>,
) -> BTreeMap<EvaluatorName, MetricConfigOptimize> {
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
/// * `variants_scores` - Map of variant names to their per-datapoint scores
/// * `datapoint_ids` - Ordered list of datapoint IDs (defines vector structure)
/// * `sorted_directions` - Sorted list of (evaluator_name, optimization_direction) tuples (defines vector structure and optimization directions)
///
/// # Returns
/// * `Vec<f32>` - Flattened vector of length D×E where each element is an imputed score
fn compute_objective_vector(
    variant_name: &str,
    variants_scores: &VariantScoresMap,
    datapoint_ids: &[DatapointId],
    sorted_directions: &[(&String, &MetricConfigOptimize)],
) -> Vec<f32> {
    let mut vec = Vec::with_capacity(datapoint_ids.len() * sorted_directions.len());

    for datapoint_id in datapoint_ids {
        for (evaluator_name, optimize_direction) in sorted_directions {
            let score = variants_scores
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
/// Efficient dominance checking using pre-computed vectors with flattened layout
/// to avoid repeated hash lookups and imputation. Vectors must be pre-computed
/// using `compute_objective_vector` with the same datapoint_ids and evaluator ordering
/// (BTreeMap order from `optimize_directions`).
///
/// # Arguments
/// * `a_vec` - Pre-computed objective vector for variant A
/// * `b_vec` - Pre-computed objective vector for variant B
/// * `sorted_directions` - Sorted list of metric optimization directions (must match vector structure)
///
/// # Returns
/// * `bool` - True if vector A dominates vector B
fn vector_dominates(
    a_vec: &[f32],
    b_vec: &[f32],
    sorted_directions: &[(&String, &MetricConfigOptimize)],
) -> bool {
    let mut better_or_equal_on_all = true;
    let mut strictly_better_on_at_least_one = false;

    // Vector is structured as: [dp0_eval0, dp0_eval1, ..., dp1_eval0, dp1_eval1, ...]
    let num_evaluators = sorted_directions.len();
    let num_datapoints = a_vec.len() / num_evaluators;

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
/// * `variant_names` - Set of variant names to calculate frequencies for
/// * `instance_optimal_variant_names` - Map from datapoint ID to set of Pareto-optimal variant names for that instance
///
/// # Returns
/// HashMap mapping each variant name to its frequency (count of instance-wise Pareto set memberships)
fn calculate_frequencies(
    variant_names: &HashSet<VariantName>,
    instance_optimal_variant_names: &HashMap<DatapointId, HashSet<VariantName>>,
) -> HashMap<VariantName, usize> {
    variant_names
        .iter()
        .map(|v| {
            let freq = instance_optimal_variant_names
                .values()
                .filter(|s| s.contains(v))
                .count();
            (v.clone(), freq)
        })
        .collect()
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
/// * `optimize_directions` - Map of evaluator names to their optimization directions
///
/// # Returns
/// * `bool` - True if variant A instance-dominates variant B on this datapoint
fn instance_dominates(
    a_scores: &HashMap<EvaluatorName, Option<f32>>,
    b_scores: &HashMap<EvaluatorName, Option<f32>>,
    optimize_directions: &BTreeMap<EvaluatorName, MetricConfigOptimize>,
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
/// * `optimize_directions` - Map of evaluator names to their optimization directions
///
/// # Returns
/// Vector of variant names that are not dominated by any other variant on this instance
fn find_non_dominated_variants(
    instance_scores: &[(VariantName, HashMap<EvaluatorName, Option<f32>>)],
    optimize_directions: &BTreeMap<EvaluatorName, MetricConfigOptimize>,
) -> Vec<VariantName> {
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
    use uuid::Uuid;

    // ============================================================================
    // Test Helper Functions
    // ============================================================================

    /// Create a HashMap of evaluator configs with specified optimize directions
    ///
    /// # Arguments
    /// * `evaluators` - Slice of (evaluator_name, optimize_direction) tuples
    ///   where optimize_direction is "max" or "min"
    fn create_test_evaluators(
        evaluators: &[(&str, &str)],
    ) -> HashMap<EvaluatorName, EvaluatorConfig> {
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
                ((*name).to_owned(), evaluator_config)
            })
            .collect()
    }

    /// Create a HashMap of test candidates with default configs and provided scores
    fn create_test_candidates(scores: &VariantScoresMap) -> HashMap<VariantName, Candidate> {
        scores
            .iter()
            .map(|(name, score)| {
                (
                    name.clone(),
                    Candidate {
                        variant: UninitializedChatCompletionConfig {
                            weight: None,
                            model: "test-model".into(),
                            ..Default::default()
                        },
                        scores: score.clone(),
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
    /// * `Vec<DatapointId>` - Sorted list of unique datapoint IDs
    fn extract_sorted_datapoint_ids(val_scores_map: &VariantScoresMap) -> Vec<DatapointId> {
        let mut datapoint_ids: Vec<DatapointId> = val_scores_map
            .values()
            .flat_map(|scores| scores.keys().copied())
            .collect();
        datapoint_ids.sort();
        datapoint_ids.dedup();
        datapoint_ids
    }

    /// Deterministic datapoint ID helper for tests
    fn dp(id: u128) -> DatapointId {
        Uuid::from_u128(id)
    }

    /// Create a HashMap of EvaluatorStats for testing is_improvement
    ///
    /// # Arguments
    /// * `stats` - Slice of (evaluator_name, mean, stderr, count) tuples
    fn create_evaluator_stats(
        stats: &[(&str, f32, f32, usize)],
    ) -> HashMap<EvaluatorName, EvaluatorStats> {
        stats
            .iter()
            .map(|(name, mean, stderr, count)| {
                (
                    (*name).to_owned(),
                    EvaluatorStats {
                        mean: *mean,
                        stderr: *stderr,
                        count: *count,
                    },
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
    fn test_sample_by_frequency_single_variant() {
        let mut frontier = ParetoFrontier::new(Vec::new(), &HashMap::new(), None);
        frontier.variant_configs.insert(
            "variant_a".to_string(),
            UninitializedChatCompletionConfig {
                model: "test-model".into(),
                ..Default::default()
            },
        );
        frontier
            .variant_frequencies_mut()
            .insert("variant_a".to_string(), 3);

        let sampled = frontier
            .sample_by_frequency()
            .expect("sampling should succeed");
        assert_eq!(sampled.name, "variant_a");
    }

    #[test]
    fn test_sample_by_frequency_zero_weights_error() {
        let mut frontier = ParetoFrontier::new(Vec::new(), &HashMap::new(), None);
        frontier
            .variant_frequencies_mut()
            .insert("variant_a".to_string(), 0);
        frontier.variant_configs.insert(
            "variant_a".to_string(),
            UninitializedChatCompletionConfig {
                model: "test-model".into(),
                ..Default::default()
            },
        );

        let result = frontier.sample_by_frequency();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("frequencies are zero")
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
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        assert!(is_improvement(&parent_stats, &child_stats, &evaluators)); // Child is better
    }

    #[test]
    fn test_is_improvement_worse() {
        // Child worse on single metric (max)
        let parent_stats = create_evaluator_stats(&[("accuracy", 0.9, 0.1, 10)]);
        let child_stats = create_evaluator_stats(&[("accuracy", 0.7, 0.1, 10)]);
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        assert!(!is_improvement(&parent_stats, &child_stats, &evaluators)); // Child is worse
    }

    #[test]
    fn test_is_improvement_equal() {
        // Child equal on all metrics - no strict improvement
        let parent_stats = create_evaluator_stats(&[("accuracy", 0.8, 0.1, 10)]);
        let child_stats = create_evaluator_stats(&[("accuracy", 0.8, 0.1, 10)]);
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        assert!(!is_improvement(&parent_stats, &child_stats, &evaluators)); // Equal is not improvement
    }

    #[test]
    fn test_is_improvement_pareto_dominant() {
        // Child better on one metric, equal on another - should be Pareto improvement
        let parent_stats =
            create_evaluator_stats(&[("accuracy", 0.8, 0.1, 10), ("f1", 0.7, 0.1, 10)]);
        let child_stats =
            create_evaluator_stats(&[("accuracy", 0.9, 0.1, 10), ("f1", 0.7, 0.1, 10)]);
        let evaluators = create_test_evaluators(&[("accuracy", "max"), ("f1", "max")]);

        assert!(is_improvement(&parent_stats, &child_stats, &evaluators)); // Better on one, equal on other = improvement
    }

    #[test]
    fn test_is_improvement_not_pareto_dominant() {
        // Child better on one metric, worse on another - not Pareto dominant
        let parent_stats =
            create_evaluator_stats(&[("accuracy", 0.9, 0.1, 10), ("f1", 0.7, 0.1, 10)]);
        let child_stats =
            create_evaluator_stats(&[("accuracy", 0.8, 0.1, 10), ("f1", 0.9, 0.1, 10)]);
        let evaluators = create_test_evaluators(&[("accuracy", "max"), ("f1", "max")]);

        assert!(!is_improvement(&parent_stats, &child_stats, &evaluators)); // Tradeoff - not an improvement
    }

    #[test]
    fn test_is_improvement_mixed_directions() {
        // Test with both max and min optimization directions
        // Child: higher accuracy (max, good), lower latency (min, good)
        let parent_stats =
            create_evaluator_stats(&[("accuracy", 0.7, 0.1, 10), ("latency", 0.5, 0.1, 10)]);
        let child_stats =
            create_evaluator_stats(&[("accuracy", 0.9, 0.1, 10), ("latency", 0.3, 0.1, 10)]);
        let evaluators = create_test_evaluators(&[("accuracy", "max"), ("latency", "min")]);

        assert!(is_improvement(&parent_stats, &child_stats, &evaluators)); // Better on both metrics
    }

    #[test]
    fn test_is_improvement_missing_stats() {
        // Child missing stats for one evaluator - should skip that metric
        let parent_stats = create_evaluator_stats(&[("accuracy", 0.8, 0.1, 10)]);
        let child_stats = create_evaluator_stats(&[("accuracy", 0.9, 0.1, 10)]);
        let evaluators = create_test_evaluators(&[("accuracy", "max"), ("f1", "max")]);

        assert!(is_improvement(&parent_stats, &child_stats, &evaluators)); // Better on accuracy, f1 skipped (missing)
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
        let evaluators =
            create_test_evaluators(&[("accuracy", "max"), ("f1", "max"), ("precision", "max")]);

        assert!(is_improvement(&parent_stats, &child_stats, &evaluators)); // Better on 2, equal on 1 = improvement
    }

    #[test]
    fn test_is_improvement_min_optimization() {
        // Test with minimize optimization - child has lower value (better)
        let parent_stats = create_evaluator_stats(&[("latency", 0.9, 0.1, 10)]);
        let child_stats = create_evaluator_stats(&[("latency", 0.3, 0.1, 10)]);
        let evaluators = create_test_evaluators(&[("latency", "min")]);

        assert!(is_improvement(&parent_stats, &child_stats, &evaluators)); // Lower latency = better for min
    }

    // ============================================================================
    // Integration Tests - Python Equivalents
    // ============================================================================

    #[test]
    fn test_basic_dominance() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // Variant A dominates B: A has higher scores on all datapoints
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), Some(0.9))])),
                    (dp(2), HashMap::from([("accuracy".to_string(), Some(0.8))])),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), Some(0.7))])),
                    (dp(2), HashMap::from([("accuracy".to_string(), Some(0.6))])),
                ]),
            ),
        ]);

        let val_scores_map = variant_scores;
        let candidates = create_test_candidates(&val_scores_map);

        let datapoint_ids = extract_sorted_datapoint_ids(&val_scores_map);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        frontier
            .update(candidates)
            .expect("pareto frontier update should succeed");

        // Only variant_a should remain
        assert_eq!(frontier.variant_configs().len(), 1);
        assert!(frontier.variant_configs().contains_key("variant_a"));
        assert!(!frontier.variant_configs().contains_key("variant_b"));

        // variant_a should be in Pareto set for both datapoints
        assert_eq!(frontier.variant_frequencies().get("variant_a"), Some(&2));
    }

    #[test]
    fn test_no_global_dominance_tradeoff() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // A is better on dp1, B is better on dp2 - neither dominates
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), Some(0.9))])),
                    (dp(2), HashMap::from([("accuracy".to_string(), Some(0.6))])),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), Some(0.7))])),
                    (dp(2), HashMap::from([("accuracy".to_string(), Some(0.8))])),
                ]),
            ),
        ]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
        assert!(result.is_ok());

        // Both should remain
        assert_eq!(frontier.variant_configs().len(), 2);
        assert!(frontier.variant_configs().contains_key("variant_a"));
        assert!(frontier.variant_configs().contains_key("variant_b"));

        // Each variant is Pareto-optimal on one datapoint
        assert_eq!(frontier.variant_frequencies().get("variant_a"), Some(&1));
        assert_eq!(frontier.variant_frequencies().get("variant_b"), Some(&1));
    }

    #[test]
    fn test_instance_wise_vs_global() {
        let evaluators = create_test_evaluators(&[("acc", "max"), ("f1", "max")]);

        // C is never instance-wise Pareto-optimal, so it should be filtered early
        // A and B are each optimal on different instances
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (
                        dp(1),
                        HashMap::from([
                            ("acc".to_string(), Some(0.9)),
                            ("f1".to_string(), Some(0.8)),
                        ]),
                    ),
                    (
                        dp(2),
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
                        dp(1),
                        HashMap::from([
                            ("acc".to_string(), Some(0.7)),
                            ("f1".to_string(), Some(0.9)),
                        ]),
                    ),
                    (
                        dp(2),
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
                        dp(1),
                        HashMap::from([
                            ("acc".to_string(), Some(0.5)),
                            ("f1".to_string(), Some(0.5)),
                        ]),
                    ),
                    (
                        dp(2),
                        HashMap::from([
                            ("acc".to_string(), Some(0.5)),
                            ("f1".to_string(), Some(0.5)),
                        ]),
                    ),
                ]),
            ),
        ]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
        assert!(result.is_ok());

        // C should be filtered out, A and B should remain
        assert!(!frontier.variant_configs().contains_key("variant_c"));
        assert!(frontier.variant_configs().contains_key("variant_a"));
        assert!(frontier.variant_configs().contains_key("variant_b"));
    }

    #[test]
    fn test_mixed_optimize_directions() {
        let evaluators = create_test_evaluators(&[("accuracy", "max"), ("latency", "min")]);

        // A has higher accuracy (max) and lower latency (min) - dominates B
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([(
                    dp(1),
                    HashMap::from([
                        ("accuracy".to_string(), Some(0.9)),
                        ("latency".to_string(), Some(0.1)),
                    ]),
                )]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([(
                    dp(1),
                    HashMap::from([
                        ("accuracy".to_string(), Some(0.7)),
                        ("latency".to_string(), Some(0.3)),
                    ]),
                )]),
            ),
        ]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
        assert!(result.is_ok());

        // Only variant_a should remain
        assert_eq!(frontier.variant_configs().len(), 1);
        assert!(frontier.variant_configs().contains_key("variant_a"));
    }

    #[test]
    fn test_none_values() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // A has None on dp1, B has None on dp2 - they're incomparable
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), None)])),
                    (dp(2), HashMap::from([("accuracy".to_string(), Some(0.8))])),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), Some(0.7))])),
                    (dp(2), HashMap::from([("accuracy".to_string(), None)])),
                ]),
            ),
        ]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
        assert!(result.is_ok());

        // Both should remain (incomparable due to None values)
        assert_eq!(frontier.variant_configs().len(), 2);
        assert!(frontier.variant_configs().contains_key("variant_a"));
        assert!(frontier.variant_configs().contains_key("variant_b"));
    }

    #[test]
    fn test_single_variant() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        let variant_scores = HashMap::from([(
            "variant_a".to_string(),
            HashMap::from([(dp(1), HashMap::from([("accuracy".to_string(), Some(0.9))]))]),
        )]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
        assert!(result.is_ok());

        // Single variant should be kept
        assert_eq!(frontier.variant_configs.len(), 1);
        assert!(frontier.variant_configs.contains_key("variant_a"));
        assert_eq!(frontier.variant_frequencies().get("variant_a"), Some(&1));
    }

    #[test]
    fn test_error_results_ignored() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // B has an error on dp1 (represented as missing datapoint)
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), Some(0.9))])),
                    (dp(2), HashMap::from([("accuracy".to_string(), Some(0.8))])),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([(dp(2), HashMap::from([("accuracy".to_string(), Some(0.6))]))]),
            ),
        ]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
        assert!(result.is_ok());

        // A dominates B globally: Pareto-optimal on dp1 (only variant) and dp2 (0.8 > 0.6)
        assert_eq!(frontier.variant_configs.len(), 1);
        assert!(frontier.variant_configs.contains_key("variant_a"));
    }

    #[test]
    fn test_all_equal() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([(dp(1), HashMap::from([("accuracy".to_string(), Some(0.8))]))]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([(dp(1), HashMap::from([("accuracy".to_string(), Some(0.8))]))]),
            ),
            (
                "variant_c".to_string(),
                HashMap::from([(dp(1), HashMap::from([("accuracy".to_string(), Some(0.8))]))]),
            ),
        ]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
        assert!(result.is_ok());

        // All should remain (no one dominates)
        assert_eq!(frontier.variant_configs().len(), 3);
        assert!(frontier.variant_configs().contains_key("variant_a"));
        assert!(frontier.variant_configs().contains_key("variant_b"));
        assert!(frontier.variant_configs().contains_key("variant_c"));
    }

    #[test]
    fn test_incomparable_variants_different_datapoints() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // Test scenario: variants evaluated on different datapoints
        // This tests the edge case where GEPA's datapoint collection from first variant matters
        // Both variants have data on both datapoints, but missing (None) on different ones
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), Some(0.9))])),
                    (dp(2), HashMap::from([("accuracy".to_string(), None)])),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), None)])),
                    (dp(2), HashMap::from([("accuracy".to_string(), Some(0.7))])),
                ]),
            ),
        ]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
        assert!(result.is_ok());

        // Both variants should remain because:
        // - On dp1: variant_a (0.9) dominates variant_b (None=-inf), so A is Pareto-optimal
        // - On dp2: variant_b (0.7) dominates variant_a (None=-inf), so B is Pareto-optimal
        // - Candidate set = {A, B}
        // - Global filtering: Neither globally dominates the other (A better on dp1, B better on dp2)
        assert_eq!(frontier.variant_configs.len(), 2);
        assert!(frontier.variant_configs.contains_key("variant_a"));
        assert!(frontier.variant_configs.contains_key("variant_b"));
    }

    #[test]
    fn test_boolean_evaluator() {
        let evaluators = create_test_evaluators(&[("exact_match", "max")]);

        // A has True (1.0), B has False (0.0) for bool evaluator
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([(
                    dp(1),
                    HashMap::from([("exact_match".to_string(), Some(1.0))]),
                )]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([(
                    dp(1),
                    HashMap::from([("exact_match".to_string(), Some(0.0))]),
                )]),
            ),
        ]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
        assert!(result.is_ok());

        // A should dominate B (1.0 > 0.0)
        assert_eq!(frontier.variant_configs.len(), 1);
        assert!(frontier.variant_configs.contains_key("variant_a"));
    }

    #[test]
    fn test_unknown_metric_rejected() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // Include an unexpected evaluator key ("weird_metric") not present in config
        let variant_scores = HashMap::from([(
            "variant_a".to_string(),
            HashMap::from([(
                dp(1),
                HashMap::from([
                    ("accuracy".to_string(), Some(0.9)),
                    ("weird_metric".to_string(), Some(0.1)),
                ]),
            )]),
        )]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);

        // Unknown metric should be dropped; update should still succeed using known metrics
        assert!(result.is_ok());
        assert_eq!(frontier.variant_configs.len(), 1);
        assert!(frontier.variant_configs.contains_key("variant_a"));
    }

    #[test]
    fn test_cache_pruning_removes_dominated_vectors() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // dp1: tie, dp2: variant_a better => candidate_set has {a,b}, global filtering keeps only A
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), Some(0.8))])),
                    (dp(2), HashMap::from([("accuracy".to_string(), Some(0.9))])),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), Some(0.8))])),
                    (dp(2), HashMap::from([("accuracy".to_string(), Some(0.7))])),
                ]),
            ),
        ]);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let candidates = create_test_candidates(&variant_scores);

        let result = frontier.update(candidates);
        assert!(result.is_ok());

        // Only variant_a should remain in frontier and cache
        assert_eq!(frontier.variant_configs.len(), 1);
        assert!(frontier.variant_configs.contains_key("variant_a"));
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
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([(dp(1), HashMap::from([("accuracy".to_string(), Some(0.9))]))]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([(dp(1), HashMap::from([("accuracy".to_string(), Some(0.7))]))]),
            ),
            (
                "variant_c".to_string(),
                HashMap::from([(dp(1), HashMap::from([("accuracy".to_string(), Some(0.5))]))]),
            ),
        ]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
        assert!(result.is_ok());

        // Only A should remain
        assert_eq!(frontier.variant_configs.len(), 1);
        assert!(frontier.variant_configs.contains_key("variant_a"));
    }

    #[test]
    fn test_runtime_performance() {
        use std::time::Instant;

        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        let num_variants = 10;
        let num_datapoints = 100;

        // Generate variant results with random scores
        let mut variant_scores = HashMap::new();
        for v in 0..num_variants {
            let variant_name = format!("variant_{v}");
            let mut datapoint_scores = HashMap::new();

            for d in 0..num_datapoints {
                let datapoint_id = dp(d as u128 + 1);
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
        let candidates = create_test_candidates(&val_scores_map);

        // Measure execution time
        let start = Instant::now();
        let datapoint_ids = extract_sorted_datapoint_ids(&val_scores_map);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
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
            !frontier.variant_configs.is_empty(),
            "Result should contain at least one non-dominated variant"
        );
    }

    // ============================================================================
    // Rust-Specific Edge Cases
    // ============================================================================

    #[test]
    fn test_empty_candidates() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        let val_scores_map: VariantScoresMap = HashMap::new();
        let candidates: HashMap<VariantName, Candidate> = HashMap::new();

        let datapoint_ids = extract_sorted_datapoint_ids(&val_scores_map);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);

        // Should return error for empty candidates
        assert!(result.is_err());
    }

    #[test]
    fn test_all_evaluations_failed() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // All val_scores entries are None (evaluation failed)
        let val_scores_map: VariantScoresMap = HashMap::new();
        let candidates: HashMap<VariantName, Candidate> = HashMap::new();

        let datapoint_ids = extract_sorted_datapoint_ids(&val_scores_map);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);

        // Should return error when all evaluations failed
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("No validation scores provided for any variant")
        );
    }

    #[test]
    fn test_all_scores_none() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // Every per_datapoint score is None
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), None)])),
                    (dp(2), HashMap::from([("accuracy".to_string(), None)])),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), None)])),
                    (dp(2), HashMap::from([("accuracy".to_string(), None)])),
                ]),
            ),
        ]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);

        // Should return error when all scores are None
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("All new variants failed to produce valid scores")
        );
    }

    #[test]
    fn test_empty_datapoints() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // Empty per_datapoint HashMap - all variants have no evaluation data
        let variant_scores = HashMap::from([
            ("variant_a".to_string(), HashMap::new()),
            ("variant_b".to_string(), HashMap::new()),
        ]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);

        // Should return error when all variants have empty datapoint maps
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("empty datapoint maps"));
    }

    #[test]
    fn test_frequency_calculation_detailed() {
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // A wins on dp1 and dp2, B wins on dp3, C never wins
        let variant_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), Some(0.9))])),
                    (dp(2), HashMap::from([("accuracy".to_string(), Some(0.9))])),
                    (dp(3), HashMap::from([("accuracy".to_string(), Some(0.5))])),
                ]),
            ),
            (
                "variant_b".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), Some(0.7))])),
                    (dp(2), HashMap::from([("accuracy".to_string(), Some(0.7))])),
                    (dp(3), HashMap::from([("accuracy".to_string(), Some(0.9))])),
                ]),
            ),
            (
                "variant_c".to_string(),
                HashMap::from([
                    (dp(1), HashMap::from([("accuracy".to_string(), Some(0.5))])),
                    (dp(2), HashMap::from([("accuracy".to_string(), Some(0.5))])),
                    (dp(3), HashMap::from([("accuracy".to_string(), Some(0.5))])),
                ]),
            ),
        ]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
        assert!(result.is_ok());

        // C is dominated, A and B remain
        assert_eq!(frontier.variant_configs.len(), 2);
        assert!(frontier.variant_configs.contains_key("variant_a"));
        assert!(frontier.variant_configs.contains_key("variant_b"));

        // A should be in Pareto set for dp1 and dp2 (freq=2)
        assert_eq!(frontier.variant_frequencies().get("variant_a"), Some(&2));
        // B should be in Pareto set for dp3 (freq=1)
        assert_eq!(frontier.variant_frequencies().get("variant_b"), Some(&1));
        // C should not be in frequencies (filtered out)
        assert!(!frontier.variant_frequencies().contains_key("variant_c"));
    }

    #[test]
    fn test_high_missing_rate_warning() {
        // This test verifies that the warning log is triggered for high missing rates
        let evaluators = create_test_evaluators(&[("accuracy", "max")]);

        // variant_a has 60% missing data (3 out of 5 datapoints have scores)
        let variant_scores = HashMap::from([(
            "variant_a".to_string(),
            HashMap::from([
                (dp(1), HashMap::from([("accuracy".to_string(), Some(0.9))])),
                (dp(2), HashMap::from([("accuracy".to_string(), Some(0.9))])),
                (dp(3), HashMap::from([("accuracy".to_string(), None)])),
                (dp(4), HashMap::from([("accuracy".to_string(), None)])),
                (dp(5), HashMap::from([("accuracy".to_string(), None)])),
            ]),
        )]);

        let candidates = create_test_candidates(&variant_scores);

        let datapoint_ids = extract_sorted_datapoint_ids(&variant_scores);
        let mut frontier = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result = frontier.update(candidates);
        assert!(result.is_ok());

        assert_eq!(frontier.variant_configs.len(), 1);

        // Note: We can't directly assert on log output, but this test documents
        // the behavior and ensures the code path executes without panicking
        // The warning should be logged: "Variant 'variant_a' has high missing score rate: 60.0%"
    }

    #[test]
    fn test_cache_correctness() {
        // This test validates that the cache produces correct results in a typical GEPA workflow:
        // - Iteration 1: Evaluate initial population
        // - Iteration 2: Keep some variants from frontier + add new mutations
        // The cache should reuse vectors for persisting variants correctly.

        let evaluators = create_test_evaluators(&[("accuracy", "max"), ("latency", "min")]);

        // Simulate Iteration 1: Initial population [A, B, C]
        let iteration1_scores = HashMap::from([
            (
                "variant_a".to_string(),
                HashMap::from([
                    (
                        dp(1),
                        HashMap::from([
                            ("accuracy".to_string(), Some(0.9)),
                            ("latency".to_string(), Some(0.2)),
                        ]),
                    ),
                    (
                        dp(2),
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
                        dp(1),
                        HashMap::from([
                            ("accuracy".to_string(), Some(0.7)),
                            ("latency".to_string(), Some(0.1)),
                        ]),
                    ),
                    (
                        dp(2),
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
                        dp(1),
                        HashMap::from([
                            ("accuracy".to_string(), Some(0.6)),
                            ("latency".to_string(), Some(0.4)),
                        ]),
                    ),
                    (
                        dp(2),
                        HashMap::from([
                            ("accuracy".to_string(), Some(0.7)),
                            ("latency".to_string(), Some(0.5)),
                        ]),
                    ),
                ]),
            ),
        ]);

        let candidates1 = create_test_candidates(&iteration1_scores);

        // Run iteration 1 with cache (frontier instance will be reused for iteration 2)
        let datapoint_ids = extract_sorted_datapoint_ids(&iteration1_scores);
        let mut frontier1_cached = ParetoFrontier::new(datapoint_ids.clone(), &evaluators, None);
        let result1_cached = frontier1_cached.update(candidates1.clone());
        assert!(result1_cached.is_ok());

        // Run iteration 1 without cache (for comparison)
        let mut frontier1_no_cache = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result1_no_cache = frontier1_no_cache.update(candidates1);
        assert!(result1_no_cache.is_ok());

        // Iteration 1 results should be identical
        assert_eq!(
            frontier1_cached
                .variant_configs
                .keys()
                .collect::<HashSet<_>>(),
            frontier1_no_cache
                .variant_configs
                .keys()
                .collect::<HashSet<_>>(),
            "Iteration 1: cached and non-cached should produce same variants"
        );

        // Simulate Iteration 2: Keep A and B (from frontier), add new variant D
        // variant_d is NEW (mutation)
        let variant_d_scores = HashMap::from([(
            "variant_d".to_string(),
            HashMap::from([
                (
                    dp(1),
                    HashMap::from([
                        ("accuracy".to_string(), Some(0.85)),
                        ("latency".to_string(), Some(0.15)),
                    ]),
                ),
                (
                    dp(2),
                    HashMap::from([
                        ("accuracy".to_string(), Some(0.75)),
                        ("latency".to_string(), Some(0.25)),
                    ]),
                ),
            ]),
        )]);

        // For cached path: only add the NEW variant D (A and B already in frontier from iteration 1)
        let candidates2_new = create_test_candidates(&variant_d_scores);

        // Run iteration 2 WITH cache (reuse same frontier1_cached instance, so cache is preserved)
        // Only pass the new variant D since A and B are already in the frontier
        let result2_cached = frontier1_cached.update(candidates2_new);
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

        let candidates2_all = create_test_candidates(&iteration2_all_scores);

        // Run iteration 2 WITHOUT cache (create fresh frontier, compute all from scratch)
        let datapoint_ids = extract_sorted_datapoint_ids(&iteration2_all_scores);
        let mut frontier2_no_cache = ParetoFrontier::new(datapoint_ids, &evaluators, None);
        let result2_no_cache = frontier2_no_cache.update(candidates2_all);
        assert!(result2_no_cache.is_ok());

        // Critical assertion: cached and non-cached should produce identical results
        assert_eq!(
            frontier1_cached
                .variant_configs
                .keys()
                .collect::<HashSet<_>>(),
            frontier2_no_cache
                .variant_configs
                .keys()
                .collect::<HashSet<_>>(),
            "Iteration 2: cached and non-cached should produce same variants"
        );

        assert_eq!(
            frontier1_cached.variant_frequencies, frontier2_no_cache.variant_frequencies,
            "Iteration 2: cached and non-cached should produce same frequencies"
        );
    }
}
