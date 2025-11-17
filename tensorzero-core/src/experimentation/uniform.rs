use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    db::{feedback::FeedbackQueries, postgres::PostgresConnectionInfo},
    error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    experimentation::get_uniform_value,
    variant::VariantInfo,
};

use super::{check_duplicates_across, check_duplicates_within, VariantSampler};

/// Pure function for uniform sampling logic.
/// Given a uniform sample in [0, 1), selects a variant from active_variants
/// using uniform sampling from candidate_variants if specified and their intersection is nonempty,
/// or sequential sampling from fallback_variants otherwise.
///
/// Returns the name of the selected variant, which is guaranteed to be in active_variants.
pub(crate) fn sample_uniform(
    active_variants: &BTreeMap<String, Arc<VariantInfo>>,
    candidate_variants: Option<&[String]>,
    fallback_variants: &[String],
    uniform_sample: f64,
) -> Result<String, Error> {
    // Determine the sampling pool based on candidate_variants
    let candidates_to_use: Vec<&str> = match candidate_variants {
        None => {
            // No candidate_variants specified, use all active variants
            active_variants.keys().map(|s| s.as_str()).collect()
        }
        Some(candidates) => {
            // Use specified candidates that are active
            candidates
                .iter()
                .filter(|c| active_variants.contains_key(c.as_str()))
                .map(|s| s.as_str())
                .collect()
        }
    };

    if !candidates_to_use.is_empty() {
        // Uniform sampling from candidates
        let num_candidates = candidates_to_use.len();
        let index = (uniform_sample * num_candidates as f64).floor() as usize;
        let index = index.min(num_candidates - 1); // Clamp to valid range
        return Ok(candidates_to_use[index].to_string());
    }

    // No active candidates, try fallback variants
    // Select the first variant from the ranked fallback_variants list that is active
    for variant_name in fallback_variants {
        if active_variants.contains_key(variant_name) {
            return Ok(variant_name.clone());
        }
    }

    // No active fallback variants found
    Err(ErrorDetails::NoFallbackVariantsRemaining.into())
}

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct UniformConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    candidate_variants: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fallback_variants: Option<Vec<String>>,
}

impl UniformConfig {
    /// Validate that the specified variants exist in the function's variants map
    pub fn load(
        &self,
        function_variants: &HashMap<String, Arc<VariantInfo>>,
    ) -> Result<Self, Error> {
        // Validate candidate_variants if specified
        if let Some(candidates) = &self.candidate_variants {
            if candidates.is_empty() && self.fallback_variants.is_none() {
                // Empty candidates with no fallbacks is an error
                return Err(Error::new(ErrorDetails::Config {
                    message: "uniform experimentation: candidate_variants cannot be empty when fallback_variants is not specified".to_string(),
                }));
            }
            // Check for duplicates within candidate_variants
            check_duplicates_within(candidates, "candidate_variants")?;

            for variant_name in candidates {
                if !function_variants.contains_key(variant_name) {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "uniform experimentation: candidate variant `{variant_name}` does not exist in function variants"
                        ),
                    }));
                }
            }
        }

        // Validate fallback_variants if specified
        if let Some(fallbacks) = &self.fallback_variants {
            if fallbacks.is_empty()
                && self
                    .candidate_variants
                    .as_ref()
                    .is_some_and(|c| c.is_empty())
            {
                // Empty fallbacks with empty candidates is an error
                return Err(Error::new(ErrorDetails::Config {
                    message: "uniform experimentation: fallback_variants cannot be empty when candidate_variants is empty".to_string(),
                }));
            }
            // Check for duplicates within fallback_variants
            check_duplicates_within(fallbacks, "fallback_variants")?;

            for variant_name in fallbacks {
                if !function_variants.contains_key(variant_name) {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "uniform experimentation: fallback variant `{variant_name}` does not exist in function variants"
                        ),
                    }));
                }
            }
        }

        // Check for duplicates across both lists
        if let (Some(candidates), Some(fallbacks)) =
            (&self.candidate_variants, &self.fallback_variants)
        {
            check_duplicates_across(candidates, fallbacks)?;
        }

        Ok(Self {
            candidate_variants: self.candidate_variants.clone(),
            fallback_variants: self.fallback_variants.clone(),
        })
    }
}

impl VariantSampler for UniformConfig {
    async fn setup(
        &self,
        _db: Arc<dyn FeedbackQueries + Send + Sync>,
        function_name: &str,
        _postgres: &PostgresConnectionInfo,
        _cancel_token: CancellationToken,
    ) -> Result<(), Error> {
        // Validate that at least one sampling strategy is available
        // This must align with the sample() semantics (lines 178-182):
        // - (None, None): candidates = all variants (ok)
        // - (None, Some(_)): candidates = empty, rely on fallbacks
        // - (Some(c), _): candidates = c
        let has_candidates = match (&self.candidate_variants, &self.fallback_variants) {
            (None, None) => true,          // All variants available as candidates
            (None, Some(_)) => false,      // Fallbacks specified, no candidates
            (Some(c), _) => !c.is_empty(), // Explicit candidates must be non-empty
        };

        let has_fallbacks = self
            .fallback_variants
            .as_ref()
            .is_some_and(|f| !f.is_empty());

        if !has_candidates && !has_fallbacks {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "Uniform config for function '{function_name}' has no valid sampling strategy. \
                    When fallback_variants is specified, either candidate_variants must be non-empty \
                    or fallback_variants must be non-empty."
                ),
            }));
        }

        Ok(())
    }

    /// Sample a variant uniformly from candidates, or sequentially from fallbacks
    /// This function pops the sampled variant from the active variants map.
    /// NOTE: We use a BTreeMap to ensure that the variants are sorted by their names and the
    /// sampling choices are deterministic given an episode ID.
    async fn sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
        _postgres: &PostgresConnectionInfo,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        // Sampling is stable per episode ID
        let uniform_sample = get_uniform_value(function_name, &episode_id);

        let fallback_variants = self.fallback_variants.as_deref().unwrap_or(&[]);

        // Determine candidate_variants based on the presence of fallback_variants
        // If fallback_variants is specified, None candidate_variants means empty (no candidates)
        // If fallback_variants is not specified, None candidate_variants means all variants
        let candidates = match (&self.candidate_variants, &self.fallback_variants) {
            (None, None) => None, // Truly unspecified, use all variants
            (None, Some(_)) => Some(&[] as &[String]), // Fallbacks specified, no candidates
            (Some(c), _) => Some(c.as_slice()), // Explicit candidates
        };

        let selected_variant_name = sample_uniform(
            active_variants,
            candidates,
            fallback_variants,
            uniform_sample,
        )?;

        active_variants
            .remove_entry(&selected_variant_name)
            .ok_or_else(|| {
                Error::new(ErrorDetails::InvalidFunctionVariants {
                    message: format!(
                        "Function `{function_name}` has no variant for the sampled variant `{selected_variant_name}`. {IMPOSSIBLE_ERROR_MESSAGE}"
                    ),
                })
            })
    }

    fn allowed_variants(&self) -> impl Iterator<Item = &str> + '_ {
        let candidates = self
            .candidate_variants
            .iter()
            .flat_map(|v| v.iter().map(String::as_str));
        let fallbacks = self
            .fallback_variants
            .iter()
            .flat_map(|v| v.iter().map(String::as_str));
        candidates.chain(fallbacks)
    }

    fn get_current_display_probabilities<'a>(
        &self,
        _function_name: &str,
        active_variants: &'a HashMap<String, Arc<VariantInfo>>,
        _postgres: &PostgresConnectionInfo,
    ) -> Result<HashMap<&'a str, f64>, Error> {
        // If there are no active variants, return an empty map
        if active_variants.is_empty() {
            return Ok(HashMap::new());
        }

        // Determine which candidates are active based on the presence of fallback_variants
        let active_candidates: Vec<&'a str> =
            match (&self.candidate_variants, &self.fallback_variants) {
                (None, None) => {
                    // Truly unspecified, all active variants are candidates
                    active_variants.keys().map(|s| s.as_str()).collect()
                }
                (None, Some(_)) => {
                    // Fallbacks specified, no candidates
                    vec![]
                }
                (Some(candidates), _) => {
                    // Use specified candidates that are active
                    active_variants
                        .keys()
                        .filter(|k| candidates.contains(k))
                        .map(|s| s.as_str())
                        .collect()
                }
            };

        if active_candidates.is_empty() {
            // No active candidates, use fallback variants
            let fallback_variants = self.fallback_variants.as_deref().unwrap_or(&[]);

            // Find the first variant from the ranked fallback_variants list that is active
            let first_active_fallback = fallback_variants
                .iter()
                .find(|variant_name| active_variants.contains_key(*variant_name));

            if let Some(selected_variant) = first_active_fallback {
                // The first active fallback variant gets 100% probability
                // All other active fallback variants get 0% probability
                let mut probabilities: HashMap<&'a str, f64> = HashMap::new();
                for key in active_variants.keys() {
                    if fallback_variants.contains(key) {
                        if key == selected_variant {
                            probabilities.insert(key.as_str(), 1.0);
                        } else {
                            probabilities.insert(key.as_str(), 0.0);
                        }
                    }
                }
                Ok(probabilities)
            } else {
                Err(ErrorDetails::NoFallbackVariantsRemaining.into())
            }
        } else {
            // Uniform distribution over active candidates
            let num_candidates = active_candidates.len();
            let uniform_prob = 1.0 / num_candidates as f64;

            let mut probabilities: HashMap<&'a str, f64> = HashMap::new();
            for variant_name in active_candidates {
                probabilities.insert(variant_name, uniform_prob);
            }

            Ok(probabilities)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ErrorContext, SchemaData};
    use crate::db::clickhouse::ClickHouseConnectionInfo;
    use crate::variant::{chat_completion::UninitializedChatCompletionConfig, VariantConfig};

    fn create_test_variants(names: &[&str]) -> BTreeMap<String, Arc<VariantInfo>> {
        names
            .iter()
            .map(|&name| {
                (
                    name.to_string(),
                    Arc::new(VariantInfo {
                        inner: VariantConfig::ChatCompletion(
                            UninitializedChatCompletionConfig {
                                weight: None,
                                model: "model-name".into(),
                                ..Default::default()
                            }
                            .load(&SchemaData::default(), &ErrorContext::new_test())
                            .unwrap(),
                        ),
                        timeouts: Default::default(),
                    }),
                )
            })
            .collect()
    }

    #[test]
    fn test_sample_uniform_all_variants() {
        // Test (None, None) - uniform over all active variants
        let active_variants = create_test_variants(&["A", "B", "C"]);

        // uniform_sample = 0.2 should select variant at index floor(0.2 * 3) = 0 (A)
        let result = sample_uniform(&active_variants, None, &[], 0.2);
        assert_eq!(result.unwrap(), "A");

        // uniform_sample = 0.5 should select variant at index floor(0.5 * 3) = 1 (B)
        let result = sample_uniform(&active_variants, None, &[], 0.5);
        assert_eq!(result.unwrap(), "B");

        // uniform_sample = 0.8 should select variant at index floor(0.8 * 3) = 2 (C)
        let result = sample_uniform(&active_variants, None, &[], 0.8);
        assert_eq!(result.unwrap(), "C");
    }

    #[test]
    fn test_sample_uniform_explicit_candidates() {
        // Test with explicit candidate_variants
        let active_variants = create_test_variants(&["A", "B", "C"]);
        let candidates = vec!["A".to_string(), "B".to_string()];

        // Should only sample from A and B
        let result = sample_uniform(&active_variants, Some(&candidates), &[], 0.2);
        assert_eq!(result.unwrap(), "A");

        let result = sample_uniform(&active_variants, Some(&candidates), &[], 0.7);
        assert_eq!(result.unwrap(), "B");
    }

    #[test]
    fn test_sample_uniform_fallback_only() {
        // Test (None, Some([...])) or (Some([]), Some([...]))
        let active_variants = create_test_variants(&["A", "B", "C"]);
        let fallbacks = vec!["B".to_string(), "C".to_string()];

        // Empty candidates, should use fallback (first active = B)
        let result = sample_uniform(&active_variants, Some(&[]), &fallbacks, 0.5);
        assert_eq!(result.unwrap(), "B");
    }

    #[test]
    fn test_sample_uniform_fallback_sequential() {
        // Test that fallbacks are sequential, not uniform
        let active_variants = create_test_variants(&["A", "B", "C"]);
        let fallbacks = vec!["B".to_string(), "C".to_string(), "A".to_string()];

        // Should always pick first active fallback (B) regardless of uniform_sample
        let result = sample_uniform(&active_variants, Some(&[]), &fallbacks, 0.1);
        assert_eq!(result.unwrap(), "B");

        let result = sample_uniform(&active_variants, Some(&[]), &fallbacks, 0.9);
        assert_eq!(result.unwrap(), "B");
    }

    #[test]
    fn test_sample_uniform_fallback_skips_inactive() {
        // Test that fallback skips inactive variants
        let active_variants = create_test_variants(&["A", "C"]);
        let fallbacks = vec!["B".to_string(), "C".to_string(), "A".to_string()];

        // B is not active, should pick C (first active fallback)
        let result = sample_uniform(&active_variants, Some(&[]), &fallbacks, 0.5);
        assert_eq!(result.unwrap(), "C");
    }

    #[test]
    fn test_sample_uniform_no_active_error() {
        // Test error when no active variants match
        let active_variants = create_test_variants(&["A", "B"]);
        let candidates = vec!["C".to_string(), "D".to_string()];
        let fallbacks = vec!["E".to_string()];

        let result = sample_uniform(&active_variants, Some(&candidates), &fallbacks, 0.5);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_uniform_config_default() {
        // Test default config (None, None)
        let config = UniformConfig::default();
        let variants = create_test_variants(&["A", "B", "C"]);
        let mut active_variants = variants.clone();
        let episode_id = Uuid::now_v7();
        let postgres = PostgresConnectionInfo::new_disabled();

        let (variant_name, _) = config
            .sample("test_function", episode_id, &mut active_variants, &postgres)
            .await
            .unwrap();

        assert!(["A", "B", "C"].contains(&variant_name.as_str()));
        assert_eq!(active_variants.len(), 2); // One variant should be removed
    }

    #[tokio::test]
    async fn test_uniform_config_explicit_candidates() {
        // Test explicit candidates
        let config = UniformConfig {
            candidate_variants: Some(vec!["A".to_string(), "B".to_string()]),
            fallback_variants: None,
        };

        let variants = create_test_variants(&["A", "B", "C"]);

        // Validate during load
        let variants_map: HashMap<_, _> = variants
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let loaded_config = config.load(&variants_map).unwrap();

        let mut active_variants = variants.clone();
        let episode_id = Uuid::now_v7();
        let postgres = PostgresConnectionInfo::new_disabled();

        let (variant_name, _) = loaded_config
            .sample("test_function", episode_id, &mut active_variants, &postgres)
            .await
            .unwrap();

        assert!(["A", "B"].contains(&variant_name.as_str()));
        assert!(!["C"].contains(&variant_name.as_str()));
    }

    #[tokio::test]
    async fn test_uniform_config_fallback_only() {
        // Test fallback only (None candidates with Some fallbacks)
        let config = UniformConfig {
            candidate_variants: None,
            fallback_variants: Some(vec!["B".to_string(), "C".to_string()]),
        };

        let variants = create_test_variants(&["A", "B", "C"]);
        let mut active_variants = variants.clone();
        let episode_id = Uuid::now_v7();
        let postgres = PostgresConnectionInfo::new_disabled();

        // First sample should pick B (first fallback)
        let (variant_name, _) = config
            .sample("test_function", episode_id, &mut active_variants, &postgres)
            .await
            .unwrap();

        assert_eq!(variant_name, "B");
    }

    #[tokio::test]
    async fn test_uniform_distribution() {
        // Test that uniform sampling produces equal distribution
        let config = UniformConfig::default();
        let variants = create_test_variants(&["A", "B", "C"]);

        let sample_size = 10_000;
        let mut counts = std::collections::HashMap::new();
        let postgres = PostgresConnectionInfo::new_disabled();

        for i in 0..sample_size {
            let mut active_variants = variants.clone();
            let episode_id = Uuid::from_u128(i as u128);
            let (variant_name, _) = config
                .sample("test_function", episode_id, &mut active_variants, &postgres)
                .await
                .unwrap();
            *counts.entry(variant_name).or_insert(0) += 1;
        }

        // Check that distribution is roughly equal (within 2% tolerance)
        let expected_prob = 1.0 / 3.0;
        let tolerance = 0.02;

        for variant in &["A", "B", "C"] {
            let actual_prob = *counts.get(*variant).unwrap_or(&0) as f64 / sample_size as f64;
            assert!(
                (actual_prob - expected_prob).abs() < tolerance,
                "Variant {variant}: expected {expected_prob:.3}, got {actual_prob:.3}"
            );
        }
    }

    #[test]
    fn test_load_validation_invalid_candidate() {
        let config = UniformConfig {
            candidate_variants: Some(vec!["A".to_string(), "INVALID".to_string()]),
            fallback_variants: None,
        };

        let variants: HashMap<_, _> = create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let result = config.load(&variants);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("candidate variant `INVALID` does not exist"));
    }

    #[test]
    fn test_load_validation_invalid_fallback() {
        let config = UniformConfig {
            candidate_variants: None,
            fallback_variants: Some(vec!["A".to_string(), "INVALID".to_string()]),
        };

        let variants: HashMap<_, _> = create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let result = config.load(&variants);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("fallback variant `INVALID` does not exist"));
    }

    #[test]
    fn test_load_validation_empty_candidates_no_fallbacks() {
        let config = UniformConfig {
            candidate_variants: Some(vec![]),
            fallback_variants: None,
        };

        let variants: HashMap<_, _> = create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let result = config.load(&variants);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("candidate_variants cannot be empty"));
    }

    #[test]
    fn test_get_current_display_probabilities_all_variants() {
        let config = UniformConfig::default();
        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "B", "C"]).into_iter().collect();
        let postgres = PostgresConnectionInfo::new_disabled();

        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // All variants should have equal probability
        assert_eq!(probs.len(), 3);
        assert!((probs["A"] - 1.0 / 3.0).abs() < 1e-9);
        assert!((probs["B"] - 1.0 / 3.0).abs() < 1e-9);
        assert!((probs["C"] - 1.0 / 3.0).abs() < 1e-9);

        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_current_display_probabilities_explicit_candidates() {
        let config = UniformConfig {
            candidate_variants: Some(vec!["A".to_string(), "B".to_string()]),
            fallback_variants: None,
        };

        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "B", "C"]).into_iter().collect();
        let postgres = PostgresConnectionInfo::new_disabled();

        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // Only A and B should be returned (candidates), not C
        assert_eq!(probs.len(), 2);
        assert!((probs["A"] - 0.5).abs() < 1e-9);
        assert!((probs["B"] - 0.5).abs() < 1e-9);
        assert!(!probs.contains_key("C"));

        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_current_display_probabilities_fallback() {
        let config = UniformConfig {
            candidate_variants: Some(vec![]),
            fallback_variants: Some(vec!["B".to_string(), "C".to_string()]),
        };

        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "B", "C"]).into_iter().collect();
        let postgres = PostgresConnectionInfo::new_disabled();

        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // First active fallback (B) gets 100%, others get 0%
        assert_eq!(probs.len(), 2);
        assert!((probs["B"] - 1.0).abs() < 1e-9);
        assert!((probs["C"] - 0.0).abs() < 1e-9);

        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_consistent_behavior_no_zero_probabilities_in_candidate_mode() {
        // Test that candidate mode doesn't include variants with 0 probability
        let config = UniformConfig {
            candidate_variants: Some(vec!["A".to_string()]),
            fallback_variants: None,
        };

        let active_variants: HashMap<_, _> = create_test_variants(&["A", "B", "C", "D"])
            .into_iter()
            .collect();
        let postgres = PostgresConnectionInfo::new_disabled();

        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // Only the candidate variant should be included
        assert_eq!(probs.len(), 1);
        assert!((probs["A"] - 1.0).abs() < 1e-9);

        // Verify non-candidate variants are not included (not even with 0.0)
        assert!(!probs.contains_key("B"));
        assert!(!probs.contains_key("C"));
        assert!(!probs.contains_key("D"));
    }

    #[test]
    fn test_consistent_behavior_no_zero_probabilities_in_fallback_mode() {
        // Test that fallback mode doesn't include variants with 0 probability
        // when there are other active variants not in the fallback list
        let config = UniformConfig {
            candidate_variants: Some(vec![]),
            fallback_variants: Some(vec!["B".to_string(), "C".to_string()]),
        };

        let active_variants: HashMap<_, _> = create_test_variants(&["A", "B", "C", "D"])
            .into_iter()
            .collect();
        let postgres = PostgresConnectionInfo::new_disabled();

        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // Only fallback variants should be included
        assert_eq!(probs.len(), 2);
        assert!((probs["B"] - 1.0).abs() < 1e-9);
        assert!((probs["C"] - 0.0).abs() < 1e-9);

        // Verify non-fallback variants are not included (not even with 0.0)
        assert!(!probs.contains_key("A"));
        assert!(!probs.contains_key("D"));
    }

    #[test]
    fn test_consistent_behavior_both_modes_exclude_irrelevant_variants() {
        // Test that both modes consistently exclude irrelevant variants
        let postgres = PostgresConnectionInfo::new_disabled();

        // Candidate mode: 2 candidates out of 5 active variants
        let candidate_config = UniformConfig {
            candidate_variants: Some(vec!["X".to_string(), "Y".to_string()]),
            fallback_variants: None,
        };
        let active_variants_1: HashMap<_, _> = create_test_variants(&["X", "Y", "Z", "W", "V"])
            .into_iter()
            .collect();

        let candidate_probs = candidate_config
            .get_current_display_probabilities("test", &active_variants_1, &postgres)
            .unwrap();

        assert_eq!(
            candidate_probs.len(),
            2,
            "Candidate mode should only return candidate variants"
        );
        assert!(candidate_probs.contains_key("X"));
        assert!(candidate_probs.contains_key("Y"));
        assert!(!candidate_probs.contains_key("Z"));
        assert!(!candidate_probs.contains_key("W"));
        assert!(!candidate_probs.contains_key("V"));

        // Fallback mode: 2 fallback variants out of 5 active variants
        let fallback_config = UniformConfig {
            candidate_variants: Some(vec![]),
            fallback_variants: Some(vec!["X".to_string(), "Y".to_string()]),
        };
        let active_variants_2: HashMap<_, _> = create_test_variants(&["X", "Y", "Z", "W", "V"])
            .into_iter()
            .collect();

        let fallback_probs = fallback_config
            .get_current_display_probabilities("test", &active_variants_2, &postgres)
            .unwrap();

        assert_eq!(
            fallback_probs.len(),
            2,
            "Fallback mode should only return fallback variants"
        );
        assert!(fallback_probs.contains_key("X"));
        assert!(fallback_probs.contains_key("Y"));
        assert!(!fallback_probs.contains_key("Z"));
        assert!(!fallback_probs.contains_key("W"));
        assert!(!fallback_probs.contains_key("V"));
    }

    #[tokio::test]
    async fn test_setup_validation_empty_both() {
        let config = UniformConfig {
            candidate_variants: Some(vec![]),
            fallback_variants: Some(vec![]),
        };

        let db = Arc::new(ClickHouseConnectionInfo::new_disabled())
            as Arc<dyn FeedbackQueries + Send + Sync>;
        let postgres = PostgresConnectionInfo::new_disabled();
        let cancel_token = CancellationToken::new();

        let result = config
            .setup(db, "test_function", &postgres, cancel_token)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_setup_validation_none_candidates_empty_fallbacks() {
        // This is the bug scenario: candidate_variants: None, fallback_variants: Some([])
        // Should be rejected because it would fail at runtime
        let config = UniformConfig {
            candidate_variants: None,
            fallback_variants: Some(vec![]),
        };

        let db = Arc::new(ClickHouseConnectionInfo::new_disabled())
            as Arc<dyn FeedbackQueries + Send + Sync>;
        let postgres = PostgresConnectionInfo::new_disabled();
        let cancel_token = CancellationToken::new();

        let result = config
            .setup(db, "test_function", &postgres, cancel_token)
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no valid sampling strategy"));
    }

    #[tokio::test]
    async fn test_setup_validation_none_candidates_with_fallbacks() {
        // This should pass: candidate_variants: None, fallback_variants: Some([...])
        // Runtime will use only fallbacks
        let config = UniformConfig {
            candidate_variants: None,
            fallback_variants: Some(vec!["A".to_string(), "B".to_string()]),
        };

        let db = Arc::new(ClickHouseConnectionInfo::new_disabled())
            as Arc<dyn FeedbackQueries + Send + Sync>;
        let postgres = PostgresConnectionInfo::new_disabled();
        let cancel_token = CancellationToken::new();

        let result = config
            .setup(db, "test_function", &postgres, cancel_token)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_setup_validation_none_none() {
        // This should pass: candidate_variants: None, fallback_variants: None
        // Runtime will use all variants
        let config = UniformConfig {
            candidate_variants: None,
            fallback_variants: None,
        };

        let db = Arc::new(ClickHouseConnectionInfo::new_disabled())
            as Arc<dyn FeedbackQueries + Send + Sync>;
        let postgres = PostgresConnectionInfo::new_disabled();
        let cancel_token = CancellationToken::new();

        let result = config
            .setup(db, "test_function", &postgres, cancel_token)
            .await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_validation_duplicate_candidates() {
        let config = UniformConfig {
            candidate_variants: Some(vec!["A".to_string(), "B".to_string(), "A".to_string()]),
            fallback_variants: None,
        };

        let variants: HashMap<_, _> = create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let result = config.load(&variants);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("`candidate_variants` contains duplicate entries"));
        assert!(err_msg.contains("A"));
    }

    #[test]
    fn test_load_validation_duplicate_fallbacks() {
        let config = UniformConfig {
            candidate_variants: Some(vec!["A".to_string()]),
            fallback_variants: Some(vec!["B".to_string(), "C".to_string(), "B".to_string()]),
        };

        let variants: HashMap<_, _> = create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let result = config.load(&variants);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("`fallback_variants` contains duplicate entries"));
        assert!(err_msg.contains("B"));
    }

    #[test]
    fn test_load_validation_duplicate_across_lists() {
        let config = UniformConfig {
            candidate_variants: Some(vec!["A".to_string(), "B".to_string()]),
            fallback_variants: Some(vec!["B".to_string(), "C".to_string()]),
        };

        let variants: HashMap<_, _> = create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let result = config.load(&variants);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot appear in both `candidate_variants` and `fallback_variants`")
        );
        assert!(err_msg.contains("B"));
    }

    #[test]
    fn test_load_validation_multiple_duplicates_across_lists() {
        let config = UniformConfig {
            candidate_variants: Some(vec!["A".to_string(), "B".to_string(), "C".to_string()]),
            fallback_variants: Some(vec!["B".to_string(), "C".to_string(), "D".to_string()]),
        };

        let variants: HashMap<_, _> = create_test_variants(&["A", "B", "C", "D"])
            .into_iter()
            .collect();

        let result = config.load(&variants);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot appear in both `candidate_variants` and `fallback_variants`")
        );
        assert!(err_msg.contains("B"));
        assert!(err_msg.contains("C"));
    }
}
