use std::collections::BTreeMap;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::db::feedback::FeedbackQueries;
use crate::db::postgres::PostgresConnectionInfo;
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::variant::VariantInfo;

pub mod asymptotic_confidence_sequences;
mod static_weights;
pub mod track_and_stop;
mod uniform;

/// Check for duplicate variants within a list
fn check_duplicates_within(variants: &[String], list_name: &str) -> Result<(), Error> {
    let mut seen = HashSet::new();
    let mut duplicates = Vec::new();

    for variant in variants {
        if !seen.insert(variant) && !duplicates.contains(&variant.as_str()) {
            duplicates.push(variant.as_str());
        }
    }

    if !duplicates.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "`{}` contains duplicate entries: {}",
                list_name,
                duplicates.join(", ")
            ),
        }));
    }

    Ok(())
}

/// Check for duplicate variants across candidate_variants and fallback_variants
fn check_duplicates_across(candidates: &[String], fallbacks: &[String]) -> Result<(), Error> {
    let candidate_set: HashSet<_> = candidates.iter().collect();
    let mut duplicates = Vec::new();

    for fallback in fallbacks {
        if candidate_set.contains(fallback) && !duplicates.contains(&fallback.as_str()) {
            duplicates.push(fallback.as_str());
        }
    }

    if !duplicates.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "variants cannot appear in both `candidate_variants` and `fallback_variants`: {}",
                duplicates.join(", ")
            ),
        }));
    }

    Ok(())
}

/// Check for duplicate variants across candidate_variants (from a map) and fallback_variants
fn check_duplicates_across_map(
    candidate_keys: impl Iterator<Item = impl AsRef<str>>,
    fallbacks: &[String],
) -> Result<(), Error> {
    let candidate_set: HashSet<_> = candidate_keys.map(|s| s.as_ref().to_string()).collect();
    let mut duplicates = Vec::new();

    for fallback in fallbacks {
        if candidate_set.contains(fallback) && !duplicates.contains(&fallback.as_str()) {
            duplicates.push(fallback.as_str());
        }
    }

    if !duplicates.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "variants cannot appear in both `candidate_variants` and `fallback_variants`: {}",
                duplicates.join(", ")
            ),
        }));
    }

    Ok(())
}

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExperimentationConfig {
    Uniform(uniform::UniformConfig),
    StaticWeights(static_weights::StaticWeightsConfig),
    // NOTE: this diverges from the spec due to technical limitations with `serde`
    // (serde enums cannot be #[serde(flatten)])
    // we can write a custom deserializer for this if we want
    TrackAndStop(track_and_stop::TrackAndStopConfig),
    #[ts(skip)]
    #[cfg(test)]
    AlwaysFails(AlwaysFailsConfig),
}

impl Default for ExperimentationConfig {
    fn default() -> Self {
        Self::Uniform(uniform::UniformConfig::default())
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UninitializedExperimentationConfig {
    StaticWeights(static_weights::StaticWeightsConfig),
    Uniform(uniform::UniformConfig),
    TrackAndStop(track_and_stop::UninitializedTrackAndStopConfig),
}

impl UninitializedExperimentationConfig {
    pub fn load(
        self,
        variants: &HashMap<String, Arc<VariantInfo>>,
        metrics: &HashMap<String, crate::config::MetricConfig>,
    ) -> Result<ExperimentationConfig, Error> {
        // Check if any variant has a weight specified
        let variants_with_weights: Vec<&str> = variants
            .iter()
            .filter(|&(_name, variant)| variant.inner.weight().is_some())
            .map(|(name, _variant)| name.as_str())
            .collect();

        if !variants_with_weights.is_empty() {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "Cannot mix `experimentation` configuration with individual variant `weight` values. \
                    The following variants have weights specified: {}. \
                    Either use the `experimentation` section to control variant sampling, or specify `weight` on individual variants, but not both.",
                    variants_with_weights.join(", ")
                ),
            }));
        }

        match self {
            UninitializedExperimentationConfig::StaticWeights(config) => {
                Ok(ExperimentationConfig::StaticWeights(config))
            }
            UninitializedExperimentationConfig::Uniform(config) => {
                Ok(ExperimentationConfig::Uniform(config.load(variants)?))
            }
            UninitializedExperimentationConfig::TrackAndStop(config) => Ok(
                ExperimentationConfig::TrackAndStop(config.load(variants, metrics)?),
            ),
        }
    }
}
pub trait VariantSampler {
    async fn setup(
        &self,
        db: Arc<dyn FeedbackQueries + Send + Sync>,
        function_name: &str,
        postgres: &PostgresConnectionInfo,
        cancel_token: CancellationToken,
    ) -> Result<(), Error>;
    async fn sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        // This gets "popped from"
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
        postgres: &PostgresConnectionInfo,
    ) -> Result<(String, Arc<VariantInfo>), Error>;

    // Return all variant names that are allowed to be used by this experimentation config
    // Used to enforce that we don't fall back to a disallowed variant.
    fn allowed_variants(&self) -> impl Iterator<Item = &str> + '_;

    fn get_current_display_probabilities<'a>(
        &self,
        function_name: &str,
        active_variants: &'a HashMap<String, Arc<VariantInfo>>,
        postgres: &PostgresConnectionInfo,
    ) -> Result<HashMap<&'a str, f64>, Error>;
}

impl ExperimentationConfig {
    /// Note: in the future when we deprecate variant.weight we can simply use #[serde(default)]
    /// and default to ExperimentationConfig::Uniform
    ///
    /// For now, we call this function from the
    pub fn legacy_from_variants_map(variants: &HashMap<String, Arc<VariantInfo>>) -> Self {
        // We loop over the variants map and if any of them have weights we output a StaticWeightsConfig
        // otherwise, we output a Uniform config
        for variant in variants.values() {
            if variant.inner.weight().is_some() {
                return Self::StaticWeights(
                    static_weights::StaticWeightsConfig::legacy_from_variants_map(variants),
                );
            }
        }
        Self::Uniform(uniform::UniformConfig::default())
    }

    pub async fn setup(
        &self,
        db: Arc<dyn FeedbackQueries + Send + Sync>,
        function_name: &str,
        postgres: &PostgresConnectionInfo,
        cancel_token: CancellationToken,
    ) -> Result<(), Error> {
        match self {
            Self::StaticWeights(config) => {
                config
                    .setup(db, function_name, postgres, cancel_token)
                    .await
            }
            Self::Uniform(config) => {
                config
                    .setup(db, function_name, postgres, cancel_token)
                    .await
            }
            Self::TrackAndStop(config) => {
                config
                    .setup(db, function_name, postgres, cancel_token)
                    .await
            }
            #[cfg(test)]
            Self::AlwaysFails(config) => {
                config
                    .setup(db, function_name, postgres, cancel_token)
                    .await
            }
        }
    }

    pub async fn sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
        postgres: &PostgresConnectionInfo,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        // First try the variant-specific sampling
        let result = match self {
            Self::StaticWeights(config) => {
                config
                    .sample(function_name, episode_id, active_variants, postgres)
                    .await
            }
            Self::Uniform(config) => {
                config
                    .sample(function_name, episode_id, active_variants, postgres)
                    .await
            }
            #[cfg(test)]
            Self::AlwaysFails(config) => {
                config
                    .sample(function_name, episode_id, active_variants, postgres)
                    .await
            }
            Self::TrackAndStop(config) => {
                config
                    .sample(function_name, episode_id, active_variants, postgres)
                    .await
            }
        };

        // If the sampler fails but there are active variants, fall back to uniform sampling
        // from the allowed variants
        result.or_else(|e| {
            if active_variants.is_empty() {
                Err(e)
            } else {
                let allowed: Vec<&str> = match self {
                    Self::StaticWeights(config) => config.allowed_variants().collect(),
                    Self::Uniform(config) => config.allowed_variants().collect(),
                    #[cfg(test)]
                    Self::AlwaysFails(config) => config.allowed_variants().collect(),
                    Self::TrackAndStop(config) => config.allowed_variants().collect(),
                };
                // If allowed is empty (UniformConfig with None, None), fall back to all variants
                if allowed.is_empty() {
                    sample_uniform(function_name, &episode_id, active_variants, None)
                } else {
                    sample_uniform(function_name, &episode_id, active_variants, Some(&allowed))
                }
            }
        })
    }

    pub fn get_current_display_probabilities<'a>(
        &self,
        function_name: &str,
        active_variants: &'a HashMap<String, Arc<VariantInfo>>,
        postgres: &PostgresConnectionInfo,
    ) -> Result<HashMap<&'a str, f64>, Error> {
        match self {
            Self::StaticWeights(config) => {
                config.get_current_display_probabilities(function_name, active_variants, postgres)
            }
            Self::Uniform(config) => {
                config.get_current_display_probabilities(function_name, active_variants, postgres)
            }
            #[cfg(test)]
            Self::AlwaysFails(config) => {
                config.get_current_display_probabilities(function_name, active_variants, postgres)
            }
            Self::TrackAndStop(config) => {
                config.get_current_display_probabilities(function_name, active_variants, postgres)
            }
        }
    }
}

fn sample_uniform(
    function_name: &str,
    episode_id: &Uuid,
    active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
    allowed_variants: Option<&[&str]>,
) -> Result<(String, Arc<VariantInfo>), Error> {
    // Filter active_variants to only include allowed variants if specified
    let sampling_pool: Vec<String> = match allowed_variants {
        Some(allowed) => active_variants
            .keys()
            .filter(|k| allowed.contains(&k.as_str()))
            .cloned()
            .collect(),
        None => active_variants.keys().cloned().collect(),
    };

    if sampling_pool.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidFunctionVariants {
            message: format!(
                "No valid variants to sample from for function `{function_name}`{}",
                if allowed_variants.is_some() {
                    " after filtering by allowed variants"
                } else {
                    ""
                }
            ),
        }));
    }

    let random_index = (get_uniform_value(function_name, episode_id) * sampling_pool.len() as f64)
        .floor() as usize;
    let Some(sampled_variant_name) = sampling_pool.get(random_index).cloned() else {
        return Err(Error::new(ErrorDetails::InvalidFunctionVariants {
            message: format!(
                "Invalid index {random_index} for function `{function_name}` with {} variants. {IMPOSSIBLE_ERROR_MESSAGE}",
                sampling_pool.len()
            ),
        }));
    };
    active_variants.remove_entry(&sampled_variant_name).ok_or_else(|| {
        Error::new(ErrorDetails::InvalidFunctionVariants {
            message: format!(
                "Function `{function_name}` has no variant for the sampled variant `{sampled_variant_name}`. {IMPOSSIBLE_ERROR_MESSAGE}"
            )
        })
    })
}

/// Implements a uniform distribution over the interval [0, 1) using a hash function.
/// This function is deterministic but should have good statistical properties.
pub(crate) fn get_uniform_value(function_name: &str, episode_id: &Uuid) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(function_name.as_bytes());
    hasher.update(episode_id.as_bytes());
    let hash_value = hasher.finalize();
    let truncated_hash =
        u32::from_be_bytes([hash_value[0], hash_value[1], hash_value[2], hash_value[3]]);
    truncated_hash as f64 / u32::MAX as f64
}

/// Test-only config that always fails during sampling to test fallback logic
#[cfg(test)]
#[derive(Debug, Serialize)]
pub struct AlwaysFailsConfig {
    allowed_variants: Vec<String>,
}

#[cfg(test)]
impl AlwaysFailsConfig {
    pub fn new(allowed_variants: Vec<String>) -> Self {
        Self { allowed_variants }
    }
}

#[cfg(test)]
impl VariantSampler for AlwaysFailsConfig {
    async fn setup(
        &self,
        _db: Arc<dyn FeedbackQueries + Send + Sync>,
        _function_name: &str,
        _postgres: &PostgresConnectionInfo,
        _cancel_token: CancellationToken,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn sample(
        &self,
        _function_name: &str,
        _episode_id: Uuid,
        _active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
        _postgres: &PostgresConnectionInfo,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        Err(Error::new(ErrorDetails::Inference {
            message: "AlwaysFails sampler always fails".to_string(),
        }))
    }

    fn allowed_variants(&self) -> impl Iterator<Item = &str> + '_ {
        self.allowed_variants.iter().map(String::as_str)
    }

    // AlwaysFailsConfig always fails and falls back to uniform sampling
    // probabilities over allowed variants
    fn get_current_display_probabilities<'a>(
        &self,
        _function_name: &str,
        active_variants: &'a HashMap<String, Arc<VariantInfo>>,
        _postgres: &PostgresConnectionInfo,
    ) -> Result<HashMap<&'a str, f64>, Error> {
        // Find intersection of active_variants and allowed_variants
        let intersection: Vec<&str> = active_variants
            .keys()
            .filter(|k| self.allowed_variants.contains(&k.to_string()))
            .map(String::as_str)
            .collect();

        if intersection.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidFunctionVariants {
                message: "No allowed variants in active variants".to_string(),
            }));
        }

        let uniform_prob = 1.0 / intersection.len() as f64;
        Ok(intersection
            .into_iter()
            .map(|variant_name| (variant_name, uniform_prob))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ErrorContext, SchemaData};
    use crate::variant::{chat_completion::UninitializedChatCompletionConfig, VariantConfig};
    use uuid::Uuid;

    #[test]
    fn test_get_uniform_value() {
        // Test with function name and episode ID
        let episode_id = Uuid::now_v7();
        let value1 = get_uniform_value("test_function", &episode_id);
        let value2 = get_uniform_value("test_function", &episode_id);

        // Values should be the same due to deterministic input
        assert_eq!(value1, value2);
        assert!((0.0..1.0).contains(&value1));
        assert!((0.0..1.0).contains(&value2));

        // Test with different function names
        let value3 = get_uniform_value("another_function", &episode_id);
        assert_ne!(value1, value3);
        assert!((0.0..1.0).contains(&value3));

        // Test with different episode IDs
        let value4 = get_uniform_value("test_function", &Uuid::now_v7());
        assert_ne!(value1, value4);
        assert_ne!(value3, value4);
        assert!((0.0..1.0).contains(&value4));
    }

    #[test]
    fn test_uniform_sampling_distribution() {
        use crate::config::{ErrorContext, SchemaData};
        use crate::variant::{chat_completion::UninitializedChatCompletionConfig, VariantConfig};
        use std::collections::HashMap;

        // Create variants with no weights (should trigger uniform sampling)
        let variant_weights = [("A", None), ("B", None), ("C", None)];
        let mut variants_map = BTreeMap::new();

        for &(name, weight) in &variant_weights {
            variants_map.insert(
                name.to_string(),
                Arc::new(VariantInfo {
                    inner: VariantConfig::ChatCompletion(
                        UninitializedChatCompletionConfig {
                            weight,
                            model: "model-name".into(),
                            ..Default::default()
                        }
                        .load(&SchemaData::default(), &ErrorContext::new_test())
                        .unwrap(),
                    ),
                    timeouts: Default::default(),
                }),
            );
        }

        let sample_size = 10_000;
        let mut counts = HashMap::new();

        // Sample many times using uniform sampling
        for i in 0..sample_size {
            let mut active_variants = variants_map.clone();
            let episode_id = Uuid::from_u128(i as u128);
            let result = sample_uniform("test_function", &episode_id, &mut active_variants, None);
            let (variant_name, _) = result.unwrap();
            *counts.entry(variant_name).or_insert(0) += 1;
        }

        // Check that each variant was sampled roughly equally (uniform distribution)
        let expected_prob = 1.0 / 3.0; // Equal probability for 3 variants
        let tolerance = 0.02; // 2% tolerance

        for variant_name in ["A", "B", "C"] {
            let actual_prob = *counts.get(variant_name).unwrap_or(&0) as f64 / sample_size as f64;
            assert!(
                (actual_prob - expected_prob).abs() < tolerance,
                "Variant {variant_name}: expected {expected_prob:.3}, got {actual_prob:.3}"
            );
        }
    }

    #[tokio::test]
    async fn test_always_fails_fallback_with_allowed_variants() {
        use crate::config::{ErrorContext, SchemaData};
        use crate::variant::{chat_completion::UninitializedChatCompletionConfig, VariantConfig};

        // Create 5 variants: A, B, C, D, E
        let variant_names = ["A", "B", "C", "D", "E"];
        let mut variants_map = BTreeMap::new();

        for &name in &variant_names {
            variants_map.insert(
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
            );
        }

        // Create AlwaysFails config that only allows A, B, C
        let allowed_variants = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let config = ExperimentationConfig::AlwaysFails(AlwaysFailsConfig::new(allowed_variants));

        let sample_size = 1_000;
        let mut counts = HashMap::new();

        // Sample many times - the sampler will always fail and fall back to uniform sampling
        // with only the allowed variants (A, B, C)
        for i in 0..sample_size {
            let mut active_variants = variants_map.clone();
            let episode_id = Uuid::from_u128(i as u128);
            let result = config
                .sample(
                    "test_function",
                    episode_id,
                    &mut active_variants,
                    &PostgresConnectionInfo::Disabled,
                )
                .await;

            // Verify sampling succeeded (fallback worked)
            assert!(result.is_ok(), "Sampling should succeed via fallback");

            let (variant_name, _) = result.unwrap();

            // Verify only allowed variants are sampled
            assert!(
                ["A", "B", "C"].contains(&variant_name.as_str()),
                "Sampled variant {variant_name} should be in allowed list (A, B, C)"
            );

            // Verify disallowed variants are never sampled
            assert!(
                !["D", "E"].contains(&variant_name.as_str()),
                "Sampled variant {variant_name} should not be D or E"
            );

            *counts.entry(variant_name).or_insert(0) += 1;

            // Verify the sampled variant was removed from active_variants
            assert_eq!(
                active_variants.len(),
                4,
                "One variant should be removed from active_variants"
            );
        }

        // Verify D and E were never sampled
        assert_eq!(
            counts.get("D").unwrap_or(&0),
            &0,
            "Variant D should never be sampled"
        );
        assert_eq!(
            counts.get("E").unwrap_or(&0),
            &0,
            "Variant E should never be sampled"
        );

        // Verify roughly uniform distribution across A, B, C
        let expected_prob = 1.0 / 3.0; // Equal probability for 3 allowed variants
        let tolerance = 0.05; // 5% tolerance (more generous for smaller sample size)

        for variant_name in ["A", "B", "C"] {
            let actual_prob = *counts.get(variant_name).unwrap_or(&0) as f64 / sample_size as f64;
            assert!(
                (actual_prob - expected_prob).abs() < tolerance,
                "Variant {variant_name}: expected {expected_prob:.3}, got {actual_prob:.3}"
            );
        }
    }

    // Tests for get_current_display_probabilities
    #[test]
    fn test_get_current_display_probabilities_uniform() {
        let mut active_variants = HashMap::new();
        for name in ["A", "B", "C"] {
            active_variants.insert(
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
            );
        }

        let config = ExperimentationConfig::Uniform(uniform::UniformConfig::default());
        let postgres = PostgresConnectionInfo::new_disabled();
        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // Should have uniform probabilities
        assert_eq!(probs.len(), 3);
        assert!((probs["A"] - 1.0 / 3.0).abs() < 1e-9);
        assert!((probs["B"] - 1.0 / 3.0).abs() < 1e-9);
        assert!((probs["C"] - 1.0 / 3.0).abs() < 1e-9);

        // Check sum
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_current_display_probabilities_uniform_empty() {
        let active_variants = HashMap::new();

        let config = ExperimentationConfig::Uniform(uniform::UniformConfig::default());
        let postgres = PostgresConnectionInfo::new_disabled();
        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // Should return empty map
        assert_eq!(probs.len(), 0);
    }
}
