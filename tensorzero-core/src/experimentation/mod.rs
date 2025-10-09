use std::collections::BTreeMap;
use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::postgres::PostgresConnectionInfo;
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::variant::VariantInfo;

mod static_weights;
pub mod track_and_stop;

#[derive(Debug, Default, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExperimentationConfig {
    StaticWeights(static_weights::StaticWeightsConfig),
    #[default]
    Uniform,
    // NOTE: this diverges from the spec due to technical limitations with `serde`
    // (serde enums cannot be #[serde(flatten)])
    // we can write a custom deserializer for this if we want
    TrackAndStop(track_and_stop::TrackAndStopConfig),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UninitializedExperimentationConfig {
    StaticWeights(static_weights::StaticWeightsConfig),
    Uniform,
    TrackAndStop(track_and_stop::UninitializedTrackAndStopConfig),
}

impl UninitializedExperimentationConfig {
    pub fn load(
        self,
        variants: &HashMap<String, Arc<VariantInfo>>,
        metrics: &HashMap<String, crate::config::MetricConfig>,
    ) -> Result<ExperimentationConfig, Error> {
        match self {
            UninitializedExperimentationConfig::StaticWeights(config) => {
                Ok(ExperimentationConfig::StaticWeights(config))
            }
            UninitializedExperimentationConfig::Uniform => Ok(ExperimentationConfig::Uniform),
            UninitializedExperimentationConfig::TrackAndStop(config) => Ok(
                ExperimentationConfig::TrackAndStop(config.load(variants, metrics)?),
            ),
        }
    }
}

pub trait VariantSampler {
    async fn setup(
        &self,
        clickhouse: &ClickHouseConnectionInfo,
        function_name: &str,
    ) -> Result<(), Error>;
    async fn sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        // This gets "popped from"
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
        postgres: &PostgresConnectionInfo,
    ) -> Result<(String, Arc<VariantInfo>), Error>;
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
        Self::Uniform
    }
}

impl VariantSampler for ExperimentationConfig {
    async fn setup(
        &self,
        clickhouse: &ClickHouseConnectionInfo,
        function_name: &str,
    ) -> Result<(), Error> {
        match self {
            Self::StaticWeights(config) => config.setup(clickhouse, function_name).await,
            Self::Uniform => Ok(()),
            Self::TrackAndStop(config) => config.setup(clickhouse, function_name).await,
        }
    }

    async fn sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
        postgres: &PostgresConnectionInfo,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        match self {
            Self::StaticWeights(config) => {
                config
                    .sample(function_name, episode_id, active_variants, postgres)
                    .await
            }
            Self::Uniform => sample_uniform(function_name, &episode_id, active_variants),
            Self::TrackAndStop(config) => {
                config
                    .sample(function_name, episode_id, active_variants, postgres)
                    .await
            }
        }
    }
}

fn sample_uniform(
    function_name: &str,
    episode_id: &Uuid,
    active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
) -> Result<(String, Arc<VariantInfo>), Error> {
    let random_index = (get_uniform_value(function_name, episode_id) * active_variants.len() as f64)
        .floor() as usize;
    let Some(sampled_variant_name) = active_variants.keys().nth(random_index).cloned() else {
        return Err(Error::new(ErrorDetails::InvalidFunctionVariants {
            message: format!(
                "Invalid index {random_index} for function `{function_name}` with {} variants. {IMPOSSIBLE_ERROR_MESSAGE}",
                active_variants.len()
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

/// Pure function for static weights sampling logic.
/// Given a uniform sample in [0, 1), selects a variant from active_variants
/// using weighted sampling from candidate_variants if their intersection is nonempty,
/// or uniform sampling from fallback_variants otherwise.
///
/// Returns the name of the selected variant, which is guaranteed to be in active_variants.
pub(crate) fn sample_static_weights(
    active_variants: &BTreeMap<String, Arc<VariantInfo>>,
    candidate_variants: &BTreeMap<String, f64>,
    fallback_variants: &[String],
    uniform_sample: f64,
) -> Result<String, Error> {
    // Compute the total weight of variants present in active_variants
    let total_weight = active_variants
        .keys()
        .map(|variant_name| candidate_variants.get(variant_name).unwrap_or(&0.0))
        .sum::<f64>();

    if total_weight <= 0.0 {
        // No active variants in the candidate set, try fallback variants
        // Take the intersection of active_variants and fallback_variants
        let intersection: Vec<&String> = active_variants
            .keys()
            .filter(|variant_name| fallback_variants.contains(variant_name))
            .collect();

        if intersection.is_empty() {
            Err(ErrorDetails::NoFallbackVariantsRemaining.into())
        } else {
            // Use uniform sample to select from intersection
            let random_index = (uniform_sample * intersection.len() as f64).floor() as usize;
            intersection
                .get(random_index)
                .ok_or_else(|| {
                    Error::new(ErrorDetails::Inference {
                        message: format!(
                            "Failed to sample variant from nonempty intersection. {IMPOSSIBLE_ERROR_MESSAGE}"
                        ),
                    })
                })
                .map(std::string::ToString::to_string)
        }
    } else {
        // Use weighted sampling from candidate variants
        let random_threshold = uniform_sample * total_weight;
        let mut cumulative_weight = 0.0;

        let variant_name = active_variants.keys().find(|variant_name| {
            cumulative_weight += candidate_variants
                .get(variant_name.as_str())
                .unwrap_or(&0.0);
            cumulative_weight > random_threshold
        });

        if let Some(name) = variant_name {
            Ok(name.clone())
        } else {
            // If we didn't find a variant (rare numerical precision issues),
            // return the first variant as a fallback
            active_variants
                .keys()
                .next()
                .ok_or_else(|| {
                    Error::new(ErrorDetails::InvalidFunctionVariants {
                        message: format!(
                            "No active variants available. {IMPOSSIBLE_ERROR_MESSAGE}"
                        ),
                    })
                })
                .cloned()
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
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
            let result = sample_uniform("test_function", &episode_id, &mut active_variants);
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

    fn create_test_variants(names: &[&str]) -> BTreeMap<String, Arc<VariantInfo>> {
        use crate::config::{ErrorContext, SchemaData};
        use crate::variant::{chat_completion::UninitializedChatCompletionConfig, VariantConfig};

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
    fn test_sample_static_weights_weighted_sampling_deterministic() {
        // Test weighted sampling with specific uniform samples
        let active_variants = create_test_variants(&["A", "B", "C"]);
        let mut candidate_variants = BTreeMap::new();
        candidate_variants.insert("A".to_string(), 1.0);
        candidate_variants.insert("B".to_string(), 2.0);
        candidate_variants.insert("C".to_string(), 3.0);
        let fallback_variants = vec![];

        // Total weight = 6.0
        // A: [0.0, 1.0/6.0) -> [0.0, 0.1667)
        // B: [1.0/6.0, 3.0/6.0) -> [0.1667, 0.5)
        // C: [3.0/6.0, 6.0/6.0) -> [0.5, 1.0)

        // Test sample that should select A
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.1,
        );
        assert_eq!(result.unwrap(), "A");

        // Test sample that should select B
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.3,
        );
        assert_eq!(result.unwrap(), "B");

        // Test sample that should select C
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.7,
        );
        assert_eq!(result.unwrap(), "C");

        // Test edge case: sample at 0.0 should select A
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.0,
        );
        assert_eq!(result.unwrap(), "A");

        // Test edge case: sample very close to 1.0 should select C
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.999,
        );
        assert_eq!(result.unwrap(), "C");
    }

    #[test]
    fn test_sample_static_weights_fallback_sampling() {
        // Test fallback sampling when candidate variants have zero weight
        let active_variants = create_test_variants(&["A", "B", "C"]);
        let candidate_variants = BTreeMap::new(); // No candidate variants
        let fallback_variants = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        // With 3 fallback variants:
        // A: [0.0, 1.0/3.0) -> [0.0, 0.333...)
        // B: [1.0/3.0, 2.0/3.0) -> [0.333..., 0.666...)
        // C: [2.0/3.0, 3.0/3.0) -> [0.666..., 1.0)

        // Test sample that should select A
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.1,
        );
        assert_eq!(result.unwrap(), "A");

        // Test sample that should select B
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.5,
        );
        assert_eq!(result.unwrap(), "B");

        // Test sample that should select C
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.9,
        );
        assert_eq!(result.unwrap(), "C");
    }

    #[test]
    fn test_sample_static_weights_only_active_variants() {
        // Test that only active variants are sampled
        let active_variants = create_test_variants(&["A", "C"]);
        let mut candidate_variants = BTreeMap::new();
        candidate_variants.insert("A".to_string(), 1.0);
        candidate_variants.insert("B".to_string(), 2.0); // B is not active
        candidate_variants.insert("C".to_string(), 3.0);
        let fallback_variants = vec![];

        // Total weight of active variants = 1.0 + 3.0 = 4.0
        // A: [0.0, 1.0/4.0) -> [0.0, 0.25)
        // C: [1.0/4.0, 4.0/4.0) -> [0.25, 1.0)

        // Test sample that should select A
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.1,
        );
        assert_eq!(result.unwrap(), "A");

        // Test sample that should select C (not B, which is not active)
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.5,
        );
        assert_eq!(result.unwrap(), "C");
    }

    #[test]
    fn test_sample_static_weights_partial_intersection_fallback() {
        // Test fallback when only some variants have weights
        let active_variants = create_test_variants(&["A", "B", "C"]);
        let mut candidate_variants = BTreeMap::new();
        candidate_variants.insert("A".to_string(), 0.0); // Zero weight, excluded from sampling
        candidate_variants.insert("B".to_string(), 0.0); // Zero weight, excluded from sampling
        let fallback_variants = vec!["B".to_string(), "C".to_string()];

        // Total weight = 0.0, should use fallback
        // Active fallbacks: B, C
        // B: [0.0, 0.5)
        // C: [0.5, 1.0)

        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.3,
        );
        assert_eq!(result.unwrap(), "B");

        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.7,
        );
        assert_eq!(result.unwrap(), "C");
    }

    #[test]
    fn test_sample_static_weights_no_fallback_error() {
        // Test error when no fallback variants are active
        let active_variants = create_test_variants(&["A", "B"]);
        let candidate_variants = BTreeMap::new(); // No weights
        let fallback_variants = vec!["C".to_string(), "D".to_string()]; // None are active

        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.5,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_static_weights_empty_active_variants() {
        // Test error when no active variants
        let active_variants = BTreeMap::new();
        let mut candidate_variants = BTreeMap::new();
        candidate_variants.insert("A".to_string(), 1.0);
        let fallback_variants = vec!["B".to_string()];

        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.5,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_static_weights_single_variant() {
        // Test with single variant
        let active_variants = create_test_variants(&["A"]);
        let mut candidate_variants = BTreeMap::new();
        candidate_variants.insert("A".to_string(), 1.0);
        let fallback_variants = vec![];

        // Should always select A regardless of sample
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.0,
        );
        assert_eq!(result.unwrap(), "A");

        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.5,
        );
        assert_eq!(result.unwrap(), "A");

        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.999,
        );
        assert_eq!(result.unwrap(), "A");
    }

    #[test]
    fn test_sample_static_weights_unequal_weights() {
        // Test with very unequal weights
        let active_variants = create_test_variants(&["A", "B"]);
        let mut candidate_variants = BTreeMap::new();
        candidate_variants.insert("A".to_string(), 0.001);
        candidate_variants.insert("B".to_string(), 999.999);
        let fallback_variants = vec![];

        // Total weight = 1000.0
        // A: [0.0, 0.000001) - very small range
        // B: [0.000001, 1.0) - almost entire range

        // Very small sample should select A
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.0000001,
        );
        assert_eq!(result.unwrap(), "A");

        // Any reasonable sample should select B
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.5,
        );
        assert_eq!(result.unwrap(), "B");
    }
}
