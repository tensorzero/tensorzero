use std::collections::BTreeMap;
use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::db::feedback::FeedbackQueries;
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
        db: Arc<dyn FeedbackQueries + Send + Sync>,
        function_name: &str,
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
        db: Arc<dyn FeedbackQueries + Send + Sync>,
        function_name: &str,
        cancel_token: CancellationToken,
    ) -> Result<(), Error> {
        match self {
            Self::StaticWeights(config) => config.setup(db, function_name, cancel_token).await,
            Self::Uniform => Ok(()),
            Self::TrackAndStop(config) => config.setup(db, function_name, cancel_token).await,
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
}
