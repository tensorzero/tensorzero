use std::collections::BTreeMap;
use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::variant::VariantInfo;

mod static_weights;

#[derive(Debug, Default, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExperimentationConfig {
    StaticWeights(static_weights::StaticWeightsConfig),
    #[default]
    Uniform,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UninitializedExperimentationConfig {
    StaticWeights(static_weights::StaticWeightsConfig),
    Uniform,
}

impl UninitializedExperimentationConfig {
    pub fn load(self) -> ExperimentationConfig {
        match self {
            UninitializedExperimentationConfig::StaticWeights(config) => {
                ExperimentationConfig::StaticWeights(config)
            }
            UninitializedExperimentationConfig::Uniform => ExperimentationConfig::Uniform,
        }
    }
}
pub trait VariantSampler {
    // TODO, when we add bandits: pass CH and PG clients here (but use opaque trait types)
    async fn setup(&self) -> Result<(), Error>;
    async fn sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
    ) -> Result<(String, Arc<VariantInfo>), Error>;

    fn allowed_variants(&self) -> impl Iterator<Item = &str> + '_;
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

    pub async fn setup(&self) -> Result<(), Error> {
        match self {
            Self::StaticWeights(config) => config.setup().await,
            Self::Uniform => Ok(()),
        }
    }

    pub async fn sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        match self {
            Self::StaticWeights(config) => {
                config
                    .sample(function_name, episode_id, active_variants)
                    .await
                    // If the sampler fails but there are active variants we sample one at uniform
                    // from the allowed variants
                    .or_else(|e| {
                        if !active_variants.is_empty() {
                            let allowed: Vec<&str> = config.allowed_variants().collect();
                            sample_uniform(function_name, &episode_id, active_variants, Some(&allowed))
                        } else {
                            Err(e)
                        }
                    })
            }
            Self::Uniform => sample_uniform(function_name, &episode_id, active_variants, None),
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
}
