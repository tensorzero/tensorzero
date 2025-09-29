use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};
use uuid::Uuid;

use crate::{
    error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    experimentation::get_uniform_value,
    variant::VariantInfo,
};

use super::VariantSampler;

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct StaticWeightsConfig {
    // Map from variant name to weight. We enforce that weights are positive at construction time.
    candidate_variants: BTreeMap<String, f64>,
    // list of fallback variants (we will uniformly sample from these at inference time)
    fallback_variants: Vec<String>,
}

impl StaticWeightsConfig {
    pub fn legacy_from_variants_map(variants: &HashMap<String, Arc<VariantInfo>>) -> Self {
        // TODO: produce a candidate variants map
        // and a list of fallback variants
        let mut candidate_variants = BTreeMap::new();
        let mut fallback_variants = Vec::new();

        for (name, variant) in variants {
            if let Some(weight) = variant.inner.weight() {
                if weight > 0.0 {
                    candidate_variants.insert(name.clone(), weight);
                }
                // If the weight is 0 then it is explicitly disabled and we don't include it
            } else {
                fallback_variants.push(name.clone());
            }
        }

        Self {
            candidate_variants,
            fallback_variants,
        }
    }
}

impl VariantSampler for StaticWeightsConfig {
    async fn setup(&self) -> Result<(), Error> {
        Ok(())
    }

    /// Sample a variant from the function based on variant weights (a categorical distribution)
    /// This function pops the sampled variant from the candidate variants map.
    /// NOTE: We use a BTreeMap to ensure that the variants are sorted by their names and the
    /// sampling choices are deterministic given an episode ID.
    async fn sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        // Compute the total weight of variants present in variant_names
        let total_weight = active_variants
            .keys()
            .map(|variant_name| self.candidate_variants.get(variant_name).unwrap_or(&0.0))
            .sum::<f64>();
        if total_weight <= 0.0 {
            // Assume that there are no active variants in the candidate set so
            // we will try the fallback variants
            // First, we take the intersection of active_variants and fallback_variants
            let intersection = active_variants
                .keys()
                .filter(|variant_name| self.fallback_variants.contains(variant_name))
                .collect::<Vec<_>>();
            if intersection.is_empty() {
                Err(ErrorDetails::NoFallbackVariantsRemaining.into())
            } else {
                // Use deterministic selection based on episode_id for consistent behavior
                let random_index = (get_uniform_value(function_name, &episode_id)
                    * intersection.len() as f64)
                    .floor() as usize;
                let selected_variant = intersection.get(random_index).ok_or_else(|| Error::new(ErrorDetails::Inference {
                    message: format!("Failed to sample variant from nonempty intersection. {IMPOSSIBLE_ERROR_MESSAGE}")
                }))?.to_string();
                let variant_data = active_variants.remove(&selected_variant).ok_or_else(|| Error::new(ErrorDetails::Inference {
                    message: format!("Failed to remove variant from active variants. {IMPOSSIBLE_ERROR_MESSAGE}")
                }))?;
                Ok((selected_variant, variant_data))
            }
        } else {
            // Sample a random threshold between 0 and the total weight using the deterministic hashing trick
            let random_threshold = get_uniform_value(function_name, &episode_id) * total_weight;
            // Iterate over the variants to find the one that corresponds to the sampled threshold
            let variant_to_remove = {
                let mut cumulative_weight = 0.0;
                active_variants
                    .iter()
                    .find(|(variant_name, _)| {
                        cumulative_weight += self
                            .candidate_variants
                            .get(variant_name.as_str())
                            .unwrap_or(&0.0);
                        cumulative_weight > random_threshold
                    })
                    .map(|(name, _)| name.clone()) // Clone the key
            };

            if let Some(variant_name) = variant_to_remove {
                return active_variants.remove_entry(&variant_name).ok_or_else(|| {
                    Error::new(ErrorDetails::InvalidFunctionVariants {
                        message: format!(
                            "Function `{function_name}` has no variant for the sampled variant `{variant_name}`. {IMPOSSIBLE_ERROR_MESSAGE}"
                        )
                    })
                });
            }
            // If we didn't find a variant (which should only happen due to rare numerical precision issues),
            // pop an arbitrary variant as a fallback
            active_variants.pop_first().ok_or_else(|| {
                Error::new(ErrorDetails::InvalidFunctionVariants {
                    message: format!(
                        "Function `{function_name}` has no variants. {IMPOSSIBLE_ERROR_MESSAGE}"
                    ),
                })
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::config::{Config, ConfigFileGlob, ErrorContext, SchemaData, TimeoutsConfig};
    use crate::variant::chat_completion::ChatCompletionConfig;
    use crate::variant::{chat_completion::UninitializedChatCompletionConfig, VariantConfig};
    use tempfile::NamedTempFile;
    use tokio;
    use uuid::Uuid;

    fn create_variants(
        variant_weights: &[(&str, Option<f64>)],
    ) -> BTreeMap<String, Arc<VariantInfo>> {
        variant_weights
            .iter()
            .map(|&(name, weight)| {
                (
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
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn test_weighted_sampling() {
        let variants_map = create_variants(&[("A", Some(1.0)), ("B", Some(2.0)), ("C", Some(3.0))]);
        let config = StaticWeightsConfig::legacy_from_variants_map(
            &variants_map
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        );
        let mut active_variants = variants_map;
        let episode_id = Uuid::now_v7();

        let (variant_name, _) = config
            .sample("test_function", episode_id, &mut active_variants)
            .await
            .unwrap();
        assert!(["A", "B", "C"].contains(&variant_name.as_str()));
        assert_eq!(active_variants.len(), 2); // One variant should be removed
    }

    #[tokio::test]
    async fn test_fallback_variants() {
        let variants_map = create_variants(&[("A", Some(0.0)), ("B", None), ("C", None)]);
        let config = StaticWeightsConfig::legacy_from_variants_map(
            &variants_map
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        );
        let mut active_variants = variants_map;
        let episode_id = Uuid::now_v7();

        let (variant_name, _) = config
            .sample("test_function", episode_id, &mut active_variants)
            .await
            .unwrap();
        assert!(["B", "C"].contains(&variant_name.as_str())); // Should pick from fallback variants
    }

    #[tokio::test]
    async fn test_empty_variants_error() {
        let config = StaticWeightsConfig {
            candidate_variants: BTreeMap::new(),
            fallback_variants: Vec::new(),
        };
        let mut active_variants = BTreeMap::new();
        let episode_id = Uuid::now_v7();

        let result = config
            .sample("test_function", episode_id, &mut active_variants)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_weighted_distribution() {
        // Test that the weighted sampling produces the expected distribution
        let variants_map = create_variants(&[("A", Some(1.0)), ("B", Some(2.0)), ("C", Some(3.0))]);
        let config = StaticWeightsConfig::legacy_from_variants_map(
            &variants_map
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        );

        let sample_size = 10_000;
        let mut counts = std::collections::HashMap::new();

        // Sample many times to build distribution
        for i in 0..sample_size {
            let mut active_variants = variants_map.clone();
            // Use different episode IDs to get different samples
            let episode_id = Uuid::from_u128(i as u128);
            let (variant_name, _) = config
                .sample("test_function", episode_id, &mut active_variants)
                .await
                .unwrap();
            *counts.entry(variant_name).or_insert(0) += 1;
        }

        // Check that the distribution roughly matches expected weights
        let total_weight = 1.0 + 2.0 + 3.0;
        let expected_a = 1.0 / total_weight;
        let expected_b = 2.0 / total_weight;
        let expected_c = 3.0 / total_weight;

        let actual_a = *counts.get("A").unwrap_or(&0) as f64 / sample_size as f64;
        let actual_b = *counts.get("B").unwrap_or(&0) as f64 / sample_size as f64;
        let actual_c = *counts.get("C").unwrap_or(&0) as f64 / sample_size as f64;

        // Allow 2% tolerance for statistical variation
        let tolerance = 0.02;
        assert!(
            (actual_a - expected_a).abs() < tolerance,
            "Variant A: expected {expected_a:.3}, got {actual_a:.3}"
        );
        assert!(
            (actual_b - expected_b).abs() < tolerance,
            "Variant B: expected {expected_b:.3}, got {actual_b:.3}"
        );
        assert!(
            (actual_c - expected_c).abs() < tolerance,
            "Variant C: expected {expected_c:.3}, got {actual_c:.3}"
        );
    }

    #[tokio::test]
    async fn test_sampling_from_config() {
        let config_str = r#"
            [functions.test]
            type = "chat"
            [functions.test.experimentation]
            type = "static_weights"
            candidate_variants = {"foo" = 5, "bar" = 1 }
            fallback_variants = ["baz"]

            [functions.test.variants.foo]
            type = "chat_completion"
            model = "openai::gpt-5"

            [functions.test.variants.bar]
            type = "chat_completion"
            model = "anthropic::claude"

            [functions.test.variants.baz]
            type = "chat_completion"
            model = "fireworks::deepseek-v3"
            "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let config = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .unwrap();
        let experiment = config.functions.get("test").unwrap().experimentation();
        // no-op but we call it for completeness
        experiment.setup().await.unwrap();

        // Test sampling distribution with many samples
        let sample_size = 10_000;
        let mut first_sample_counts = std::collections::HashMap::new();
        let mut third_sample_counts = std::collections::HashMap::new();

        for i in 0..sample_size {
            let mut variants = BTreeMap::from([
                (
                    "foo".to_string(),
                    Arc::new(VariantInfo {
                        inner: VariantConfig::ChatCompletion(ChatCompletionConfig::default()),
                        timeouts: TimeoutsConfig::default(),
                    }),
                ),
                (
                    "bar".to_string(),
                    Arc::new(VariantInfo {
                        inner: VariantConfig::ChatCompletion(ChatCompletionConfig::default()),
                        timeouts: TimeoutsConfig::default(),
                    }),
                ),
                (
                    "baz".to_string(),
                    Arc::new(VariantInfo {
                        inner: VariantConfig::ChatCompletion(ChatCompletionConfig::default()),
                        timeouts: TimeoutsConfig::default(),
                    }),
                ),
            ]);

            // Use different episode IDs to get different samples
            let episode_id = Uuid::from_u128(i as u128);

            // Sample first variant (should be from candidate variants: foo or bar)
            let (first_sample_name, _) = experiment
                .sample("test", episode_id, &mut variants)
                .await
                .unwrap();
            *first_sample_counts
                .entry(first_sample_name.clone())
                .or_insert(0) += 1;

            // Sample second variant
            let (second_sample_name, _) = experiment
                .sample("test", episode_id, &mut variants)
                .await
                .unwrap();

            assert_ne!(first_sample_name, second_sample_name);

            // Sample third variant (should always be baz since it's the fallback)
            let (third_sample_name, _) = experiment
                .sample("test", episode_id, &mut variants)
                .await
                .unwrap();
            *third_sample_counts.entry(third_sample_name).or_insert(0) += 1;
        }

        // Check that "foo" was selected first ~83.3% of the time (5/6)
        let total_weight = 5.0 + 1.0; // foo=5, bar=1
        let expected_foo_first = 5.0 / total_weight; // ~0.833
        let actual_foo_first =
            *first_sample_counts.get("foo").unwrap_or(&0) as f64 / sample_size as f64;

        // Allow 2% tolerance for statistical variation
        let tolerance = 0.02;
        assert!(
            (actual_foo_first - expected_foo_first).abs() < tolerance,
            "Foo selected first: expected {expected_foo_first:.3}, got {actual_foo_first:.3}"
        );

        // Check that "baz" was always selected third (as the fallback variant)
        assert_eq!(*third_sample_counts.get("baz").unwrap(), sample_size);
        assert_eq!(third_sample_counts.len(), 1); // Only "baz" should appear in third samples
    }
}
