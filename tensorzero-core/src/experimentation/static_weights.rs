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

use super::{check_duplicates_across_map, check_duplicates_within, VariantSampler};

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
        // Select the first variant from the ranked fallback_variants list that is active
        for variant_name in fallback_variants {
            if active_variants.contains_key(variant_name) {
                return Ok(variant_name.clone());
            }
        }

        // No active fallback variants found
        Err(ErrorDetails::NoFallbackVariantsRemaining.into())
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

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct StaticWeightsConfig {
    // Map from variant name to weight. Zero weights exclude variants from weighted sampling.
    // We enforce that weights are non-negative during setup validation.
    candidate_variants: BTreeMap<String, f64>,
    // list of fallback variants (we will uniformly sample from these at inference time)
    #[serde(default)]
    fallback_variants: Vec<String>,
}

impl StaticWeightsConfig {
    pub fn legacy_from_variants_map(variants: &HashMap<String, Arc<VariantInfo>>) -> Self {
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
    async fn setup(
        &self,
        _db: Arc<dyn FeedbackQueries + Send + Sync>,
        _function_name: &str,
        _postgres: &PostgresConnectionInfo,
        _cancel_token: CancellationToken,
    ) -> Result<(), Error> {
        // Check for duplicates within fallback_variants
        check_duplicates_within(&self.fallback_variants, "fallback_variants")?;

        // Check for duplicates across candidate_variants and fallback_variants
        check_duplicates_across_map(self.candidate_variants.keys(), &self.fallback_variants)?;

        // Validate that all weights are non-negative
        for weight in self.candidate_variants.values() {
            if *weight < 0.0 {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!("Invalid weight in static weights config: {weight}"),
                }));
            }
        }

        // Make sure there are candidate variants with positive weight or fallback variants
        let has_positive_weight = self.candidate_variants.values().any(|&w| w > 0.0);
        if !has_positive_weight && self.fallback_variants.is_empty() {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "Static weights config for function '{_function_name}' has no candidate variants with positive weight and no fallback variants. At least one is required."
                ),
            }));
        }

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
        _postgres: &PostgresConnectionInfo,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        // Sampling is stable per episode ID
        let uniform_sample = get_uniform_value(function_name, &episode_id);
        let selected_variant_name = sample_static_weights(
            active_variants,
            &self.candidate_variants,
            &self.fallback_variants,
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
        self.candidate_variants
            .keys()
            .map(String::as_str)
            .chain(self.fallback_variants.iter().map(String::as_str))
    }

    fn get_current_display_probabilities<'a>(
        &self,
        _function_name: &str,
        active_variants: &'a HashMap<String, Arc<VariantInfo>>,
        _postgres: &PostgresConnectionInfo,
    ) -> Result<HashMap<&'a str, f64>, Error> {
        // Compute the total weight of variants present in active_variants
        let total_weight: f64 = active_variants
            .keys()
            .map(|variant_name| self.candidate_variants.get(variant_name).unwrap_or(&0.0))
            .sum();

        if total_weight <= 0.0 {
            // No active variants in the candidate set, use fallback variants
            // Find the first variant from the ranked fallback_variants list that is active
            let first_active_fallback = self
                .fallback_variants
                .iter()
                .find(|variant_name| active_variants.contains_key(*variant_name));

            if let Some(selected_variant) = first_active_fallback {
                // The first active fallback variant gets 100% probability
                // All other active fallback variants get 0% probability
                let mut probabilities: HashMap<&'a str, f64> = HashMap::new();
                for key in active_variants.keys() {
                    if self.fallback_variants.contains(key) {
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
            // Use weighted probabilities from candidate variants
            let probabilities: HashMap<&'a str, f64> = active_variants
                .keys()
                .map(|variant_name| {
                    let weight = self.candidate_variants.get(variant_name).unwrap_or(&0.0);
                    (variant_name.as_str(), weight / total_weight)
                })
                .collect();
            Ok(probabilities)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::config::{
        Config, ConfigFileGlob, ConfigLoadInfo, ErrorContext, SchemaData, TimeoutsConfig,
    };
    use crate::db::clickhouse::ClickHouseConnectionInfo;
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
        let postgres = PostgresConnectionInfo::new_disabled();

        let (variant_name, _) = config
            .sample("test_function", episode_id, &mut active_variants, &postgres)
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
        let postgres = PostgresConnectionInfo::new_disabled();

        let (variant_name, _) = config
            .sample("test_function", episode_id, &mut active_variants, &postgres)
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
        let postgres = PostgresConnectionInfo::new_disabled();

        let result = config
            .sample("test_function", episode_id, &mut active_variants, &postgres)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_setup_error_no_positive_weights_no_fallbacks() {
        let config = StaticWeightsConfig {
            candidate_variants: BTreeMap::new(),
            fallback_variants: Vec::new(),
        };

        let db = Arc::new(ClickHouseConnectionInfo::new_disabled())
            as Arc<dyn FeedbackQueries + Send + Sync>;
        let postgres = PostgresConnectionInfo::new_disabled();
        let cancel_token = CancellationToken::new();

        let result = config
            .setup(db, "test_function", &postgres, cancel_token)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("no candidate variants with positive weight"));
        assert!(err.to_string().contains("no fallback variants"));
    }

    #[tokio::test]
    async fn test_setup_error_zero_weights_no_fallbacks() {
        let mut candidate_variants = BTreeMap::new();
        candidate_variants.insert("A".to_string(), 0.0);
        candidate_variants.insert("B".to_string(), 0.0);

        let config = StaticWeightsConfig {
            candidate_variants,
            fallback_variants: Vec::new(),
        };

        let db = Arc::new(ClickHouseConnectionInfo::new_disabled())
            as Arc<dyn FeedbackQueries + Send + Sync>;
        let postgres = PostgresConnectionInfo::new_disabled();
        let cancel_token = CancellationToken::new();

        let result = config
            .setup(db, "test_function", &postgres, cancel_token)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("no candidate variants with positive weight"));
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
        let postgres = PostgresConnectionInfo::new_disabled();

        // Sample many times to build distribution
        for i in 0..sample_size {
            let mut active_variants = variants_map.clone();
            // Use different episode IDs to get different samples
            let episode_id = Uuid::from_u128(i as u128);
            let (variant_name, _) = config
                .sample("test_function", episode_id, &mut active_variants, &postgres)
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

        let ConfigLoadInfo { config, .. } = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .unwrap();
        let experiment = config.functions.get("test").unwrap().experimentation();
        let postgres = PostgresConnectionInfo::new_disabled();
        // no-op but we call it for completeness
        experiment
            .setup(
                Arc::new(ClickHouseConnectionInfo::new_disabled())
                    as Arc<dyn FeedbackQueries + Send + Sync>,
                "test",
                &postgres,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        // Test sampling distribution with many samples
        let sample_size = 10_000;
        let mut first_sample_counts = std::collections::HashMap::new();
        let mut third_sample_counts = std::collections::HashMap::new();
        let postgres = PostgresConnectionInfo::new_disabled();

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
                .sample("test", episode_id, &mut variants, &postgres)
                .await
                .unwrap();
            *first_sample_counts
                .entry(first_sample_name.clone())
                .or_insert(0) += 1;

            // Sample second variant
            let (second_sample_name, _) = experiment
                .sample("test", episode_id, &mut variants, &postgres)
                .await
                .unwrap();

            assert_ne!(first_sample_name, second_sample_name);

            // Sample third variant (should always be baz since it's the fallback)
            let (third_sample_name, _) = experiment
                .sample("test", episode_id, &mut variants, &postgres)
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

        // Test sample that should select A
        // With ranked list, always returns the first active variant (A)
        let result = sample_static_weights(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.1,
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
            0.9,
        );
        assert_eq!(result.unwrap(), "A");
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
        // With ranked list, always returns the first active variant (B)

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
        assert_eq!(result.unwrap(), "B");
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

    // Tests for get_current_display_probabilities
    #[test]
    fn test_get_current_display_probabilities_weighted() {
        // Test weighted probabilities
        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "B", "C"]).into_iter().collect();
        let mut candidate_variants = BTreeMap::new();
        candidate_variants.insert("A".to_string(), 1.0);
        candidate_variants.insert("B".to_string(), 2.0);
        candidate_variants.insert("C".to_string(), 3.0);

        let config = StaticWeightsConfig {
            candidate_variants,
            fallback_variants: vec![],
        };

        let postgres = PostgresConnectionInfo::new_disabled();
        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // Total weight = 6.0
        // Expected: A=1/6, B=2/6, C=3/6
        assert_eq!(probs.len(), 3);
        assert!((probs["A"] - 1.0 / 6.0).abs() < 1e-9);
        assert!((probs["B"] - 2.0 / 6.0).abs() < 1e-9);
        assert!((probs["C"] - 3.0 / 6.0).abs() < 1e-9);

        // Check sum
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_current_display_probabilities_fallback() {
        // Test fallback (uniform) probabilities
        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let config = StaticWeightsConfig {
            candidate_variants: BTreeMap::new(), // No weights
            fallback_variants: vec!["A".to_string(), "B".to_string(), "C".to_string()],
        };

        let postgres = PostgresConnectionInfo::new_disabled();
        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // With ranked list, first active variant gets 100% probability
        assert_eq!(probs.len(), 3);
        assert!((probs["A"] - 1.0).abs() < 1e-9);
        assert!((probs["B"] - 0.0).abs() < 1e-9);
        assert!((probs["C"] - 0.0).abs() < 1e-9);

        // Check sum
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_current_display_probabilities_partial_intersection() {
        // Test with only some active variants having weights
        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "C"]).into_iter().collect();
        let mut candidate_variants = BTreeMap::new();
        candidate_variants.insert("A".to_string(), 1.0);
        candidate_variants.insert("B".to_string(), 2.0); // Not active
        candidate_variants.insert("C".to_string(), 3.0);

        let config = StaticWeightsConfig {
            candidate_variants,
            fallback_variants: vec![],
        };

        let postgres = PostgresConnectionInfo::new_disabled();
        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // Only A and C should appear, with normalized weights
        // Total weight = 1.0 + 3.0 = 4.0
        assert_eq!(probs.len(), 2);
        assert!((probs["A"] - 1.0 / 4.0).abs() < 1e-9);
        assert!((probs["C"] - 3.0 / 4.0).abs() < 1e-9);

        // Check sum
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_current_display_probabilities_fallback_partial() {
        // Test fallback with only some variants active
        let active_variants: HashMap<_, _> =
            create_test_variants(&["B", "C"]).into_iter().collect();

        let config = StaticWeightsConfig {
            candidate_variants: BTreeMap::new(),
            fallback_variants: vec!["A".to_string(), "B".to_string(), "C".to_string()],
        };

        let postgres = PostgresConnectionInfo::new_disabled();
        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // Only B and C are active and in fallback. With ranked list, first active (B) gets 100%
        assert_eq!(probs.len(), 2);
        assert!((probs["B"] - 1.0).abs() < 1e-9);
        assert!((probs["C"] - 0.0).abs() < 1e-9);

        // Check sum
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_current_display_probabilities_no_fallback_error() {
        // Test error when no fallback variants match
        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "B"]).into_iter().collect();

        let config = StaticWeightsConfig {
            candidate_variants: BTreeMap::new(),
            fallback_variants: vec!["C".to_string(), "D".to_string()],
        };

        let postgres = PostgresConnectionInfo::new_disabled();
        let result = config.get_current_display_probabilities("test", &active_variants, &postgres);

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_setup_validation_duplicate_fallbacks() {
        let config = StaticWeightsConfig {
            candidate_variants: BTreeMap::from([("A".to_string(), 1.0)]),
            fallback_variants: vec!["B".to_string(), "C".to_string(), "B".to_string()],
        };

        let db = Arc::new(ClickHouseConnectionInfo::new_disabled())
            as Arc<dyn FeedbackQueries + Send + Sync>;
        let postgres = PostgresConnectionInfo::new_disabled();
        let cancel_token = CancellationToken::new();

        let result = config
            .setup(db, "test_function", &postgres, cancel_token)
            .await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("`fallback_variants` contains duplicate entries"));
        assert!(err_msg.contains("B"));
    }

    #[tokio::test]
    async fn test_setup_validation_duplicate_across_lists() {
        let config = StaticWeightsConfig {
            candidate_variants: BTreeMap::from([("A".to_string(), 1.0), ("B".to_string(), 2.0)]),
            fallback_variants: vec!["B".to_string(), "C".to_string()],
        };

        let db = Arc::new(ClickHouseConnectionInfo::new_disabled())
            as Arc<dyn FeedbackQueries + Send + Sync>;
        let postgres = PostgresConnectionInfo::new_disabled();
        let cancel_token = CancellationToken::new();

        let result = config
            .setup(db, "test_function", &postgres, cancel_token)
            .await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot appear in both `candidate_variants` and `fallback_variants`")
        );
        assert!(err_msg.contains("B"));
    }

    #[tokio::test]
    async fn test_setup_validation_multiple_duplicates_across_lists() {
        let config = StaticWeightsConfig {
            candidate_variants: BTreeMap::from([
                ("A".to_string(), 1.0),
                ("B".to_string(), 2.0),
                ("C".to_string(), 3.0),
            ]),
            fallback_variants: vec!["B".to_string(), "C".to_string(), "D".to_string()],
        };

        let db = Arc::new(ClickHouseConnectionInfo::new_disabled())
            as Arc<dyn FeedbackQueries + Send + Sync>;
        let postgres = PostgresConnectionInfo::new_disabled();
        let cancel_token = CancellationToken::new();

        let result = config
            .setup(db, "test_function", &postgres, cancel_token)
            .await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot appear in both `candidate_variants` and `fallback_variants`")
        );
        assert!(err_msg.contains("B"));
        assert!(err_msg.contains("C"));
    }
}
