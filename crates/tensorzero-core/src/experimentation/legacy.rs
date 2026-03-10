//! Legacy experimentation config types for backward compatibility.
//!
//! These types exist solely to deserialize old config formats (`type = "uniform"` and
//! `type = "static_weights"`) and convert them into `StaticExperimentationConfig`.
//! They contain no runtime logic — all validation and sampling is handled by `StaticExperimentationConfig`.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use super::static_experimentation::{StaticExperimentationConfig, WeightedVariants};
use crate::variant::VariantInfo;

/// Legacy `type = "uniform"` config. Converts to `StaticExperimentationConfig` with equal weights.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct LegacyUniformExperimentationConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    candidate_variants: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fallback_variants: Option<Vec<String>>,
}

impl LegacyUniformExperimentationConfig {
    /// Convert to `StaticExperimentationConfig`, or `None` if neither candidates
    /// nor fallbacks are specified (meaning "use all variants uniformly").
    /// Validation is deferred to `StaticExperimentationConfig::validate()`.
    pub fn into_static_config(self) -> Option<StaticExperimentationConfig> {
        if self.candidate_variants.is_none() && self.fallback_variants.is_none() {
            return None;
        }

        let candidate_variants =
            WeightedVariants::from_equal_weights(self.candidate_variants.unwrap_or_default());
        Some(StaticExperimentationConfig {
            candidate_variants,
            fallback_variants: self.fallback_variants.unwrap_or_default(),
        })
    }
}

/// Legacy `type = "static_weights"` config. Converts to `StaticExperimentationConfig` directly.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct LegacyStaticWeightsExperimentationConfig {
    candidate_variants: BTreeMap<String, f64>,
    #[serde(default)]
    fallback_variants: Vec<String>,
}

impl LegacyStaticWeightsExperimentationConfig {
    /// Convert to `StaticExperimentationConfig`.
    /// Validation is deferred to `StaticExperimentationConfig::validate()`.
    pub fn into_static_config(self) -> StaticExperimentationConfig {
        StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_map(self.candidate_variants),
            fallback_variants: self.fallback_variants,
        }
    }

    /// Build a legacy config from the per-variant `weight` fields in a function's variants map.
    /// Used when no `[experimentation]` section is present in the config.
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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::config::{Config, ConfigFileGlob, TimeoutsConfig};
    use crate::db::clickhouse::ClickHouseConnectionInfo;
    use crate::db::feedback::FeedbackQueries;
    use crate::db::postgres::PostgresConnectionInfo;
    use crate::variant::VariantConfig;
    use crate::variant::chat_completion::ChatCompletionConfig;
    use tempfile::NamedTempFile;
    use tokio_util::sync::CancellationToken;

    /// End-to-end test: legacy `static_weights` TOML is deserialized, converted to
    /// `StaticExperimentationConfig`, validated, and then used for sampling.
    #[tokio::test]
    async fn test_static_weights_end_to_end() {
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
        let postgres = PostgresConnectionInfo::new_disabled();
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

            let episode_id = uuid::Uuid::from_u128(i as u128);

            let (first_sample_name, _) = experiment
                .sample("test", episode_id, &mut variants, &postgres)
                .await
                .unwrap();
            *first_sample_counts
                .entry(first_sample_name.clone())
                .or_insert(0) += 1;

            let (second_sample_name, _) = experiment
                .sample("test", episode_id, &mut variants, &postgres)
                .await
                .unwrap();
            assert_ne!(first_sample_name, second_sample_name);

            let (third_sample_name, _) = experiment
                .sample("test", episode_id, &mut variants, &postgres)
                .await
                .unwrap();
            *third_sample_counts.entry(third_sample_name).or_insert(0) += 1;
        }

        // foo should be selected first ~83.3% of the time (5/6)
        let expected_foo_first = 5.0 / 6.0;
        let actual_foo_first =
            *first_sample_counts.get("foo").unwrap_or(&0) as f64 / sample_size as f64;
        let tolerance = 0.02;
        assert!(
            (actual_foo_first - expected_foo_first).abs() < tolerance,
            "Foo selected first: expected {expected_foo_first:.3}, got {actual_foo_first:.3}"
        );

        // baz should always be selected third (as the fallback variant)
        assert_eq!(
            *third_sample_counts.get("baz").unwrap(),
            sample_size,
            "Fallback variant `baz` should always be selected third"
        );
        assert_eq!(
            third_sample_counts.len(),
            1,
            "Only `baz` should appear in third samples"
        );
    }
}
