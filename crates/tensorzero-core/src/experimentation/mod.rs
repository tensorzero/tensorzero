use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::Arc,
};
use tensorzero_derive::TensorZeroDeserialize;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::config::Namespace;
use crate::db::feedback::FeedbackQueries;
use crate::db::postgres::PostgresConnectionInfo;
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::variant::VariantInfo;
pub use adaptive_experimentation::{
    AdaptiveExperimentationAlgorithm, AdaptiveExperimentationConfig,
    UninitializedAdaptiveExperimentationConfig,
};
pub use static_experimentation::{StaticExperimentationConfig, WeightedVariants};

pub mod adaptive_experimentation;
pub mod asymptotic_confidence_sequences;
mod legacy;
pub mod static_experimentation;
pub mod track_and_stop;

/// Check for duplicate variants within a list
fn check_duplicates_within(variants: &[String], list_name: &str) -> Result<(), Error> {
    let mut seen = HashSet::new();
    let duplicates: Vec<&str> = variants
        .iter()
        .filter(|v| !seen.insert(v.as_str()))
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    if !duplicates.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "`{list_name}` contains duplicate entries: {}",
                duplicates.join(", ")
            ),
        }));
    }

    Ok(())
}

/// Check for duplicate variants across candidate_variants and fallback_variants.
/// Candidates can be provided as any iterator of string-like items.
fn check_duplicates_across(
    candidates: impl Iterator<Item = impl AsRef<str>>,
    fallbacks: &[String],
) -> Result<(), Error> {
    let candidate_set: HashSet<String> = candidates.map(|s| s.as_ref().to_string()).collect();
    let duplicates: Vec<&str> = fallbacks
        .iter()
        .filter(|f| candidate_set.contains(f.as_str()))
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

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

/// Runtime experimentation config — only two variants (plus test-only AlwaysFails).
/// Legacy types (`Uniform`, `StaticWeights`, `TrackAndStop`) are converted to these
/// during `load()`.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Default, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExperimentationConfig {
    /// No experimentation configured — sample uniformly from all active variants.
    #[default]
    Default,
    Static(StaticExperimentationConfig),
    Adaptive(AdaptiveExperimentationConfig),
    #[cfg_attr(feature = "ts-bindings", ts(skip))]
    #[cfg(test)]
    AlwaysFails(AlwaysFailsConfig),
}

/// Holds the base experimentation config plus namespace-specific configs (loaded version).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[derive(Default)]
pub struct ExperimentationConfigWithNamespaces {
    /// The base experimentation config used when no namespace is provided
    /// or when the provided namespace doesn't have a specific config
    pub base: ExperimentationConfig,
    /// Namespace-specific experimentation configs
    pub namespaces: HashMap<String, ExperimentationConfig>,
}

impl ExperimentationConfigWithNamespaces {
    /// Get the experimentation config for a given namespace.
    /// If namespace is None or the namespace doesn't have a specific config,
    /// returns the base config.
    pub fn get_for_namespace(&self, namespace: Option<&Namespace>) -> &ExperimentationConfig {
        match namespace {
            Some(ns) => self.namespaces.get(ns.as_str()).unwrap_or(&self.base),
            None => &self.base,
        }
    }

    /// Check if a specific namespace has a dedicated config
    pub fn has_namespace_config(&self, namespace: &str) -> bool {
        self.namespaces.contains_key(namespace)
    }

    /// Create an ExperimentationConfigWithNamespaces from a legacy variants map.
    /// This creates a config with only a base experimentation config (no namespace overrides).
    pub fn legacy_from_variants_map(variants: &HashMap<String, Arc<VariantInfo>>) -> Self {
        Self {
            base: ExperimentationConfig::legacy_from_variants_map(variants),
            namespaces: HashMap::new(),
        }
    }
}

/// Uninitialized experimentation config — 5 variants for backward compatibility.
/// The 3 legacy variants (`Uniform`, `StaticWeights`, `TrackAndStop`) are converted
/// to the 2 new variants (`Static`, `Adaptive`) during `load()`.
#[derive(Clone, Debug, Serialize, TensorZeroDeserialize, JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum UninitializedExperimentationConfig {
    // New types
    Static(StaticExperimentationConfig),
    Adaptive(UninitializedAdaptiveExperimentationConfig),
    // Legacy types (backward compat)
    StaticWeights(legacy::LegacyStaticWeightsExperimentationConfig),
    Uniform(legacy::LegacyUniformExperimentationConfig),
    TrackAndStop(track_and_stop::UninitializedTrackAndStopExperimentationConfig),
}

/// Wrapper struct that holds the base experimentation config plus namespace-specific configs.
/// This is the type used in the TOML config to allow both a default config and per-namespace overrides.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct UninitializedExperimentationConfigWithNamespaces {
    /// The base experimentation config (type, candidate_variants, etc.)
    #[serde(flatten)]
    pub base: UninitializedExperimentationConfig,
    /// Namespace-specific experimentation configs
    #[serde(default)]
    pub namespaces: HashMap<String, UninitializedExperimentationConfig>,
}

impl UninitializedExperimentationConfigWithNamespaces {
    pub fn load(
        self,
        variants: &HashMap<String, Arc<VariantInfo>>,
        metrics: &HashMap<String, crate::config::MetricConfig>,
        _function_name: &str,
        warn_deprecated: bool,
    ) -> Result<ExperimentationConfigWithNamespaces, Error> {
        // Load the base config (no namespace)
        let base = self.base.load(variants, metrics, None, warn_deprecated)?;

        // Load namespace-specific configs
        let namespaces = self
            .namespaces
            .into_iter()
            .map(|(namespace, config)| {
                let loaded =
                    config.load(variants, metrics, Some(namespace.clone()), warn_deprecated)?;
                Ok((namespace, loaded))
            })
            .collect::<Result<HashMap<_, _>, Error>>()?;

        Ok(ExperimentationConfigWithNamespaces { base, namespaces })
    }
}

impl UninitializedExperimentationConfig {
    pub fn load(
        self,
        variants: &HashMap<String, Arc<VariantInfo>>,
        metrics: &HashMap<String, crate::config::MetricConfig>,
        namespace: Option<String>,
        warn_deprecated: bool,
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
            // New types — validate and pass through
            UninitializedExperimentationConfig::Static(config) => {
                config.validate(variants)?;
                Ok(ExperimentationConfig::Static(config))
            }
            UninitializedExperimentationConfig::Adaptive(config) => Ok(
                ExperimentationConfig::Adaptive(config.load(variants, metrics, namespace)?),
            ),
            // Legacy types — convert to new types with optional deprecation warnings
            UninitializedExperimentationConfig::Uniform(config) => {
                if warn_deprecated {
                    tracing::warn!(
                        "Experimentation type `uniform` is deprecated. Use `static` instead."
                    );
                }
                match config.into_static_config() {
                    Some(static_config) => {
                        static_config.validate(variants)?;
                        Ok(ExperimentationConfig::Static(static_config))
                    }
                    None => Ok(ExperimentationConfig::Default),
                }
            }
            UninitializedExperimentationConfig::StaticWeights(config) => {
                if warn_deprecated {
                    tracing::warn!(
                        "Experimentation type `static_weights` is deprecated. Use `static` instead."
                    );
                }
                let static_config = config.into_static_config();
                static_config.validate(variants)?;
                Ok(ExperimentationConfig::Static(static_config))
            }
            UninitializedExperimentationConfig::TrackAndStop(config) => {
                if warn_deprecated {
                    tracing::warn!(
                        "Experimentation type `track_and_stop` is deprecated. Use `adaptive` instead."
                    );
                }
                let adaptive = UninitializedAdaptiveExperimentationConfig {
                    algorithm: AdaptiveExperimentationAlgorithm::TrackAndStop,
                    inner: config,
                };
                Ok(ExperimentationConfig::Adaptive(
                    adaptive.load(variants, metrics, namespace)?,
                ))
            }
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
    /// Build an `ExperimentationConfig` from the legacy `variant.weight` map.
    /// Used when no `[experimentation]` section is present in config.
    pub fn legacy_from_variants_map(variants: &HashMap<String, Arc<VariantInfo>>) -> Self {
        // If any variant has an explicit weight, build a Static config with those weights
        for variant in variants.values() {
            if variant.inner.weight().is_some() {
                let config =
                    legacy::LegacyStaticWeightsExperimentationConfig::legacy_from_variants_map(
                        variants,
                    );
                return Self::Static(config.into_static_config());
            }
        }
        // Otherwise, default to uniform sampling over all variants
        Self::Default
    }

    pub async fn setup(
        &self,
        db: Arc<dyn FeedbackQueries + Send + Sync>,
        function_name: &str,
        postgres: &PostgresConnectionInfo,
        cancel_token: CancellationToken,
    ) -> Result<(), Error> {
        match self {
            Self::Default => Ok(()),
            Self::Static(config) => {
                config
                    .setup(db, function_name, postgres, cancel_token)
                    .await
            }
            Self::Adaptive(config) => {
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
            Self::Default => {
                return sample_uniform(function_name, &episode_id, active_variants, None);
            }
            Self::Static(config) => {
                config
                    .sample(function_name, episode_id, active_variants, postgres)
                    .await
            }
            Self::Adaptive(config) => {
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
        };

        // If the sampler fails but there are active variants, fall back to uniform sampling
        // from the allowed variants
        result.or_else(|e| {
            if active_variants.is_empty() {
                Err(e)
            } else {
                let allowed: Vec<&str> = match self {
                    Self::Default => {
                        return Err(Error::new(ErrorDetails::Inference {
                            message: format!(
                                "Default experimentation config should not reach fallback logic for function `{function_name}`. {IMPOSSIBLE_ERROR_MESSAGE}"
                            ),
                        }));
                    }
                    Self::Static(config) => config.allowed_variants().collect(),
                    Self::Adaptive(config) => config.allowed_variants().collect(),
                    #[cfg(test)]
                    Self::AlwaysFails(config) => config.allowed_variants().collect(),
                };
                let filter = if allowed.is_empty() {
                    None
                } else {
                    Some(allowed.as_slice())
                };
                sample_uniform(function_name, &episode_id, active_variants, filter)
            }
        })
    }

    /// Returns whether a variant name could be sampled by this experimentation config.
    /// `Default` means all variants are eligible, so we return `true`.
    pub fn could_sample_variant(&self, variant_name: &str) -> bool {
        let allowed: Vec<&str> = match self {
            Self::Default => return true,
            Self::Static(c) => c.allowed_variants().collect(),
            Self::Adaptive(c) => c.allowed_variants().collect(),
            #[cfg(test)]
            Self::AlwaysFails(c) => c.allowed_variants().collect(),
        };
        allowed.is_empty() || allowed.contains(&variant_name)
    }

    pub fn get_current_display_probabilities<'a>(
        &self,
        function_name: &str,
        active_variants: &'a HashMap<String, Arc<VariantInfo>>,
        postgres: &PostgresConnectionInfo,
    ) -> Result<HashMap<&'a str, f64>, Error> {
        match self {
            Self::Default => {
                if active_variants.is_empty() {
                    return Ok(HashMap::new());
                }
                let uniform_prob = 1.0 / active_variants.len() as f64;
                Ok(active_variants
                    .keys()
                    .map(|k| (k.as_str(), uniform_prob))
                    .collect())
            }
            Self::Static(config) => {
                config.get_current_display_probabilities(function_name, active_variants, postgres)
            }
            Self::Adaptive(config) => {
                config.get_current_display_probabilities(function_name, active_variants, postgres)
            }
            #[cfg(test)]
            Self::AlwaysFails(config) => {
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
    // Divide by 2^32 (not u32::MAX) so the result is strictly in [0, 1).
    // Using u32::MAX would allow exactly 1.0, which causes `sample_weighted`
    // to miss all candidates when `cumulative_weight > random_threshold` is
    // never satisfied (random_threshold == total_weight).
    truncated_hash as f64 / (u32::MAX as f64 + 1.0)
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
            .filter(|k| self.allowed_variants.contains(*k))
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
    use std::io::Write;

    use crate::config::{Config, ConfigFileGlob, ErrorContext, Namespace, SchemaData};
    use crate::variant::{VariantConfig, chat_completion::UninitializedChatCompletionConfig};
    use tempfile::NamedTempFile;
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

    #[test]
    fn test_get_current_display_probabilities_static_empty() {
        let active_variants = HashMap::new();

        let config = ExperimentationConfig::default();
        let postgres = PostgresConnectionInfo::new_disabled();
        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // Should return empty map
        assert_eq!(probs.len(), 0);
    }

    /// Helper to create a variants map with the given variant names (no weights).
    fn make_variants_map(names: &[&str]) -> HashMap<String, Arc<VariantInfo>> {
        let mut map = HashMap::new();
        for &name in names {
            map.insert(
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
        map
    }

    // =========================================================================
    // Tests for ExperimentationConfigWithNamespaces
    // =========================================================================

    #[test]
    fn test_get_for_namespace_none_returns_base() {
        let config = ExperimentationConfigWithNamespaces::default();
        let result = config.get_for_namespace(None);
        assert!(
            matches!(result, ExperimentationConfig::Default),
            "None namespace should return the base config"
        );
    }

    #[test]
    fn test_get_for_namespace_unknown_returns_base() {
        let config = ExperimentationConfigWithNamespaces::default();
        let ns = Namespace::new("unknown").unwrap();
        let result = config.get_for_namespace(Some(&ns));
        assert!(
            matches!(result, ExperimentationConfig::Default),
            "Unknown namespace should fall back to the base config"
        );
    }

    #[test]
    fn test_get_for_namespace_known_returns_override() {
        let mut namespaces = HashMap::new();
        namespaces.insert(
            "mobile".to_string(),
            ExperimentationConfig::Static(StaticExperimentationConfig {
                candidate_variants: WeightedVariants::from_map(BTreeMap::from([(
                    "variant_a".to_string(),
                    1.0,
                )])),
                fallback_variants: vec![],
            }),
        );
        let config = ExperimentationConfigWithNamespaces {
            base: ExperimentationConfig::default(),
            namespaces,
        };
        let ns = Namespace::new("mobile").unwrap();
        let result = config.get_for_namespace(Some(&ns));
        assert!(
            matches!(result, ExperimentationConfig::Static(_)),
            "Known namespace should return the namespace-specific config"
        );
    }

    #[test]
    fn test_has_namespace_config() {
        let mut namespaces = HashMap::new();
        namespaces.insert("mobile".to_string(), ExperimentationConfig::default());
        let config = ExperimentationConfigWithNamespaces {
            base: ExperimentationConfig::default(),
            namespaces,
        };
        assert!(
            config.has_namespace_config("mobile"),
            "`has_namespace_config` should return true for an existing namespace"
        );
        assert!(
            !config.has_namespace_config("web"),
            "`has_namespace_config` should return false for a missing namespace"
        );
    }

    #[test]
    fn test_legacy_from_variants_map_has_empty_namespaces() {
        let variants = make_variants_map(&["a", "b"]);
        let config = ExperimentationConfigWithNamespaces::legacy_from_variants_map(&variants);
        assert!(
            config.namespaces.is_empty(),
            "Legacy creation should produce no namespace configs"
        );
    }

    // =========================================================================
    // Tests for UninitializedExperimentationConfigWithNamespaces::load()
    // =========================================================================

    #[test]
    fn test_load_namespace_config_uniform_legacy() {
        let variants = make_variants_map(&["variant_a", "variant_b"]);
        let metrics = HashMap::new();

        let uninitialized = UninitializedExperimentationConfigWithNamespaces {
            base: UninitializedExperimentationConfig::Uniform(
                legacy::LegacyUniformExperimentationConfig::default(),
            ),
            namespaces: HashMap::from([(
                "mobile".to_string(),
                UninitializedExperimentationConfig::Uniform(
                    legacy::LegacyUniformExperimentationConfig::default(),
                ),
            )]),
        };

        let loaded = uninitialized.load(&variants, &metrics, "test_fn", false);
        assert!(
            loaded.is_ok(),
            "Namespace with legacy `uniform` type should load successfully"
        );
        let loaded = loaded.unwrap();
        assert!(
            matches!(
                loaded.namespaces.get("mobile"),
                Some(ExperimentationConfig::Default)
            ),
            "Loaded legacy `uniform` namespace with no candidates/fallbacks should become Default"
        );
    }

    #[test]
    fn test_load_namespace_config_static_weights_legacy() {
        let variants = make_variants_map(&["variant_a", "variant_b"]);
        let metrics = HashMap::new();

        let uninitialized = UninitializedExperimentationConfigWithNamespaces {
            base: UninitializedExperimentationConfig::Uniform(
                legacy::LegacyUniformExperimentationConfig::default(),
            ),
            namespaces: HashMap::from([(
                "mobile".to_string(),
                UninitializedExperimentationConfig::StaticWeights(
                    serde_json::from_value(serde_json::json!({
                        "candidate_variants": {"variant_a": 1.0}
                    }))
                    .unwrap(),
                ),
            )]),
        };

        let loaded = uninitialized.load(&variants, &metrics, "test_fn", false);
        assert!(
            loaded.is_ok(),
            "Namespace with legacy `static_weights` type should load successfully"
        );
        let loaded = loaded.unwrap();
        assert!(
            matches!(
                loaded.namespaces.get("mobile"),
                Some(ExperimentationConfig::Static(_))
            ),
            "Loaded legacy `static_weights` namespace should become Static"
        );
    }

    #[test]
    fn test_load_namespace_config_track_and_stop_legacy() {
        let variants = make_variants_map(&["variant_a", "variant_b"]);
        let mut metrics = HashMap::new();
        metrics.insert(
            "test_metric".to_string(),
            crate::config::MetricConfig {
                r#type: crate::config::MetricConfigType::Boolean,
                level: crate::config::MetricConfigLevel::Inference,
                optimize: crate::config::MetricConfigOptimize::Max,
                description: None,
            },
        );

        let uninitialized = UninitializedExperimentationConfigWithNamespaces {
            base: UninitializedExperimentationConfig::Uniform(
                legacy::LegacyUniformExperimentationConfig::default(),
            ),
            namespaces: HashMap::from([(
                "mobile".to_string(),
                UninitializedExperimentationConfig::TrackAndStop(
                    serde_json::from_value(serde_json::json!({
                        "metric": "test_metric",
                        "candidate_variants": ["variant_a", "variant_b"],
                        "min_samples_per_variant": 10,
                        "delta": 0.05,
                        "epsilon": 0.1,
                        "update_period_s": 60
                    }))
                    .unwrap(),
                ),
            )]),
        };

        let result = uninitialized.load(&variants, &metrics, "test_fn", false);
        assert!(
            result.is_ok(),
            "Namespace with legacy `track_and_stop` type should load successfully"
        );

        let loaded = result.unwrap();
        assert!(
            loaded.has_namespace_config("mobile"),
            "Should have a `mobile` namespace config"
        );
        assert!(
            matches!(
                loaded.get_for_namespace(Some(&Namespace::new("mobile").unwrap())),
                ExperimentationConfig::Adaptive(_)
            ),
            "The `mobile` namespace config should be `Adaptive`"
        );
    }

    #[test]
    fn test_load_static_type() {
        let variants = make_variants_map(&["variant_a", "variant_b"]);
        let metrics = HashMap::new();

        let config: UninitializedExperimentationConfig =
            serde_json::from_value(serde_json::json!({
                "type": "static",
                "candidate_variants": ["variant_a", "variant_b"]
            }))
            .unwrap();

        let loaded = config.load(&variants, &metrics, None, false);
        assert!(loaded.is_ok(), "Static type should load successfully");
        assert!(
            matches!(loaded.unwrap(), ExperimentationConfig::Static(_)),
            "Should produce a Static config"
        );
    }

    #[test]
    fn test_load_static_type_with_weights() {
        let variants = make_variants_map(&["variant_a", "variant_b"]);
        let metrics = HashMap::new();

        let config: UninitializedExperimentationConfig =
            serde_json::from_value(serde_json::json!({
                "type": "static",
                "candidate_variants": {"variant_a": 0.7, "variant_b": 0.3}
            }))
            .unwrap();

        let loaded = config.load(&variants, &metrics, None, false);
        assert!(
            loaded.is_ok(),
            "Static type with weights should load successfully"
        );
        assert!(
            matches!(loaded.unwrap(), ExperimentationConfig::Static(_)),
            "Should produce a Static config"
        );
    }

    #[test]
    fn test_load_adaptive_type() {
        let variants = make_variants_map(&["variant_a", "variant_b"]);
        let mut metrics = HashMap::new();
        metrics.insert(
            "test_metric".to_string(),
            crate::config::MetricConfig {
                r#type: crate::config::MetricConfigType::Boolean,
                level: crate::config::MetricConfigLevel::Inference,
                optimize: crate::config::MetricConfigOptimize::Max,
                description: None,
            },
        );

        let config: UninitializedExperimentationConfig =
            serde_json::from_value(serde_json::json!({
                "type": "adaptive",
                "metric": "test_metric",
                "candidate_variants": ["variant_a", "variant_b"],
                "min_samples_per_variant": 10,
                "delta": 0.05,
                "epsilon": 0.1,
                "update_period_s": 60
            }))
            .unwrap();

        let loaded = config.load(&variants, &metrics, None, false);
        assert!(loaded.is_ok(), "Adaptive type should load successfully");
        assert!(
            matches!(loaded.unwrap(), ExperimentationConfig::Adaptive(_)),
            "Should produce an Adaptive config"
        );
    }

    #[test]
    fn test_backward_compat_uniform_deserializes() {
        let config: UninitializedExperimentationConfig =
            serde_json::from_value(serde_json::json!({
                "type": "uniform"
            }))
            .unwrap();
        assert!(
            matches!(config, UninitializedExperimentationConfig::Uniform(_)),
            "Legacy `uniform` type should still deserialize"
        );
    }

    #[test]
    fn test_backward_compat_static_weights_deserializes() {
        let config: UninitializedExperimentationConfig =
            serde_json::from_value(serde_json::json!({
                "type": "static_weights",
                "candidate_variants": {"a": 1.0}
            }))
            .unwrap();
        assert!(
            matches!(config, UninitializedExperimentationConfig::StaticWeights(_)),
            "Legacy `static_weights` type should still deserialize"
        );
    }

    #[test]
    fn test_backward_compat_track_and_stop_deserializes() {
        let config: UninitializedExperimentationConfig =
            serde_json::from_value(serde_json::json!({
                "type": "track_and_stop",
                "metric": "test_metric",
                "candidate_variants": ["a", "b"]
            }))
            .unwrap();
        assert!(
            matches!(config, UninitializedExperimentationConfig::TrackAndStop(_)),
            "Legacy `track_and_stop` type should still deserialize"
        );
    }

    #[tokio::test]
    async fn test_experimentation_with_variant_weights_error_static() {
        let config_str = r#"
            [models.test]
            routing = ["test"]

            [models.test.providers.test]
            type = "dummy"
            model_name = "test"

            [functions.test_function]
            type = "chat"

            [functions.test_function.variants.variant_a]
            type = "chat_completion"
            model = "test"
            weight = 0.5

            [functions.test_function.variants.variant_b]
            type = "chat_completion"
            model = "test"
            weight = 0.5

            [functions.test_function.experimentation]
            type = "static"
            candidate_variants = ["variant_a", "variant_b"]
            "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let err = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .expect_err("Config should fail to load");

        let err_msg = err.to_string();
        assert!(
            err_msg.contains(
                "Cannot mix `experimentation` configuration with individual variant `weight` values"
            ),
            "Unexpected error message: {err_msg}"
        );
        assert!(
            err_msg.contains("variant_a") && err_msg.contains("variant_b"),
            "Error should list both variants with weights: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_experimentation_with_variant_weights_error_adaptive() {
        let config_str = r#"
            [models.test]
            routing = ["test"]

            [models.test.providers.test]
            type = "dummy"
            model_name = "test"

            [metrics.test_metric]
            type = "boolean"
            optimize = "max"
            level = "inference"

            [functions.test_function]
            type = "chat"

            [functions.test_function.variants.variant_a]
            type = "chat_completion"
            model = "test"
            weight = 0.6

            [functions.test_function.variants.variant_b]
            type = "chat_completion"
            model = "test"

            [functions.test_function.experimentation]
            type = "adaptive"
            metric = "test_metric"
            candidate_variants = ["variant_a", "variant_b"]
            min_samples_per_variant = 100
            delta = 0.05
            epsilon = 0.1
            update_period_s = 60
            "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let err = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .expect_err("Config should fail to load");

        let err_msg = err.to_string();
        assert!(
            err_msg.contains(
                "Cannot mix `experimentation` configuration with individual variant `weight` values"
            ),
            "Unexpected error message: {err_msg}"
        );
        assert!(
            err_msg.contains("variant_a"),
            "Error should list the variant with weight: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_experimentation_with_namespaces_valid() {
        let config_str = r#"
            [models.test]
            routing = ["test"]

            [models.test.providers.test]
            type = "dummy"
            model_name = "test"

            [functions.test_function]
            type = "chat"

            [functions.test_function.variants.variant_a]
            type = "chat_completion"
            model = "test"

            [functions.test_function.variants.variant_b]
            type = "chat_completion"
            model = "test"

            [functions.test_function.experimentation]
            type = "static"
            candidate_variants = ["variant_a", "variant_b"]

            [functions.test_function.experimentation.namespaces.mobile]
            type = "static"
            candidate_variants = {"variant_a" = 1.0}

            [functions.test_function.experimentation.namespaces.web]
            type = "static"
            candidate_variants = ["variant_a", "variant_b"]
            "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let loaded = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .expect("Config with valid namespace experimentation should load successfully");

        let func = loaded
            .functions
            .get("test_function")
            .expect("test_function should exist");
        let exp = func.experimentation_with_namespaces();
        assert!(
            exp.has_namespace_config("mobile"),
            "Should have a `mobile` namespace config"
        );
        assert!(
            exp.has_namespace_config("web"),
            "Should have a `web` namespace config"
        );
    }

    #[tokio::test]
    async fn test_experimentation_namespace_adaptive_loads() {
        let config_str = r#"
            [models.test]
            routing = ["test"]

            [models.test.providers.test]
            type = "dummy"
            model_name = "test"

            [metrics.test_metric]
            type = "boolean"
            optimize = "max"
            level = "inference"

            [functions.test_function]
            type = "chat"

            [functions.test_function.variants.variant_a]
            type = "chat_completion"
            model = "test"

            [functions.test_function.variants.variant_b]
            type = "chat_completion"
            model = "test"

            [functions.test_function.experimentation]
            type = "static"
            candidate_variants = ["variant_a", "variant_b"]

            [functions.test_function.experimentation.namespaces.mobile]
            type = "adaptive"
            metric = "test_metric"
            candidate_variants = ["variant_a", "variant_b"]
            min_samples_per_variant = 10
            delta = 0.05
            epsilon = 0.1
            update_period_s = 60
            "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let loaded = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .expect("Namespace with adaptive should load successfully");

        let function_config = loaded.functions.get("test_function").unwrap();
        let experimentation = function_config.experimentation_with_namespaces();
        assert!(
            experimentation.has_namespace_config("mobile"),
            "Should have a `mobile` namespace config"
        );
        assert!(
            matches!(
                experimentation.get_for_namespace(Some(&Namespace::new("mobile").unwrap())),
                ExperimentationConfig::Adaptive(_)
            ),
            "The `mobile` namespace config should be `Adaptive`"
        );
    }
}
