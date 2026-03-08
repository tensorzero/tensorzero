use schemars::JsonSchema;
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

use super::{VariantSampler, check_duplicates_across, check_duplicates_within};

/// Unified static experimentation config that replaces both `uniform` and `static_weights`.
///
/// `candidate_variants` is mandatory and stored as a `BTreeMap<String, f64>`.
/// It can be deserialized from either a list of variant names (equal weights of 1.0 each)
/// or a map of variant names to explicit weights.
///
/// `fallback_variants` is optional and defaults to an empty list.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct StaticExperimentationConfig {
    pub candidate_variants: WeightedVariants,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallback_variants: Vec<String>,
}

/// A newtype wrapping `BTreeMap<String, f64>` that can be deserialized from either:
/// - A JSON/TOML array of strings `["a", "b"]` → each gets weight 1.0
/// - A JSON/TOML map `{"a": 0.7, "b": 0.3}` → explicit weights
///
/// Always serializes as a map.
#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct WeightedVariants(BTreeMap<String, f64>);

impl WeightedVariants {
    pub fn from_map(map: BTreeMap<String, f64>) -> Self {
        Self(map)
    }

    pub fn from_equal_weights(names: Vec<String>) -> Self {
        Self(names.into_iter().map(|n| (n, 1.0)).collect())
    }

    pub fn inner(&self) -> &BTreeMap<String, f64> {
        &self.0
    }

    pub fn into_inner(self) -> BTreeMap<String, f64> {
        self.0
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(feature = "ts-bindings")]
impl ts_rs::TS for WeightedVariants {
    type WithoutGenerics = Self;
    type OptionInnerType = Self;

    fn name(_cfg: &ts_rs::Config) -> String {
        "WeightedVariants".to_string()
    }

    fn decl(cfg: &ts_rs::Config) -> String {
        format!("type WeightedVariants = {};", Self::inline(cfg))
    }

    fn decl_concrete(cfg: &ts_rs::Config) -> String {
        Self::decl(cfg)
    }

    fn inline(_cfg: &ts_rs::Config) -> String {
        "string[] | Record<string, number>".to_string()
    }

    fn output_path() -> Option<std::path::PathBuf> {
        Some(std::path::PathBuf::from("WeightedVariants.ts"))
    }
}

impl<'de> Deserialize<'de> for WeightedVariants {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, SeqAccess, Visitor};
        use std::fmt;

        struct WeightedVariantsVisitor;

        impl<'de> Visitor<'de> for WeightedVariantsVisitor {
            type Value = WeightedVariants;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a list of variant names or a map of variant names to weights")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut names = Vec::new();
                while let Some(name) = seq.next_element::<String>()? {
                    if names.contains(&name) {
                        return Err(de::Error::custom(format!(
                            "duplicate variant name in `candidate_variants`: `{name}`"
                        )));
                    }
                    names.push(name);
                }
                Ok(WeightedVariants::from_equal_weights(names))
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut result = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, f64>()? {
                    if result.contains_key(&key) {
                        return Err(de::Error::custom(format!(
                            "duplicate variant name in `candidate_variants`: `{key}`"
                        )));
                    }
                    result.insert(key, value);
                }
                Ok(WeightedVariants::from_map(result))
            }
        }

        deserializer.deserialize_any(WeightedVariantsVisitor)
    }
}

/// Pure function for weighted sampling logic.
/// Given a uniform sample in [0, 1), selects a variant from active_variants
/// using weighted sampling from candidate_variants if their intersection is nonempty,
/// or sequential sampling from fallback_variants otherwise.
///
/// Returns the name of the selected variant, which is guaranteed to be in active_variants.
pub(crate) fn sample_weighted(
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
            // Fallback for rare floating-point precision edge cases where
            // `cumulative_weight` never exceeds `random_threshold`.
            // We pick the last active variant with positive candidate weight
            // to stay within the candidate set rather than returning an
            // arbitrary active variant that might not be a candidate.
            active_variants
                .keys()
                .rev()
                .find(|name| {
                    candidate_variants
                        .get(name.as_str())
                        .is_some_and(|&w| w > 0.0)
                })
                .or_else(|| active_variants.keys().next())
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

impl StaticExperimentationConfig {
    /// Validate that candidate and fallback variant names exist in the function's variants map,
    /// check for duplicates, and validate weights.
    /// This should be called during config loading to catch typos early.
    pub fn validate(
        &self,
        function_variants: &HashMap<String, Arc<VariantInfo>>,
    ) -> Result<(), Error> {
        // Validate candidate variant names exist
        for name in self.candidate_variants.keys() {
            if !function_variants.contains_key(name) {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "static experimentation: candidate variant `{name}` does not exist in function variants"
                    ),
                }));
            }
        }

        // Validate fallback variant names exist
        for name in &self.fallback_variants {
            if !function_variants.contains_key(name) {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "static experimentation: fallback variant `{name}` does not exist in function variants"
                    ),
                }));
            }
        }

        // Check for duplicates within fallback_variants
        check_duplicates_within(&self.fallback_variants, "fallback_variants")?;

        // Check for duplicates across candidate_variants and fallback_variants
        check_duplicates_across(self.candidate_variants.keys(), &self.fallback_variants)?;

        // Validate that all weights are non-negative
        for (name, weight) in self.candidate_variants.inner() {
            if *weight < 0.0 {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "Invalid weight for variant `{name}` in static experimentation config: {weight}"
                    ),
                }));
            }
        }

        // If no candidate has positive weight and there are no fallback variants,
        // sampling will always fail at runtime. Reject this at load time.
        // (This also catches explicitly empty `candidate_variants = []`.)
        if !self.candidate_variants.inner().values().any(|&w| w > 0.0)
            && self.fallback_variants.is_empty()
        {
            return Err(Error::new(ErrorDetails::Config {
                message:
                    "Experimentation config has no active candidate variants and no fallback variants. At least one is required."
                        .to_string(),
            }));
        }

        Ok(())
    }
}

impl VariantSampler for StaticExperimentationConfig {
    async fn setup(
        &self,
        _db: Arc<dyn FeedbackQueries + Send + Sync>,
        _function_name: &str,
        _postgres: &PostgresConnectionInfo,
        _cancel_token: CancellationToken,
    ) -> Result<(), Error> {
        // Validation (duplicates, weights, variant existence) is done in
        // StaticConfig::validate() at config load time.
        Ok(())
    }

    /// Sample a variant using weighted sampling from candidates, or sequential from fallbacks.
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
        let selected_variant_name = sample_weighted(
            active_variants,
            self.candidate_variants.inner(),
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
            .map(|variant_name| {
                self.candidate_variants
                    .inner()
                    .get(variant_name)
                    .unwrap_or(&0.0)
            })
            .sum();

        if total_weight <= 0.0 {
            // No active variants in the candidate set, use fallback variants
            let first_active_fallback = self
                .fallback_variants
                .iter()
                .find(|variant_name| active_variants.contains_key(*variant_name));

            if let Some(selected_variant) = first_active_fallback {
                let probabilities: HashMap<&'a str, f64> = active_variants
                    .keys()
                    .filter(|key| self.fallback_variants.contains(key))
                    .map(|key| {
                        let prob = if key == selected_variant { 1.0 } else { 0.0 };
                        (key.as_str(), prob)
                    })
                    .collect();
                Ok(probabilities)
            } else {
                Err(ErrorDetails::NoFallbackVariantsRemaining.into())
            }
        } else {
            // Use weighted probabilities from candidate variants
            let probabilities: HashMap<&'a str, f64> = active_variants
                .keys()
                .map(|variant_name| {
                    let weight = self
                        .candidate_variants
                        .inner()
                        .get(variant_name)
                        .unwrap_or(&0.0);
                    (variant_name.as_str(), weight / total_weight)
                })
                .collect();
            Ok(probabilities)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ErrorContext, SchemaData};
    use crate::variant::{VariantConfig, chat_completion::UninitializedChatCompletionConfig};

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
    fn test_weighted_variants_deserialize_from_vec() {
        let json = r#"["a", "b", "c"]"#;
        let wv: WeightedVariants = serde_json::from_str(json).unwrap();
        assert_eq!(wv.inner().len(), 3, "Should have 3 entries");
        assert_eq!(wv.inner()["a"], 1.0, "All weights should be 1.0");
        assert_eq!(wv.inner()["b"], 1.0, "All weights should be 1.0");
        assert_eq!(wv.inner()["c"], 1.0, "All weights should be 1.0");
    }

    #[test]
    fn test_weighted_variants_deserialize_from_map() {
        let json = r#"{"a": 0.7, "b": 0.3}"#;
        let wv: WeightedVariants = serde_json::from_str(json).unwrap();
        assert_eq!(wv.inner().len(), 2, "Should have 2 entries");
        assert!(
            (wv.inner()["a"] - 0.7).abs() < 1e-9,
            "Weight for `a` should be 0.7"
        );
        assert!(
            (wv.inner()["b"] - 0.3).abs() < 1e-9,
            "Weight for `b` should be 0.3"
        );
    }

    #[test]
    fn test_weighted_variants_deserialize_from_vec_rejects_duplicates() {
        let json = r#"["a", "b", "a"]"#;
        let result = serde_json::from_str::<WeightedVariants>(json);
        assert!(
            result.is_err(),
            "Should reject duplicate variant names in list form"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("duplicate variant name"),
            "Error should mention duplicate: {err}"
        );
    }

    #[test]
    fn test_weighted_variants_deserialize_from_map_rejects_duplicates() {
        // JSON maps with duplicate keys: serde_json passes both to the visitor
        let json = r#"{"a": 0.5, "b": 0.3, "a": 0.2}"#;
        let result = serde_json::from_str::<WeightedVariants>(json);
        assert!(
            result.is_err(),
            "Should reject duplicate variant names in map form"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("duplicate variant name"),
            "Error should mention duplicate: {err}"
        );
    }

    #[test]
    fn test_weighted_variants_serialize_as_map() {
        let wv = WeightedVariants::from_equal_weights(vec!["a".into(), "b".into()]);
        let json = serde_json::to_value(&wv).unwrap();
        assert!(json.is_object(), "Should serialize as a map");
        assert_eq!(json["a"], 1.0);
        assert_eq!(json["b"], 1.0);
    }

    #[test]
    fn test_static_config_deserialize_with_vec_candidates() {
        let json = r#"{"candidate_variants": ["a", "b"]}"#;
        let config: StaticExperimentationConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            config.candidate_variants.inner().len(),
            2,
            "Should have 2 candidate variants"
        );
        assert!(
            config.fallback_variants.is_empty(),
            "Fallback variants should be empty by default"
        );
    }

    #[test]
    fn test_static_config_deserialize_with_map_candidates() {
        let json = r#"{"candidate_variants": {"a": 0.5, "b": 0.5}, "fallback_variants": ["c"]}"#;
        let config: StaticExperimentationConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            config.candidate_variants.inner().len(),
            2,
            "Should have 2 candidate variants"
        );
        assert_eq!(
            config.fallback_variants.len(),
            1,
            "Should have 1 fallback variant"
        );
    }

    #[test]
    fn test_sample_weighted_deterministic() {
        let active_variants = create_test_variants(&["A", "B", "C"]);
        let mut candidate_variants = BTreeMap::new();
        candidate_variants.insert("A".to_string(), 1.0);
        candidate_variants.insert("B".to_string(), 2.0);
        candidate_variants.insert("C".to_string(), 3.0);

        // Total weight = 6.0
        // A: [0.0, 1.0/6.0)
        // B: [1.0/6.0, 3.0/6.0)
        // C: [3.0/6.0, 6.0/6.0)

        let result = sample_weighted(&active_variants, &candidate_variants, &[], 0.1);
        assert_eq!(result.unwrap(), "A", "Sample 0.1 should select A");

        let result = sample_weighted(&active_variants, &candidate_variants, &[], 0.3);
        assert_eq!(result.unwrap(), "B", "Sample 0.3 should select B");

        let result = sample_weighted(&active_variants, &candidate_variants, &[], 0.7);
        assert_eq!(result.unwrap(), "C", "Sample 0.7 should select C");
    }

    #[test]
    fn test_sample_weighted_fallback() {
        let active_variants = create_test_variants(&["A", "B", "C"]);
        let candidate_variants = BTreeMap::new();
        let fallback_variants = vec!["B".to_string(), "C".to_string()];

        let result = sample_weighted(
            &active_variants,
            &candidate_variants,
            &fallback_variants,
            0.5,
        );
        assert_eq!(
            result.unwrap(),
            "B",
            "Should select first active fallback variant"
        );
    }

    #[tokio::test]
    async fn test_static_config_sample_weighted() {
        let config = StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_map(BTreeMap::from([
                ("A".to_string(), 1.0),
                ("B".to_string(), 2.0),
            ])),
            fallback_variants: vec!["C".to_string()],
        };

        let variants = create_test_variants(&["A", "B", "C"]);
        let mut active_variants = variants.clone();
        let episode_id = Uuid::now_v7();
        let postgres = PostgresConnectionInfo::new_disabled();

        let (variant_name, _) = config
            .sample("test_function", episode_id, &mut active_variants, &postgres)
            .await
            .unwrap();

        assert!(
            ["A", "B"].contains(&variant_name.as_str()),
            "Should sample from candidate variants"
        );
    }

    #[test]
    fn test_static_config_validate_negative_weight() {
        let config = StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_map(BTreeMap::from([(
                "A".to_string(),
                -1.0,
            )])),
            fallback_variants: vec![],
        };

        let function_variants: HashMap<_, _> = create_test_variants(&["A"]).into_iter().collect();

        let result = config.validate(&function_variants);
        assert!(
            result.is_err(),
            "Should reject negative weights during validation"
        );
    }

    #[test]
    fn test_static_config_validate_unknown_candidate() {
        let config = StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_equal_weights(vec![
                "A".to_string(),
                "nonexistent".to_string(),
            ]),
            fallback_variants: vec![],
        };

        let function_variants: HashMap<_, _> =
            create_test_variants(&["A", "B"]).into_iter().collect();

        let result = config.validate(&function_variants);
        assert!(
            result.is_err(),
            "Should reject unknown candidate variant at validation time"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("nonexistent"),
            "Error should mention the unknown variant: {err}"
        );
    }

    #[test]
    fn test_static_config_validate_unknown_fallback() {
        let config = StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_equal_weights(vec!["A".to_string()]),
            fallback_variants: vec!["missing".to_string()],
        };

        let function_variants: HashMap<_, _> =
            create_test_variants(&["A", "B"]).into_iter().collect();

        let result = config.validate(&function_variants);
        assert!(
            result.is_err(),
            "Should reject unknown fallback variant at validation time"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("missing"),
            "Error should mention the unknown variant: {err}"
        );
    }

    #[test]
    fn test_static_config_validate_duplicate_fallbacks() {
        let config = StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_equal_weights(vec!["A".to_string()]),
            fallback_variants: vec!["B".to_string(), "C".to_string(), "B".to_string()],
        };

        let function_variants: HashMap<_, _> =
            create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let result = config.validate(&function_variants);
        assert!(
            result.is_err(),
            "Should reject duplicate fallback variant names"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("duplicate"),
            "Error should mention duplicate: {err}"
        );
    }

    #[test]
    fn test_static_config_validate_duplicate_across_candidate_and_fallback() {
        let config = StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_equal_weights(vec![
                "A".to_string(),
                "B".to_string(),
            ]),
            fallback_variants: vec!["B".to_string(), "C".to_string()],
        };

        let function_variants: HashMap<_, _> =
            create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let result = config.validate(&function_variants);
        assert!(
            result.is_err(),
            "Should reject variants appearing in both candidate and fallback lists"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("cannot appear in both"),
            "Error should mention cross-list duplicate: {err}"
        );
    }

    #[test]
    fn test_static_config_validate_zero_weights_no_fallbacks() {
        let config = StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_map(BTreeMap::from([
                ("A".to_string(), 0.0),
                ("B".to_string(), 0.0),
            ])),
            fallback_variants: vec![],
        };

        let function_variants: HashMap<_, _> =
            create_test_variants(&["A", "B"]).into_iter().collect();

        let result = config.validate(&function_variants);
        assert!(
            result.is_err(),
            "Should reject config with all-zero candidate weights and no fallbacks"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("no active candidate variants and no fallback variants"),
            "Error should explain the issue: {err}"
        );
    }

    #[test]
    fn test_static_config_validate_empty_candidates_no_fallbacks() {
        let config = StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_equal_weights(vec![]),
            fallback_variants: vec![],
        };

        let function_variants: HashMap<_, _> =
            create_test_variants(&["A", "B"]).into_iter().collect();

        let result = config.validate(&function_variants);
        assert!(
            result.is_err(),
            "Should reject config with empty candidates and no fallbacks"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("no active candidate variants and no fallback variants"),
            "Error should explain the issue: {err}"
        );
    }

    #[test]
    fn test_static_config_validate_empty_candidates_with_fallbacks_ok() {
        let config = StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_equal_weights(vec![]),
            fallback_variants: vec!["A".to_string()],
        };

        let function_variants: HashMap<_, _> =
            create_test_variants(&["A", "B"]).into_iter().collect();

        let result = config.validate(&function_variants);
        assert!(
            result.is_ok(),
            "Empty candidates with fallbacks should be valid"
        );
    }

    #[test]
    fn test_static_config_validate_zero_weights_with_fallbacks_ok() {
        let config = StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_map(BTreeMap::from([(
                "A".to_string(),
                0.0,
            )])),
            fallback_variants: vec!["B".to_string()],
        };

        let function_variants: HashMap<_, _> =
            create_test_variants(&["A", "B"]).into_iter().collect();

        let result = config.validate(&function_variants);
        assert!(
            result.is_ok(),
            "Zero-weight candidates with fallbacks should be valid"
        );
    }

    #[test]
    fn test_get_current_display_probabilities_weighted() {
        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let config = StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_map(BTreeMap::from([
                ("A".to_string(), 1.0),
                ("B".to_string(), 2.0),
                ("C".to_string(), 3.0),
            ])),
            fallback_variants: vec![],
        };

        let postgres = PostgresConnectionInfo::new_disabled();
        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        assert_eq!(probs.len(), 3);
        assert!((probs["A"] - 1.0 / 6.0).abs() < 1e-9);
        assert!((probs["B"] - 2.0 / 6.0).abs() < 1e-9);
        assert!((probs["C"] - 3.0 / 6.0).abs() < 1e-9);
    }
}
