use std::collections::HashMap;
use std::sync::Arc;

use schemars::JsonSchema;

use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::model::ModelTable;

/// A validated namespace identifier.
///
/// Namespace identifiers must be non-empty strings.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[derive(ts_rs::TS)]
#[ts(export, type = "string")]
pub struct Namespace(String);

impl JsonSchema for Namespace {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("Namespace")
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        String::json_schema(generator)
    }
}

impl Namespace {
    /// Creates a new Namespace, validating that it is non-empty.
    pub fn new(namespace: impl Into<String>) -> Result<Self, Error> {
        let namespace = namespace.into();
        if namespace.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Namespace identifier cannot be empty".to_string(),
            }));
        }
        if namespace.starts_with("tensorzero::") {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Namespace cannot start with `tensorzero::`: {namespace}"),
            }));
        }
        Ok(Self(namespace))
    }

    /// Returns the namespace as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the Namespace and returns the inner String.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::fmt::Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for Namespace {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl serde::Serialize for Namespace {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Namespace {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Namespace::new(s).map_err(|e| serde::de::Error::custom(e.to_string()))
    }
}

/// Validates that namespaced models are only used in matching namespace experimentation configs.
///
/// A variant using a namespaced model must not be reachable from the base experimentation config
/// or from a different namespace's experimentation config. This ensures that namespace isolation
/// is maintained: a model scoped to namespace "A" can only be sampled by the experimentation
/// config for namespace "A".
pub fn validate_namespaced_model_usage(
    functions: &HashMap<String, Arc<FunctionConfig>>,
    models: &ModelTable,
) -> Result<(), Error> {
    for (function_name, function) in functions {
        let experimentation = function.experimentation_with_namespaces();
        let all_variants = function.variants();

        for (variant_name, variant_info) in all_variants {
            for model_name in variant_info.inner.all_model_names(all_variants) {
                let Some(model_namespace) = models.get_namespace(model_name) else {
                    continue; // Unnamespaced models have no restrictions
                };

                // Check: base experimentation must not sample this variant
                if experimentation.base.could_sample_variant(variant_name) {
                    return Err(ErrorDetails::Config {
                        message: format!(
                            "Variant `{variant_name}` of function `{function_name}` uses model `{model_name}` \
                            which has namespace `{model_namespace}`, but the variant is reachable from the \
                            base experimentation config. Namespaced model variants must only be reachable \
                            from a matching namespace experimentation config."
                        ),
                    }
                    .into());
                }

                // Check: no other namespace experimentation config should sample this variant
                // unless its namespace matches the model's namespace
                for (ns_name, ns_config) in &experimentation.namespaces {
                    if ns_name != model_namespace.as_str()
                        && ns_config.could_sample_variant(variant_name)
                    {
                        return Err(ErrorDetails::Config {
                            message: format!(
                                "Variant `{variant_name}` of function `{function_name}` uses model `{model_name}` \
                                which has namespace `{model_namespace}`, but the variant is reachable from \
                                namespace `{ns_name}` experimentation config. Namespaced model variants must only \
                                be reachable from a matching namespace experimentation config."
                            ),
                        }
                        .into());
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validates that namespaced variants are only used in matching namespace experimentation configs.
///
/// A variant with a namespace must not be reachable from the base experimentation config
/// or from a different namespace's experimentation config. This ensures that namespace isolation
/// is maintained: a variant scoped to namespace "A" can only be sampled by the experimentation
/// config for namespace "A".
pub fn validate_namespaced_variant_usage(
    functions: &HashMap<String, Arc<FunctionConfig>>,
) -> Result<(), Error> {
    for (function_name, function) in functions {
        let experimentation = function.experimentation_with_namespaces();
        let all_variants = function.variants();

        for (variant_name, variant_info) in all_variants {
            let Some(variant_namespace) = variant_info.namespace.as_ref() else {
                continue; // Unnamespaced variants have no restrictions
            };

            // Check: base experimentation must not sample this variant
            if experimentation.base.could_sample_variant(variant_name) {
                return Err(ErrorDetails::Config {
                    message: format!(
                        "Variant `{variant_name}` of function `{function_name}` has namespace `{variant_namespace}`, \
                        but is reachable from the base experimentation config. Namespaced variants must only be \
                        reachable from a matching namespace experimentation config."
                    ),
                }
                .into());
            }

            // Check: no other namespace experimentation config should sample this variant
            for (ns_name, ns_config) in &experimentation.namespaces {
                if ns_name != variant_namespace.as_str()
                    && ns_config.could_sample_variant(variant_name)
                {
                    return Err(ErrorDetails::Config {
                        message: format!(
                            "Variant `{variant_name}` of function `{function_name}` has namespace `{variant_namespace}`, \
                            but is reachable from namespace `{ns_name}` experimentation config. Namespaced variants \
                            must only be reachable from a matching namespace experimentation config."
                        ),
                    }
                    .into());
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use crate::experimentation::static_experimentation::{
        StaticExperimentationConfig, WeightedVariants,
    };
    use crate::experimentation::{ExperimentationConfig, ExperimentationConfigWithNamespaces};
    use crate::function::FunctionConfigChat;
    use crate::tool::ToolChoice;
    use crate::variant::{VariantConfig, VariantInfo};

    #[test]
    fn test_namespace_valid() {
        let cases = ["abc", "my_namespace_2", "a", "UPPER", "with-hyphens"];
        for case in cases {
            let ns = Namespace::new(case);
            assert!(ns.is_ok(), "Namespace `{case}` should be valid");
            assert_eq!(ns.unwrap().as_str(), case);
        }
    }

    #[test]
    fn test_namespace_empty_rejected() {
        let ns = Namespace::new("");
        assert!(ns.is_err(), "Empty namespace should be rejected");
    }

    #[test]
    fn test_namespace_tensorzero_prefix_rejected() {
        let ns = Namespace::new("tensorzero::internal");
        assert!(
            ns.is_err(),
            "Namespace starting with `tensorzero::` should be rejected"
        );
        let err_msg = ns.unwrap_err().to_string();
        assert!(
            err_msg.contains("tensorzero::"),
            "Error should mention the prefix, got: {err_msg}"
        );
    }

    #[test]
    fn test_namespace_serde_roundtrip() {
        let ns = Namespace::new("my_namespace").unwrap();
        let serialized = serde_json::to_string(&ns).unwrap();
        assert_eq!(serialized, "\"my_namespace\"");
        let deserialized: Namespace = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            ns, deserialized,
            "Namespace should survive serialize + deserialize roundtrip"
        );
    }

    #[test]
    fn test_namespace_deserialize_empty_rejected() {
        let result = serde_json::from_str::<Namespace>("\"\"");
        assert!(result.is_err(), "Deserializing an empty string should fail");
    }

    #[test]
    fn test_namespace_deserialize_tensorzero_prefix_rejected() {
        let result = serde_json::from_str::<Namespace>("\"tensorzero::foo\"");
        assert!(
            result.is_err(),
            "Deserializing a namespace starting with `tensorzero::` should fail"
        );
    }

    /// Helper: build a minimal Chat function with the given variants and experimentation config.
    fn make_chat_function(
        variants: HashMap<String, Arc<VariantInfo>>,
        experimentation: ExperimentationConfigWithNamespaces,
    ) -> Arc<FunctionConfig> {
        Arc::new(FunctionConfig::Chat(FunctionConfigChat {
            variants,
            schemas: Default::default(),
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation,
            evaluators: HashMap::new(),
        }))
    }

    /// Helper: build a VariantInfo with an optional namespace.
    fn make_variant(namespace: Option<Namespace>) -> Arc<VariantInfo> {
        Arc::new(VariantInfo {
            inner: VariantConfig::ChatCompletion(Default::default()),
            timeouts: Default::default(),
            namespace,
        })
    }

    #[test]
    fn test_validate_namespaced_variant_no_namespace_passes() {
        // Variants without namespace should always pass
        let variants: HashMap<String, Arc<VariantInfo>> = [
            ("v1".to_string(), make_variant(None)),
            ("v2".to_string(), make_variant(None)),
        ]
        .into();

        let experimentation = ExperimentationConfigWithNamespaces {
            base: ExperimentationConfig::Default,
            namespaces: HashMap::new(),
        };

        let functions: HashMap<String, Arc<FunctionConfig>> = [(
            "fn1".to_string(),
            make_chat_function(variants, experimentation),
        )]
        .into();

        let result = validate_namespaced_variant_usage(&functions);
        assert!(
            result.is_ok(),
            "Variants without namespace should pass validation"
        );
    }

    #[test]
    fn test_validate_namespaced_variant_reachable_from_base_rejected() {
        // A variant with namespace "mobile" is reachable from the base (Default) config → error
        let variants: HashMap<String, Arc<VariantInfo>> = [(
            "v1".to_string(),
            make_variant(Some(Namespace::new("mobile").unwrap())),
        )]
        .into();

        let experimentation = ExperimentationConfigWithNamespaces {
            base: ExperimentationConfig::Default,
            namespaces: HashMap::new(),
        };

        let functions: HashMap<String, Arc<FunctionConfig>> = [(
            "fn1".to_string(),
            make_chat_function(variants, experimentation),
        )]
        .into();

        let result = validate_namespaced_variant_usage(&functions);
        assert!(
            result.is_err(),
            "Namespaced variant reachable from base should be rejected"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("mobile") && err_msg.contains("base experimentation"),
            "Error should mention the namespace and base config, got: {err_msg}"
        );
    }

    #[test]
    fn test_validate_namespaced_variant_matching_namespace_passes() {
        // A variant with namespace "mobile" is only reachable from the "mobile" namespace config → ok
        // We also need a non-namespaced variant "v_base" to give the base config something to point at.
        let variants: HashMap<String, Arc<VariantInfo>> = [
            (
                "v1".to_string(),
                make_variant(Some(Namespace::new("mobile").unwrap())),
            ),
            ("v_base".to_string(), make_variant(None)),
        ]
        .into();

        let mobile_config = ExperimentationConfig::Static(StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_map(
                [("v1".to_string(), 1.0)].into_iter().collect(),
            ),
            fallback_variants: vec![],
        });

        // Base config explicitly lists only v_base, so it cannot sample v1
        let base_config = ExperimentationConfig::Static(StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_map(
                [("v_base".to_string(), 1.0)].into_iter().collect(),
            ),
            fallback_variants: vec![],
        });

        let experimentation = ExperimentationConfigWithNamespaces {
            base: base_config,
            namespaces: [("mobile".to_string(), mobile_config)].into(),
        };

        let functions: HashMap<String, Arc<FunctionConfig>> = [(
            "fn1".to_string(),
            make_chat_function(variants, experimentation),
        )]
        .into();

        let result = validate_namespaced_variant_usage(&functions);
        assert!(
            result.is_ok(),
            "Namespaced variant reachable only from matching namespace should pass"
        );
    }

    #[test]
    fn test_validate_namespaced_variant_wrong_namespace_rejected() {
        // A variant with namespace "mobile" is reachable from the "web" namespace config → error
        let variants: HashMap<String, Arc<VariantInfo>> = [
            (
                "v1".to_string(),
                make_variant(Some(Namespace::new("mobile").unwrap())),
            ),
            ("v_base".to_string(), make_variant(None)),
        ]
        .into();

        let web_config = ExperimentationConfig::Static(StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_map(
                [("v1".to_string(), 1.0)].into_iter().collect(),
            ),
            fallback_variants: vec![],
        });

        // Base config explicitly lists only v_base, so it cannot sample v1
        let base_config = ExperimentationConfig::Static(StaticExperimentationConfig {
            candidate_variants: WeightedVariants::from_map(
                [("v_base".to_string(), 1.0)].into_iter().collect(),
            ),
            fallback_variants: vec![],
        });

        let experimentation = ExperimentationConfigWithNamespaces {
            base: base_config,
            namespaces: [("web".to_string(), web_config)].into(),
        };

        let functions: HashMap<String, Arc<FunctionConfig>> = [(
            "fn1".to_string(),
            make_chat_function(variants, experimentation),
        )]
        .into();

        let result = validate_namespaced_variant_usage(&functions);
        assert!(
            result.is_err(),
            "Namespaced variant reachable from wrong namespace should be rejected"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("mobile") && err_msg.contains("web"),
            "Error should mention both namespaces, got: {err_msg}"
        );
    }
}
