use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::model::ModelTable;

/// A validated namespace identifier.
///
/// Namespace identifiers must be non-empty strings.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export, type = "string"))]
pub struct Namespace(String);

impl Namespace {
    /// Creates a new Namespace, validating that it is non-empty.
    pub fn new(namespace: impl Into<String>) -> Result<Self, Error> {
        let namespace = namespace.into();
        if namespace.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Namespace identifier cannot be empty".to_string(),
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
        for (variant_name, variant_info) in function.variants() {
            for model_name in variant_info.inner.direct_model_names() {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
