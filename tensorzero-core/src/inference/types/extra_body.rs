use super::{deserialize_delete, serialize_delete};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(transparent)]
pub struct ExtraBodyConfig {
    pub data: Vec<ExtraBodyReplacement>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
pub struct ExtraBodyReplacement {
    pub pointer: String,
    #[serde(flatten)]
    pub kind: ExtraBodyReplacementKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(rename_all = "snake_case")]
pub enum ExtraBodyReplacementKind {
    Value(Value),
    // We only allow `"delete": true` to be set - deserializing `"delete": false` will error
    #[serde(
        serialize_with = "serialize_delete",
        deserialize_with = "deserialize_delete"
    )]
    Delete,
}

/// The 'InferenceExtraBody' options provided directly in an inference request.
/// These have not yet been filtered by variant name
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(transparent)]
pub struct UnfilteredInferenceExtraBody {
    extra_body: Vec<InferenceExtraBody>,
}

impl UnfilteredInferenceExtraBody {
    pub fn is_empty(&self) -> bool {
        self.extra_body.is_empty()
    }

    /// Get a reference to the extra_body vector
    pub fn as_slice(&self) -> &[InferenceExtraBody] {
        &self.extra_body
    }

    /// Filter the 'InferenceExtraBody' options by variant name
    /// If the variant name is `None`, then all variant-specific extra body options are removed
    pub fn filter(self, variant_name: &str) -> FilteredInferenceExtraBody {
        FilteredInferenceExtraBody {
            data: self
                .extra_body
                .into_iter()
                .filter(|config| config.should_apply_variant(variant_name))
                .collect(),
        }
    }
}

/// The result of filtering `InferenceExtraBody` by variant name.
/// All `InferenceExtraBody::Variant` options with a non-matching variant have
/// been removed, while all `InferenceExtraBody::Provider` options have been retained.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(transparent)]
pub struct FilteredInferenceExtraBody {
    pub data: Vec<InferenceExtraBody>,
}

/// Holds the config-level and inference-level extra body options
#[derive(Clone, Debug, Default, PartialEq, Serialize, ts_rs::TS)]
pub struct FullExtraBodyConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<ExtraBodyConfig>,
    pub inference_extra_body: FilteredInferenceExtraBody,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(untagged)]
pub enum InferenceExtraBody {
    Provider {
        model_provider_name: String,
        pointer: String,
        #[serde(flatten)]
        kind: ExtraBodyReplacementKind,
    },
    Variant {
        variant_name: String,
        pointer: String,
        #[serde(flatten)]
        kind: ExtraBodyReplacementKind,
    },
    ModelProvider {
        model_name: String,
        provider_name: String,
        pointer: String,
        #[serde(flatten)]
        kind: ExtraBodyReplacementKind,
    },
    Always {
        pointer: String,
        #[serde(flatten)]
        kind: ExtraBodyReplacementKind,
    },
}

impl InferenceExtraBody {
    pub fn should_apply_variant(&self, variant_name: &str) -> bool {
        match self {
            InferenceExtraBody::Provider { .. } => true,
            InferenceExtraBody::Variant {
                variant_name: v, ..
            } => v == variant_name,
            InferenceExtraBody::ModelProvider { .. } => true,
            InferenceExtraBody::Always { .. } => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_inference_extra_body_all_deserialize() {
        let json = r#"{"pointer": "/test", "value": {"key": "value"}}"#;
        let result: InferenceExtraBody = serde_json::from_str(json).unwrap();
        match result {
            InferenceExtraBody::Always { pointer, kind } => {
                assert_eq!(pointer, "/test");
                match kind {
                    ExtraBodyReplacementKind::Value(v) => {
                        assert_eq!(v, json!({"key": "value"}));
                    }
                    ExtraBodyReplacementKind::Delete => panic!("Expected Value kind"),
                }
            }
            _ => panic!("Expected Always variant"),
        }
    }

    #[test]
    fn test_inference_extra_body_all_with_delete() {
        let json = r#"{"pointer": "/test", "delete": true}"#;
        let result: InferenceExtraBody = serde_json::from_str(json).unwrap();
        match result {
            InferenceExtraBody::Always { pointer, kind } => {
                assert_eq!(pointer, "/test");
                assert_eq!(kind, ExtraBodyReplacementKind::Delete);
            }
            _ => panic!("Expected Always variant"),
        }
    }

    #[test]
    fn test_should_apply_variant_all() {
        let all_variant = InferenceExtraBody::Always {
            pointer: "/test".to_string(),
            kind: ExtraBodyReplacementKind::Value(json!({"key": "value"})),
        };

        // Always should apply to any variant
        assert!(all_variant.should_apply_variant("variant1"));
        assert!(all_variant.should_apply_variant("variant2"));
        assert!(all_variant.should_apply_variant("any_variant"));
    }

    #[test]
    fn test_should_apply_variant_provider() {
        let provider_variant = InferenceExtraBody::Provider {
            model_provider_name: "openai".to_string(),
            pointer: "/test".to_string(),
            kind: ExtraBodyReplacementKind::Value(json!(1)),
        };

        // Provider should apply to any variant
        assert!(provider_variant.should_apply_variant("variant1"));
        assert!(provider_variant.should_apply_variant("variant2"));
    }

    #[test]
    fn test_should_apply_variant_variant_match() {
        let variant = InferenceExtraBody::Variant {
            variant_name: "variant1".to_string(),
            pointer: "/test".to_string(),
            kind: ExtraBodyReplacementKind::Value(json!(1)),
        };

        // Should apply to matching variant
        assert!(variant.should_apply_variant("variant1"));
    }

    #[test]
    fn test_should_apply_variant_variant_no_match() {
        let variant = InferenceExtraBody::Variant {
            variant_name: "variant1".to_string(),
            pointer: "/test".to_string(),
            kind: ExtraBodyReplacementKind::Value(json!(1)),
        };

        // Should NOT apply to non-matching variant
        assert!(!variant.should_apply_variant("variant2"));
    }

    #[test]
    fn test_filter_includes_all_variant() {
        let unfiltered = UnfilteredInferenceExtraBody {
            extra_body: vec![
                InferenceExtraBody::Variant {
                    variant_name: "variant1".to_string(),
                    pointer: "/v1".to_string(),
                    kind: ExtraBodyReplacementKind::Value(json!(1)),
                },
                InferenceExtraBody::Always {
                    pointer: "/all".to_string(),
                    kind: ExtraBodyReplacementKind::Value(json!(2)),
                },
                InferenceExtraBody::Variant {
                    variant_name: "variant2".to_string(),
                    pointer: "/v2".to_string(),
                    kind: ExtraBodyReplacementKind::Value(json!(3)),
                },
            ],
        };

        let filtered = unfiltered.filter("variant1");
        assert_eq!(filtered.data.len(), 2); // variant1 + All

        // Verify All is included
        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, InferenceExtraBody::Always { .. })));
    }

    #[test]
    fn test_filter_mixed_variants() {
        let unfiltered = UnfilteredInferenceExtraBody {
            extra_body: vec![
                InferenceExtraBody::Provider {
                    model_provider_name: "openai".to_string(),
                    pointer: "/provider".to_string(),
                    kind: ExtraBodyReplacementKind::Value(json!("provider")),
                },
                InferenceExtraBody::Variant {
                    variant_name: "variant1".to_string(),
                    pointer: "/v1".to_string(),
                    kind: ExtraBodyReplacementKind::Value(json!("v1")),
                },
                InferenceExtraBody::Always {
                    pointer: "/all".to_string(),
                    kind: ExtraBodyReplacementKind::Value(json!("all")),
                },
                InferenceExtraBody::Variant {
                    variant_name: "variant2".to_string(),
                    pointer: "/v2".to_string(),
                    kind: ExtraBodyReplacementKind::Value(json!("v2")),
                },
            ],
        };

        let filtered = unfiltered.filter("variant1");
        // Should include: Provider, Variant(variant1), All
        assert_eq!(filtered.data.len(), 3);

        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, InferenceExtraBody::Provider { .. })));
        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, InferenceExtraBody::Always { .. })));
        assert!(filtered.data.iter().any(|item| match item {
            InferenceExtraBody::Variant { variant_name, .. } => variant_name == "variant1",
            _ => false,
        }));
    }

    #[test]
    fn test_inference_extra_body_all_roundtrip() {
        let original = InferenceExtraBody::Always {
            pointer: "/test".to_string(),
            kind: ExtraBodyReplacementKind::Value(json!({"test": "data"})),
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: InferenceExtraBody = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_inference_extra_body_model_provider_deserialize() {
        let json = r#"{"model_name": "gpt-4o", "provider_name": "openai", "pointer": "/test", "value": {"key": "value"}}"#;
        let result: InferenceExtraBody = serde_json::from_str(json).unwrap();
        match result {
            InferenceExtraBody::ModelProvider {
                model_name,
                provider_name,
                pointer,
                kind,
            } => {
                assert_eq!(model_name, "gpt-4o");
                assert_eq!(provider_name, "openai");
                assert_eq!(pointer, "/test");
                match kind {
                    ExtraBodyReplacementKind::Value(v) => {
                        assert_eq!(v, json!({"key": "value"}));
                    }
                    ExtraBodyReplacementKind::Delete => panic!("Expected Value kind"),
                }
            }
            _ => panic!("Expected ModelProvider variant"),
        }
    }

    #[test]
    fn test_inference_extra_body_model_provider_with_delete() {
        let json = r#"{"model_name": "gpt-4o", "provider_name": "openai", "pointer": "/test", "delete": true}"#;
        let result: InferenceExtraBody = serde_json::from_str(json).unwrap();
        match result {
            InferenceExtraBody::ModelProvider {
                model_name,
                provider_name,
                pointer,
                kind,
            } => {
                assert_eq!(model_name, "gpt-4o");
                assert_eq!(provider_name, "openai");
                assert_eq!(pointer, "/test");
                assert_eq!(kind, ExtraBodyReplacementKind::Delete);
            }
            _ => panic!("Expected ModelProvider variant"),
        }
    }

    #[test]
    fn test_should_apply_variant_model_provider() {
        let model_provider_variant = InferenceExtraBody::ModelProvider {
            model_name: "gpt-4o".to_string(),
            provider_name: "openai".to_string(),
            pointer: "/test".to_string(),
            kind: ExtraBodyReplacementKind::Value(json!(1)),
        };

        // ModelProvider should apply to any variant
        assert!(model_provider_variant.should_apply_variant("variant1"));
        assert!(model_provider_variant.should_apply_variant("variant2"));
    }

    #[test]
    fn test_filter_includes_model_provider() {
        let unfiltered = UnfilteredInferenceExtraBody {
            extra_body: vec![
                InferenceExtraBody::Variant {
                    variant_name: "variant1".to_string(),
                    pointer: "/v1".to_string(),
                    kind: ExtraBodyReplacementKind::Value(json!(1)),
                },
                InferenceExtraBody::ModelProvider {
                    model_name: "gpt-4o".to_string(),
                    provider_name: "openai".to_string(),
                    pointer: "/mp".to_string(),
                    kind: ExtraBodyReplacementKind::Value(json!(2)),
                },
                InferenceExtraBody::Variant {
                    variant_name: "variant2".to_string(),
                    pointer: "/v2".to_string(),
                    kind: ExtraBodyReplacementKind::Value(json!(3)),
                },
            ],
        };

        let filtered = unfiltered.filter("variant1");
        assert_eq!(filtered.data.len(), 2); // variant1 + ModelProvider

        // Verify ModelProvider is included
        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, InferenceExtraBody::ModelProvider { .. })));
    }

    #[test]
    fn test_inference_extra_body_model_provider_roundtrip() {
        let original = InferenceExtraBody::ModelProvider {
            model_name: "gpt-4o".to_string(),
            provider_name: "openai".to_string(),
            pointer: "/test".to_string(),
            kind: ExtraBodyReplacementKind::Value(json!({"test": "data"})),
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: InferenceExtraBody = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }
}
