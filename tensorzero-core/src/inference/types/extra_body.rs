use super::{deserialize_delete, serialize_delete};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_derive::export_schema;

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

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize, ts_rs::TS)]
#[export_schema]
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
    extra_body: Vec<DynamicExtraBody>,
}

impl UnfilteredInferenceExtraBody {
    pub fn is_empty(&self) -> bool {
        self.extra_body.is_empty()
    }

    /// Get a reference to the extra_body vector
    pub fn as_slice(&self) -> &[DynamicExtraBody] {
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
    pub data: Vec<DynamicExtraBody>,
}

/// Holds the config-level and inference-level extra body options
#[derive(Clone, Debug, Default, PartialEq, Serialize, ts_rs::TS)]
pub struct FullExtraBodyConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<ExtraBodyConfig>,
    pub inference_extra_body: FilteredInferenceExtraBody,
}

pub mod dynamic {
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use tensorzero_derive::export_schema;

    #[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize, ts_rs::TS)]
    #[ts(optional_fields)]
    #[export_schema]
    #[serde(untagged, deny_unknown_fields)]
    pub enum ExtraBody {
        #[schemars(title = "ProviderExtraBody")]
        #[deprecated(note = "Migrate to `ModelProvider` and remove in 2026.2+. (#4640)")]
        /// DEPRECATED: Use `ModelProvider` instead.
        Provider {
            /// A fully-qualified model provider name in your configuration (e.g. `tensorzero::model_name::my_model::provider_name::my_provider`)
            model_provider_name: String,
            /// A JSON Pointer to the field to update (e.g. `/enable_agi`)
            pointer: String,
            /// The value to set the field to
            value: serde_json::Value,
        },
        #[schemars(title = "ProviderExtraBodyDelete")]
        #[deprecated(note = "Migrate to `ModelProviderDelete` and remove in 2026.2+. (#4640)")]
        /// DEPRECATED: Use `ModelProviderDelete` instead.
        ProviderDelete {
            /// A fully-qualified model provider name in your configuration (e.g. `tensorzero::model_name::my_model::provider_name::my_provider`)
            model_provider_name: String,
            /// A JSON Pointer to the field to update (e.g. `/enable_agi`)
            pointer: String,
            #[serde(
                serialize_with = "super::super::serialize_delete_field",
                deserialize_with = "super::super::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::schema_for_delete_field")]
            /// Set to true to remove the field from the model provider request's body
            delete: (),
        },
        #[schemars(title = "VariantExtraBody")]
        Variant {
            /// A variant name in your configuration (e.g. `my_variant`)
            variant_name: String,
            /// A JSON Pointer to the field to update (e.g. `/enable_agi`)
            pointer: String,
            /// The value to set the field to
            value: serde_json::Value,
        },
        #[schemars(title = "VariantExtraBodyDelete")]
        VariantDelete {
            /// A variant name in your configuration (e.g. `my_variant`)
            variant_name: String,
            /// A JSON Pointer to the field to update (e.g. `/enable_agi`)
            pointer: String,
            #[serde(
                serialize_with = "super::super::serialize_delete_field",
                deserialize_with = "super::super::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::schema_for_delete_field")]
            /// Set to true to remove the field from the model provider request's body
            delete: (),
        },
        #[schemars(title = "ModelProviderExtraBody")]
        ModelProvider {
            /// A model name in your configuration (e.g. `my_gpt_5`) or a short-hand model name (e.g. `openai::gpt-5`)
            model_name: String,
            /// A provider name for the model you specified (e.g. `my_openai`)
            provider_name: Option<String>,
            /// A JSON Pointer to the field to update (e.g. `/enable_agi`)
            pointer: String,
            /// The value to set the field to
            value: serde_json::Value,
        },
        #[schemars(title = "ModelProviderExtraBodyDelete")]
        ModelProviderDelete {
            /// A model name in your configuration (e.g. `my_gpt_5`) or a short-hand model name (e.g. `openai::gpt-5`)
            model_name: String,
            /// A provider name for the model you specified (e.g. `my_openai`)
            provider_name: Option<String>,
            /// A JSON Pointer to the field to update (e.g. `/enable_agi`)
            pointer: String,
            #[serde(
                serialize_with = "super::super::serialize_delete_field",
                deserialize_with = "super::super::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::schema_for_delete_field")]
            /// Set to true to remove the field from the model provider request's body
            delete: (),
        },
        #[schemars(title = "AlwaysExtraBody")]
        Always {
            /// A JSON Pointer to the field to update (e.g. `/enable_agi`)
            pointer: String,
            /// The value to set the field to
            value: serde_json::Value,
        },
        #[schemars(title = "AlwaysExtraBodyDelete")]
        AlwaysDelete {
            /// A JSON Pointer to the field to update (e.g. `/enable_agi`)
            pointer: String,
            #[serde(
                serialize_with = "super::super::serialize_delete_field",
                deserialize_with = "super::super::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::schema_for_delete_field")]
            /// Set to true to remove the field from the model provider request's body
            delete: (),
        },
    }

    impl ExtraBody {
        pub fn should_apply_variant(&self, variant_name: &str) -> bool {
            match self {
                #[expect(deprecated)]
                ExtraBody::Provider { .. } | ExtraBody::ProviderDelete { .. } => true,
                ExtraBody::Variant {
                    variant_name: v, ..
                }
                | ExtraBody::VariantDelete {
                    variant_name: v, ..
                } => v == variant_name,
                ExtraBody::ModelProvider { .. } | ExtraBody::ModelProviderDelete { .. } => true,
                ExtraBody::Always { .. } | ExtraBody::AlwaysDelete { .. } => true,
            }
        }
    }
}

pub use dynamic::ExtraBody as DynamicExtraBody;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_inference_extra_body_all_deserialize() {
        let json = r#"{"pointer": "/test", "value": {"key": "value"}}"#;
        let result: DynamicExtraBody = serde_json::from_str(json).unwrap();
        match result {
            DynamicExtraBody::Always { pointer, value } => {
                assert_eq!(pointer, "/test");
                assert_eq!(value, json!({"key": "value"}));
            }
            _ => panic!("Expected Always variant"),
        }
    }

    #[test]
    fn test_inference_extra_body_all_with_delete() {
        let json = r#"{"pointer": "/test", "delete": true}"#;
        let result: DynamicExtraBody = serde_json::from_str(json).unwrap();
        match result {
            DynamicExtraBody::AlwaysDelete { pointer, .. } => {
                assert_eq!(pointer, "/test");
            }
            _ => panic!("Expected AlwaysDelete variant"),
        }
    }

    #[test]
    fn test_should_apply_variant_all() {
        let all_variant = DynamicExtraBody::Always {
            pointer: "/test".to_string(),
            value: json!({"key": "value"}),
        };

        // Always should apply to any variant
        assert!(all_variant.should_apply_variant("variant1"));
        assert!(all_variant.should_apply_variant("variant2"));
        assert!(all_variant.should_apply_variant("any_variant"));
    }

    #[test]
    fn test_should_apply_variant_provider() {
        #[expect(deprecated)]
        let provider_variant = DynamicExtraBody::Provider {
            model_provider_name: "openai".to_string(),
            pointer: "/test".to_string(),
            value: json!(1),
        };

        // Provider should apply to any variant
        assert!(provider_variant.should_apply_variant("variant1"));
        assert!(provider_variant.should_apply_variant("variant2"));
    }

    #[test]
    fn test_should_apply_variant_variant_match() {
        let variant = DynamicExtraBody::Variant {
            variant_name: "variant1".to_string(),
            pointer: "/test".to_string(),
            value: json!(1),
        };

        // Should apply to matching variant
        assert!(variant.should_apply_variant("variant1"));
    }

    #[test]
    fn test_should_apply_variant_variant_no_match() {
        let variant = DynamicExtraBody::Variant {
            variant_name: "variant1".to_string(),
            pointer: "/test".to_string(),
            value: json!(1),
        };

        // Should NOT apply to non-matching variant
        assert!(!variant.should_apply_variant("variant2"));
    }

    #[test]
    fn test_filter_includes_all_variant() {
        let unfiltered = UnfilteredInferenceExtraBody {
            extra_body: vec![
                DynamicExtraBody::Variant {
                    variant_name: "variant1".to_string(),
                    pointer: "/v1".to_string(),
                    value: json!(1),
                },
                DynamicExtraBody::Always {
                    pointer: "/all".to_string(),
                    value: json!(2),
                },
                DynamicExtraBody::Variant {
                    variant_name: "variant2".to_string(),
                    pointer: "/v2".to_string(),
                    value: json!(3),
                },
            ],
        };

        let filtered = unfiltered.filter("variant1");
        assert_eq!(filtered.data.len(), 2); // variant1 + All

        // Verify All is included
        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, DynamicExtraBody::Always { .. })));
    }

    #[test]
    fn test_filter_mixed_variants() {
        let unfiltered = UnfilteredInferenceExtraBody {
            extra_body: vec![
                #[expect(deprecated)]
                DynamicExtraBody::Provider {
                    model_provider_name: "openai".to_string(),
                    pointer: "/provider".to_string(),
                    value: json!("provider"),
                },
                DynamicExtraBody::Variant {
                    variant_name: "variant1".to_string(),
                    pointer: "/v1".to_string(),
                    value: json!("v1"),
                },
                DynamicExtraBody::Always {
                    pointer: "/all".to_string(),
                    value: json!("all"),
                },
                DynamicExtraBody::Variant {
                    variant_name: "variant2".to_string(),
                    pointer: "/v2".to_string(),
                    value: json!("v2"),
                },
            ],
        };

        let filtered = unfiltered.filter("variant1");
        // Should include: Provider, Variant(variant1), All
        assert_eq!(filtered.data.len(), 3);

        #[expect(deprecated)]
        {
            assert!(filtered
                .data
                .iter()
                .any(|item| matches!(item, DynamicExtraBody::Provider { .. })));
        }
        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, DynamicExtraBody::Always { .. })));
        assert!(filtered.data.iter().any(|item| match item {
            DynamicExtraBody::Variant { variant_name, .. } => variant_name == "variant1",
            _ => false,
        }));
    }

    #[test]
    fn test_inference_extra_body_all_roundtrip() {
        let original = DynamicExtraBody::Always {
            pointer: "/test".to_string(),
            value: json!({"test": "data"}),
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: DynamicExtraBody = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_inference_extra_body_model_provider_deserialize() {
        let json = r#"{"model_name": "gpt-4o", "provider_name": "openai", "pointer": "/test", "value": {"key": "value"}}"#;
        let result: DynamicExtraBody = serde_json::from_str(json).unwrap();
        match result {
            DynamicExtraBody::ModelProvider {
                model_name,
                provider_name,
                pointer,
                value,
            } => {
                assert_eq!(model_name, "gpt-4o");
                assert_eq!(provider_name, Some("openai".to_string()));
                assert_eq!(pointer, "/test");
                assert_eq!(value, json!({"key": "value"}));
            }
            _ => panic!("Expected ModelProvider variant"),
        }
    }

    #[test]
    fn test_inference_extra_body_model_provider_with_delete() {
        let json = r#"{"model_name": "gpt-4o", "provider_name": "openai", "pointer": "/test", "delete": true}"#;
        let result: DynamicExtraBody = serde_json::from_str(json).unwrap();
        match result {
            DynamicExtraBody::ModelProviderDelete {
                model_name,
                provider_name,
                pointer,
                ..
            } => {
                assert_eq!(model_name, "gpt-4o");
                assert_eq!(provider_name, Some("openai".to_string()));
                assert_eq!(pointer, "/test");
            }
            _ => panic!("Expected ModelProviderDelete variant"),
        }
    }

    #[test]
    fn test_should_apply_variant_model_provider() {
        let model_provider_variant = DynamicExtraBody::ModelProvider {
            model_name: "gpt-4o".to_string(),
            provider_name: Some("openai".to_string()),
            pointer: "/test".to_string(),
            value: json!(1),
        };

        // ModelProvider should apply to any variant
        assert!(model_provider_variant.should_apply_variant("variant1"));
        assert!(model_provider_variant.should_apply_variant("variant2"));
    }

    #[test]
    fn test_filter_includes_model_provider() {
        let unfiltered = UnfilteredInferenceExtraBody {
            extra_body: vec![
                DynamicExtraBody::Variant {
                    variant_name: "variant1".to_string(),
                    pointer: "/v1".to_string(),
                    value: json!(1),
                },
                DynamicExtraBody::ModelProvider {
                    model_name: "gpt-4o".to_string(),
                    provider_name: Some("openai".to_string()),
                    pointer: "/mp".to_string(),
                    value: json!(2),
                },
                DynamicExtraBody::Variant {
                    variant_name: "variant2".to_string(),
                    pointer: "/v2".to_string(),
                    value: json!(3),
                },
            ],
        };

        let filtered = unfiltered.filter("variant1");
        assert_eq!(filtered.data.len(), 2); // variant1 + ModelProvider

        // Verify ModelProvider is included
        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, DynamicExtraBody::ModelProvider { .. })));
    }

    #[test]
    fn test_inference_extra_body_model_provider_roundtrip() {
        let original = DynamicExtraBody::ModelProvider {
            model_name: "gpt-4o".to_string(),
            provider_name: Some("openai".to_string()),
            pointer: "/test".to_string(),
            value: json!({"test": "data"}),
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: DynamicExtraBody = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_inference_extra_body_rejects_partial_model_provider_missing_model() {
        // Missing model_name should be rejected, not silently become Always
        let json = r#"{"provider_name": "openai", "pointer": "/test", "value": {"key": "value"}}"#;
        let result: Result<DynamicExtraBody, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "Expected error when provider_name is present but model_name is missing"
        );
    }

    #[test]
    fn test_inference_extra_body_rejects_extra_fields() {
        // Extra fields should be rejected
        let json = r#"{"pointer": "/test", "value": {"key": "value"}, "unknown_field": "value"}"#;
        let result: Result<DynamicExtraBody, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "Expected error when unknown fields are present"
        );
    }
}
