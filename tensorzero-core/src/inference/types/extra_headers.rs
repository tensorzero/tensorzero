use super::{deserialize_delete, serialize_delete};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(transparent)]
pub struct ExtraHeadersConfig {
    pub data: Vec<ExtraHeader>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
pub struct ExtraHeader {
    pub name: String,
    #[serde(flatten)]
    pub kind: ExtraHeaderKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ExtraHeaderKind {
    Value(String),
    // We only allow `"delete": true` to be set - deserializing `"delete": false` will error
    #[serde(
        serialize_with = "serialize_delete",
        deserialize_with = "deserialize_delete"
    )]
    Delete,
}

/// The 'InferenceExtraHeaders' options provided directly in an inference request
/// These have not yet been filtered by variant name
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(transparent)]
pub struct UnfilteredInferenceExtraHeaders {
    pub extra_headers: Vec<DynamicExtraHeader>,
}

impl UnfilteredInferenceExtraHeaders {
    pub fn is_empty(&self) -> bool {
        self.extra_headers.is_empty()
    }

    /// Get a reference to the extra_headers vector
    pub fn as_slice(&self) -> &[DynamicExtraHeader] {
        &self.extra_headers
    }

    /// Filter the 'InferenceExtraHeader' options by variant name
    /// If the variant name is `None`, then all variant-specific extra header options are removed
    pub fn filter(self, variant_name: &str) -> FilteredInferenceExtraHeaders {
        FilteredInferenceExtraHeaders {
            data: self
                .extra_headers
                .into_iter()
                .filter(|config| config.should_apply_variant(variant_name))
                .collect(),
        }
    }
}

/// The result of filtering `InferenceExtraHeader` by variant name.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(transparent)]
pub struct FilteredInferenceExtraHeaders {
    pub data: Vec<DynamicExtraHeader>,
}

/// Holds the config-level and inference-level extra headers options
#[derive(Clone, Debug, Default, PartialEq, Serialize, ts_rs::TS)]
pub struct FullExtraHeadersConfig {
    pub variant_extra_headers: Option<ExtraHeadersConfig>,
    pub inference_extra_headers: FilteredInferenceExtraHeaders,
}

pub mod dynamic {
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use tensorzero_derive::export_schema;

    #[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize, ts_rs::TS)]
    #[ts(optional_fields)]
    #[export_schema]
    #[serde(untagged, deny_unknown_fields)]
    pub enum ExtraHeader {
        #[schemars(title = "ProviderExtraHeader")]
        #[deprecated(note = "Migrate to `ModelProvider` and remove in 2026.2+. (#4640)")]
        /// DEPRECATED: Use `ModelProvider` instead.
        Provider {
            /// A fully-qualified model provider name in your configuration (e.g. `tensorzero::model_name::my_model::provider_name::my_provider`)
            model_provider_name: String,
            /// The name of the HTTP header (e.g. `anthropic-beta`)
            name: String,
            /// The value of the HTTP header (e.g. `feature1,feature2,feature3`)
            value: String,
        },
        #[schemars(title = "ProviderExtraHeaderDelete")]
        #[deprecated(note = "Migrate to `ModelProviderDelete` and remove in 2026.2+. (#4640)")]
        /// DEPRECATED: Use `ModelProviderDelete` instead.
        ProviderDelete {
            /// A fully-qualified model provider name in your configuration (e.g. `tensorzero::model_name::my_model::provider_name::my_provider`)
            model_provider_name: String,
            /// The name of the HTTP header (e.g. `anthropic-beta`)
            name: String,
            #[serde(
                serialize_with = "super::super::serialize_delete_field",
                deserialize_with = "super::super::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::schema_for_delete_field")]
            /// Set to true to remove the header from the model provider request
            delete: (),
        },
        #[schemars(title = "VariantExtraHeader")]
        Variant {
            /// A variant name in your configuration (e.g. `my_variant`)
            variant_name: String,
            /// The name of the HTTP header (e.g. `anthropic-beta`)
            name: String,
            /// The value of the HTTP header (e.g. `feature1,feature2,feature3`)
            value: String,
        },
        #[schemars(title = "VariantExtraHeaderDelete")]
        VariantDelete {
            /// A variant name in your configuration (e.g. `my_variant`)
            variant_name: String,
            /// The name of the HTTP header (e.g. `anthropic-beta`)
            name: String,
            #[serde(
                serialize_with = "super::super::serialize_delete_field",
                deserialize_with = "super::super::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::schema_for_delete_field")]
            /// Set to true to remove the header from the model provider request
            delete: (),
        },
        #[schemars(title = "ModelProviderExtraHeader")]
        ModelProvider {
            /// A model name in your configuration (e.g. `my_gpt_5`) or a short-hand model name (e.g. `openai::gpt-5`)
            model_name: String,
            /// A provider name for the model you specified (e.g. `my_openai`).
            provider_name: Option<String>,
            /// The name of the HTTP header (e.g. `anthropic-beta`)
            name: String,
            /// The value of the HTTP header (e.g. `feature1,feature2,feature3`)
            value: String,
        },
        #[schemars(title = "ModelProviderExtraHeaderDelete")]
        ModelProviderDelete {
            /// A model name in your configuration (e.g. `my_gpt_5`) or a short-hand model name (e.g. `openai::gpt-5`)
            model_name: String,
            /// A provider name for the model you specified (e.g. `my_openai`)
            provider_name: Option<String>,
            /// The name of the HTTP header (e.g. `anthropic-beta`)
            name: String,
            #[serde(
                serialize_with = "super::super::serialize_delete_field",
                deserialize_with = "super::super::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::schema_for_delete_field")]
            /// Set to true to remove the header from the model provider request
            delete: (),
        },
        #[schemars(title = "AlwaysExtraHeader")]
        Always {
            /// The name of the HTTP header (e.g. `anthropic-beta`)
            name: String,
            /// The value of the HTTP header (e.g. `feature1,feature2,feature3`)
            value: String,
        },
        #[schemars(title = "AlwaysExtraHeaderDelete")]
        AlwaysDelete {
            /// The name of the HTTP header (e.g. `anthropic-beta`)
            name: String,
            #[serde(
                serialize_with = "super::super::serialize_delete_field",
                deserialize_with = "super::super::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::schema_for_delete_field")]
            /// Set to true to remove the header from the model provider request
            delete: (),
        },
    }

    impl ExtraHeader {
        pub fn should_apply_variant(&self, variant_name: &str) -> bool {
            match self {
                #[expect(deprecated)]
                ExtraHeader::Provider { .. } | ExtraHeader::ProviderDelete { .. } => true,
                ExtraHeader::Variant {
                    variant_name: v, ..
                }
                | ExtraHeader::VariantDelete {
                    variant_name: v, ..
                } => v == variant_name,
                ExtraHeader::ModelProvider { .. } | ExtraHeader::ModelProviderDelete { .. } => true,
                ExtraHeader::Always { .. } | ExtraHeader::AlwaysDelete { .. } => true,
            }
        }
    }
}

pub use dynamic::ExtraHeader as DynamicExtraHeader;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_extra_header_all_deserialize() {
        let json = r#"{"name": "X-Custom-Header", "value": "custom-value"}"#;
        let result: DynamicExtraHeader = serde_json::from_str(json).unwrap();
        match result {
            DynamicExtraHeader::Always { name, value } => {
                assert_eq!(name, "X-Custom-Header");
                assert_eq!(value, "custom-value");
            }
            _ => panic!("Expected Always variant"),
        }
    }

    #[test]
    fn test_inference_extra_header_all_with_delete() {
        let json = r#"{"name": "X-Custom-Header", "delete": true}"#;
        let result: DynamicExtraHeader = serde_json::from_str(json).unwrap();
        match result {
            DynamicExtraHeader::AlwaysDelete { name, .. } => {
                assert_eq!(name, "X-Custom-Header");
            }
            _ => panic!("Expected AlwaysDelete variant"),
        }
    }

    #[test]
    fn test_should_apply_variant_all() {
        let all_variant = DynamicExtraHeader::Always {
            name: "X-Custom-Header".to_string(),
            value: "value".to_string(),
        };

        // Always should apply to any variant
        assert!(all_variant.should_apply_variant("variant1"));
        assert!(all_variant.should_apply_variant("variant2"));
        assert!(all_variant.should_apply_variant("any_variant"));
    }

    #[test]
    fn test_should_apply_variant_provider() {
        #[expect(deprecated)]
        let provider_variant = DynamicExtraHeader::Provider {
            model_provider_name: "openai".to_string(),
            name: "Authorization".to_string(),
            value: "Bearer token".to_string(),
        };

        // Provider should apply to any variant
        assert!(provider_variant.should_apply_variant("variant1"));
        assert!(provider_variant.should_apply_variant("variant2"));
    }

    #[test]
    fn test_should_apply_variant_variant_match() {
        let variant = DynamicExtraHeader::Variant {
            variant_name: "variant1".to_string(),
            name: "X-Variant-Header".to_string(),
            value: "value".to_string(),
        };

        // Should apply to matching variant
        assert!(variant.should_apply_variant("variant1"));
    }

    #[test]
    fn test_should_apply_variant_variant_no_match() {
        let variant = DynamicExtraHeader::Variant {
            variant_name: "variant1".to_string(),
            name: "X-Variant-Header".to_string(),
            value: "value".to_string(),
        };

        // Should NOT apply to non-matching variant
        assert!(!variant.should_apply_variant("variant2"));
    }

    #[test]
    fn test_filter_includes_all_variant() {
        let unfiltered = UnfilteredInferenceExtraHeaders {
            extra_headers: vec![
                DynamicExtraHeader::Variant {
                    variant_name: "variant1".to_string(),
                    name: "X-V1".to_string(),
                    value: "v1".to_string(),
                },
                DynamicExtraHeader::Always {
                    name: "X-All".to_string(),
                    value: "all".to_string(),
                },
                DynamicExtraHeader::Variant {
                    variant_name: "variant2".to_string(),
                    name: "X-V2".to_string(),
                    value: "v2".to_string(),
                },
            ],
        };

        let filtered = unfiltered.filter("variant1");
        assert_eq!(filtered.data.len(), 2); // variant1 + Always

        // Verify Always is included
        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, DynamicExtraHeader::Always { .. })));
    }

    #[test]
    fn test_filter_mixed_variants() {
        let unfiltered = UnfilteredInferenceExtraHeaders {
            extra_headers: vec![
                #[expect(deprecated)]
                DynamicExtraHeader::Provider {
                    model_provider_name: "openai".to_string(),
                    name: "X-Provider".to_string(),
                    value: "provider".to_string(),
                },
                DynamicExtraHeader::Variant {
                    variant_name: "variant1".to_string(),
                    name: "X-V1".to_string(),
                    value: "v1".to_string(),
                },
                DynamicExtraHeader::Always {
                    name: "X-All".to_string(),
                    value: "all".to_string(),
                },
                DynamicExtraHeader::Variant {
                    variant_name: "variant2".to_string(),
                    name: "X-V2".to_string(),
                    value: "v2".to_string(),
                },
            ],
        };

        let filtered = unfiltered.filter("variant1");
        // Should include: Provider, Variant(variant1), Always
        assert_eq!(filtered.data.len(), 3);

        #[expect(deprecated)]
        {
            assert!(filtered
                .data
                .iter()
                .any(|item| matches!(item, DynamicExtraHeader::Provider { .. })));
        }
        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, DynamicExtraHeader::Always { .. })));
        assert!(filtered.data.iter().any(|item| match item {
            DynamicExtraHeader::Variant { variant_name, .. } => variant_name == "variant1",
            _ => false,
        }));
    }

    #[test]
    fn test_inference_extra_header_all_roundtrip() {
        let original = DynamicExtraHeader::Always {
            name: "X-Test-Header".to_string(),
            value: "test-value".to_string(),
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: DynamicExtraHeader = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_inference_extra_header_model_provider_deserialize() {
        let json = r#"{"model_name": "gpt-4o", "provider_name": "openai", "name": "X-Custom-Header", "value": "custom-value"}"#;
        let result: DynamicExtraHeader = serde_json::from_str(json).unwrap();
        match result {
            DynamicExtraHeader::ModelProvider {
                model_name,
                provider_name,
                name,
                value,
            } => {
                assert_eq!(model_name, "gpt-4o");
                assert_eq!(provider_name, Some("openai".to_string()));
                assert_eq!(name, "X-Custom-Header");
                assert_eq!(value, "custom-value");
            }
            _ => panic!("Expected ModelProvider variant"),
        }
    }

    #[test]
    fn test_inference_extra_header_model_provider_with_delete() {
        let json = r#"{"model_name": "gpt-4o", "provider_name": "openai", "name": "X-Custom-Header", "delete": true}"#;
        let result: DynamicExtraHeader = serde_json::from_str(json).unwrap();
        match result {
            DynamicExtraHeader::ModelProviderDelete {
                model_name,
                provider_name,
                name,
                ..
            } => {
                assert_eq!(model_name, "gpt-4o");
                assert_eq!(provider_name, Some("openai".to_string()));
                assert_eq!(name, "X-Custom-Header");
            }
            _ => panic!("Expected ModelProviderDelete variant"),
        }
    }

    #[test]
    fn test_should_apply_variant_model_provider() {
        let model_provider_variant = DynamicExtraHeader::ModelProvider {
            model_name: "gpt-4o".to_string(),
            provider_name: Some("openai".to_string()),
            name: "X-Custom-Header".to_string(),
            value: "value".to_string(),
        };

        // ModelProvider should apply to any variant
        assert!(model_provider_variant.should_apply_variant("variant1"));
        assert!(model_provider_variant.should_apply_variant("variant2"));
    }

    #[test]
    fn test_filter_includes_model_provider() {
        let unfiltered = UnfilteredInferenceExtraHeaders {
            extra_headers: vec![
                DynamicExtraHeader::Variant {
                    variant_name: "variant1".to_string(),
                    name: "X-V1".to_string(),
                    value: "v1".to_string(),
                },
                DynamicExtraHeader::ModelProvider {
                    model_name: "gpt-4o".to_string(),
                    provider_name: Some("openai".to_string()),
                    name: "X-MP".to_string(),
                    value: "mp".to_string(),
                },
                DynamicExtraHeader::Variant {
                    variant_name: "variant2".to_string(),
                    name: "X-V2".to_string(),
                    value: "v2".to_string(),
                },
            ],
        };

        let filtered = unfiltered.filter("variant1");
        assert_eq!(filtered.data.len(), 2); // variant1 + ModelProvider

        // Verify ModelProvider is included
        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, DynamicExtraHeader::ModelProvider { .. })));
    }

    #[test]
    fn test_inference_extra_header_model_provider_roundtrip() {
        let original = DynamicExtraHeader::ModelProvider {
            model_name: "gpt-4o".to_string(),
            provider_name: Some("openai".to_string()),
            name: "X-Test-Header".to_string(),
            value: "test-value".to_string(),
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: DynamicExtraHeader = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_inference_extra_header_rejects_partial_model_provider_missing_model() {
        // Missing model_name should be rejected, not silently become Always
        let json = r#"{"provider_name": "openai", "name": "X-Custom-Header", "value": "test"}"#;
        let result: Result<DynamicExtraHeader, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "Expected error when provider_name is present but model_name is missing"
        );
    }

    #[test]
    fn test_inference_extra_header_rejects_extra_fields() {
        // Extra fields should be rejected
        let json = r#"{"name": "X-Custom-Header", "value": "test", "unknown_field": "value"}"#;
        let result: Result<DynamicExtraHeader, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "Expected error when unknown fields are present"
        );
    }
}
