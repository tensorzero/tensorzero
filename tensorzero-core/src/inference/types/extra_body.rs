use crate::inference::types::extra_headers::{
    DynamicExtraHeader, ExtraHeaderKind, FullExtraHeadersConfig, UnfilteredInferenceExtraHeaders,
};

use super::{deserialize_delete, serialize_delete};
use crate::inference::types::extra_body::dynamic::ExtraBody;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_derive::export_schema;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ExtraBodyConfig {
    pub data: Vec<ExtraBodyReplacement>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct ExtraBodyReplacement {
    pub pointer: String,
    #[serde(flatten)]
    pub kind: ExtraBodyReplacementKind,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[export_schema]
#[serde(rename_all = "snake_case")]
pub enum ExtraBodyReplacementKind {
    #[schemars(title = "ExtraBodyReplacementKindValue")]
    Value(Value),
    // We only allow `"delete": true` to be set - deserializing `"delete": false` will error
    #[serde(
        serialize_with = "serialize_delete",
        deserialize_with = "deserialize_delete"
    )]
    Delete,
}

/// In relay mode, we perform special handling of extra_body options:
/// * Variant-level filtering is applied on the relay gateway
/// * All of the extra_body options are forwarded to the downstream gateway,
/// * We skip validation of model/provider filters on the relay gateway
///   (see `validate_inference_filters`), since the downstream gateway
///   is where they actually get applied. We don't want to require creating
///   fake models/providers on the relay gateway when they're never actually
///   going to get invoked on the relay
pub fn prepare_relay_extra_body(extra_body: &FullExtraBodyConfig) -> UnfilteredInferenceExtraBody {
    let FullExtraBodyConfig {
        extra_body,
        inference_extra_body,
    } = extra_body;

    // Forward any static extra_body options directly to the downstream gateway,
    // which is what actually applies them when the model gets invoked
    let mut new_extra_body = extra_body
        .as_ref()
        .map(|b| {
            b.data
                .iter()
                .map(|replacement| match &replacement.kind {
                    ExtraBodyReplacementKind::Value(value) => ExtraBody::Always {
                        pointer: replacement.pointer.clone(),
                        value: value.clone(),
                    },
                    ExtraBodyReplacementKind::Delete => ExtraBody::AlwaysDelete {
                        pointer: replacement.pointer.clone(),
                        delete: (),
                    },
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    #[expect(deprecated)]
    new_extra_body.extend(
        inference_extra_body
            .data
            .iter()
            .map(|replacement| match &replacement {
                // We've already filtered `inference_extra_body` to apply to the variant
                // that we're invoking on the *relay* gateway
                // We want these variant-level ExtraBody replacements to get applied on
                // the *downstream* gateway, so we convert them to 'always' (since they
                // passed the variant filter on the relay gateway)
                ExtraBody::Variant {
                    variant_name: _,
                    pointer,
                    value,
                } => ExtraBody::Always {
                    pointer: pointer.clone(),
                    value: value.clone(),
                },
                ExtraBody::VariantDelete {
                    variant_name: _,
                    pointer,
                    delete: (),
                } => ExtraBody::AlwaysDelete {
                    pointer: pointer.clone(),
                    delete: (),
                },
                // We forward all other `ExtraBody` replacements as-is to the downstream gateway
                // This will allow the downstream gateway to apply model/provider filtering,
                // since the models are actually invoked on the downstream gateway
                ExtraBody::ModelProvider { .. }
                | ExtraBody::ModelProviderDelete { .. }
                | ExtraBody::Provider { .. }
                | ExtraBody::ProviderDelete { .. }
                | ExtraBody::Always { .. }
                | ExtraBody::AlwaysDelete { .. } => replacement.clone(),
            }),
    );
    UnfilteredInferenceExtraBody {
        extra_body: new_extra_body,
    }
}

/// See `prepare_relay_extra_body` for more details - the logic is virtually identical
pub fn prepare_relay_extra_headers(
    extra_headers: &FullExtraHeadersConfig,
) -> UnfilteredInferenceExtraHeaders {
    let FullExtraHeadersConfig {
        variant_extra_headers,
        inference_extra_headers,
    } = extra_headers;

    let mut new_extra_headers = variant_extra_headers
        .as_ref()
        .map(|b| {
            b.data
                .iter()
                .map(|header| match &header.kind {
                    ExtraHeaderKind::Value(value) => DynamicExtraHeader::Always {
                        name: header.name.clone(),
                        value: value.clone(),
                    },
                    ExtraHeaderKind::Delete => DynamicExtraHeader::AlwaysDelete {
                        name: header.name.clone(),
                        delete: (),
                    },
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    // The variant/model/model-provider handling for extra_header is identical to that of extra_body
    // See `prepare_relay_extra_body` for more details
    #[expect(deprecated)]
    new_extra_headers.extend(
        inference_extra_headers
            .data
            .iter()
            .map(|header| match &header {
                DynamicExtraHeader::Variant {
                    variant_name: _,
                    name,
                    value,
                } => DynamicExtraHeader::Always {
                    name: name.clone(),
                    value: value.clone(),
                },
                DynamicExtraHeader::VariantDelete {
                    variant_name: _,
                    name,
                    delete: (),
                } => DynamicExtraHeader::AlwaysDelete {
                    name: name.clone(),
                    delete: (),
                },
                DynamicExtraHeader::Provider { .. }
                | DynamicExtraHeader::ProviderDelete { .. }
                | DynamicExtraHeader::ModelProvider { .. }
                | DynamicExtraHeader::ModelProviderDelete { .. }
                | DynamicExtraHeader::Always { .. }
                | DynamicExtraHeader::AlwaysDelete { .. } => header.clone(),
            }),
    );
    UnfilteredInferenceExtraHeaders {
        extra_headers: new_extra_headers,
    }
}

/// The 'InferenceExtraBody' options provided directly in an inference request.
/// These have not yet been filtered by variant name
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
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
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct FilteredInferenceExtraBody {
    pub data: Vec<DynamicExtraBody>,
}

/// Holds the config-level and inference-level extra body options
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FullExtraBodyConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<ExtraBodyConfig>,
    pub inference_extra_body: FilteredInferenceExtraBody,
}

pub mod dynamic {
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use tensorzero_derive::export_schema;

    #[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
    #[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
    #[cfg_attr(feature = "ts-bindings", ts(optional_fields))]
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
    use crate::inference::types::extra_headers::{
        ExtraHeader, ExtraHeadersConfig, FilteredInferenceExtraHeaders,
    };

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
        assert!(
            filtered
                .data
                .iter()
                .any(|item| matches!(item, DynamicExtraBody::Always { .. }))
        );
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
            assert!(
                filtered
                    .data
                    .iter()
                    .any(|item| matches!(item, DynamicExtraBody::Provider { .. }))
            );
        }
        assert!(
            filtered
                .data
                .iter()
                .any(|item| matches!(item, DynamicExtraBody::Always { .. }))
        );
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
        assert!(
            filtered
                .data
                .iter()
                .any(|item| matches!(item, DynamicExtraBody::ModelProvider { .. }))
        );
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

    #[test]
    fn test_prepare_relay_extra_body() {
        let extra_body = FullExtraBodyConfig {
            extra_body: Some(ExtraBodyConfig {
                data: vec![
                    ExtraBodyReplacement {
                        pointer: "/add_me".to_string(),
                        kind: ExtraBodyReplacementKind::Value(json!("my_value")),
                    },
                    ExtraBodyReplacement {
                        pointer: "/delete_me".to_string(),
                        kind: ExtraBodyReplacementKind::Delete,
                    },
                ],
            }),
            inference_extra_body: FilteredInferenceExtraBody {
                data: vec![
                    DynamicExtraBody::Variant {
                        variant_name: "first_variant".to_string(),
                        pointer: "/v1".to_string(),
                        value: json!(1),
                    },
                    DynamicExtraBody::VariantDelete {
                        variant_name: "second_variant".to_string(),
                        pointer: "/v2".to_string(),
                        delete: (),
                    },
                    DynamicExtraBody::ModelProvider {
                        model_name: "gpt-4o".to_string(),
                        provider_name: Some("openai".to_string()),
                        pointer: "/mp".to_string(),
                        value: json!(2),
                    },
                ],
            },
        };

        let prepared = prepare_relay_extra_body(&extra_body);
        assert_eq!(
            prepared,
            UnfilteredInferenceExtraBody {
                extra_body: vec![
                    DynamicExtraBody::Always {
                        pointer: "/add_me".to_string(),
                        value: json!("my_value"),
                    },
                    DynamicExtraBody::AlwaysDelete {
                        pointer: "/delete_me".to_string(),
                        delete: (),
                    },
                    DynamicExtraBody::Always {
                        pointer: "/v1".to_string(),
                        value: json!(1),
                    },
                    DynamicExtraBody::AlwaysDelete {
                        pointer: "/v2".to_string(),
                        delete: (),
                    },
                    DynamicExtraBody::ModelProvider {
                        model_name: "gpt-4o".to_string(),
                        provider_name: Some("openai".to_string()),
                        pointer: "/mp".to_string(),
                        value: json!(2),
                    },
                ]
            }
        );
    }

    #[test]
    fn test_prepare_relay_extra_headers() {
        let extra_headers = FullExtraHeadersConfig {
            variant_extra_headers: Some(ExtraHeadersConfig {
                data: vec![
                    ExtraHeader {
                        name: "add_me".to_string(),
                        kind: ExtraHeaderKind::Value("my_value".to_string()),
                    },
                    ExtraHeader {
                        name: "delete_me".to_string(),
                        kind: ExtraHeaderKind::Delete,
                    },
                ],
            }),
            inference_extra_headers: FilteredInferenceExtraHeaders {
                data: vec![
                    DynamicExtraHeader::Variant {
                        variant_name: "first_variant".to_string(),
                        name: "v1".to_string(),
                        value: "v1_value".to_string(),
                    },
                    DynamicExtraHeader::VariantDelete {
                        variant_name: "second_variant".to_string(),
                        name: "v2".to_string(),
                        delete: (),
                    },
                    DynamicExtraHeader::ModelProvider {
                        model_name: "gpt-4o".to_string(),
                        provider_name: Some("openai".to_string()),
                        name: "mp".to_string(),
                        value: "mp_value".to_string(),
                    },
                ],
            },
        };

        let prepared = prepare_relay_extra_headers(&extra_headers);
        assert_eq!(
            prepared,
            UnfilteredInferenceExtraHeaders {
                extra_headers: vec![
                    DynamicExtraHeader::Always {
                        name: "add_me".to_string(),
                        value: "my_value".to_string(),
                    },
                    DynamicExtraHeader::AlwaysDelete {
                        name: "delete_me".to_string(),
                        delete: (),
                    },
                    DynamicExtraHeader::Always {
                        name: "v1".to_string(),
                        value: "v1_value".to_string(),
                    },
                    DynamicExtraHeader::AlwaysDelete {
                        name: "v2".to_string(),
                        delete: (),
                    },
                    DynamicExtraHeader::ModelProvider {
                        model_name: "gpt-4o".to_string(),
                        provider_name: Some("openai".to_string()),
                        name: "mp".to_string(),
                        value: "mp_value".to_string(),
                    },
                ],
            }
        );
    }
}
