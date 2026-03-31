use crate::inference::types::extra_headers::{
    DynamicExtraHeader, ExtraHeaderKind, FullExtraHeadersConfig, UnfilteredInferenceExtraHeaders,
};

use super::{deserialize_delete, serialize_delete};
use crate::inference::types::extra_body::dynamic::ExtraBody;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_derive::export_schema;
use tensorzero_stored_config::{
    StoredExtraBodyConfig, StoredExtraBodyReplacement, StoredExtraBodyReplacementKind,
};

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
                DynamicExtraHeader::ModelProvider { .. }
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
#[derive(Clone, Debug, Default, JsonSchema, PartialEq, Serialize)]
#[serde(transparent)]
pub struct UnfilteredInferenceExtraBody {
    extra_body: Vec<DynamicExtraBody>,
}

/// Custom `Deserialize` for `UnfilteredInferenceExtraBody` that handles the legacy
/// `Provider`/`ProviderDelete` format stored in the database. These legacy rows contain
/// a `model_provider_name` field which no longer exists in `DynamicExtraBody`.
/// We map them to `Always`/`AlwaysDelete` (dropping the provider filter).
impl<'de> Deserialize<'de> for UnfilteredInferenceExtraBody {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw_values: Vec<serde_json::Value> = Vec::deserialize(deserializer)?;
        let mut extra_body = Vec::with_capacity(raw_values.len());

        for value in raw_values {
            match serde_json::from_value::<DynamicExtraBody>(value.clone()) {
                Ok(parsed) => extra_body.push(parsed),
                Err(original_err) => {
                    // Check if this is a legacy format with `model_provider_name`
                    let Some(obj) = value.as_object() else {
                        return Err(serde::de::Error::custom(original_err));
                    };
                    if !obj.contains_key("model_provider_name") {
                        return Err(serde::de::Error::custom(original_err));
                    }

                    let Some(pointer) = obj.get("pointer").and_then(|v| v.as_str()) else {
                        return Err(serde::de::Error::custom(
                            "legacy `extra_body` entry with `model_provider_name` is missing `pointer`",
                        ));
                    };
                    let pointer = pointer.to_string();
                    let model_provider_name = obj
                        .get("model_provider_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("<unknown>");

                    if let Some(val) = obj.get("value") {
                        tracing::warn!(
                            model_provider_name,
                            pointer,
                            "Dropping `model_provider_name` filter from legacy `extra_body` entry — applying unconditionally",
                        );
                        extra_body.push(DynamicExtraBody::Always {
                            pointer,
                            value: val.clone(),
                        });
                    } else if obj.get("delete").and_then(|v| v.as_bool()) == Some(true) {
                        tracing::warn!(
                            model_provider_name,
                            pointer,
                            "Dropping `model_provider_name` filter from legacy `extra_body` delete entry — applying unconditionally",
                        );
                        extra_body.push(DynamicExtraBody::AlwaysDelete {
                            pointer,
                            delete: (),
                        });
                    } else {
                        return Err(serde::de::Error::custom(
                            "legacy `extra_body` entry with `model_provider_name` has neither `value` nor `delete: true`",
                        ));
                    }
                }
            }
        }

        Ok(UnfilteredInferenceExtraBody { extra_body })
    }
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
/// been removed, while all other options have been retained.
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

// ─── Stored → Uninitialized conversions ──────────────────────────────────────

impl From<StoredExtraBodyReplacementKind> for ExtraBodyReplacementKind {
    fn from(stored: StoredExtraBodyReplacementKind) -> Self {
        match stored {
            StoredExtraBodyReplacementKind::Value(v) => ExtraBodyReplacementKind::Value(v),
            StoredExtraBodyReplacementKind::Delete => ExtraBodyReplacementKind::Delete,
        }
    }
}

impl From<StoredExtraBodyReplacement> for ExtraBodyReplacement {
    fn from(stored: StoredExtraBodyReplacement) -> Self {
        ExtraBodyReplacement {
            pointer: stored.pointer,
            kind: stored.kind.into(),
        }
    }
}

impl From<StoredExtraBodyConfig> for ExtraBodyConfig {
    fn from(stored: StoredExtraBodyConfig) -> Self {
        ExtraBodyConfig {
            data: stored.data.into_iter().map(Into::into).collect(),
        }
    }
}

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
                DynamicExtraBody::ModelProvider {
                    model_name: "gpt-4o".to_string(),
                    provider_name: Some("openai".to_string()),
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
        // Should include: ModelProvider, Variant(variant1), Always
        assert_eq!(filtered.data.len(), 3);

        assert!(
            filtered
                .data
                .iter()
                .any(|item| matches!(item, DynamicExtraBody::ModelProvider { .. }))
        );
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

    #[test]
    fn test_unfiltered_extra_body_legacy_provider_with_value() {
        let json = r#"[{"model_provider_name": "tensorzero::model_name::X::provider_name::Y", "pointer": "/field", "value": "hello"}]"#;
        let result: UnfilteredInferenceExtraBody = serde_json::from_str(json).unwrap();
        assert_eq!(result.extra_body.len(), 1);
        assert_eq!(
            result.extra_body[0],
            DynamicExtraBody::Always {
                pointer: "/field".to_string(),
                value: json!("hello"),
            }
        );
    }

    #[test]
    fn test_unfiltered_extra_body_legacy_provider_with_delete() {
        let json = r#"[{"model_provider_name": "tensorzero::model_name::X::provider_name::Y", "pointer": "/field", "delete": true}]"#;
        let result: UnfilteredInferenceExtraBody = serde_json::from_str(json).unwrap();
        assert_eq!(result.extra_body.len(), 1);
        assert_eq!(
            result.extra_body[0],
            DynamicExtraBody::AlwaysDelete {
                pointer: "/field".to_string(),
                delete: (),
            }
        );
    }

    #[test]
    fn test_unfiltered_extra_body_mixed_legacy_and_current() {
        let json = r#"[
            {"model_provider_name": "tensorzero::model_name::X::provider_name::Y", "pointer": "/legacy", "value": 42},
            {"pointer": "/current", "value": "new_format"},
            {"model_provider_name": "tensorzero::model_name::A::provider_name::B", "pointer": "/legacy_delete", "delete": true},
            {"variant_name": "my_variant", "pointer": "/variant", "value": true}
        ]"#;
        let result: UnfilteredInferenceExtraBody = serde_json::from_str(json).unwrap();
        assert_eq!(result.extra_body.len(), 4);
        assert_eq!(
            result.extra_body[0],
            DynamicExtraBody::Always {
                pointer: "/legacy".to_string(),
                value: json!(42),
            }
        );
        assert_eq!(
            result.extra_body[1],
            DynamicExtraBody::Always {
                pointer: "/current".to_string(),
                value: json!("new_format"),
            }
        );
        assert_eq!(
            result.extra_body[2],
            DynamicExtraBody::AlwaysDelete {
                pointer: "/legacy_delete".to_string(),
                delete: (),
            }
        );
        assert_eq!(
            result.extra_body[3],
            DynamicExtraBody::Variant {
                variant_name: "my_variant".to_string(),
                pointer: "/variant".to_string(),
                value: json!(true),
            }
        );
    }

    #[test]
    fn test_unfiltered_extra_body_current_format_regression() {
        let json = r#"[
            {"pointer": "/test", "value": {"key": "value"}},
            {"pointer": "/delete_me", "delete": true},
            {"model_name": "gpt-4o", "provider_name": "openai", "pointer": "/mp", "value": 1}
        ]"#;
        let result: UnfilteredInferenceExtraBody = serde_json::from_str(json).unwrap();
        assert_eq!(result.extra_body.len(), 3);
        assert_eq!(
            result.extra_body[0],
            DynamicExtraBody::Always {
                pointer: "/test".to_string(),
                value: json!({"key": "value"}),
            }
        );
        assert_eq!(
            result.extra_body[1],
            DynamicExtraBody::AlwaysDelete {
                pointer: "/delete_me".to_string(),
                delete: (),
            }
        );
        assert_eq!(
            result.extra_body[2],
            DynamicExtraBody::ModelProvider {
                model_name: "gpt-4o".to_string(),
                provider_name: Some("openai".to_string()),
                pointer: "/mp".to_string(),
                value: json!(1),
            }
        );
    }

    #[test]
    fn test_unfiltered_extra_body_invalid_input_errors() {
        let json = r#"[{"unknown_field": "value"}]"#;
        let result: Result<UnfilteredInferenceExtraBody, _> = serde_json::from_str(json);
        assert!(result.is_err(), "Expected error for invalid input");
    }

    #[test]
    fn test_unfiltered_extra_body_empty_array() {
        let json = r"[]";
        let result: UnfilteredInferenceExtraBody = serde_json::from_str(json).unwrap();
        assert!(result.extra_body.is_empty());
    }
}
