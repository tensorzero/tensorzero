use crate::extra_body::dynamic::ExtraBody;
use crate::extra_headers::{
    DynamicExtraHeader, ExtraHeaderKind, FullExtraHeadersConfig, UnfilteredInferenceExtraHeaders,
};
use crate::serde_helpers::{deserialize_delete, serialize_delete};
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
                serialize_with = "super::super::serde_helpers::serialize_delete_field",
                deserialize_with = "super::super::serde_helpers::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::serde_helpers::schema_for_delete_field")]
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
                serialize_with = "super::super::serde_helpers::serialize_delete_field",
                deserialize_with = "super::super::serde_helpers::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::serde_helpers::schema_for_delete_field")]
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
                serialize_with = "super::super::serde_helpers::serialize_delete_field",
                deserialize_with = "super::super::serde_helpers::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::serde_helpers::schema_for_delete_field")]
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
