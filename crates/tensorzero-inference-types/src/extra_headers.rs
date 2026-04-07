use crate::serde_helpers::{deserialize_delete, serialize_delete};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ExtraHeadersConfig {
    pub data: Vec<ExtraHeader>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct ExtraHeader {
    pub name: String,
    #[serde(flatten)]
    pub kind: ExtraHeaderKind,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
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
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct FilteredInferenceExtraHeaders {
    pub data: Vec<DynamicExtraHeader>,
}

/// Holds the config-level and inference-level extra headers options
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FullExtraHeadersConfig {
    pub variant_extra_headers: Option<ExtraHeadersConfig>,
    pub inference_extra_headers: FilteredInferenceExtraHeaders,
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
    pub enum ExtraHeader {
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
                serialize_with = "super::super::serde_helpers::serialize_delete_field",
                deserialize_with = "super::super::serde_helpers::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::serde_helpers::schema_for_delete_field")]
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
                serialize_with = "super::super::serde_helpers::serialize_delete_field",
                deserialize_with = "super::super::serde_helpers::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::serde_helpers::schema_for_delete_field")]
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
                serialize_with = "super::super::serde_helpers::serialize_delete_field",
                deserialize_with = "super::super::serde_helpers::deserialize_delete_field"
            )]
            #[schemars(schema_with = "super::super::serde_helpers::schema_for_delete_field")]
            /// Set to true to remove the header from the model provider request
            delete: (),
        },
    }

    impl ExtraHeader {
        pub fn should_apply_variant(&self, variant_name: &str) -> bool {
            match self {
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
