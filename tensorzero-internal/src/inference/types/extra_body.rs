use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ExtraHeadersConfig {
    pub data: Vec<ExtraHeader>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ExtraHeader {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ExtraBodyConfig {
    pub data: Vec<ExtraBodyReplacement>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ExtraBodyReplacement {
    pub pointer: String,
    pub value: Value,
}

/// The 'InferenceExtraBody' options provided directly in an inference request
/// These have not yet been filtered by variant name
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct UnfilteredInferenceExtraBody {
    extra_body: Vec<InferenceExtraBody>,
}

impl UnfilteredInferenceExtraBody {
    pub fn is_empty(&self) -> bool {
        self.extra_body.is_empty()
    }
    /// Filter the 'InferenceExtraBody' options by variant name
    /// If the variant name is `None`, then all variant-specific extra body options are removed
    pub fn filter(self, variant_name: Option<&str>) -> FilteredInferenceExtraBody {
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
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct FilteredInferenceExtraBody {
    pub data: Vec<InferenceExtraBody>,
}

/// Holds the config-level and inference-level extra body options
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FullExtraBodyConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant_extra_headers: Option<ExtraHeadersConfig>,
    pub extra_body: Option<ExtraBodyConfig>,
    pub inference_extra_body: FilteredInferenceExtraBody,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum InferenceExtraBody {
    Provider {
        model_provider_name: String,
        pointer: String,
        value: serde_json::Value,
    },
    Variant {
        variant_name: String,
        pointer: String,
        value: serde_json::Value,
    },
}

impl InferenceExtraBody {
    pub fn should_apply_variant(&self, variant_name: Option<&str>) -> bool {
        match (self, variant_name) {
            (InferenceExtraBody::Provider { .. }, _) => true,
            (
                InferenceExtraBody::Variant {
                    variant_name: v, ..
                },
                Some(expected_name),
            ) => v == expected_name,
            (InferenceExtraBody::Variant { .. }, None) => false,
        }
    }
}
