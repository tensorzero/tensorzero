use serde::{Deserialize, Serialize};

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

/// The 'InferenceExtraHeaders' options provided directly in an inference request
/// These have not yet been filtered by variant name
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct UnfilteredInferenceExtraHeaders {
    pub headers: Vec<InferenceExtraHeader>,
}

impl UnfilteredInferenceExtraHeaders {
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }

    /// Filter the 'InferenceExtraHeader' options by variant name
    /// If the variant name is `None`, then all variant-specific extra header options are removed
    pub fn filter(self, variant_name: Option<&str>) -> FilteredInferenceExtraHeaders {
        FilteredInferenceExtraHeaders {
            data: self
                .headers
                .into_iter()
                .filter(|config| config.should_apply_variant(variant_name))
                .collect(),
        }
    }
}

/// The result of filtering `InferenceExtraHeader` by variant name.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct FilteredInferenceExtraHeaders {
    pub data: Vec<InferenceExtraHeader>,
}

/// Holds the config-level and inference-level extra headers options
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FullExtraHeadersConfig {
    pub variant_extra_headers: Option<ExtraHeadersConfig>,
    pub inference_extra_headers: FilteredInferenceExtraHeaders,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum InferenceExtraHeader {
    Provider {
        model_provider_name: String,
        name: String,
        value: String,
    },
    Variant {
        variant_name: String,
        name: String,
        value: String,
    },
}

impl InferenceExtraHeader {
    pub fn should_apply_variant(&self, variant_name: Option<&str>) -> bool {
        match (self, variant_name) {
            (InferenceExtraHeader::Provider { .. }, _) => true,
            (
                InferenceExtraHeader::Variant {
                    variant_name: v, ..
                },
                Some(expected_name),
            ) => v == expected_name,
            (InferenceExtraHeader::Variant { .. }, None) => false,
        }
    }
}
