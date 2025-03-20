use serde::{Deserialize, Serialize};

use super::InferenceExtraBody;

/// The 'InferenceExtraBody' options provided directly in an inference request
/// These have not yet been filtered by variant name
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(transparent)]
pub struct UnfilteredInferenceExtraBody {
    extra_body: Vec<InferenceExtraBody>,
}

impl UnfilteredInferenceExtraBody {
    /// Filter the 'InferenceExtraBody' options by variant name
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
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(transparent)]
pub struct FilteredInferenceExtraBody {
    pub data: Vec<InferenceExtraBody>,
}
