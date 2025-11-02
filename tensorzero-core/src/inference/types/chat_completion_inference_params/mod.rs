//! TODO (GabrielBianconi):
//! We are migrating the inference parameters to a struct that must be explicitly handled by every model provider.
//! To avoid a massive PR, I'll start with a small struct as an extension, and gradually migrate the rest of the parameters.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionInferenceParamsV2 {
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
}

pub fn warn_inference_parameter_not_supported(model_provider_name: &str, parameter_name: &str) {
    tracing::warn!(
        "{} does not support the inference parameter `{}`, so it'll be ignored.",
        model_provider_name,
        parameter_name
    );
}
