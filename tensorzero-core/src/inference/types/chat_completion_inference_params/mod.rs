//! TODO (GabrielBianconi):
//! We are migrating the inference parameters to a struct that must be explicitly handled by every model provider.
//! To avoid a massive PR, I'll start with a small struct as an extension, and gradually migrate the rest of the parameters.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct ChatCompletionInferenceParamsV2 {
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<i32>,
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
}

pub fn warn_inference_parameter_not_supported(
    model_provider_name: &str,
    parameter_name: &str,
    suffix: Option<&str>,
) {
    let mut message = format!(
        "{model_provider_name} does not support the inference parameter `{parameter_name}`, so it will be ignored."
    );
    if let Some(suffix) = suffix {
        message.push_str(&format!(" {suffix}"));
    }
    tracing::warn!("{}", message);
}
