//! TODO (GabrielBianconi):
//! We are migrating the inference parameters to a struct that must be explicitly handled by every model provider.
//! To avoid a massive PR, I'll start with a small struct as an extension, and gradually migrate the rest of the parameters.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionInferenceParamsV2 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
}
