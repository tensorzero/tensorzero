use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_core::config::UninitializedVariantInfo;
use tensorzero_core::evaluations::{UninitializedEvaluationConfig, UninitializedEvaluatorConfig};
use tensorzero_core::experimentation::UninitializedExperimentationConfig;

/// Represents a targeted edit operation to apply to a TensorZero config.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum EditPayload {
    UpsertVariant(Box<UpsertVariantPayload>),
    UpsertExperimentation(UpsertExperimentationPayload),
    UpsertEvaluation(UpsertEvaluationPayload),
    UpsertEvaluator(UpsertEvaluatorPayload),
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct UpsertVariantPayload {
    pub function_name: String,
    pub variant_name: String,
    pub variant: UninitializedVariantInfo,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct UpsertExperimentationPayload {
    pub function_name: String,
    pub experimentation: UninitializedExperimentationConfig,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct UpsertEvaluationPayload {
    pub evaluation_name: String,
    pub evaluation: UninitializedEvaluationConfig,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct UpsertEvaluatorPayload {
    pub evaluation_name: String,
    pub evaluator_name: String,
    pub evaluator: UninitializedEvaluatorConfig,
}

impl EditPayload {
    /// Returns a description of the edit operation for logging/debugging.
    pub fn describe(&self) -> String {
        match self {
            EditPayload::UpsertVariant(payload) => {
                format!(
                    "upsert variant `{}` in function `{}`",
                    payload.variant_name, payload.function_name
                )
            }
            EditPayload::UpsertExperimentation(payload) => {
                format!(
                    "upsert experimentation config in function `{}`",
                    payload.function_name
                )
            }
            EditPayload::UpsertEvaluation(payload) => {
                format!("upsert evaluation `{}`", payload.evaluation_name)
            }
            EditPayload::UpsertEvaluator(payload) => {
                format!(
                    "upsert evaluator `{}` in evaluation `{}`",
                    payload.evaluator_name, payload.evaluation_name
                )
            }
        }
    }
}
