use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_core::config::UninitializedVariantInfo;
use tensorzero_core::evaluations::{UninitializedEvaluationConfig, UninitializedEvaluatorConfig};
use tensorzero_core::experimentation::UninitializedExperimentationConfig;
use tensorzero_derive::TensorZeroDeserialize;

/// Represents a targeted edit operation to apply to a TensorZero config.
#[derive(ts_rs::TS, Clone, Debug, Serialize, TensorZeroDeserialize, JsonSchema)]
#[ts(export)]
#[serde(tag = "operation")]
#[serde(rename_all = "snake_case")]
pub enum EditPayload {
    UpsertVariant(Box<UpsertVariantPayload>),
    UpsertExperimentation(UpsertExperimentationPayload),
    UpsertEvaluation(UpsertEvaluationPayload),
    UpsertEvaluator(UpsertEvaluatorPayload),
}

#[derive(ts_rs::TS, Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[ts(export)]
pub struct UpsertVariantPayload {
    pub function_name: String,
    pub variant_name: String,
    pub variant: UninitializedVariantInfo,
}

#[derive(ts_rs::TS, Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[ts(export)]
pub struct UpsertExperimentationPayload {
    pub function_name: String,
    pub experimentation: UninitializedExperimentationConfig,
}

#[derive(ts_rs::TS, Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[ts(export)]
pub struct UpsertEvaluationPayload {
    pub evaluation_name: String,
    pub evaluation: UninitializedEvaluationConfig,
}

#[derive(ts_rs::TS)]
#[ts(export)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct UpsertEvaluatorPayload {
    pub evaluation_name: String,
    pub evaluator_name: String,
    pub evaluator: UninitializedEvaluatorConfig,
}
