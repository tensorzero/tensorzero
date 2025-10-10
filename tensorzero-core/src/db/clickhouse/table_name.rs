use enum_map::Enum;

/// Defines all of the ClickHouse tables that we write to from Rust
/// This will be used to implement per-table ClickHouse write batching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Enum)]
pub enum TableName {
    BatchModelInference,
    BatchRequest,
    ChatInference,
    ChatInferenceDatapoint,
    JsonInference,
    JsonInferenceDatapoint,
    ModelInference,
    ModelInferenceCache,
    DeploymentID,
    TensorZeroMigration,
    BooleanMetricFeedback,
    FloatMetricFeedback,
    DemonstrationFeedback,
    CommentFeedback,
    StaticEvaluationHumanFeedback,
}

impl TableName {
    pub fn as_str(self) -> &'static str {
        match self {
            TableName::BatchModelInference => "BatchModelInference",
            TableName::BatchRequest => "BatchRequest",
            TableName::ChatInference => "ChatInference",
            TableName::ChatInferenceDatapoint => "ChatInferenceDatapoint",
            TableName::JsonInference => "JsonInference",
            TableName::JsonInferenceDatapoint => "JsonInferenceDatapoint",
            TableName::ModelInference => "ModelInference",
            TableName::ModelInferenceCache => "ModelInferenceCache",
            TableName::DeploymentID => "DeploymentID",
            TableName::TensorZeroMigration => "TensorZeroMigration",
            TableName::BooleanMetricFeedback => "BooleanMetricFeedback",
            TableName::FloatMetricFeedback => "FloatMetricFeedback",
            TableName::DemonstrationFeedback => "DemonstrationFeedback",
            TableName::CommentFeedback => "CommentFeedback",
            TableName::StaticEvaluationHumanFeedback => "StaticEvaluationHumanFeedback",
        }
    }
}
