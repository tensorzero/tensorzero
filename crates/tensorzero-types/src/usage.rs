use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The type of API used for a model inference.
/// Used in raw usage reporting to help consumers interpret provider-specific usage data.
#[derive(ts_rs::TS, Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum ApiType {
    ChatCompletions,
    Responses,
    Embeddings,
    Other,
}

/// A single entry in the raw response array, representing raw response data from one model inference.
/// This preserves the original provider-specific response string that TensorZero normalizes.
#[derive(ts_rs::TS, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export, optional_fields)]
pub struct RawResponseEntry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_inference_id: Option<Uuid>,
    pub provider_type: String,
    pub api_type: ApiType,
    pub data: String,
}
