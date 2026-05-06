use serde::{Deserialize, Serialize};

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
