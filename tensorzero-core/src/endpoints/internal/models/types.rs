//! Types for model statistics endpoints.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Response containing the count of distinct models used.
#[derive(Debug, Serialize, Deserialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
pub struct CountModelsResponse {
    /// The count of distinct models used.
    pub model_count: u32,
}
