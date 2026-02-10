use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Error;
use crate::inference::types::FunctionType;
use tensorzero_derive::TensorZeroDeserialize;

/// A single resolved object type for a given UUID.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub enum ResolvedObject {
    Inference {
        function_name: String,
        function_type: FunctionType,
        variant_name: String,
        episode_id: Uuid,
    },
    Episode,
    BooleanFeedback,
    FloatFeedback,
    CommentFeedback,
    DemonstrationFeedback,
    ChatDatapoint {
        dataset_name: String,
        function_name: String,
    },
    JsonDatapoint {
        dataset_name: String,
        function_name: String,
    },
}

/// Response type for the resolve_uuid endpoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ResolveUuidResponse {
    pub id: Uuid,
    pub object_types: Vec<ResolvedObject>,
}

/// Trait for resolving a UUID to its object type(s).
#[async_trait]
pub trait ResolveUuidQueries {
    async fn resolve_uuid(&self, id: &Uuid) -> Result<Vec<ResolvedObject>, Error>;
}
