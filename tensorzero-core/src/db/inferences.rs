/// Definitions for inference-related traits and types.
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

#[cfg(test)]
use mockall::automock;

use crate::config::Config;
use crate::db::clickhouse::query_builder::{InferenceFilter, OrderBy};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{ContentBlockChatOutput, JsonInferenceOutput, StoredInput};
use crate::serde_util::{deserialize_defaulted_string, deserialize_json_string};
use crate::stored_inference::{
    StoredChatInferenceDatabase, StoredInferenceDatabase, StoredJsonInference,
};
use crate::tool::ToolCallConfigDatabaseInsert;

#[derive(Debug, Deserialize)]
pub(super) struct ClickHouseStoredChatInferenceWithDispreferredOutputs {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub timestamp: DateTime<Utc>,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub input: StoredInput,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output: Vec<ContentBlockChatOutput>,
    #[serde(default)]
    pub dispreferred_outputs: Vec<String>,
    #[serde(deserialize_with = "deserialize_defaulted_string")]
    pub tool_params: ToolCallConfigDatabaseInsert,
    pub tags: HashMap<String, String>,
}

impl TryFrom<ClickHouseStoredChatInferenceWithDispreferredOutputs> for StoredChatInferenceDatabase {
    type Error = Error;

    fn try_from(
        value: ClickHouseStoredChatInferenceWithDispreferredOutputs,
    ) -> Result<Self, Self::Error> {
        let dispreferred_outputs = value
            .dispreferred_outputs
            .into_iter()
            .map(|dispreferred_output| {
                serde_json::from_str(&dispreferred_output).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to deserialize dispreferred output: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<Vec<ContentBlockChatOutput>>, Error>>()?;

        Ok(StoredChatInferenceDatabase {
            function_name: value.function_name,
            variant_name: value.variant_name,
            input: value.input,
            output: value.output,
            dispreferred_outputs,
            episode_id: value.episode_id,
            inference_id: value.inference_id,
            tool_params: value.tool_params,
            tags: value.tags,
            timestamp: value.timestamp,
        })
    }
}

#[derive(Debug, Deserialize)]
pub(super) struct ClickHouseStoredJsonInferenceWithDispreferredOutputs {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub timestamp: DateTime<Utc>,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub input: StoredInput,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output: JsonInferenceOutput,
    #[serde(default)]
    pub dispreferred_outputs: Vec<String>,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output_schema: Value,
    pub tags: HashMap<String, String>,
}

impl TryFrom<ClickHouseStoredJsonInferenceWithDispreferredOutputs> for StoredJsonInference {
    type Error = Error;

    fn try_from(
        value: ClickHouseStoredJsonInferenceWithDispreferredOutputs,
    ) -> Result<Self, Self::Error> {
        let dispreferred_outputs = value
            .dispreferred_outputs
            .into_iter()
            .map(|dispreferred_output| {
                serde_json::from_str(&dispreferred_output).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to deserialize dispreferred output: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<JsonInferenceOutput>, Error>>()?;
        Ok(StoredJsonInference {
            function_name: value.function_name,
            variant_name: value.variant_name,
            input: value.input,
            output: value.output,
            dispreferred_outputs,
            episode_id: value.episode_id,
            inference_id: value.inference_id,
            output_schema: value.output_schema,
            tags: value.tags,
            timestamp: value.timestamp,
        })
    }
}

/// Structs that almost map to the storage format of inferences, but contains a dispreferred_outputs field.
/// When querying inferences, if the user requests a join with the DemonstrationFeedback table, we use the
/// demonstration feedback as `output` and set the original output as `dispreferred_outputs`.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum ClickHouseStoredInferenceWithDispreferredOutputs {
    Json(ClickHouseStoredJsonInferenceWithDispreferredOutputs),
    Chat(ClickHouseStoredChatInferenceWithDispreferredOutputs),
}

impl TryFrom<ClickHouseStoredInferenceWithDispreferredOutputs> for StoredInferenceDatabase {
    type Error = Error;

    fn try_from(
        value: ClickHouseStoredInferenceWithDispreferredOutputs,
    ) -> Result<Self, Self::Error> {
        Ok(match value {
            ClickHouseStoredInferenceWithDispreferredOutputs::Json(inference) => {
                StoredInferenceDatabase::Json(inference.try_into()?)
            }
            ClickHouseStoredInferenceWithDispreferredOutputs::Chat(inference) => {
                StoredInferenceDatabase::Chat(inference.try_into()?)
            }
        })
    }
}

// TODO(shuyangli): Move to tensorzero-core/src/endpoints/stored_inferences/v1/types.rs
/// Source of an inference output when querying inferences. Users can choose this because there may be
/// demonstration feedback (manually-curated output) for the inference that should be preferred.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum InferenceOutputSource {
    /// The inference output is the original output from the inference.
    Inference,
    /// The inference output is the demonstration feedback for the inference.
    Demonstration,
}

impl TryFrom<&str> for InferenceOutputSource {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "inference" => Ok(InferenceOutputSource::Inference),
            "demonstration" => Ok(InferenceOutputSource::Demonstration),
            _ => Err(Error::new(ErrorDetails::InvalidInferenceOutputSource {
                source_kind: value.to_string(),
            })),
        }
    }
}
/// Parameters for a ListInferences query.
#[derive(Debug, Clone)]
pub struct ListInferencesParams<'a> {
    /// Function name to query. If provided, only inferences from this function will be returned.
    pub function_name: Option<&'a str>,
    /// Inference IDs. If provided, only inferences with these IDs will be returned.
    pub ids: Option<&'a [Uuid]>,
    /// Variant name to query.
    pub variant_name: Option<&'a str>,
    /// Episode ID to query.
    pub episode_id: Option<&'a Uuid>,
    /// Filters to apply to the query.
    pub filters: Option<&'a InferenceFilter>,
    /// Source of the inference output to query.
    pub output_source: InferenceOutputSource,
    /// Maximum number of inferences to return.
    pub limit: Option<u64>,
    /// Number of inferences to skip.
    pub offset: Option<u64>,
    pub order_by: Option<&'a [OrderBy]>,
}

impl Default for ListInferencesParams<'_> {
    fn default() -> Self {
        Self {
            function_name: None,
            ids: None,
            variant_name: None,
            episode_id: None,
            filters: None,
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: None,
        }
    }
}

#[async_trait]
#[cfg_attr(test, automock)]
pub trait InferenceQueries {
    async fn list_inferences(
        &self,
        // config is used for identifying the type of the function.
        config: &Config,
        params: &ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInferenceDatabase>, Error>;
}
