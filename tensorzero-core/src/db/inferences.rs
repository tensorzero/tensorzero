/// Definitions for inference-related traits and types.
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tensorzero_derive::TensorZeroDeserialize;
use uuid::Uuid;

#[cfg(test)]
use mockall::automock;

use crate::config::{Config, MetricConfigLevel};
use crate::db::clickhouse::query_builder::{InferenceFilter, OrderBy};
use crate::endpoints::inference::InferenceParams;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::{
    ContentBlockChatOutput, FunctionType, JsonInferenceOutput, StoredInput,
};
use crate::serde_util::{deserialize_defaulted_json_string, deserialize_json_string};
use crate::stored_inference::{
    StoredChatInferenceDatabase, StoredInferenceDatabase, StoredJsonInference,
};
use crate::tool::{ToolCallConfigDatabaseInsert, deserialize_tool_info};

pub(crate) const DEFAULT_INFERENCE_QUERY_LIMIT: u32 = 20;

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
    #[serde(flatten, deserialize_with = "deserialize_tool_info")]
    pub tool_params: ToolCallConfigDatabaseInsert,
    pub tags: HashMap<String, String>,
    #[serde(default, deserialize_with = "deserialize_defaulted_json_string")]
    pub extra_body: UnfilteredInferenceExtraBody,
    #[serde(default, deserialize_with = "deserialize_defaulted_json_string")]
    pub inference_params: InferenceParams,
    pub processing_time_ms: Option<u64>,
    pub ttft_ms: Option<u64>,
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
            extra_body: value.extra_body,
            inference_params: value.inference_params,
            processing_time_ms: value.processing_time_ms,
            ttft_ms: value.ttft_ms,
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
    #[serde(default, deserialize_with = "deserialize_defaulted_json_string")]
    pub extra_body: UnfilteredInferenceExtraBody,
    #[serde(default, deserialize_with = "deserialize_defaulted_json_string")]
    pub inference_params: InferenceParams,
    pub processing_time_ms: Option<u64>,
    pub ttft_ms: Option<u64>,
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
            extra_body: value.extra_body,
            inference_params: value.inference_params,
            processing_time_ms: value.processing_time_ms,
            ttft_ms: value.ttft_ms,
        })
    }
}

/// Structs that almost map to the storage format of inferences, but contains a dispreferred_outputs field.
/// When querying inferences, if the user requests a join with the DemonstrationFeedback table, we use the
/// demonstration feedback as `output` and set the original output as `dispreferred_outputs`.
#[derive(Debug, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
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
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum InferenceOutputSource {
    /// No output - used when creating datapoints without output.
    None,
    /// The inference output is the original output from the inference.
    #[default]
    Inference,
    /// The inference output is the demonstration feedback for the inference.
    Demonstration,
}

impl TryFrom<&str> for InferenceOutputSource {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "none" => Ok(InferenceOutputSource::None),
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
    /// We always enforce a limit at the database level to avoid unbounded queries.
    pub limit: u32,
    /// Number of inferences to skip before starting to return results.
    /// This is mutually exclusive with cursor pagination. If both are provided, we return an error.
    pub offset: u32,
    /// Optional cursor-based pagination condition.
    /// This supports 2 types: "before a given ID" and "after a given ID".
    /// This is mutually exclusive with offset pagination. If both are provided, we return an error.
    pub pagination: Option<PaginationParams>,
    /// Ordering criteria for the results.
    pub order_by: Option<&'a [OrderBy]>,
    /// Experimental: search query to filter inferences by.
    pub search_query_experimental: Option<&'a str>,
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
            limit: DEFAULT_INFERENCE_QUERY_LIMIT,
            offset: 0,
            pagination: None,
            order_by: None,
            search_query_experimental: None,
        }
    }
}

/// Parameters for cursor-based pagination.
/// Currently it only supports paginating before/after a given ID. In the future, we can extend this
/// to support paginating with additional metrics at the page boundary.
#[derive(Debug, Clone)]
pub enum PaginationParams {
    /// Return the latest inferences before the given ID.
    Before { id: Uuid },
    /// Return the oldest inferences after the given ID.
    After { id: Uuid },
}

/// Inference metadata from the InferenceById table.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct InferenceMetadata {
    pub id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub function_type: FunctionType,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot_hash: Option<String>,
}

/// Parameters for listing inference metadata.
#[derive(Debug, Clone, Default)]
pub struct ListInferenceMetadataParams {
    /// Optional cursor-based pagination condition.
    pub pagination: Option<PaginationParams>,
    /// Maximum number of records to return.
    pub limit: u32,
    /// Optional function name to filter by.
    pub function_name: Option<String>,
    /// Optional variant name to filter by.
    pub variant_name: Option<String>,
    /// Optional episode ID to filter by.
    pub episode_id: Option<Uuid>,
}

/// Parameters for a CountInferences query.
/// Similar to ListInferencesParams but without pagination fields since we only need a count.
#[derive(Debug, Clone, Default)]
pub struct CountInferencesParams<'a> {
    /// Function name to query. If provided, only inferences from this function will be counted.
    pub function_name: Option<&'a str>,
    /// Variant name to query.
    pub variant_name: Option<&'a str>,
    /// Episode ID to query.
    pub episode_id: Option<&'a Uuid>,
    /// Filters to apply to the query.
    pub filters: Option<&'a InferenceFilter>,
    /// Experimental: search query to filter inferences by.
    pub search_query_experimental: Option<&'a str>,
}

/// Function information retrieved for feedback validation.
/// Contains the function name, type, variant, and episode ID associated with an inference or episode.
#[derive(Debug, Deserialize, PartialEq)]
pub struct FunctionInfo {
    pub function_name: String,
    pub function_type: FunctionType,
    pub variant_name: String,
    pub episode_id: Uuid,
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

    /// List inference metadata from the InferenceById table.
    async fn list_inference_metadata(
        &self,
        params: &ListInferenceMetadataParams,
    ) -> Result<Vec<InferenceMetadata>, Error>;

    /// Count inferences matching the given parameters.
    async fn count_inferences(
        &self,
        config: &Config,
        params: &CountInferencesParams<'_>,
    ) -> Result<u64, Error>;

    /// Get function information for feedback validation by target_id.
    ///
    /// When `level` is `Inference`, queries by inference_id from `InferenceById`.
    /// When `level` is `Episode`, queries by episode_id from `InferenceByEpisodeId`.
    ///
    /// Returns `None` if the target doesn't exist.
    async fn get_function_info(
        &self,
        target_id: &Uuid,
        level: MetricConfigLevel,
    ) -> Result<Option<FunctionInfo>, Error>;

    /// Get tool parameters from a chat inference for demonstration validation.
    ///
    /// Returns the tool configuration that was used at inference time, which is needed
    /// to validate demonstration feedback against the actual tools available.
    ///
    /// Returns `None` if the inference doesn't exist.
    async fn get_chat_inference_tool_params(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<ToolCallConfigDatabaseInsert>, Error>;

    /// Get output schema from a json inference for demonstration validation.
    ///
    /// Returns the output schema that was used at inference time, which is needed
    /// to validate demonstration feedback against the actual schema.
    ///
    /// Returns `None` if the inference doesn't exist.
    async fn get_json_inference_output_schema(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<Value>, Error>;

    /// Get the output string from an inference for human feedback context.
    ///
    /// Returns the serialized output of the inference, which is needed when
    /// writing static evaluation human feedback records.
    ///
    /// Returns `None` if the inference doesn't exist.
    async fn get_inference_output(
        &self,
        function_info: &FunctionInfo,
        inference_id: Uuid,
    ) -> Result<Option<String>, Error>;
}
