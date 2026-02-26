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

use crate::config::{Config, MetricConfig, MetricConfigLevel};
use crate::db::TimeWindow;
use crate::db::clickhouse::query_builder::{InferenceFilter, OrderBy, OrderByTerm};
use crate::endpoints::inference::InferenceParams;
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfigType;
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::{
    ChatInferenceDatabaseInsert, ContentBlockChatOutput, FunctionType, JsonInferenceDatabaseInsert,
    JsonInferenceOutput, StoredInput,
};
use crate::serde_util::{
    deserialize_defaulted_json_string, deserialize_json_string, deserialize_u64,
    serialize_utc_datetime_rfc_3339_with_millis,
};
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
    pub snapshot_hash: Option<String>,
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
            input: Some(value.input),
            output: Some(value.output),
            dispreferred_outputs,
            episode_id: value.episode_id,
            inference_id: value.inference_id,
            tool_params: Some(value.tool_params),
            tags: value.tags,
            timestamp: value.timestamp,
            extra_body: Some(value.extra_body),
            inference_params: Some(value.inference_params),
            processing_time_ms: value.processing_time_ms,
            ttft_ms: value.ttft_ms,
            snapshot_hash: value.snapshot_hash,
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
    pub snapshot_hash: Option<String>,
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
            input: Some(value.input),
            output: Some(value.output),
            dispreferred_outputs,
            episode_id: value.episode_id,
            inference_id: value.inference_id,
            output_schema: Some(value.output_schema),
            tags: value.tags,
            timestamp: value.timestamp,
            extra_body: Some(value.extra_body),
            inference_params: Some(value.inference_params),
            processing_time_ms: value.processing_time_ms,
            ttft_ms: value.ttft_ms,
            snapshot_hash: value.snapshot_hash,
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

impl ListInferencesParams<'_> {
    /// Validates that before/after pagination works with the rest of the request:
    /// - If order_by is provided, only timestamp ordering is supported.
    /// - Offset must not be provided.
    ///
    /// Returns an error if the request is invalid.
    pub fn validate_pagination(&self) -> Result<(), Error> {
        if self.pagination.is_none() {
            return Ok(());
        };
        let Some(order_by) = self.order_by else {
            return Ok(());
        };

        for order in order_by {
            match &order.term {
                OrderByTerm::Timestamp => {
                    // Timestamp ordering is compatible with before/after pagination (UUIDv7 is time-ordered)
                    continue;
                }
                OrderByTerm::Metric { name } => {
                    return Err(Error::new(ErrorDetails::InvalidRequest {
                        message: format!(
                            "Cannot order by metric '{name}'; only ordering by timestamp is supported with before/after pagination.",
                        ),
                    }));
                }
                OrderByTerm::SearchRelevance => {
                    return Err(Error::new(ErrorDetails::InvalidRequest {
                        message: "Cannot order by search relevance; only ordering by timestamp is supported with before/after pagination.".to_string(),
                    }));
                }
            }
        }

        if self.offset != 0 {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "OFFSET is not supported when using before/after pagination".to_string(),
            }));
        }

        Ok(())
    }
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
#[derive(Debug, Deserialize, PartialEq, sqlx::FromRow)]
pub struct FunctionInfo {
    pub function_name: String,
    pub function_type: FunctionType,
    pub variant_name: String,
    pub episode_id: Uuid,
}

// ===== Inference count types (merged from inference_count module) =====

/// Parameters for counting inferences for a function.
#[derive(Debug)]
pub struct CountInferencesForFunctionParams<'a> {
    pub function_name: &'a str,
    pub function_type: FunctionConfigType,
    pub variant_name: Option<&'a str>,
}

/// Row returned from the count_inferences_by_variant query.
#[derive(Debug, Deserialize)]
pub struct CountByVariant {
    pub variant_name: String,
    /// Number of inferences for this variant
    #[serde(deserialize_with = "deserialize_u64")]
    pub inference_count: u64,
    /// ISO 8601 timestamp of the last inference for this variant
    pub last_used_at: String,
}

/// Parameters for counting inferences with feedback.
/// If `metric_threshold` is Some, only counts inferences with feedback meeting the threshold criteria.
pub struct CountInferencesWithFeedbackParams<'a> {
    pub function_name: &'a str,
    pub function_type: FunctionConfigType,
    pub metric_name: &'a str,
    pub metric_config: &'a MetricConfig,
    /// If present, only counts inferences with feedback meeting the threshold criteria.
    pub metric_threshold: Option<f64>,
}

/// Parameters for getting function throughput by variant.
#[derive(Debug)]
pub struct GetFunctionThroughputByVariantParams<'a> {
    pub function_name: &'a str,
    pub time_window: TimeWindow,
    pub max_periods: u32,
}

/// Row returned from the get_function_throughput_by_variant query.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct VariantThroughput {
    /// Start datetime of the period in RFC 3339 format with milliseconds
    #[serde(serialize_with = "serialize_utc_datetime_rfc_3339_with_millis")]
    pub period_start: DateTime<Utc>,
    pub variant_name: String,
    /// Number of inferences for this (period, variant) combination
    pub count: u32,
}

/// Row returned from the list_functions_with_inference_count query.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct FunctionInferenceCount {
    pub function_name: String,
    /// ISO 8601 timestamp of the most recent inference for this function
    #[serde(serialize_with = "serialize_utc_datetime_rfc_3339_with_millis")]
    pub last_inference_timestamp: DateTime<Utc>,
    /// Total number of inferences for this function
    pub inference_count: u32,
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

    // ===== Write methods =====

    /// Insert chat inference records.
    async fn insert_chat_inferences(
        &self,
        rows: &[ChatInferenceDatabaseInsert],
    ) -> Result<(), Error>;

    /// Insert JSON inference records.
    async fn insert_json_inferences(
        &self,
        rows: &[JsonInferenceDatabaseInsert],
    ) -> Result<(), Error>;

    // ===== Inference count methods (merged from InferenceCountQueries trait) =====
    // Note: count_inferences_for_function, count_inferences_with_demonstration_feedback, and
    // count_inferences_for_episode were removed as they can be achieved via count_inferences with filters.

    /// Counts inferences for a function, optionally filtered by variant, grouped by variant.
    /// Returns grouped data with variant name, count, and last_used_at timestamps.
    async fn count_inferences_by_variant(
        &self,
        params: CountInferencesForFunctionParams<'_>,
    ) -> Result<Vec<CountByVariant>, Error>;

    /// Count the number of inferences with feedback for a metric.
    /// If `metric_threshold` is Some, only counts inferences with feedback meeting the threshold criteria
    /// based on the metric config's optimize direction (max/min).
    async fn count_inferences_with_feedback(
        &self,
        params: CountInferencesWithFeedbackParams<'_>,
    ) -> Result<u64, Error>;

    /// Get function throughput (inference counts) grouped by variant and time period.
    /// Returns throughput data for the last `max_periods` time periods, grouped by variant.
    /// For cumulative time window, returns all-time data with a fixed period_start.
    async fn get_function_throughput_by_variant(
        &self,
        params: GetFunctionThroughputByVariantParams<'_>,
    ) -> Result<Vec<VariantThroughput>, Error>;

    /// List all functions with their inference counts, ordered by most recent inference.
    /// Returns the function name, count of inferences, and timestamp of the most recent inference.
    async fn list_functions_with_inference_count(
        &self,
    ) -> Result<Vec<FunctionInferenceCount>, Error>;
}
