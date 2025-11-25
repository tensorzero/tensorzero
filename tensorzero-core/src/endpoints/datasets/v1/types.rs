use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_derive::export_schema;
use uuid::Uuid;

pub use crate::db::clickhouse::query_builder::{
    DatapointFilter, InferenceFilter, OrderBy, OrderByTerm, OrderDirection, TagFilter, TimeFilter,
};
use crate::endpoints::datasets::Datapoint;
use crate::inference::types::{ContentBlockChatOutput, Input};
use crate::serde_util::deserialize_double_option;
use crate::tool::{DynamicToolParams, ProviderTool, Tool, ToolChoice};

/// The property to order datapoints by.
/// This is flattened in the public API inside the `DatapointOrderBy` struct.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "by", rename_all = "snake_case")]
pub enum DatapointOrderByTerm {
    /// Creation timestamp of the datapoint.
    #[schemars(title = "DatapointOrderByTimestamp")]
    Timestamp,

    /// Relevance score of the search query in the input and output of the datapoint.
    /// Requires a search query (experimental). If it's not provided, we return an error.
    ///
    /// Current relevance metric is very rudimentary (just term frequency), but we plan
    /// to improve it in the future.
    #[schemars(title = "DatapointOrderBySearchRelevance")]
    SearchRelevance,
}

/// Order by clauses for querying datapoints.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq, ts_rs::TS)]
#[ts(export)]
#[export_schema]
pub struct DatapointOrderBy {
    /// The property to order by.
    #[serde(flatten)]
    pub term: DatapointOrderByTerm,

    /// The ordering direction.
    pub direction: OrderDirection,
}

/// Request to update one or more datapoints in a dataset.
#[derive(Debug, Serialize, Deserialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
#[export_schema]
pub struct UpdateDatapointsRequest {
    /// The datapoints to update.
    pub datapoints: Vec<UpdateDatapointRequest>,
}

/// A tagged request to update a single datapoint in a dataset.
#[derive(Debug, Serialize, Deserialize, JsonSchema, ts_rs::TS)]
#[serde(tag = "type", rename_all = "snake_case")]
#[ts(export, tag = "type", rename_all = "snake_case")]
#[export_schema]
pub enum UpdateDatapointRequest {
    /// Request to update a chat datapoint.
    #[schemars(title = "UpdateChatDatapointRequest")]
    Chat(UpdateChatDatapointRequest),
    /// Request to update a JSON datapoint.
    #[schemars(title = "UpdateJsonDatapointRequest")]
    Json(UpdateJsonDatapointRequest),
}

/// An update request for a chat datapoint.
#[derive(Clone, Debug, JsonSchema, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
#[export_schema]
#[schemars(title = "UpdateChatDatapointRequestInternal")]
pub struct UpdateChatDatapointRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Datapoint input. If omitted, it will be left unchanged.
    #[serde(default)]
    pub input: Option<Input>,

    /// Chat datapoint output. If omitted, it will be left unchanged. If specified as `null`, it will be set to
    /// `null`. Otherwise, it will overwrite the existing output (and can be an empty array).
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "Chat datapoint output.

If omitted (which uses the default value `UNSET`), it will be left unchanged. If set to `None`, it will be cleared.
Otherwise, it will overwrite the existing output (and can be an empty list).")]
    pub output: Option<Option<Vec<ContentBlockChatOutput>>>,

    /// Datapoint tool parameters.
    #[serde(flatten)]
    pub tool_params: UpdateDynamicToolParamsRequest,

    /// DEPRECATED (#4725 / 2026.2+): Datapoint tool parameters.
    /// Moving forward, don't nest these fields.
    #[serde(
        default,
        rename = "tool_params",
        skip_serializing_if = "Option::is_none"
    )]
    #[deprecated(note = "Use flattened fields instead. (#4725)")]
    #[ts(skip)]
    #[schemars(rename = "tool_params")]
    pub deprecated_do_not_use_tool_params: Option<UpdateDynamicToolParamsRequest>,

    /// Datapoint tags. If omitted, it will be left unchanged. If empty, it will be cleared. Otherwise,
    /// it will be overwrite the existing tags.
    #[serde(default)]
    #[schemars(description = "Datapoint tags.

If omitted (which uses the default value `UNSET`), it will be left unchanged. If set to `None`, it will be cleared.
Otherwise, it will overwrite the existing tags.")]
    pub tags: Option<HashMap<String, String>>,

    /// Metadata fields to update.
    #[serde(flatten)]
    pub metadata: DatapointMetadataUpdate,

    /// DEPRECATED (#4725 / 2026.2+): Metadata fields to update.
    /// Moving forward, don't nest these fields.
    #[serde(default, rename = "metadata", skip_serializing_if = "Option::is_none")]
    #[deprecated(note = "Use flattened fields instead. (#4725)")]
    #[ts(skip)]
    #[schemars(rename = "metadata")]
    pub deprecated_do_not_use_metadata: Option<DatapointMetadataUpdate>,
}

impl<'de> Deserialize<'de> for UpdateChatDatapointRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(rename = "UpdateChatDatapointRequest")]
        struct Helper {
            id: Uuid,
            #[serde(default)]
            input: Option<Input>,
            #[serde(default, deserialize_with = "deserialize_double_option")]
            output: Option<Option<Vec<ContentBlockChatOutput>>>,
            #[serde(flatten)]
            tool_params_new: UpdateDynamicToolParamsRequest,
            #[serde(default)]
            tool_params: Option<UpdateDynamicToolParamsRequest>,
            #[serde(default)]
            tags: Option<HashMap<String, String>>,
            #[serde(flatten)]
            metadata_new: DatapointMetadataUpdate,
            #[serde(default)]
            metadata: Option<DatapointMetadataUpdate>,
        }

        let helper = Helper::deserialize(deserializer)?;

        // Helper function to check if UpdateDynamicToolParamsRequest is default
        fn is_tool_params_default(params: &UpdateDynamicToolParamsRequest) -> bool {
            params.allowed_tools.is_none()
                && params.additional_tools.is_none()
                && params.tool_choice.is_none()
                && params.parallel_tool_calls.is_none()
                && params.provider_tools.is_none()
        }

        // Check if the deprecated tool_params field is being used
        let tool_params_is_default = helper
            .tool_params
            .as_ref()
            .is_none_or(is_tool_params_default);
        let tool_params_new_is_default = is_tool_params_default(&helper.tool_params_new);

        let mut tool_params_new = helper.tool_params_new;

        if !tool_params_is_default {
            if !tool_params_new_is_default {
                return Err(serde::de::Error::custom(
                    "Cannot specify both `tool_params` (deprecated) and flattened tool parameter fields. Use only the flattened fields."
                ));
            }
            // Emit deprecation warning
            crate::utils::deprecation_warning(
                "The `tool_params` field is deprecated. Please use flattened tool parameter fields instead. (#4725)"
            );
            // Copy tool_params to tool_params_new
            if let Some(tool_params) = &helper.tool_params {
                tool_params_new = tool_params.clone();
            }
        }

        // Check if the deprecated metadata field is being used
        let metadata_is_default =
            helper.metadata.is_none() || helper.metadata.as_ref().is_none_or(|m| m.name.is_none());
        let metadata_new_is_default = helper.metadata_new.name.is_none();

        let mut metadata_new = helper.metadata_new;

        if !metadata_is_default {
            if !metadata_new_is_default {
                return Err(serde::de::Error::custom(
                    "Cannot specify both `metadata` (deprecated) and flattened metadata fields. Use only the flattened fields."
                ));
            }
            // Emit deprecation warning
            crate::utils::deprecation_warning(
                "The `metadata` field is deprecated. Please use flattened metadata fields instead. (#4725)"
            );
            // Copy metadata to metadata_new
            if let Some(metadata) = &helper.metadata {
                metadata_new = metadata.clone();
            }
        }

        Ok(UpdateChatDatapointRequest {
            id: helper.id,
            input: helper.input,
            output: helper.output,
            tool_params: tool_params_new,
            #[expect(deprecated)]
            deprecated_do_not_use_tool_params: None,
            tags: helper.tags,
            metadata: metadata_new,
            #[expect(deprecated)]
            deprecated_do_not_use_metadata: None,
        })
    }
}

/// A request to update the dynamic tool parameters of a datapoint.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq, JsonSchema, ts_rs::TS)]
#[ts(export, optional_fields)]
#[export_schema]
pub struct UpdateDynamicToolParamsRequest {
    /// A subset of static tools configured for the function that the inference is explicitly allowed to use.
    /// If omitted, it will be left unchanged. If specified as `null`, it will be cleared (we allow function-configured tools plus additional tools
    /// provided at inference time). If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "A subset of static tools configured for the function that the inference is explicitly allowed to use.

If omitted (which uses the default value `UNSET`), it will be left unchanged. If set to `None`, it will be cleared (we allow function-configured tools
plus additional tools provided at inference time). If specified as a value, it will be set to the provided value.")]
    pub allowed_tools: Option<Option<Vec<String>>>,

    /// Tools that the user provided at inference time (not in function config), in addition to the function-configured tools, that are also allowed.
    /// Modifying `additional_tools` DOES NOT automatically modify `allowed_tools`; `allowed_tools` must be explicitly updated to include
    /// new tools or exclude removed tools.
    /// If omitted, it will be left unchanged. If specified as a value, it will be set to the provided value.
    pub additional_tools: Option<Vec<Tool>>,

    /// User-specified tool choice strategy.
    /// If omitted, it will be left unchanged. If specified as `null`, we will clear the dynamic tool choice and use function-configured tool choice.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "User-specified tool choice strategy.

If omitted (which uses the default value `UNSET`), it will be left unchanged. If set to `None`, it will be cleared (we will use function-configured
tool choice). If specified as a value, it will be set to the provided value.")]
    pub tool_choice: Option<Option<ToolChoice>>,

    /// Whether to use parallel tool calls in the inference.
    /// If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "Whether to use parallel tool calls in the inference.

If omitted (which uses the default value `UNSET`), it will be left unchanged. If set to `None`, it will be cleared (we will use function-configured
parallel tool calls). If specified as a value, it will be set to the provided value.")]
    pub parallel_tool_calls: Option<Option<bool>>,

    /// Provider-specific tool configurations
    /// If omitted, it will be left unchanged. If specified as a value, it will be set to the provided value.
    pub provider_tools: Option<Vec<ProviderTool>>,
}

/// An update request for a JSON datapoint.
#[derive(Clone, Debug, JsonSchema, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
#[export_schema]
#[schemars(title = "UpdateJsonDatapointRequestInternal")]
pub struct UpdateJsonDatapointRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Datapoint input. If omitted, it will be left unchanged.
    #[serde(default)]
    pub input: Option<Input>,

    /// JSON datapoint output. If omitted, it will be left unchanged. If `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "JSON datapoint output.
If omitted (which uses the default value `UNSET`), it will be left unchanged. If set to `None`, it will be cleared (represents edge case where
inference succeeded but model didn't output relevant content blocks). Otherwise, it will overwrite the existing output.")]
    pub output: Option<Option<JsonDatapointOutputUpdate>>,

    /// The output schema of the JSON datapoint. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    /// If not provided, the function's output schema will be used.
    #[serde(default)]
    pub output_schema: Option<Value>,

    /// Datapoint tags. If omitted, it will be left unchanged. If empty, it will be cleared. Otherwise,
    /// it will be overwrite the existing tags.
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,

    /// Metadata fields to update.
    #[serde(flatten)]
    pub metadata: DatapointMetadataUpdate,

    /// DEPRECATED (#4725 / 2026.2+): Metadata fields to update.
    /// Moving forward, don't nest these fields.
    #[serde(default, rename = "metadata", skip_serializing_if = "Option::is_none")]
    #[deprecated(note = "Use flattened fields instead. (#4725)")]
    #[ts(skip)]
    #[schemars(rename = "metadata")]
    pub deprecated_do_not_use_metadata: Option<DatapointMetadataUpdate>,
}

impl<'de> Deserialize<'de> for UpdateJsonDatapointRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(rename = "UpdateJsonDatapointRequest")]
        struct Helper {
            id: Uuid,
            #[serde(default)]
            input: Option<Input>,
            #[serde(default, deserialize_with = "deserialize_double_option")]
            output: Option<Option<JsonDatapointOutputUpdate>>,
            #[serde(default)]
            output_schema: Option<Value>,
            #[serde(default)]
            tags: Option<HashMap<String, String>>,
            #[serde(flatten)]
            metadata_new: DatapointMetadataUpdate,
            #[serde(default)]
            metadata: Option<DatapointMetadataUpdate>,
        }

        let helper = Helper::deserialize(deserializer)?;

        // Check if the deprecated metadata field is being used
        let metadata_is_default =
            helper.metadata.is_none() || helper.metadata.as_ref().is_none_or(|m| m.name.is_none());
        let metadata_new_is_default = helper.metadata_new.name.is_none();

        let mut metadata_new = helper.metadata_new;

        if !metadata_is_default {
            if !metadata_new_is_default {
                return Err(serde::de::Error::custom(
                    "Cannot specify both `metadata` (deprecated) and flattened metadata fields. Use only the flattened fields."
                ));
            }
            // Emit deprecation warning
            crate::utils::deprecation_warning(
                "The `metadata` field is deprecated. Please use flattened metadata fields instead. (#4725)"
            );
            // Copy metadata to metadata_new
            if let Some(metadata) = &helper.metadata {
                metadata_new = metadata.clone();
            }
        }

        Ok(UpdateJsonDatapointRequest {
            id: helper.id,
            input: helper.input,
            output: helper.output,
            output_schema: helper.output_schema,
            tags: helper.tags,
            metadata: metadata_new,
            #[expect(deprecated)]
            deprecated_do_not_use_metadata: None,
        })
    }
}

/// A request to update the output of a JSON datapoint.
///
/// We intentionally only accept the `raw` field, because JSON datapoints can contain invalid or malformed JSON for eval purposes.
#[derive(Debug, Serialize, Deserialize, Clone, JsonSchema, ts_rs::TS)]
#[ts(export)]
#[export_schema]
pub struct JsonDatapointOutputUpdate {
    /// The raw output of the datapoint. For valid JSON outputs, this should be a JSON-serialized string.
    ///
    /// This will be parsed and validated against the datapoint's `output_schema`. Valid `raw` values will be parsed and stored as `parsed`, and
    /// invalid `raw` values will be stored as-is, because we allow invalid outputs in datapoints by design.
    pub raw: Option<String>,
}

/// A request to update the metadata of a datapoint.
#[derive(Debug, Serialize, Default, Deserialize, Clone, PartialEq, JsonSchema, ts_rs::TS)]
#[ts(export, optional_fields)]
#[export_schema]
pub struct DatapointMetadataUpdate {
    /// Datapoint name. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "Datapoint name.

If omitted (which uses the default value `UNSET`), it will be left unchanged. If set to `None`, it will be cleared. If specified as a value, it will
be set to the provided value.")]
    pub name: Option<Option<String>>,
}

/// A response to a request to update one or more datapoints in a dataset.
#[derive(Debug, Serialize, Deserialize, Clone, JsonSchema, ts_rs::TS)]
#[ts(export)]
#[export_schema]
pub struct UpdateDatapointsResponse {
    /// The IDs of the datapoints that were updated.
    /// These are newly generated IDs for UpdateDatapoint requests, and they are the same IDs for UpdateDatapointMetadata requests.
    pub ids: Vec<Uuid>,
}

/// Request to update metadata for one or more datapoints in a dataset.
/// Used by the `PATCH /v1/datasets/{dataset_id}/datapoints/metadata` endpoint.
#[derive(Debug, Serialize, Deserialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
#[export_schema]
pub struct UpdateDatapointsMetadataRequest {
    /// The datapoints to update metadata for.
    pub datapoints: Vec<UpdateDatapointMetadataRequest>,
}

/// A request to update the metadata of a single datapoint.
#[derive(Debug, Serialize, Deserialize, JsonSchema, ts_rs::TS)]
#[ts(export, optional_fields)]
#[export_schema]
pub struct UpdateDatapointMetadataRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Metadata fields to update.
    #[serde(flatten)]
    pub metadata: DatapointMetadataUpdate,
}

/// Request to list datapoints from a dataset with pagination and filters.
/// Used by the `POST /v1/datasets/{dataset_id}/list_datapoints` endpoint.
#[derive(Debug, Default, Serialize, Deserialize, JsonSchema, ts_rs::TS)]
#[ts(export, optional_fields)]
#[export_schema]
pub struct ListDatapointsRequest {
    /// Optional function name to filter datapoints by.
    /// If provided, only datapoints from this function will be returned.
    pub function_name: Option<String>,

    /// The maximum number of datapoints to return.
    /// Defaults to 20.
    pub limit: Option<u32>,

    /// The maximum number of datapoints to return. Defaults to 20.
    /// Deprecated: please use `limit`. If `limit` is provided, `page_size` is ignored.
    #[deprecated(since = "2025.11.1", note = "Use `limit` instead")]
    pub page_size: Option<u32>,

    /// The number of datapoints to skip before starting to return results.
    /// Defaults to 0.
    pub offset: Option<u32>,

    /// Optional filter to apply when querying datapoints.
    /// Supports filtering by tags, time, and logical combinations (AND/OR/NOT).
    pub filter: Option<DatapointFilter>,

    /// Optional ordering criteria for the results.
    /// Supports multiple sort criteria (e.g., sort by timestamp then by search relevance).
    pub order_by: Option<Vec<DatapointOrderBy>>,

    /// Text query to filter. Case-insensitive substring search over the datapoints' input and output.
    ///
    /// THIS FEATURE IS EXPERIMENTAL, and we may change or remove it at any time.
    /// We recommend against depending on this feature for critical use cases.
    ///
    /// Important limitations:
    /// - This requires an exact substring match; we do not tokenize this query string.
    /// - This doesn't search for any content in the template itself.
    /// - Quality is based on term frequency > 0, without any relevance scoring.
    /// - There are no performance guarantees (it's best effort only). Today, with no other
    ///   filters, it will perform a full table scan, which may be extremely slow depending
    ///   on the data volume.
    pub search_query_experimental: Option<String>,
}

/// Request to get specific datapoints by their IDs.
/// Used by the `POST /v1/datasets/{dataset_name}/get_datapoints` endpoint.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS, JsonSchema)]
#[export_schema]
#[ts(export)]
pub struct GetDatapointsRequest {
    /// The IDs of the datapoints to retrieve. Required.
    pub ids: Vec<Uuid>,
}

/// Response containing the requested datapoints.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS, JsonSchema)]
#[export_schema]
#[ts(export)]
pub struct GetDatapointsResponse {
    /// The retrieved datapoints.
    pub datapoints: Vec<Datapoint>,
}

/// Specifies the source of the output for the datapoint when creating datapoints from inferences.
/// - `None`: Do not include any output in the datapoint.
/// - `Inference`: Include the original inference output in the datapoint.
/// - `Demonstration`: Include the latest demonstration feedback as output in the datapoint.
#[derive(Debug, Deserialize, Serialize, ts_rs::TS, JsonSchema)]
#[export_schema]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum CreateDatapointsFromInferenceOutputSource {
    /// Do not include any output in the datapoint.
    None,
    /// Include the original inference output in the datapoint.
    Inference,
    /// Include the latest demonstration feedback as output in the datapoint.
    Demonstration,
}

/// Request to create datapoints from inferences.
#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct CreateDatapointsFromInferenceRequest {
    #[serde(flatten)]
    pub params: CreateDatapointsFromInferenceRequestParams,

    /// When creating the datapoint, this specifies the source of the output for the datapoint.
    /// If not provided, by default we will use the original inference output as the datapoint's output
    /// (equivalent to `inference`).
    pub output_source: Option<CreateDatapointsFromInferenceOutputSource>,
}

/// Parameters for creating datapoints from inferences.
/// Can specify either a list of inference IDs or a query to find inferences.
#[derive(Debug, Deserialize, Serialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
#[export_schema]
pub enum CreateDatapointsFromInferenceRequestParams {
    /// Create datapoints from specific inference IDs.
    #[schemars(title = "CreateDatapointsFromInferenceRequestParamsInferenceIds")]
    InferenceIds {
        /// The inference IDs to create datapoints from.
        inference_ids: Vec<Uuid>,
    },

    /// Create datapoints from an inference query.
    #[schemars(title = "CreateDatapointsFromInferenceRequestParamsInferenceQuery")]
    InferenceQuery {
        /// The function name to filter inferences by.
        function_name: String,

        /// Variant name to filter inferences by, optional.
        #[ts(optional)]
        variant_name: Option<String>,

        /// Filters to apply when querying inferences, optional.
        #[ts(optional)]
        filters: Option<InferenceFilter>,
    },
}

/// Response from creating datapoints.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS, JsonSchema)]
#[ts(export)]
#[export_schema]
pub struct CreateDatapointsResponse {
    /// The IDs of the newly-generated datapoints.
    pub ids: Vec<Uuid>,
}

/// Request to create datapoints manually.
/// Used by the `POST /v1/datasets/{dataset_id}/datapoints` endpoint.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS, JsonSchema)]
#[ts(export)]
#[export_schema]
pub struct CreateDatapointsRequest {
    /// The datapoints to create.
    pub datapoints: Vec<CreateDatapointRequest>,
}

/// A tagged request to create a single datapoint.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[derive(ts_rs::TS)]
#[export_schema]
#[ts(export, tag = "type", rename_all = "snake_case")]
pub enum CreateDatapointRequest {
    /// Request to create a chat datapoint.
    #[schemars(title = "CreateDatapointRequestChat")]
    Chat(CreateChatDatapointRequest),
    /// Request to create a JSON datapoint.
    #[schemars(title = "CreateDatapointRequestJson")]
    Json(CreateJsonDatapointRequest),
}

/// A request to create a chat datapoint.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS, JsonSchema)]
#[export_schema]
#[ts(export, optional_fields)]
pub struct CreateChatDatapointRequest {
    /// The function name for this datapoint. Required.
    pub function_name: String,

    /// Episode ID that the datapoint belongs to. Optional.
    #[serde(default)]
    pub episode_id: Option<Uuid>,

    /// Input to the function. Required.
    pub input: Input,

    /// Chat datapoint output. Optional.
    #[serde(default)]
    pub output: Option<Vec<ContentBlockChatOutput>>,

    /// Dynamic tool parameters for the datapoint. Optional.
    /// This is flattened to mirror inference requests.
    #[serde(flatten)]
    pub dynamic_tool_params: DynamicToolParams,

    /// Tags associated with this datapoint. Optional.
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,

    /// Human-readable name for the datapoint. Optional.
    #[serde(default)]
    pub name: Option<String>,
}

/// A request to create a JSON datapoint.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS, JsonSchema)]
#[export_schema]
#[ts(export, optional_fields)]
pub struct CreateJsonDatapointRequest {
    /// The function name for this datapoint. Required.
    pub function_name: String,

    /// Episode ID that the datapoint belongs to. Optional.
    #[serde(default)]
    pub episode_id: Option<Uuid>,

    /// Input to the function. Required.
    pub input: Input,

    /// JSON datapoint output. Optional.
    pub output: Option<JsonDatapointOutputUpdate>,

    /// The output schema of the JSON datapoint. Optional.
    /// If not provided, the function's output schema will be used. If provided, it will be validated.
    #[serde(default)]
    pub output_schema: Option<Value>,

    /// Tags associated with this datapoint. Optional.
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,

    /// Human-readable name for the datapoint. Optional.
    #[serde(default)]
    pub name: Option<String>,
}

/// Request to delete datapoints from a dataset.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS, JsonSchema)]
#[export_schema]
#[ts(export)]
pub struct DeleteDatapointsRequest {
    /// The IDs of the datapoints to delete.
    pub ids: Vec<Uuid>,
}

/// Response containing the number of deleted datapoints.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS, JsonSchema)]
#[export_schema]
#[ts(export)]
pub struct DeleteDatapointsResponse {
    /// The number of deleted datapoints.
    pub num_deleted_datapoints: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Test deserialization of deprecated `metadata` field (#4725 / 2026.2+).
    // When only the deprecated nested `metadata` field is provided, it should be copied to the flattened metadata fields.
    #[test]
    fn test_chat_deprecation_4725_metadata_only_deprecated() {
        let logs_contain = crate::utils::testing::capture_logs();
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001",
            "metadata": {
                "name": "test_name"
            }
        });

        let result: UpdateChatDatapointRequest = serde_json::from_value(json).unwrap();
        assert_eq!(result.metadata.name, Some(Some("test_name".to_string())));
        #[expect(deprecated)]
        {
            assert_eq!(result.deprecated_do_not_use_metadata, None);
        }
        assert!(logs_contain("The `metadata` field is deprecated. Please use flattened metadata fields instead. (#4725)"));
    }

    // Test deserialization of deprecated `metadata` field (#4725 / 2026.2+).
    // When only the new flattened `name` field is provided, it should work normally.
    #[test]
    fn test_chat_deprecation_4725_metadata_only_new() {
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "test_name"
        });

        let result: UpdateChatDatapointRequest = serde_json::from_value(json).unwrap();
        assert_eq!(result.metadata.name, Some(Some("test_name".to_string())));
        #[expect(deprecated)]
        {
            assert_eq!(result.deprecated_do_not_use_metadata, None);
        }
    }

    // Test deserialization of deprecated `metadata` field (#4725 / 2026.2+).
    // When both deprecated `metadata` and new flattened fields are provided, it should error.
    #[test]
    fn test_chat_deprecation_4725_metadata_both_provided() {
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001",
            "metadata": {
                "name": "old_name"
            },
            "name": "new_name"
        });

        let result: Result<UpdateChatDatapointRequest, _> = serde_json::from_value(json);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot specify both"));
    }

    // Test deserialization of deprecated `metadata` field (#4725 / 2026.2+).
    // When neither `metadata` field is provided, deserialization should succeed with defaults.
    #[test]
    fn test_chat_deprecation_4725_metadata_neither_provided() {
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001"
        });

        let result: UpdateChatDatapointRequest = serde_json::from_value(json).unwrap();
        assert_eq!(result.metadata.name, None);
        #[expect(deprecated)]
        {
            assert_eq!(result.deprecated_do_not_use_metadata, None);
        }
    }

    // Test deserialization of deprecated `tool_params` field (#4725 / 2026.2+).
    // When only the deprecated nested `tool_params` field is provided, it should be copied to the flattened tool parameter fields.
    #[test]
    fn test_chat_deprecation_4725_tool_params_only_deprecated() {
        let logs_contain = crate::utils::testing::capture_logs();
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001",
            "tool_params": {
                "allowed_tools": ["tool1", "tool2"]
            }
        });

        let result: UpdateChatDatapointRequest = serde_json::from_value(json).unwrap();
        assert_eq!(
            result.tool_params.allowed_tools,
            Some(Some(vec!["tool1".to_string(), "tool2".to_string()]))
        );
        #[expect(deprecated)]
        {
            assert_eq!(result.deprecated_do_not_use_tool_params, None);
        }
        assert!(logs_contain("The `tool_params` field is deprecated. Please use flattened tool parameter fields instead. (#4725)"));
    }

    // Test deserialization of deprecated `tool_params` field (#4725 / 2026.2+).
    // When only the new flattened `allowed_tools` field is provided, it should work normally.
    #[test]
    fn test_chat_deprecation_4725_tool_params_only_new() {
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001",
            "allowed_tools": ["tool1", "tool2"]
        });

        let result: UpdateChatDatapointRequest = serde_json::from_value(json).unwrap();
        assert_eq!(
            result.tool_params.allowed_tools,
            Some(Some(vec!["tool1".to_string(), "tool2".to_string()]))
        );
        #[expect(deprecated)]
        {
            assert_eq!(result.deprecated_do_not_use_tool_params, None);
        }
    }

    // Test deserialization of deprecated `tool_params` field (#4725 / 2026.2+).
    // When both deprecated `tool_params` and new flattened fields are provided, it should error.
    #[test]
    fn test_chat_deprecation_4725_tool_params_both_provided() {
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001",
            "tool_params": {
                "allowed_tools": ["tool1"]
            },
            "allowed_tools": ["tool2"]
        });

        let result: Result<UpdateChatDatapointRequest, _> = serde_json::from_value(json);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot specify both"));
    }

    // Test deserialization of deprecated `tool_params` field (#4725 / 2026.2+).
    // When neither `tool_params` field is provided, deserialization should succeed with defaults.
    #[test]
    fn test_chat_deprecation_4725_tool_params_neither_provided() {
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001"
        });

        let result: UpdateChatDatapointRequest = serde_json::from_value(json).unwrap();
        assert_eq!(result.tool_params.allowed_tools, None);
        #[expect(deprecated)]
        {
            assert_eq!(result.deprecated_do_not_use_tool_params, None);
        }
    }

    // Test deserialization of deprecated `metadata` field (#4725 / 2026.2+).
    // When only the deprecated nested `metadata` field is provided, it should be copied to the flattened metadata fields.
    #[test]
    fn test_json_deprecation_4725_metadata_only_deprecated() {
        let logs_contain = crate::utils::testing::capture_logs();
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001",
            "metadata": {
                "name": "test_name"
            }
        });

        let result: UpdateJsonDatapointRequest = serde_json::from_value(json).unwrap();
        assert_eq!(result.metadata.name, Some(Some("test_name".to_string())));
        #[expect(deprecated)]
        {
            assert_eq!(result.deprecated_do_not_use_metadata, None);
        }
        assert!(logs_contain("The `metadata` field is deprecated. Please use flattened metadata fields instead. (#4725)"));
    }

    // Test deserialization of deprecated `metadata` field (#4725 / 2026.2+).
    // When only the new flattened `name` field is provided, it should work normally.
    #[test]
    fn test_json_deprecation_4725_metadata_only_new() {
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "test_name"
        });

        let result: UpdateJsonDatapointRequest = serde_json::from_value(json).unwrap();
        assert_eq!(result.metadata.name, Some(Some("test_name".to_string())));
        #[expect(deprecated)]
        {
            assert_eq!(result.deprecated_do_not_use_metadata, None);
        }
    }

    // Test deserialization of deprecated `metadata` field (#4725 / 2026.2+).
    // When both deprecated `metadata` and new flattened fields are provided, it should error.
    #[test]
    fn test_json_deprecation_4725_metadata_both_provided() {
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001",
            "metadata": {
                "name": "old_name"
            },
            "name": "new_name"
        });

        let result: Result<UpdateJsonDatapointRequest, _> = serde_json::from_value(json);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot specify both"));
    }

    // Test deserialization of deprecated `metadata` field (#4725 / 2026.2+).
    // When neither `metadata` field is provided, deserialization should succeed with defaults.
    #[test]
    fn test_json_deprecation_4725_metadata_neither_provided() {
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001"
        });

        let result: UpdateJsonDatapointRequest = serde_json::from_value(json).unwrap();
        assert_eq!(result.metadata.name, None);
        #[expect(deprecated)]
        {
            assert_eq!(result.deprecated_do_not_use_metadata, None);
        }
    }

    // Test deserialization of deprecated `metadata` field (#4725 / 2026.2+).
    // Verify that empty deprecated `metadata` object is treated as default (not an error).
    #[test]
    fn test_chat_deprecation_4725_metadata_empty_deprecated() {
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001",
            "metadata": {}
        });

        let result: UpdateChatDatapointRequest = serde_json::from_value(json).unwrap();
        assert_eq!(result.metadata.name, None);
        #[expect(deprecated)]
        {
            assert_eq!(result.deprecated_do_not_use_metadata, None);
        }
    }

    // Test deserialization of deprecated `tool_params` field (#4725 / 2026.2+).
    // Verify that empty deprecated `tool_params` object is treated as default (not an error).
    #[test]
    fn test_chat_deprecation_4725_tool_params_empty_deprecated() {
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001",
            "tool_params": {}
        });

        let result: UpdateChatDatapointRequest = serde_json::from_value(json).unwrap();
        assert_eq!(result.tool_params.allowed_tools, None);
        #[expect(deprecated)]
        {
            assert_eq!(result.deprecated_do_not_use_tool_params, None);
        }
    }

    // Test deserialization of deprecated `tool_params` field (#4725 / 2026.2+).
    // Verify that multiple `tool_params` fields can be provided in the deprecated nested form.
    #[test]
    fn test_chat_deprecation_4725_tool_params_multiple_fields_deprecated() {
        let logs_contain = crate::utils::testing::capture_logs();
        let json = json!({
            "id": "00000000-0000-0000-0000-000000000001",
            "tool_params": {
                "allowed_tools": ["tool1"],
                "tool_choice": "auto",
                "parallel_tool_calls": true
            }
        });

        let result: UpdateChatDatapointRequest = serde_json::from_value(json).unwrap();
        assert_eq!(
            result.tool_params.allowed_tools,
            Some(Some(vec!["tool1".to_string()]))
        );
        assert!(result.tool_params.tool_choice.is_some());
        assert_eq!(result.tool_params.parallel_tool_calls, Some(Some(true)));
        #[expect(deprecated)]
        {
            assert_eq!(result.deprecated_do_not_use_tool_params, None);
        }
        assert!(logs_contain("The `tool_params` field is deprecated. Please use flattened tool parameter fields instead. (#4725)"));
    }
}
