use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::db::clickhouse::query_builder::{DatapointFilter, InferenceFilter};
use crate::endpoints::datasets::Datapoint;
use crate::inference::types::{ContentBlockChatOutput, Input};
use crate::serde_util::deserialize_double_option;
use crate::tool::{DynamicToolParams, ToolCallConfigDatabaseInsert};

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
/// Request to update one or more datapoints in a dataset.
pub struct UpdateDatapointsRequest {
    /// The datapoints to update.
    pub datapoints: Vec<UpdateDatapointRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, tag = "type", rename_all = "snake_case"))]
/// A tagged request to update a single datapoint in a dataset.
pub enum UpdateDatapointRequest {
    /// Request to update a chat datapoint.
    Chat(UpdateChatDatapointRequest),
    /// Request to update a JSON datapoint.
    Json(UpdateJsonDatapointRequest),
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
/// An update request for a chat datapoint.
/// For any fields that are optional in ChatInferenceDatapoint, the request field distinguishes between an omitted field, `null`, and a value:
/// - If the field is omitted, it will be left unchanged.
/// - If the field is specified as `null`, it will be set to `null`.
/// - If the field has a value, it will be set to the provided value.
///
/// In Rust this is modeled as an `Option<Option<T>>`, where `None` means "unchanged" and `Some(None)` means "set to `null`" and `Some(Some(T))` means "set to the provided value".
pub struct UpdateChatDatapointRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Datapoint input. If omitted, it will be left unchanged.
    #[serde(default)]
    pub input: Option<Input>,

    /// Chat datapoint output. If omitted, it will be left unchanged. If empty, it will be cleared. Otherwise,
    /// it will overwrite the existing output.
    #[serde(default)]
    pub output: Option<Vec<ContentBlockChatOutput>>,

    /// Datapoint tool parameters. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    pub tool_params: Option<Option<ToolCallConfigDatabaseInsert>>,

    /// Datapoint tags. If omitted, it will be left unchanged. If empty, it will be cleared. Otherwise,
    /// it will be overwrite the existing tags.
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,

    /// Metadata fields. If omitted, it will be left unchanged.
    #[serde(default)]
    pub metadata: Option<DatapointMetadataUpdate>,
}

/// An update request for a JSON datapoint.
/// For any fields that are optional in JsonInferenceDatapoint, the request field distinguishes between an omitted field, `null`, and a value:
/// - If the field is omitted, it will be left unchanged.
/// - If the field is specified as `null`, it will be set to `null`.
/// - If the field has a value, it will be set to the provided value.
///
/// In Rust this is modeled as an `Option<Option<T>>`, where `None` means "unchanged" and `Some(None)` means "set to `null`" and `Some(Some(T))` means "set to the provided value".
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct UpdateJsonDatapointRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Datapoint input. If omitted, it will be left unchanged.
    #[serde(default)]
    pub input: Option<Input>,

    /// JSON datapoint output. If omitted, it will be left unchanged. If `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    /// This will be validated against `output_schema` or the function's output schema.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    pub output: Option<Option<Value>>,

    /// The output schema of the JSON datapoint. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    /// If not provided, the function's output schema will be used.
    #[serde(default)]
    pub output_schema: Option<Value>,

    /// Datapoint tags. If omitted, it will be left unchanged. If empty, it will be cleared. Otherwise,
    /// it will be overwrite the existing tags.
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,

    /// Metadata fields. If omitted, it will be left unchanged.
    #[serde(default)]
    pub metadata: Option<DatapointMetadataUpdate>,
}

/// A request to update the metadata of a datapoint.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct DatapointMetadataUpdate {
    /// Datapoint name. If omitted, it will be left unchanged. If specified as `null`, it will be set to `null`. If specified as a value, it will be set to the provided value.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    pub name: Option<Option<String>>,
}

/// A response to a request to update one or more datapoints in a dataset.
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct UpdateDatapointsResponse {
    /// The IDs of the datapoints that were updated.
    /// These are newly generated IDs for UpdateDatapoint requests, and they are the same IDs for UpdateDatapointMetadata requests.
    pub ids: Vec<Uuid>,
}

/// Request to update metadata for one or more datapoints in a dataset.
/// Used by the `PATCH /v1/datasets/{dataset_id}/datapoints/metadata` endpoint.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct UpdateDatapointsMetadataRequest {
    /// The datapoints to update metadata for.
    pub datapoints: Vec<UpdateDatapointMetadataRequest>,
}

/// A request to update the metadata of a single datapoint.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct UpdateDatapointMetadataRequest {
    /// The ID of the datapoint to update. Required.
    pub id: Uuid,

    /// Metadata fields to update. If omitted, no metadata changes will be made.
    #[serde(default)]
    pub metadata: Option<DatapointMetadataUpdate>,
}

/// Request to list datapoints from a dataset with pagination and filters.
/// Used by the `POST /v1/datasets/{dataset_id}/list_datapoints` endpoint.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct ListDatapointsRequest {
    /// Optional function name to filter datapoints by.
    /// If provided, only datapoints from this function will be returned.
    pub function_name: Option<String>,

    /// The maximum number of datapoints to return.
    /// Defaults to 20.
    pub page_size: Option<u32>,

    /// The number of datapoints to skip before starting to return results.
    /// Defaults to 0.
    pub offset: Option<u32>,

    /// Optional filter to apply when querying datapoints.
    /// Supports filtering by tags, time, and logical combinations (AND/OR/NOT).
    pub filter: Option<DatapointFilter>,
}

/// Request to get specific datapoints by their IDs.
/// Used by the `POST /v1/datasets/get_datapoints` endpoint.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct GetDatapointsRequest {
    /// The IDs of the datapoints to retrieve. Required.
    pub ids: Vec<Uuid>,
}

/// Response containing the requested datapoints.
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct GetDatapointsResponse {
    /// The retrieved datapoints.
    pub datapoints: Vec<Datapoint>,
}

/// Specifies the source of the output for the datapoint when creating datapoints from inferences.
/// - `None`: Do not include any output in the datapoint.
/// - `Inference`: Include the original inference output in the datapoint.
/// - `Demonstration`: Include the latest demonstration feedback as output in the datapoint.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
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
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
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
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CreateDatapointsFromInferenceRequestParams {
    /// Create datapoints from specific inference IDs.
    InferenceIds {
        /// The inference IDs to create datapoints from.
        inference_ids: Vec<Uuid>,
    },

    /// Create datapoints from an inference query.
    InferenceQuery {
        /// The function name to filter inferences by.
        function_name: String,

        /// Variant name to filter inferences by, optional.
        #[cfg_attr(test, ts(optional))]
        variant_name: Option<String>,

        /// Filters to apply when querying inferences, optional.
        #[cfg_attr(test, ts(optional))]
        filters: Option<InferenceFilter>,
    },
}

/// Response from creating datapoints (either manually or from inferences).
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct CreateDatapointsResponse {
    /// The IDs of the newly-generated datapoints.
    pub ids: Vec<Uuid>,
}

/// Request to create datapoints manually.
/// Used by the `POST /v1/datasets/{dataset_id}/datapoints` endpoint.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct CreateDatapointsRequest {
    /// The datapoints to create.
    pub datapoints: Vec<CreateDatapointRequest>,
}

/// A tagged request to create a single datapoint.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, tag = "type", rename_all = "snake_case"))]
pub enum CreateDatapointRequest {
    /// Request to create a chat datapoint.
    Chat(CreateChatDatapointRequest),
    /// Request to create a JSON datapoint.
    Json(CreateJsonDatapointRequest),
}

/// A request to create a chat datapoint.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct CreateChatDatapointRequest {
    /// The function name for this datapoint. Required.
    pub function_name: String,

    /// Episode ID that the datapoint belongs to. Optional.
    #[serde(default)]
    pub episode_id: Option<Uuid>,

    /// Input to the function. Required.
    pub input: Input,

    /// Chat datapoint output. Optional.
    /// This can be provided as either a string or a list of content blocks.
    /// If provided, it will be validated against the function's output schema.
    #[serde(default)]
    pub output: Option<Value>,

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

impl CreateChatDatapointRequest {
    /// Validates and prepares this request for insertion into the database.
    /// Returns the datapoint insert struct and the generated UUID.
    pub async fn prepare(
        self,
        config: &crate::config::Config,
        fetch_context: &crate::inference::types::FetchContext<'_>,
        dataset_name: &str,
    ) -> Result<(crate::db::datasets::ChatInferenceDatapointInsert, Uuid), crate::error::Error> {
        use crate::endpoints::feedback::{
            validate_parse_demonstration, DemonstrationOutput, DynamicDemonstrationInfo,
        };
        use crate::error::{Error, ErrorDetails};
        use crate::function::FunctionConfig;

        // Validate function exists and is a chat function
        let function_config = config.get_function(&self.function_name)?;
        let FunctionConfig::Chat(_) = &**function_config else {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!(
                    "Function '{}' is not configured as a chat function",
                    self.function_name
                ),
            }));
        };

        // Validate and convert input
        function_config.validate_input(&self.input)?;
        let stored_input = self
            .input
            .into_lazy_resolved_input(crate::inference::types::FetchContext {
                client: fetch_context.client,
                object_store_info: fetch_context.object_store_info,
            })?
            .into_stored_input(fetch_context.object_store_info)
            .await?;

        // Prepare the tool config
        let tool_config =
            function_config.prepare_tool_config(self.dynamic_tool_params, &config.tools)?;
        let dynamic_demonstration_info =
            DynamicDemonstrationInfo::Chat(tool_config.clone().unwrap_or_default());

        // Validate and parse output if provided
        let output = if let Some(output_value) = self.output {
            let validated_output =
                validate_parse_demonstration(&function_config, &output_value, dynamic_demonstration_info)
                    .await?;

            let DemonstrationOutput::Chat(output) = validated_output else {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: "Expected chat output from validate_parse_demonstration".to_string(),
                }));
            };

            Some(output)
        } else {
            None
        };

        let id = Uuid::now_v7();

        let insert = crate::db::datasets::ChatInferenceDatapointInsert {
            dataset_name: dataset_name.to_string(),
            function_name: self.function_name,
            name: self.name,
            id,
            episode_id: self.episode_id,
            input: stored_input,
            output,
            tool_params: tool_config.as_ref().map(|x| x.clone().into()),
            tags: self.tags,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        Ok((insert, id))
    }
}

/// A request to create a JSON datapoint.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct CreateJsonDatapointRequest {
    /// The function name for this datapoint. Required.
    pub function_name: String,

    /// Episode ID that the datapoint belongs to. Optional.
    #[serde(default)]
    pub episode_id: Option<Uuid>,

    /// Input to the function. Required.
    pub input: Input,

    /// JSON datapoint output. Optional.
    /// If provided, it will be validated against the output_schema (or the function's output schema if not provided).
    #[serde(default)]
    pub output: Option<Value>,

    /// The output schema of the JSON datapoint. Optional.
    /// If not provided, the function's output schema will be used.
    #[serde(default)]
    pub output_schema: Option<Value>,

    /// Tags associated with this datapoint. Optional.
    #[serde(default)]
    pub tags: Option<HashMap<String, String>>,

    /// Human-readable name for the datapoint. Optional.
    #[serde(default)]
    pub name: Option<String>,
}

impl CreateJsonDatapointRequest {
    /// Validates and prepares this request for insertion into the database.
    /// Returns the datapoint insert struct and the generated UUID.
    pub async fn prepare(
        self,
        config: &crate::config::Config,
        fetch_context: &crate::inference::types::FetchContext<'_>,
        dataset_name: &str,
    ) -> Result<(crate::db::datasets::JsonInferenceDatapointInsert, Uuid), crate::error::Error> {
        use crate::endpoints::feedback::{
            validate_parse_demonstration, DemonstrationOutput, DynamicDemonstrationInfo,
        };
        use crate::error::{Error, ErrorDetails};
        use crate::function::FunctionConfig;
        use crate::inference::types::JsonInferenceOutput;

        // Validate function exists and is a JSON function
        let function_config = config.get_function(&self.function_name)?;
        let FunctionConfig::Json(json_function_config) = &**function_config else {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!(
                    "Function '{}' is not configured as a JSON function",
                    self.function_name
                ),
            }));
        };

        // Validate and convert input
        function_config.validate_input(&self.input)?;
        let stored_input = self
            .input
            .into_lazy_resolved_input(crate::inference::types::FetchContext {
                client: fetch_context.client,
                object_store_info: fetch_context.object_store_info,
            })?
            .into_stored_input(fetch_context.object_store_info)
            .await?;

        // Determine the output schema (use provided or default to function's schema)
        let output_schema = self
            .output_schema
            .unwrap_or_else(|| json_function_config.output_schema.value.clone());
        let dynamic_demonstration_info = DynamicDemonstrationInfo::Json(output_schema.clone());

        // Validate and parse output if provided
        let output = if let Some(output_value) = self.output {
            let validated_output =
                validate_parse_demonstration(&function_config, &output_value, dynamic_demonstration_info)
                    .await?;

            let DemonstrationOutput::Json(output) = validated_output else {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: "Expected JSON output from validate_parse_demonstration".to_string(),
                }));
            };

            Some(JsonInferenceOutput {
                raw: output
                    .get("raw")
                    .and_then(|v| v.as_str().map(str::to_string)),
                parsed: output.get("parsed").cloned(),
            })
        } else {
            None
        };

        let id = Uuid::now_v7();

        let insert = crate::db::datasets::JsonInferenceDatapointInsert {
            dataset_name: dataset_name.to_string(),
            function_name: self.function_name,
            name: self.name,
            id,
            episode_id: self.episode_id,
            input: stored_input,
            output,
            output_schema,
            tags: self.tags,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        Ok((insert, id))
    }
}

/// Request to delete datapoints from a dataset.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct DeleteDatapointsRequest {
    /// The IDs of the datapoints to delete.
    pub ids: Vec<Uuid>,
}

/// Response containing the number of deleted datapoints.
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct DeleteDatapointsResponse {
    /// The number of deleted datapoints.
    pub num_deleted_datapoints: u64,
}
