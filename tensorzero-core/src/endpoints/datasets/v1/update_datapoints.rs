use std::collections::{HashMap, HashSet};

use axum::extract::{Path, State};
use axum::Json;
use chrono::Utc;
use serde::Deserialize;
use tracing::instrument;
use uuid::Uuid;

use crate::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries, GetDatapointsParams,
    JsonInferenceDatapointInsert,
};
use crate::endpoints::datasets::{
    validate_dataset_name, StoredChatInferenceDatapoint, StoredDatapoint,
    StoredJsonInferenceDatapoint, CLICKHOUSE_DATETIME_FORMAT,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{FetchContext, Input};
use crate::jsonschema_util::{DynamicJSONSchema, JsonSchemaRef};
use crate::tool::apply_dynamic_tool_params_update_to_tool_call_config;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{
    UpdateChatDatapointRequest, UpdateDatapointRequest, UpdateDatapointsMetadataRequest,
    UpdateDatapointsRequest, UpdateDatapointsResponse, UpdateJsonDatapointRequest,
};

impl UpdateDatapointRequest {
    pub fn id(&self) -> Uuid {
        match self {
            UpdateDatapointRequest::Chat(chat) => chat.id,
            UpdateDatapointRequest::Json(json) => json.id,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct UpdateDatapointsPathParams {
    pub dataset_name: String,
}

#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.update_datapoints", skip(app_state, request))]
pub async fn update_datapoints_handler(
    State(app_state): AppState,
    Path(path_params): Path<UpdateDatapointsPathParams>,
    StructuredJson(request): StructuredJson<UpdateDatapointsRequest>,
) -> Result<Json<UpdateDatapointsResponse>, Error> {
    let response = update_datapoints(&app_state, &path_params.dataset_name, request).await?;
    Ok(Json(response))
}

/// Business logic for updating datapoints in a dataset.
/// This function validates the request, fetches existing datapoints, prepares updates,
/// and inserts the updated datapoints into ClickHouse.
///
/// Returns an error if there are no datapoints, or if there are duplicate datapoint IDs.
pub async fn update_datapoints(
    app_state: &AppStateData,
    dataset_name: &str,
    request: UpdateDatapointsRequest,
) -> Result<UpdateDatapointsResponse, Error> {
    validate_dataset_name(dataset_name)?;

    if request.datapoints.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "At least one datapoint must be provided".to_string(),
        }));
    }

    let mut seen_ids = HashSet::new();
    for datapoint in &request.datapoints {
        if !seen_ids.insert(datapoint.id()) {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Duplicate datapoint id provided: {}", datapoint.id()),
            }));
        }
    }

    let fetch_context = FetchContext {
        client: &app_state.http_client,
        object_store_info: &app_state.config.object_store_info,
    };

    // Fetch all datapoints in a single batch query
    let datapoint_ids: Vec<Uuid> = request
        .datapoints
        .iter()
        .map(UpdateDatapointRequest::id)
        .collect();
    let datapoints_vec = app_state
        .clickhouse_connection_info
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.to_string()),
            function_name: None,
            ids: Some(datapoint_ids),
            limit: u32::MAX, // No limit - fetch all matching datapoints
            offset: 0,
            allow_stale: false,
            filter: None, // No filtering when updating datapoints
            order_by: None,
            search_query_experimental: None,
        })
        .await?;

    // Build a HashMap to construct new DatapointInserts
    let mut datapoints_map: HashMap<Uuid, StoredDatapoint> =
        datapoints_vec.into_iter().map(|dp| (dp.id(), dp)).collect();

    // Each update will produce two DatapointInserts: one stale and one updated
    let mut datapoints: Vec<DatapointInsert> = Vec::with_capacity(request.datapoints.len() * 2);
    let mut new_ids: Vec<Uuid> = Vec::with_capacity(request.datapoints.len());

    // Create a timestamp for all the staled_at fields in the query.
    let now_timestamp = Utc::now().format(CLICKHOUSE_DATETIME_FORMAT).to_string();

    for update in request.datapoints {
        let datapoint_id = update.id();
        let existing = datapoints_map.remove(&datapoint_id).ok_or_else(|| {
            Error::new(ErrorDetails::DatapointNotFound {
                dataset_name: dataset_name.to_string(),
                datapoint_id,
            })
        })?;

        match (update, existing) {
            (UpdateDatapointRequest::Chat(_), StoredDatapoint::Json(_)) => {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Datapoint {datapoint_id} is a JSON datapoint but a chat update was provided"
                    ),
                }));
            }
            (UpdateDatapointRequest::Json(_), StoredDatapoint::Chat(_)) => {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Datapoint {datapoint_id} is a chat datapoint but a JSON update was provided"
                    ),
                }));
            }
            (UpdateDatapointRequest::Chat(update), StoredDatapoint::Chat(existing)) => {
                let prepared = prepare_chat_update(
                    app_state,
                    &fetch_context,
                    dataset_name,
                    update,
                    existing,
                    &now_timestamp,
                )
                .await?;
                datapoints.extend([prepared.stale, prepared.updated]);
                new_ids.push(prepared.new_id);
            }
            (UpdateDatapointRequest::Json(update), StoredDatapoint::Json(existing)) => {
                let prepared = prepare_json_update(
                    app_state,
                    &fetch_context,
                    dataset_name,
                    update,
                    existing,
                    &now_timestamp,
                )
                .await?;
                datapoints.extend([prepared.stale, prepared.updated]);
                new_ids.push(prepared.new_id);
            }
        }
    }

    app_state
        .clickhouse_connection_info
        .insert_datapoints(&datapoints)
        .await?;
    Ok(UpdateDatapointsResponse { ids: new_ids })
}

#[derive(Debug)]
struct PreparedUpdate {
    stale: DatapointInsert,
    updated: DatapointInsert,
    new_id: Uuid,
}

async fn prepare_chat_update(
    app_state: &AppStateData,
    fetch_context: &FetchContext<'_>,
    dataset_name: &str,
    update: UpdateChatDatapointRequest,
    existing_datapoint: StoredChatInferenceDatapoint,
    now_timestamp: &str,
) -> Result<PreparedUpdate, Error> {
    if existing_datapoint.dataset_name != dataset_name {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Datapoint {} belongs to dataset '{}' instead of '{}'",
                existing_datapoint.id, existing_datapoint.dataset_name, dataset_name
            ),
        }));
    }

    // If provided, convert the provided input into a StoredInput.
    let function_config = app_state
        .config
        .get_function(&existing_datapoint.function_name)?;
    let FunctionConfig::Chat(_) = &**function_config else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Function '{}' is not configured as a chat function",
                existing_datapoint.function_name
            ),
        }));
    };

    let updated_datapoint_id = Uuid::now_v7();

    // Update old datapoint as staled, and create new datapoint.
    let mut staled_existing_datapoint: ChatInferenceDatapointInsert =
        existing_datapoint.clone().into();
    staled_existing_datapoint.staled_at = Some(now_timestamp.to_owned());

    // Update the datapoint with new data
    let mut updated_datapoint: ChatInferenceDatapointInsert = existing_datapoint.into();
    updated_datapoint.id = updated_datapoint_id;
    updated_datapoint.is_custom = true;

    let maybe_new_input =
        convert_input_to_stored_input(update.input, fetch_context, function_config.as_ref())
            .await?;
    if let Some(new_input) = maybe_new_input {
        updated_datapoint.input = new_input;
    }

    if let Some(new_output) = update.output {
        updated_datapoint.output = new_output;
    }

    // Apply the dynamic tool params update to the tool call config.
    updated_datapoint.tool_params = apply_dynamic_tool_params_update_to_tool_call_config(
        updated_datapoint.tool_params,
        update.tool_params,
        function_config.as_ref(),
        &app_state.config.tools,
    )?;

    if let Some(new_tags) = update.tags {
        updated_datapoint.tags = Some(new_tags);
    }
    if let Some(new_name) = update.metadata.name {
        updated_datapoint.name = new_name;
    }

    Ok(PreparedUpdate {
        new_id: updated_datapoint_id,
        stale: DatapointInsert::Chat(staled_existing_datapoint),
        updated: DatapointInsert::Chat(updated_datapoint),
    })
}

async fn prepare_json_update(
    app_state: &AppStateData,
    fetch_context: &FetchContext<'_>,
    dataset_name: &str,
    update: UpdateJsonDatapointRequest,
    existing_datapoint: StoredJsonInferenceDatapoint,
    now_timestamp: &str,
) -> Result<PreparedUpdate, Error> {
    if existing_datapoint.dataset_name != dataset_name {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Datapoint {} belongs to dataset '{}' instead of '{}'",
                existing_datapoint.id, existing_datapoint.dataset_name, dataset_name
            ),
        }));
    }

    // If provided, convert the provided input into a StoredInput.
    let function_config = app_state
        .config
        .get_function(&existing_datapoint.function_name)?;
    let FunctionConfig::Json(_) = &**function_config else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Function '{}' is not configured as a JSON function",
                existing_datapoint.function_name
            ),
        }));
    };

    let updated_datapoint_id = Uuid::now_v7();

    // Update old datapoint as staled, and create new datapoint.
    let mut staled_existing_datapoint: JsonInferenceDatapointInsert =
        existing_datapoint.clone().into();
    staled_existing_datapoint.staled_at = Some(now_timestamp.to_owned());

    // Update the datapoint with new data
    let mut updated_datapoint: JsonInferenceDatapointInsert = existing_datapoint.into();
    updated_datapoint.id = updated_datapoint_id;
    updated_datapoint.is_custom = true;

    let maybe_new_input =
        convert_input_to_stored_input(update.input, fetch_context, function_config.as_ref())
            .await?;
    if let Some(new_input) = maybe_new_input {
        updated_datapoint.input = new_input;
    }

    // Validate and update output_schema if provided
    let output_schema = if let Some(new_output_schema) = update.output_schema {
        // Validate the new schema by converting it to DynamicJSONSchema
        let schema_str = serde_json::to_string(&new_output_schema).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize output_schema: {e}"),
            })
        })?;
        let validated_schema = DynamicJSONSchema::parse_from_str(&schema_str)?;
        // Ensure the schema is valid by forcing compilation
        validated_schema.ensure_valid().await?;
        updated_datapoint.output_schema = new_output_schema;
        validated_schema
    } else {
        // Use existing schema, convert it to DynamicJSONSchema
        let schema_str = serde_json::to_string(&updated_datapoint.output_schema).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize existing output_schema: {e}"),
            })
        })?;
        let schema = DynamicJSONSchema::parse_from_str(&schema_str)?;
        // Ensure the schema is valid by forcing compilation
        schema.ensure_valid().await?;
        schema
    };

    // Validate the output against the output schema. If the output is invalid, we only store the raw output.
    if let Some(new_output) = update.output {
        updated_datapoint.output = match new_output {
            Some(output) => Some(
                output
                    .into_json_inference_output(JsonSchemaRef::Dynamic(&output_schema))
                    .await,
            ),
            None => None,
        };
    }

    if let Some(new_tags) = update.tags {
        updated_datapoint.tags = Some(new_tags);
    }

    if let Some(new_name) = update.metadata.name {
        updated_datapoint.name = new_name;
    }

    Ok(PreparedUpdate {
        new_id: updated_datapoint_id,
        stale: DatapointInsert::Json(staled_existing_datapoint),
        updated: DatapointInsert::Json(updated_datapoint),
    })
}

async fn convert_input_to_stored_input(
    input: Option<Input>,
    fetch_context: &FetchContext<'_>,
    function_config: &FunctionConfig,
) -> Result<Option<StoredInput>, Error> {
    match input {
        None => Ok(None),
        Some(input) => {
            function_config.validate_input(&input)?;

            // If the input file is already in ObjectStorage format, do not resolve; and directly skip into StoredInput.
            // Otherwise, if the file needs to be fetched (because it's new), fetch it and convert to StoredInput.
            // This all happens behind the scene in `into_stored_input()`.
            let stored_input = input
                .into_lazy_resolved_input(fetch_context)?
                // This call may trigger requests to write newly-provided files to object storage.
                //
                // TODO(shuyangli): consider refactoring file writing logic so it's hard to forget making these calls.
                // Should we put it into `into_stored_file`?
                .into_stored_input(fetch_context.object_store_info)
                .await?;
            Ok(Some(stored_input))
        }
    }
}

// ============================================================================
// Update Datapoint Metadata Endpoint
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct UpdateDatapointsMetadataPathParams {
    pub dataset_name: String,
}

#[axum::debug_handler(state = AppStateData)]
#[instrument(
    name = "datasets.v1.update_datapoints_metadata",
    skip(app_state, request)
)]
pub async fn update_datapoints_metadata_handler(
    State(app_state): AppState,
    Path(path_params): Path<UpdateDatapointsMetadataPathParams>,
    StructuredJson(request): StructuredJson<UpdateDatapointsMetadataRequest>,
) -> Result<Json<UpdateDatapointsResponse>, Error> {
    let response = update_datapoints_metadata(
        &app_state.clickhouse_connection_info,
        &path_params.dataset_name,
        request,
    )
    .await?;
    Ok(Json(response))
}

/// Business logic for updating datapoint metadata in a dataset.
/// This function only updates metadata fields (like name) without creating new datapoint IDs.
/// Unlike update_datapoints, this does NOT stale the old datapoint or create a new ID.
pub async fn update_datapoints_metadata(
    clickhouse_handler: &impl DatasetQueries,
    dataset_name: &str,
    request: UpdateDatapointsMetadataRequest,
) -> Result<UpdateDatapointsResponse, Error> {
    validate_dataset_name(dataset_name)?;

    if request.datapoints.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "At least one datapoint must be provided".to_string(),
        }));
    }

    let mut seen_ids = HashSet::new();
    for datapoint in &request.datapoints {
        if !seen_ids.insert(datapoint.id) {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Duplicate datapoint id provided: {}", datapoint.id),
            }));
        }
    }

    // Fetch all datapoints in a single batch query
    let datapoint_ids: Vec<Uuid> = request.datapoints.iter().map(|d| d.id).collect();
    let datapoints_vec = clickhouse_handler
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.to_string()),
            function_name: None,
            ids: Some(datapoint_ids.clone()),
            limit: u32::MAX,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await?;

    // Build a HashMap for quick lookup
    let mut datapoints_map: HashMap<Uuid, StoredDatapoint> =
        datapoints_vec.into_iter().map(|dp| (dp.id(), dp)).collect();

    let mut datapoints: Vec<DatapointInsert> = Vec::with_capacity(request.datapoints.len());

    for update in request.datapoints {
        let datapoint_id = update.id;
        let existing = datapoints_map.remove(&datapoint_id).ok_or_else(|| {
            Error::new(ErrorDetails::DatapointNotFound {
                dataset_name: dataset_name.to_string(),
                datapoint_id,
            })
        })?;

        match existing {
            StoredDatapoint::Chat(mut existing_datapoint) => {
                if existing_datapoint.dataset_name != dataset_name {
                    return Err(Error::new(ErrorDetails::InvalidRequest {
                        message: format!(
                            "Datapoint {datapoint_id} belongs to dataset '{}' instead of '{dataset_name}'",
                            existing_datapoint.dataset_name
                        ),
                    }));
                }

                if let Some(new_name) = update.metadata.name {
                    existing_datapoint.name = new_name;
                }

                datapoints.push(DatapointInsert::Chat(existing_datapoint.into()));
            }
            StoredDatapoint::Json(mut existing_datapoint) => {
                if existing_datapoint.dataset_name != dataset_name {
                    return Err(Error::new(ErrorDetails::InvalidRequest {
                        message: format!(
                            "Datapoint {datapoint_id} belongs to dataset '{}' instead of '{dataset_name}'",
                            existing_datapoint.dataset_name
                        ),
                    }));
                }

                if let Some(new_name) = update.metadata.name {
                    existing_datapoint.name = new_name;
                }

                datapoints.push(DatapointInsert::Json(existing_datapoint.into()));
            }
        }
    }

    clickhouse_handler.insert_datapoints(&datapoints).await?;

    // Return the same IDs (not new ones, since we didn't create new datapoints)
    Ok(UpdateDatapointsResponse { ids: datapoint_ids })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ObjectStoreInfo, SchemaData};
    use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
    use crate::endpoints::datasets::v1::types::{
        DatapointMetadataUpdate, JsonDatapointOutputUpdate,
    };
    use crate::endpoints::datasets::StoredChatInferenceDatapoint;
    use crate::experimentation::ExperimentationConfig;
    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::http::TensorzeroHttpClient;
    use crate::inference::types::storage::{StorageKind, StoragePath};
    use crate::inference::types::{
        Base64File, ContentBlockChatOutput, File, Input, InputMessage, InputMessageContent,
        JsonInferenceOutput, ObjectStoragePointer, Role, StoredInputMessage,
        StoredInputMessageContent, Text,
    };
    use crate::jsonschema_util::StaticJSONSchema;
    use crate::tool::{AllowedTools, AllowedToolsChoice, ToolCallConfigDatabaseInsert, ToolChoice};
    use crate::utils::gateway::{AppStateData, GatewayHandle, GatewayHandleTestOptions};
    use object_store::path::Path as ObjectStorePath;
    use serde_json::json;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    mod file_conversion_tests {
        use super::*;

        #[tokio::test]
        async fn test_convert_input_with_object_storage_does_not_refetch() {
            // This test verifies that File::ObjectStorage inputs bypass object storage access entirely.
            // We use StorageKind::Disabled with object_store: None to ensure that if the code
            // tried to access object storage, it would fail. The fact that this test passes
            // proves that File::ObjectStorage is handled specially and never triggers storage access.

            // TODO(shuyangli): Provide proper object storage mocks for tests. This requires a mock for
            // `ObjectStore` which we don't own, so it's a little complicated.

            // Create a File::ObjectStorage that should NOT be fetched
            let storage_path = StoragePath {
                kind: StorageKind::Disabled,
                path: ObjectStorePath::parse("test/path/image.png").unwrap(),
            };

            let file = File::ObjectStoragePointer(ObjectStoragePointer {
                source_url: Some("https://example.com/original.png".parse().unwrap()),
                mime_type: mime::IMAGE_PNG,
                storage_path: storage_path.clone(),
                detail: None,
                filename: None,
            });

            let input = Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::File(file.clone())],
                }],
            };

            // Create minimal function config
            let function_config = FunctionConfig::Chat(FunctionConfigChat {
                variants: HashMap::new(),
                schemas: SchemaData::default(),
                tools: vec![],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: Some(true),
                description: None,
                experimentation: ExperimentationConfig::default(),
                all_explicit_templates_names: HashSet::new(),
            });

            // Create fetch context with NO actual object storage info.
            // If the code tries to access object storage, it will fail with an error.
            let http_client = TensorzeroHttpClient::new_testing().unwrap();
            let object_store_info: Option<ObjectStoreInfo> = None;
            let fetch_context = FetchContext {
                client: &http_client,
                object_store_info: &object_store_info,
            };

            // Convert input to stored input
            // This succeeds ONLY because File::ObjectStorage bypasses storage access
            let result =
                convert_input_to_stored_input(Some(input), &fetch_context, &function_config)
                    .await
                    .unwrap();

            // Verify the result
            assert!(result.is_some());
            let stored_input = result.unwrap();
            assert_eq!(stored_input.messages.len(), 1);
            assert_eq!(stored_input.messages[0].content.len(), 1);

            // Verify that the File::ObjectStorage was passed through without fetching
            match &stored_input.messages[0].content[0] {
                StoredInputMessageContent::File(stored_file) => {
                    assert_eq!(stored_file.storage_path.path, storage_path.path);
                    assert_eq!(stored_file.mime_type, mime::IMAGE_PNG);
                    assert_eq!(
                        stored_file.source_url,
                        Some("https://example.com/original.png".parse().unwrap())
                    );
                }
                _ => panic!("Expected File content"),
            }
        }

        #[tokio::test]
        async fn test_convert_input_with_base64_processes_without_actual_storage() {
            // This test verifies that File::Base64 goes through the write_file() code path,
            // but gracefully handles disabled storage (for testing). In contrast,
            // File::ObjectStorage completely bypasses the write_file() path.
            //
            // The key difference tested here vs test_convert_input_with_object_storage_does_not_refetch:
            // - File::ObjectStorage: future is discarded, no async operations, just metadata passthrough
            // - File::Base64: goes through async resolve() -> write_file() -> storage write (or no-op if disabled)

            let file = File::Base64(
                Base64File::new(
                    None,
                    mime::IMAGE_PNG,
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==".to_string(),
                    None,
                    None,
                )
                .expect("test data should be valid"),
            );

            let input = Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::File(file.clone())],
                }],
            };

            // Create minimal function config
            let function_config = FunctionConfig::Chat(FunctionConfigChat {
                variants: HashMap::new(),
                schemas: SchemaData::default(),
                tools: vec![],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: Some(true),
                description: None,
                experimentation: ExperimentationConfig::default(),
                all_explicit_templates_names: HashSet::new(),
            });

            // Create fetch context with disabled storage
            // File::Base64 will call write_file() but it no-ops with disabled storage
            let http_client = TensorzeroHttpClient::new_testing().unwrap();
            let object_store_info = Some(ObjectStoreInfo {
                object_store: None, // Disabled storage - write_file() returns Ok(()) without writing
                kind: StorageKind::Disabled,
            });
            let fetch_context = FetchContext {
                client: &http_client,
                object_store_info: &object_store_info,
            };

            // Convert input to stored input
            // This succeeds because write_file() gracefully handles disabled storage
            let result =
                convert_input_to_stored_input(Some(input), &fetch_context, &function_config)
                    .await
                    .unwrap();

            // Verify the result
            assert!(result.is_some());
            let stored_input = result.unwrap();
            assert_eq!(stored_input.messages.len(), 1);
            assert_eq!(stored_input.messages[0].content.len(), 1);

            // Verify that the File::Base64 was converted to StoredFile
            match &stored_input.messages[0].content[0] {
                StoredInputMessageContent::File(stored_file) => {
                    // Should have been processed into a stored file
                    assert_eq!(stored_file.mime_type, mime::IMAGE_PNG);
                    // With disabled storage, path should still be generated
                    assert!(!stored_file.storage_path.path.as_ref().is_empty());
                    // URL should be None since this came from Base64
                    assert_eq!(stored_file.source_url, None);
                }
                _ => panic!("Expected File content"),
            }
        }
    }

    mod update_utils {
        use super::*;

        /// Helper to create a sample ChatInferenceDatapoint
        pub fn create_sample_chat_datapoint(dataset_name: &str) -> StoredChatInferenceDatapoint {
            StoredChatInferenceDatapoint {
                id: Uuid::now_v7(),
                dataset_name: dataset_name.to_string(),
                function_name: "test_chat_function".to_string(),
                name: Some("test_datapoint".to_string()),
                episode_id: Some(Uuid::now_v7()),
                input: StoredInput {
                    system: None,
                    messages: vec![crate::inference::types::StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text(Text {
                            text: "original input".to_string(),
                        })],
                    }],
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "original output".to_string(),
                })]),
                tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
                    vec![],
                    vec![],
                    AllowedTools {
                        tools: vec![],
                        choice: AllowedToolsChoice::FunctionDefault,
                    },
                    ToolChoice::Auto,
                    Some(true),
                )),
                tags: Some(HashMap::from([("key".to_string(), "value".to_string())])),
                auxiliary: "{}".to_string(),
                staled_at: None,
                source_inference_id: None,
                is_custom: true,
                is_deleted: false,
                updated_at: chrono::Utc::now()
                    .format(CLICKHOUSE_DATETIME_FORMAT)
                    .to_string(),
            }
        }

        /// Helper to create a sample JsonInferenceDatapoint
        pub fn create_sample_json_datapoint(dataset_name: &str) -> StoredJsonInferenceDatapoint {
            StoredJsonInferenceDatapoint {
                id: Uuid::now_v7(),
                dataset_name: dataset_name.to_string(),
                function_name: "test_json_function".to_string(),
                name: Some("test_datapoint".to_string()),
                episode_id: Some(Uuid::now_v7()),
                input: StoredInput {
                    system: None,
                    messages: vec![crate::inference::types::StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text(Text {
                            text: "original input".to_string(),
                        })],
                    }],
                },
                output: Some(JsonInferenceOutput {
                    raw: Some(r#"{"value":"original"}"#.to_string()),
                    parsed: Some(json!({"value": "original"})),
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": false
                }),
                tags: Some(HashMap::from([("key".to_string(), "value".to_string())])),
                auxiliary: "{}".to_string(),
                staled_at: None,
                source_inference_id: None,
                is_custom: true,
                is_deleted: false,
                updated_at: chrono::Utc::now()
                    .format(CLICKHOUSE_DATETIME_FORMAT)
                    .to_string(),
            }
        }
    }

    mod prepare_update_tests {
        use crate::{
            endpoints::datasets::v1::types::UpdateDynamicToolParamsRequest,
            tool::{FunctionTool, Tool, ToolChoice},
        };

        use super::*;

        /// Helper to create a minimal AppStateData for testing
        fn create_test_app_state() -> AppStateData {
            let mut mock_client = MockClickHouseClient::new();
            mock_client.expect_batcher_join_handle().returning(|| None);

            let mut config = Config::default();

            // Add a chat function
            config.functions.insert(
                "test_chat_function".to_string(),
                Arc::new(FunctionConfig::Chat(FunctionConfigChat {
                    variants: HashMap::new(),
                    schemas: SchemaData::default(),
                    tools: vec![],
                    tool_choice: ToolChoice::Auto,
                    parallel_tool_calls: Some(true),
                    description: None,
                    experimentation: ExperimentationConfig::default(),
                    all_explicit_templates_names: HashSet::new(),
                })),
            );

            // Add a JSON function
            config.functions.insert(
                "test_json_function".to_string(),
                Arc::new(FunctionConfig::Json(FunctionConfigJson {
                    variants: HashMap::new(),
                    schemas: SchemaData::default(),
                    output_schema: StaticJSONSchema::from_value(json!({
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": false
                    }))
                    .unwrap(),
                    json_mode_tool_call_config: crate::tool::ToolCallConfig::default(),
                    description: None,
                    experimentation: ExperimentationConfig::default(),
                    all_explicit_template_names: HashSet::new(),
                })),
            );

            let gateway_handle = GatewayHandle::new_unit_test_data(
                Arc::new(config),
                GatewayHandleTestOptions {
                    clickhouse_client: Arc::new(mock_client),
                    postgres_healthy: true,
                },
            );
            gateway_handle.app_state.clone()
        }

        /// Helper to create a sample ChatInferenceDatapoint
        fn create_sample_chat_datapoint(dataset_name: &str) -> StoredChatInferenceDatapoint {
            StoredChatInferenceDatapoint {
                id: Uuid::now_v7(),
                dataset_name: dataset_name.to_string(),
                function_name: "test_chat_function".to_string(),
                name: Some("test_datapoint".to_string()),
                episode_id: Some(Uuid::now_v7()),
                input: StoredInput {
                    system: None,
                    messages: vec![StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text(Text {
                            text: "original input".to_string(),
                        })],
                    }],
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "original output".to_string(),
                })]),
                tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
                    vec![],
                    vec![],
                    AllowedTools {
                        tools: vec![],
                        choice: AllowedToolsChoice::FunctionDefault,
                    },
                    ToolChoice::Auto,
                    Some(true),
                )),
                tags: Some(HashMap::from([("key".to_string(), "value".to_string())])),
                auxiliary: "{}".to_string(),
                staled_at: None,
                source_inference_id: None,
                is_custom: true,
                is_deleted: false,
                updated_at: chrono::Utc::now()
                    .format(CLICKHOUSE_DATETIME_FORMAT)
                    .to_string(),
            }
        }

        /// Helper to create a sample JsonInferenceDatapoint
        fn create_sample_json_datapoint(dataset_name: &str) -> StoredJsonInferenceDatapoint {
            StoredJsonInferenceDatapoint {
                id: Uuid::now_v7(),
                dataset_name: dataset_name.to_string(),
                function_name: "test_json_function".to_string(),
                name: Some("test_datapoint".to_string()),
                episode_id: Some(Uuid::now_v7()),
                input: StoredInput {
                    system: None,
                    messages: vec![StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text(Text {
                            text: "original input".to_string(),
                        })],
                    }],
                },
                output: Some(JsonInferenceOutput {
                    raw: Some(r#"{"value":"original"}"#.to_string()),
                    parsed: Some(json!({"value": "original"})),
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": false
                }),
                tags: Some(HashMap::from([("key".to_string(), "value".to_string())])),
                auxiliary: "{}".to_string(),
                staled_at: None,
                source_inference_id: None,
                is_custom: true,
                is_deleted: false,
                updated_at: chrono::Utc::now()
                    .format(CLICKHOUSE_DATETIME_FORMAT)
                    .to_string(),
            }
        }

        fn create_fetch_context(http_client: &'_ TensorzeroHttpClient) -> FetchContext<'_> {
            FetchContext {
                client: http_client,
                object_store_info: &None,
            }
        }

        // ============================================================================
        // Tests for prepare_chat_update
        // ============================================================================

        #[tokio::test]
        async fn test_prepare_chat_update_no_updates() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_chat_datapoint(dataset_name);
            let original_id = existing.id;

            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                tool_params: UpdateDynamicToolParamsRequest::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing.clone(),
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            // Verify stale datapoint
            let DatapointInsert::Chat(stale) = result.stale else {
                panic!("Expected Chat insert");
            };
            assert_eq!(stale.id, original_id);
            assert_eq!(stale.staled_at, Some("2025-01-01 00:00:00".to_string()));

            // Verify updated datapoint - should have new ID but all fields unchanged
            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };
            assert_ne!(updated.id, original_id);
            assert_eq!(updated.id, result.new_id);
            assert_eq!(updated.name, existing.name);
            assert_eq!(
                updated.output.as_ref().unwrap()[0],
                existing.output.unwrap()[0]
            );
            assert!(updated.tool_params.is_some());
            assert!(updated.tags.is_some());
        }

        #[tokio::test]
        async fn test_prepare_chat_update_input_only() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_chat_datapoint(dataset_name);

            let new_input = Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "new input text".into(),
                    })],
                }],
            };

            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: Some(new_input),
                output: None,
                tool_params: UpdateDynamicToolParamsRequest::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing.clone(),
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };

            // Input should be updated
            assert_eq!(updated.input.messages.len(), 1);
            match &updated.input.messages[0].content[0] {
                StoredInputMessageContent::Text(text) => {
                    assert_eq!(text.text, "new input text");
                }
                _ => panic!("Expected text content"),
            }

            // Other fields unchanged
            assert_eq!(
                updated.output.as_ref().unwrap()[0],
                existing.output.unwrap()[0]
            );
        }

        #[tokio::test]
        async fn test_prepare_chat_update_output_only() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_chat_datapoint(dataset_name);

            let new_output = vec![ContentBlockChatOutput::Text(Text {
                text: "new output".to_string(),
            })];

            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: Some(Some(new_output.clone())),
                tool_params: UpdateDynamicToolParamsRequest::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing.clone(),
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };

            // Output should be updated
            assert_eq!(updated.output, Some(new_output));
        }

        #[tokio::test]
        async fn test_prepare_chat_update_output_set_to_null() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_chat_datapoint(dataset_name);

            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: Some(None),
                tool_params: Default::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: None,
                metadata: Default::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing.clone(),
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };

            // Output should be cleared
            assert_eq!(updated.output, None);
        }

        #[tokio::test]
        async fn test_prepare_chat_update_tool_params_omitted() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_chat_datapoint(dataset_name);
            let original_tool_params = existing.tool_params.clone();

            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                tool_params: UpdateDynamicToolParamsRequest::default(), // Omitted - should remain unchanged
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };

            assert_eq!(updated.tool_params, original_tool_params);
        }

        #[tokio::test]
        async fn test_prepare_chat_update_tool_params_set_to_value() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_chat_datapoint(dataset_name);

            // Create DynamicToolParams directly instead of round-tripping through database_insert_to_dynamic_tool_params
            // This represents a user setting allowed_tools to an empty list with tool_choice None
            // When there are no tools available, the result should be None (tools disabled)
            let new_function_tool = FunctionTool {
                name: "test_tool".to_string(),
                description: "Test tool".to_string(),
                parameters: json!({}),
                strict: false,
            };
            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                tool_params: UpdateDynamicToolParamsRequest {
                    allowed_tools: Some(Some(vec!["test_tool".to_string()])),
                    additional_tools: Some(vec![Tool::Function(new_function_tool.clone())]),
                    tool_choice: Some(Some(ToolChoice::None)),
                    parallel_tool_calls: Some(Some(false)),
                    provider_tools: Some(vec![]),
                },
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };
            let Some(tool_params) = updated.tool_params else {
                panic!("Expected tool params in prepared update");
            };

            // Verify that tool params are transformed correctly into database type
            assert_eq!(
                tool_params.dynamic_tools,
                vec![Tool::Function(new_function_tool)],
                "Dynamic tools should be transformed correctly"
            );
            assert_eq!(
                tool_params.dynamic_provider_tools,
                vec![],
                "Dynamic provider tools should be transformed correctly"
            );
            assert_eq!(
                tool_params.allowed_tools,
                AllowedTools {
                    tools: vec!["test_tool".to_string()],
                    choice: AllowedToolsChoice::Explicit,
                },
                "Allowed tools should be transformed correctly"
            );
            assert_eq!(
                tool_params.tool_choice,
                ToolChoice::None,
                "Tool choice should be transformed correctly"
            );
            assert_eq!(
                tool_params.parallel_tool_calls,
                Some(false),
                "Parallel tool calls should be transformed correctly"
            );

            // Don't check the legacy tool params field.
        }

        #[tokio::test]
        async fn test_prepare_chat_update_tags_cases() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";

            // Case 1: Omitted - should remain unchanged
            let existing = create_sample_chat_datapoint(dataset_name);
            let original_tags = existing.tags.clone();
            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                tool_params: UpdateDynamicToolParamsRequest::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };
            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();
            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };
            assert_eq!(updated.tags, original_tags);

            // Case 2: Set to empty HashMap (will clear tags)
            let existing = create_sample_chat_datapoint(dataset_name);
            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                tool_params: UpdateDynamicToolParamsRequest::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: Some(HashMap::new()),
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };
            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();
            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };
            assert_eq!(updated.tags, Some(HashMap::new()));

            // Case 3: Set to value
            let existing = create_sample_chat_datapoint(dataset_name);
            let new_tags = HashMap::from([("new_key".to_string(), "new_value".to_string())]);
            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                tool_params: UpdateDynamicToolParamsRequest::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: Some(new_tags.clone()),
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };
            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();
            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };
            assert_eq!(updated.tags, Some(new_tags));
        }

        #[tokio::test]
        async fn test_prepare_chat_update_metadata_name_cases() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";

            // Case 1: Metadata omitted - name unchanged
            let existing = create_sample_chat_datapoint(dataset_name);
            let original_name = existing.name.clone();
            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                tool_params: UpdateDynamicToolParamsRequest::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };
            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();
            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };
            assert_eq!(updated.name, original_name);

            // Case 2: Metadata.name set to null
            let existing = create_sample_chat_datapoint(dataset_name);
            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                tool_params: UpdateDynamicToolParamsRequest::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: None,
                metadata: DatapointMetadataUpdate { name: Some(None) },
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };
            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();
            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };
            assert_eq!(updated.name, None);

            // Case 3: Metadata.name set to value
            let existing = create_sample_chat_datapoint(dataset_name);
            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                tool_params: UpdateDynamicToolParamsRequest::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: None,
                metadata: DatapointMetadataUpdate {
                    name: Some(Some("new_name".to_string())),
                },
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };
            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();
            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };
            assert_eq!(updated.name, Some("new_name".to_string()));
        }

        #[tokio::test]
        async fn test_prepare_chat_update_all_fields() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_chat_datapoint(dataset_name);

            let new_input = Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "new input".into(),
                    })],
                }],
            };
            let new_output = vec![ContentBlockChatOutput::Text(Text {
                text: "new output".to_string(),
            })];
            let new_tags = HashMap::from([("new".to_string(), "tag".to_string())]);

            // Create DynamicToolParams directly instead of round-tripping
            // Setting allowed_tools to empty list means "no tools" which results in None
            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: Some(new_input),
                output: Some(Some(new_output.clone())),
                tool_params: UpdateDynamicToolParamsRequest {
                    allowed_tools: Some(Some(vec![])),
                    additional_tools: None,
                    tool_choice: Some(Some(ToolChoice::None)),
                    parallel_tool_calls: Some(Some(false)),
                    provider_tools: Some(vec![]),
                },
                tags: Some(new_tags.clone()),
                metadata: DatapointMetadataUpdate {
                    name: Some(Some("updated_name".to_string())),
                },
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };

            // Verify all fields were updated
            assert_eq!(updated.output, Some(new_output));
            assert_eq!(updated.tool_params, None); // Empty allowed_tools results in None
            assert_eq!(updated.tags, Some(new_tags));
            assert_eq!(updated.name, Some("updated_name".to_string()));
        }

        // ============================================================================
        // Tests for prepare_json_update
        // ============================================================================

        #[tokio::test]
        async fn test_prepare_json_update_no_updates() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_json_datapoint(dataset_name);
            let original_id = existing.id;

            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                output_schema: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_json_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing.clone(),
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            // Verify stale datapoint
            let DatapointInsert::Json(stale) = result.stale else {
                panic!("Expected Json insert");
            };
            assert_eq!(stale.id, original_id);
            assert_eq!(stale.staled_at, Some("2025-01-01 00:00:00".to_string()));

            // Verify updated datapoint
            let DatapointInsert::Json(updated) = result.updated else {
                panic!("Expected Json insert");
            };
            assert_ne!(updated.id, original_id);
            assert_eq!(updated.name, existing.name);
            assert_eq!(
                updated.output.as_ref().unwrap().parsed,
                existing.output.unwrap().parsed
            );
        }

        #[tokio::test]
        async fn test_prepare_json_update_output_omitted() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_json_datapoint(dataset_name);
            let original_output = existing.output.clone();

            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: None,
                output: None, // Omitted
                output_schema: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_json_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Json(updated) = result.updated else {
                panic!("Expected Json insert");
            };

            assert_eq!(updated.output, original_output);
        }

        #[tokio::test]
        async fn test_prepare_json_update_output_set_to_null() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_json_datapoint(dataset_name);

            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: None,
                output: Some(None), // Set to null
                output_schema: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_json_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Json(updated) = result.updated else {
                panic!("Expected Json insert");
            };

            assert_eq!(updated.output, None);
        }

        #[tokio::test]
        async fn test_prepare_json_update_output_set_to_value() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_json_datapoint(dataset_name);

            let new_output_value = json!({"value": "new"});

            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: None,
                output: Some(Some(JsonDatapointOutputUpdate {
                    raw: Some(serde_json::to_string(&new_output_value).unwrap()),
                })),
                output_schema: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_json_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Json(updated) = result.updated else {
                panic!("Expected Json insert");
            };

            assert_eq!(
                updated.output.as_ref().unwrap().parsed,
                Some(new_output_value)
            );
            assert_eq!(
                updated.output.as_ref().unwrap().raw,
                Some(r#"{"value":"new"}"#.to_string())
            );
        }

        #[tokio::test]
        async fn test_prepare_json_update_output_schema_only() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_json_datapoint(dataset_name);
            let original_output = existing.output.clone();

            let new_schema =
                json!({"type": "object", "properties": {"newField": {"type": "number"}}});

            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                output_schema: Some(new_schema.clone()),
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_json_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Json(updated) = result.updated else {
                panic!("Expected Json insert");
            };

            assert_eq!(updated.output_schema, new_schema);
            assert_eq!(updated.output, original_output);
        }

        #[tokio::test]
        async fn test_prepare_json_update_output_schema_and_output() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_json_datapoint(dataset_name);

            let new_schema = json!({"type": "object", "properties": {"count": {"type": "number"}}});
            let new_output = json!({"count": 42});

            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: None,
                output: Some(Some(JsonDatapointOutputUpdate {
                    raw: Some(serde_json::to_string(&new_output).unwrap()),
                })),
                output_schema: Some(new_schema.clone()),
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_json_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Json(updated) = result.updated else {
                panic!("Expected Json insert");
            };

            assert_eq!(updated.output_schema, new_schema);
            assert_eq!(updated.output.as_ref().unwrap().parsed, Some(new_output));
        }

        #[tokio::test]
        async fn test_prepare_json_update_output_validation_failure() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_json_datapoint(dataset_name);

            // Output doesn't match the schema (expects {value: string}, providing {count: number})
            let bad_output = json!({"count": 123});

            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: None,
                output: Some(Some(JsonDatapointOutputUpdate {
                    raw: Some(serde_json::to_string(&bad_output).unwrap()),
                })),
                output_schema: None, // Will use existing schema which expects {value: string}
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_json_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Json(updated) = result.updated else {
                panic!("Expected Json insert");
            };

            assert_eq!(updated.output.as_ref().unwrap().parsed, None);
        }

        #[tokio::test]
        async fn test_prepare_json_update_invalid_output_schema() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_json_datapoint(dataset_name);

            // Provide an invalid schema
            let invalid_schema = json!({
                "type": "invalid_type",  // This is not a valid JSON Schema type
                "properties": {"value": {"type": "string"}}
            });

            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                output_schema: Some(invalid_schema),
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_json_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await;

            // Should return an error because the schema is invalid
            assert!(result.is_err());
            let error = result.unwrap_err();
            // Verify the error is related to JSON schema validation
            assert!(matches!(
                error.get_details(),
                ErrorDetails::DynamicJsonSchema { .. }
            ));
        }

        #[tokio::test]
        async fn test_prepare_json_update_tags_cases() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";

            // Similar to chat tests - omitted, null, value
            let existing = create_sample_json_datapoint(dataset_name);
            let original_tags = existing.tags.clone();

            // Omitted
            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                output_schema: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };
            let result = prepare_json_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();
            let DatapointInsert::Json(updated) = result.updated else {
                panic!("Expected Json insert");
            };
            assert_eq!(updated.tags, original_tags);
        }

        #[tokio::test]
        async fn test_prepare_json_update_all_fields() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_json_datapoint(dataset_name);

            let new_input = Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "new json input".into(),
                    })],
                }],
            };
            let new_schema =
                json!({"type": "object", "properties": {"result": {"type": "boolean"}}});
            let new_output = json!({"result": true});
            let new_tags = HashMap::from([("json_tag".to_string(), "value".to_string())]);

            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: Some(new_input),
                output: Some(Some(JsonDatapointOutputUpdate {
                    raw: Some(serde_json::to_string(&new_output).unwrap()),
                })),
                output_schema: Some(new_schema.clone()),
                tags: Some(new_tags.clone()),
                metadata: DatapointMetadataUpdate {
                    name: Some(Some("json_updated".to_string())),
                },
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_json_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Json(updated) = result.updated else {
                panic!("Expected Json insert");
            };

            assert_eq!(updated.output_schema, new_schema);
            assert_eq!(updated.output.as_ref().unwrap().parsed, Some(new_output));
            assert_eq!(updated.tags, Some(new_tags));
            assert_eq!(updated.name, Some("json_updated".to_string()));
        }

        #[tokio::test]
        async fn test_prepare_chat_update_sets_is_custom_true_from_inference() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";

            // Create a datapoint from an inference (is_custom = false, has source_inference_id)
            let mut existing = create_sample_chat_datapoint(dataset_name);
            existing.is_custom = false;
            existing.source_inference_id = Some(Uuid::now_v7());
            let source_inference_id = existing.source_inference_id;

            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: Some(Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "edited output".to_string(),
                })])),
                tool_params: UpdateDynamicToolParamsRequest::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };

            // After update, is_custom should be true
            assert!(updated.is_custom);
            // source_inference_id should be preserved
            assert_eq!(updated.source_inference_id, source_inference_id);
        }

        #[tokio::test]
        async fn test_prepare_chat_update_keeps_is_custom_true() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";

            // Create a custom datapoint (is_custom = true, no source_inference_id)
            let mut existing = create_sample_chat_datapoint(dataset_name);
            existing.is_custom = true;
            existing.source_inference_id = None;

            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: Some(Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "edited output".to_string(),
                })])),
                tool_params: UpdateDynamicToolParamsRequest::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_tool_params: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_chat_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Chat(updated) = result.updated else {
                panic!("Expected Chat insert");
            };

            // is_custom should remain true
            assert!(updated.is_custom);
            // source_inference_id should remain None
            assert_eq!(updated.source_inference_id, None);
        }

        #[tokio::test]
        async fn test_prepare_json_update_sets_is_custom_true_from_inference() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";

            // Create a datapoint from an inference (is_custom = false, has source_inference_id)
            let mut existing = create_sample_json_datapoint(dataset_name);
            existing.is_custom = false;
            existing.source_inference_id = Some(Uuid::now_v7());
            let source_inference_id = existing.source_inference_id;

            let new_output = json!({"value": "edited"});
            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: None,
                output: Some(Some(JsonDatapointOutputUpdate {
                    raw: Some(serde_json::to_string(&new_output).unwrap()),
                })),
                output_schema: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_json_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Json(updated) = result.updated else {
                panic!("Expected Json insert");
            };

            // After update, is_custom should be true
            assert!(updated.is_custom);
            // source_inference_id should be preserved
            assert_eq!(updated.source_inference_id, source_inference_id);
        }

        #[tokio::test]
        async fn test_prepare_json_update_keeps_is_custom_true() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";

            // Create a custom datapoint (is_custom = true, no source_inference_id)
            let mut existing = create_sample_json_datapoint(dataset_name);
            existing.is_custom = true;
            existing.source_inference_id = None;

            let new_output = json!({"value": "edited"});
            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: None,
                output: Some(Some(JsonDatapointOutputUpdate {
                    raw: Some(serde_json::to_string(&new_output).unwrap()),
                })),
                output_schema: None,
                tags: None,
                metadata: DatapointMetadataUpdate::default(),
                #[expect(deprecated)]
                deprecated_do_not_use_metadata: None,
            };

            let result = prepare_json_update(
                &app_state,
                &fetch_context,
                dataset_name,
                update,
                existing,
                "2025-01-01 00:00:00",
            )
            .await
            .unwrap();

            let DatapointInsert::Json(updated) = result.updated else {
                panic!("Expected Json insert");
            };

            // is_custom should remain true
            assert!(updated.is_custom);
            // source_inference_id should remain None
            assert_eq!(updated.source_inference_id, None);
        }
    }

    mod update_datapoints_metadata_tests {
        use super::update_utils::*;
        use super::*;
        use crate::db::datasets::MockDatasetQueries;
        use crate::endpoints::datasets::v1::types::UpdateDatapointMetadataRequest;

        #[tokio::test]
        async fn test_update_metadata_chat_datapoint() {
            let dataset_name = "test_dataset";
            let existing_datapoint = create_sample_chat_datapoint(dataset_name);
            let datapoint_id = existing_datapoint.id;

            let mut mock_db = MockDatasetQueries::new();
            let existing_datapoint_clone = existing_datapoint.clone();
            mock_db.expect_get_datapoints().returning(move |_| {
                let cloned_datapoint = existing_datapoint_clone.clone();
                Box::pin(async move { Ok(vec![StoredDatapoint::Chat(cloned_datapoint)]) })
            });
            mock_db
                .expect_insert_datapoints()
                .withf(move |datapoints_inserts| {
                    assert_eq!(datapoints_inserts.len(), 1, "Expected 1 datapoint insert");
                    let datapoint_insert = &datapoints_inserts[0];
                    // ID should stay the same.
                    assert_eq!(datapoint_insert.id(), datapoint_id);
                    let DatapointInsert::Chat(dp) = datapoint_insert else {
                        panic!("Expected Chat insert");
                    };
                    // Name should be updated.
                    assert_eq!(dp.name, Some("new_name".to_string()));
                    // The other fields should stay the same.
                    assert_eq!(dp.input, existing_datapoint.input);
                    assert_eq!(dp.output, existing_datapoint.output);
                    assert_eq!(dp.tool_params, existing_datapoint.tool_params.clone());
                    assert_eq!(dp.tags, existing_datapoint.tags);
                    assert_eq!(dp.staled_at, existing_datapoint.staled_at);
                    assert_eq!(
                        dp.source_inference_id,
                        existing_datapoint.source_inference_id
                    );
                    true
                })
                .returning(|_| Box::pin(async move { Ok(1) }));

            let request = UpdateDatapointsMetadataRequest {
                datapoints: vec![UpdateDatapointMetadataRequest {
                    id: datapoint_id,
                    metadata: DatapointMetadataUpdate {
                        name: Some(Some("new_name".to_string())),
                    },
                }],
            };

            let result = update_datapoints_metadata(&mock_db, dataset_name, request).await;
            assert!(result.is_ok());
            let response = result.unwrap();
            assert_eq!(response.ids.len(), 1);
            assert_eq!(response.ids[0], datapoint_id);
        }

        #[tokio::test]
        async fn test_update_metadata_json_datapoint() {
            let dataset_name = "test_dataset";
            let existing_datapoint = create_sample_json_datapoint(dataset_name);
            let datapoint_id = existing_datapoint.id;

            let mut mock_db = MockDatasetQueries::new();
            let existing_datapoint_clone = existing_datapoint.clone();
            mock_db.expect_get_datapoints().returning(move |_| {
                let dp = existing_datapoint_clone.clone();
                Box::pin(async move { Ok(vec![StoredDatapoint::Json(dp)]) })
            });
            mock_db
                .expect_insert_datapoints()
                .withf(move |datapoints_inserts| {
                    assert_eq!(datapoints_inserts.len(), 1, "Expected 1 datapoint insert");
                    let datapoint_insert = &datapoints_inserts[0];
                    // ID should stay the same.
                    assert_eq!(datapoint_insert.id(), datapoint_id);
                    let DatapointInsert::Json(dp) = datapoint_insert else {
                        panic!("Expected Json insert");
                    };
                    // Name should be updated.
                    assert_eq!(dp.name, Some("updated_json_name".to_string()));
                    // The other fields should stay the same.
                    assert_eq!(dp.input, existing_datapoint.input);
                    assert_eq!(dp.output, existing_datapoint.output);
                    assert_eq!(dp.output_schema, existing_datapoint.output_schema);
                    assert_eq!(dp.tags, existing_datapoint.tags);
                    assert_eq!(dp.staled_at, existing_datapoint.staled_at);
                    assert_eq!(
                        dp.source_inference_id,
                        existing_datapoint.source_inference_id
                    );
                    true
                })
                .returning(|_| Box::pin(async move { Ok(1) }));

            let request = UpdateDatapointsMetadataRequest {
                datapoints: vec![UpdateDatapointMetadataRequest {
                    id: datapoint_id,
                    metadata: DatapointMetadataUpdate {
                        name: Some(Some("updated_json_name".to_string())),
                    },
                }],
            };

            let result = update_datapoints_metadata(&mock_db, dataset_name, request).await;
            assert!(result.is_ok());
            let response = result.unwrap();
            assert_eq!(response.ids.len(), 1);
            assert_eq!(response.ids[0], datapoint_id);
        }

        #[tokio::test]
        async fn test_update_metadata_set_name_to_null() {
            let dataset_name = "test_dataset";
            let existing_datapoint = create_sample_chat_datapoint(dataset_name);
            let datapoint_id = existing_datapoint.id;

            let mut mock_db = MockDatasetQueries::new();
            let existing_datapoint_clone = existing_datapoint.clone();
            mock_db.expect_get_datapoints().returning(move |_| {
                let dp = existing_datapoint_clone.clone();
                Box::pin(async move { Ok(vec![StoredDatapoint::Chat(dp)]) })
            });
            mock_db
                .expect_insert_datapoints()
                .withf(|datapoints| {
                    datapoints.len() == 1
                        && matches!(&datapoints[0], DatapointInsert::Chat(dp) if dp.name.is_none())
                })
                .returning(|_| Box::pin(async move { Ok(1) }));

            let request = UpdateDatapointsMetadataRequest {
                datapoints: vec![UpdateDatapointMetadataRequest {
                    id: datapoint_id,
                    metadata: DatapointMetadataUpdate { name: Some(None) },
                }],
            };

            let result = update_datapoints_metadata(&mock_db, dataset_name, request).await;
            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_update_metadata_datapoint_not_found() {
            let dataset_name = "test_dataset";
            let non_existent_id = Uuid::now_v7();

            let mut mock_db = MockDatasetQueries::new();
            mock_db
                .expect_get_datapoints()
                .returning(|_| Box::pin(async move { Ok(vec![]) }));

            let request = UpdateDatapointsMetadataRequest {
                datapoints: vec![UpdateDatapointMetadataRequest {
                    id: non_existent_id,
                    metadata: DatapointMetadataUpdate {
                        name: Some(Some("new_name".to_string())),
                    },
                }],
            };

            let result = update_datapoints_metadata(&mock_db, dataset_name, request).await;
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err().get_details(),
                ErrorDetails::DatapointNotFound { .. }
            ));
        }

        #[tokio::test]
        async fn test_update_metadata_duplicate_ids() {
            let dataset_name = "test_dataset";
            let duplicate_id = Uuid::now_v7();

            let mock_db = MockDatasetQueries::new();

            let request = UpdateDatapointsMetadataRequest {
                datapoints: vec![
                    UpdateDatapointMetadataRequest {
                        id: duplicate_id,
                        metadata: DatapointMetadataUpdate {
                            name: Some(Some("name1".to_string())),
                        },
                    },
                    UpdateDatapointMetadataRequest {
                        id: duplicate_id,
                        metadata: DatapointMetadataUpdate {
                            name: Some(Some("name2".to_string())),
                        },
                    },
                ],
            };

            let result = update_datapoints_metadata(&mock_db, dataset_name, request).await;
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err().get_details(),
                ErrorDetails::InvalidRequest { .. }
            ));
        }

        #[tokio::test]
        async fn test_update_metadata_empty_datapoints() {
            let dataset_name = "test_dataset";
            let mock_db = MockDatasetQueries::new();

            let request = UpdateDatapointsMetadataRequest { datapoints: vec![] };

            let result = update_datapoints_metadata(&mock_db, dataset_name, request).await;
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err().get_details(),
                ErrorDetails::InvalidRequest { .. }
            ));
        }

        #[tokio::test]
        async fn test_update_metadata_batch() {
            let dataset_name = "test_dataset";
            let datapoint1 = create_sample_chat_datapoint(dataset_name);
            let datapoint2 = create_sample_json_datapoint(dataset_name);
            let id1 = datapoint1.id;
            let id2 = datapoint2.id;

            let datapoint1_clone = datapoint1.clone();
            let datapoint2_clone = datapoint2.clone();

            let mut mock_db = MockDatasetQueries::new();
            mock_db.expect_get_datapoints().returning(move |_| {
                let dp1 = datapoint1_clone.clone();
                let dp2 = datapoint2_clone.clone();
                Box::pin(
                    async move { Ok(vec![StoredDatapoint::Chat(dp1), StoredDatapoint::Json(dp2)]) },
                )
            });
            mock_db
                .expect_insert_datapoints()
                .withf(|datapoints| {
                    datapoints.len() == 2
                        && matches!(&datapoints[0], DatapointInsert::Chat(dp) if dp.name == Some("updated_name1".to_string()))
                        && matches!(&datapoints[1], DatapointInsert::Json(dp) if dp.name == Some("updated_name2".to_string()))
                })
                .returning(|_| Box::pin(async move { Ok(2) }));

            let request = UpdateDatapointsMetadataRequest {
                datapoints: vec![
                    UpdateDatapointMetadataRequest {
                        id: id1,
                        metadata: DatapointMetadataUpdate {
                            name: Some(Some("updated_name1".to_string())),
                        },
                    },
                    UpdateDatapointMetadataRequest {
                        id: id2,
                        metadata: DatapointMetadataUpdate {
                            name: Some(Some("updated_name2".to_string())),
                        },
                    },
                ],
            };

            let result = update_datapoints_metadata(&mock_db, dataset_name, request).await;
            assert!(result.is_ok());
            let response = result.unwrap();
            assert_eq!(response.ids.len(), 2);
            assert_eq!(response.ids[0], id1);
            assert_eq!(response.ids[1], id2);
        }
    }
}
