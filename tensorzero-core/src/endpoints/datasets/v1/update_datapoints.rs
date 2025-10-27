use std::collections::{HashMap, HashSet};

use axum::extract::{Path, State};
use axum::Json;
use chrono::Utc;
use serde::Deserialize;
use serde_json;
use tracing::instrument;
use uuid::Uuid;

use crate::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries, GetDatapointsParams,
    JsonInferenceDatapointInsert,
};
use crate::endpoints::datasets::{
    validate_dataset_name, ChatInferenceDatapoint, Datapoint, JsonInferenceDatapoint,
    CLICKHOUSE_DATETIME_FORMAT,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{FetchContext, Input, JsonInferenceOutput};
use crate::jsonschema_util::StaticJSONSchema;
use crate::tool::ToolCallConfigDatabaseInsert;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{
    UpdateChatDatapointRequest, UpdateDatapointRequest, UpdateDatapointsRequest,
    UpdateDatapointsResponse, UpdateJsonDatapointRequest,
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
async fn update_datapoints(
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
            page_size: u32::MAX, // No limit - fetch all matching datapoints
            offset: 0,
            allow_stale: false,
            filter: None, // No filtering when updating datapoints
        })
        .await?;

    // Build a HashMap to construct new DatapointInserts
    let mut datapoints_map: HashMap<Uuid, Datapoint> =
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
            (UpdateDatapointRequest::Chat(_), Datapoint::Json(_)) => {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Datapoint {datapoint_id} is a JSON datapoint but a chat update was provided"
                    ),
                }));
            }
            (UpdateDatapointRequest::Json(_), Datapoint::Chat(_)) => {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Datapoint {datapoint_id} is a chat datapoint but a JSON update was provided"
                    ),
                }));
            }
            (UpdateDatapointRequest::Chat(update), Datapoint::Chat(existing)) => {
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
            (UpdateDatapointRequest::Json(update), Datapoint::Json(existing)) => {
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
    existing_datapoint: ChatInferenceDatapoint,
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

    let maybe_new_input =
        convert_input_to_stored_input(update.input, fetch_context, function_config.as_ref())
            .await?;
    if let Some(new_input) = maybe_new_input {
        updated_datapoint.input = new_input;
    }

    if let Some(new_output) = update.output {
        updated_datapoint.output = Some(new_output);
    }

    // Check if any tool info updates are provided
    let has_tool_updates = update.dynamic_tools.is_some()
        || update.dynamic_provider_tools.is_some()
        || update.allowed_tools.is_some()
        || update.tool_choice.is_some()
        || update.parallel_tool_calls.is_some();

    if has_tool_updates {
        // If tool_info doesn't exist but we have updates, create a default one
        if updated_datapoint.tool_info.is_none() {
            updated_datapoint.tool_info = Some(ToolCallConfigDatabaseInsert::default());
        }

        // Apply the updates to the tool_info
        if let Some(tool_info) = updated_datapoint.tool_info.as_mut() {
            tool_info.update(
                update.dynamic_tools,
                update.dynamic_provider_tools,
                update.allowed_tools,
                update.tool_choice,
                update.parallel_tool_calls,
            );
        }
    }

    if let Some(new_tags) = update.tags {
        updated_datapoint.tags = Some(new_tags);
    }
    if let Some(new_metadata) = update.metadata {
        if let Some(new_name) = new_metadata.name {
            updated_datapoint.name = new_name;
        }
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
    existing_datapoint: JsonInferenceDatapoint,
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

    // Grab a copy of IDs for logging.
    let existing_datapoint_id = existing_datapoint.id;
    let updated_datapoint_id = Uuid::now_v7();

    // Update old datapoint as staled, and create new datapoint.
    let mut staled_existing_datapoint: JsonInferenceDatapointInsert =
        existing_datapoint.clone().into();
    staled_existing_datapoint.staled_at = Some(now_timestamp.to_owned());

    // Update the datapoint with new data
    let mut updated_datapoint: JsonInferenceDatapointInsert = existing_datapoint.into();
    updated_datapoint.id = updated_datapoint_id;

    let maybe_new_input =
        convert_input_to_stored_input(update.input, fetch_context, function_config.as_ref())
            .await?;
    if let Some(new_input) = maybe_new_input {
        updated_datapoint.input = new_input;
    }

    if let Some(new_output_schema) = update.output_schema {
        updated_datapoint.output_schema = new_output_schema;
    }
    if let Some(new_output) = update.output {
        updated_datapoint.output = match new_output {
            None => None,
            Some(value) => {
                // Validate the output with schema before saving.
                StaticJSONSchema::from_value(updated_datapoint.output_schema.clone())?
            .validate(&value)
            .map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Provided output for datapoint {existing_datapoint_id} does not match function output schema: {e}",
                    ),
                })
            })?;

                Some(JsonInferenceOutput {
                    raw: Some(serde_json::to_string(&value).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Failed to serialize provided output for datapoint {existing_datapoint_id}: {e}",
                            )
                        })
                    })?),
                    parsed: Some(value),
                })
            }
        }
    }

    if let Some(new_tags) = update.tags {
        updated_datapoint.tags = Some(new_tags);
    }

    if let Some(new_metadata) = update.metadata {
        if let Some(new_name) = new_metadata.name {
            updated_datapoint.name = new_name;
        }
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
                .into_lazy_resolved_input(FetchContext {
                    client: fetch_context.client,
                    object_store_info: fetch_context.object_store_info,
                })?
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ObjectStoreInfo, SchemaData};
    use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
    use crate::endpoints::datasets::v1::types::DatapointMetadataUpdate;
    use crate::endpoints::datasets::{ChatInferenceDatapoint, JsonInferenceDatapoint};
    use crate::experimentation::ExperimentationConfig;
    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::http::TensorzeroHttpClient;
    use crate::inference::types::storage::{StorageKind, StoragePath};
    use crate::inference::types::{
        ContentBlockChatOutput, File, Input, InputMessage, InputMessageContent,
        JsonInferenceOutput, Role, StoredInputMessageContent, Text,
    };
    use crate::jsonschema_util::StaticJSONSchema;
    use crate::tool::{AllowedTools, ToolCallConfigDatabaseInsert, ToolChoice};
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

            let file = File::ObjectStorage {
                source_url: Some("https://example.com/original.png".parse().unwrap()),
                mime_type: mime::IMAGE_PNG,
                storage_path: storage_path.clone(),
            };

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
                experimentation: ExperimentationConfig::Uniform,
                all_explicit_templates_names: HashSet::new(),
            });

            // Create fetch context with NO actual object storage info.
            // If the code tries to access object storage, it will fail with an error.
            let http_client = TensorzeroHttpClient::new().unwrap();
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
                    assert_eq!(stored_file.file.mime_type, mime::IMAGE_PNG);
                    assert_eq!(
                        stored_file.file.url,
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

            let file = File::Base64 {
            mime_type: mime::IMAGE_PNG,
            data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==".to_string(),
        };

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
                experimentation: ExperimentationConfig::Uniform,
                all_explicit_templates_names: HashSet::new(),
            });

            // Create fetch context with disabled storage
            // File::Base64 will call write_file() but it no-ops with disabled storage
            let http_client = TensorzeroHttpClient::new().unwrap();
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
                    assert_eq!(stored_file.file.mime_type, mime::IMAGE_PNG);
                    // With disabled storage, path should still be generated
                    assert!(!stored_file.storage_path.path.as_ref().is_empty());
                    // URL should be None since this came from Base64
                    assert_eq!(stored_file.file.url, None);
                }
                _ => panic!("Expected File content"),
            }
        }
    }
    // ============================================================================
    // Test helpers for prepare_chat_update and prepare_json_update
    // ============================================================================

    mod prepare_update_tests {
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
                    experimentation: ExperimentationConfig::Uniform,
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
                    implicit_tool_call_config: crate::tool::ToolCallConfig::default(),
                    description: None,
                    experimentation: ExperimentationConfig::Uniform,
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
        fn create_sample_chat_datapoint(dataset_name: &str) -> ChatInferenceDatapoint {
            ChatInferenceDatapoint {
                id: Uuid::now_v7(),
                dataset_name: dataset_name.to_string(),
                function_name: "test_chat_function".to_string(),
                name: Some("test_datapoint".to_string()),
                episode_id: Some(Uuid::now_v7()),
                input: StoredInput {
                    system: None,
                    messages: vec![crate::inference::types::StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text {
                            value: json!("original input"),
                        }],
                    }],
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "original output".to_string(),
                })]),
                tool_info: Some(ToolCallConfigDatabaseInsert {
                    dynamic_tools: vec![],
                    dynamic_provider_tools: vec![],
                    allowed_tools: AllowedTools::default(),
                    tool_choice: ToolChoice::Auto,
                    parallel_tool_calls: Some(true),
                    ..Default::default()
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

        /// Helper to create a sample JsonInferenceDatapoint
        fn create_sample_json_datapoint(dataset_name: &str) -> JsonInferenceDatapoint {
            JsonInferenceDatapoint {
                id: Uuid::now_v7(),
                dataset_name: dataset_name.to_string(),
                function_name: "test_json_function".to_string(),
                name: Some("test_datapoint".to_string()),
                episode_id: Some(Uuid::now_v7()),
                input: StoredInput {
                    system: None,
                    messages: vec![crate::inference::types::StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text {
                            value: json!("original input"),
                        }],
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
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                tags: None,
                metadata: None,
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
            assert!(updated.tool_info.is_some());
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
                    content: vec![InputMessageContent::Text(
                        crate::inference::types::TextKind::Text {
                            text: "new input text".into(),
                        },
                    )],
                }],
            };

            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: Some(new_input),
                output: None,
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                tags: None,
                metadata: None,
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
                StoredInputMessageContent::Text { value } => {
                    let text: String = serde_json::from_value(value.clone()).unwrap();
                    assert_eq!(text, "new input text");
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
                output: Some(new_output.clone()),
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                tags: None,
                metadata: None,
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
        async fn test_prepare_chat_update_tool_info_omitted() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_chat_datapoint(dataset_name);
            let original_tool_info = existing.tool_info.clone();

            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: None,
                parallel_tool_calls: None, // All omitted - should remain unchanged
                tags: None,
                metadata: None,
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

            assert_eq!(updated.tool_info, original_tool_info);
        }

        #[tokio::test]
        async fn test_prepare_chat_update_tool_choice_set_to_none() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_chat_datapoint(dataset_name);

            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: Some(Some(ToolChoice::None)), // Explicitly set to None choice
                parallel_tool_calls: None,
                tags: None,
                metadata: None,
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

            assert_eq!(
                updated.tool_info.as_ref().unwrap().tool_choice,
                ToolChoice::None
            );
        }

        #[tokio::test]
        async fn test_prepare_chat_update_multiple_tool_fields_updated() {
            let app_state = create_test_app_state();
            let fetch_context = create_fetch_context(&app_state.http_client);
            let dataset_name = "test_dataset";
            let existing = create_sample_chat_datapoint(dataset_name);

            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: None,
                output: None,
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: Some(Some(ToolChoice::Required)),
                parallel_tool_calls: Some(Some(false)),
                tags: None,
                metadata: None,
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

            let tool_info = updated.tool_info.as_ref().unwrap();
            assert_eq!(tool_info.tool_choice, ToolChoice::Required);
            assert_eq!(tool_info.parallel_tool_calls, Some(false));
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
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                tags: None,
                metadata: None,
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
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                tags: Some(HashMap::new()),
                metadata: None,
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
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                tags: Some(new_tags.clone()),
                metadata: None,
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
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                tags: None,
                metadata: None,
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
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                tags: None,
                metadata: Some(DatapointMetadataUpdate { name: Some(None) }),
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
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                tags: None,
                metadata: Some(DatapointMetadataUpdate {
                    name: Some(Some("new_name".to_string())),
                }),
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
                    content: vec![InputMessageContent::Text(
                        crate::inference::types::TextKind::Text {
                            text: "new input".into(),
                        },
                    )],
                }],
            };
            let new_output = vec![ContentBlockChatOutput::Text(Text {
                text: "new output".to_string(),
            })];
            let new_tags = HashMap::from([("new".to_string(), "tag".to_string())]);

            let update = UpdateChatDatapointRequest {
                id: existing.id,
                input: Some(new_input),
                output: Some(new_output.clone()),
                dynamic_tools: None,
                dynamic_provider_tools: None,
                allowed_tools: None,
                tool_choice: Some(Some(ToolChoice::Required)),
                parallel_tool_calls: Some(Some(false)),
                tags: Some(new_tags.clone()),
                metadata: Some(DatapointMetadataUpdate {
                    name: Some(Some("updated_name".to_string())),
                }),
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
            let tool_info = updated.tool_info.as_ref().unwrap();
            assert_eq!(tool_info.tool_choice, ToolChoice::Required);
            assert_eq!(tool_info.parallel_tool_calls, Some(false));
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
                metadata: None,
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
                metadata: None,
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
                metadata: None,
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
                output: Some(Some(new_output_value.clone())),
                output_schema: None,
                tags: None,
                metadata: None,
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
                metadata: None,
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
                output: Some(Some(new_output.clone())),
                output_schema: Some(new_schema.clone()),
                tags: None,
                metadata: None,
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
                output: Some(Some(bad_output)),
                output_schema: None, // Will use existing schema which expects {value: string}
                tags: None,
                metadata: None,
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

            assert!(result.is_err(), "Expected validation error");
            let err = result.unwrap_err();
            let err_msg = format!("{err:?}");
            assert!(
                err_msg.contains("does not match") || err_msg.contains("schema"),
                "Expected schema validation error, got: {err_msg}"
            );
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
                metadata: None,
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
                    content: vec![InputMessageContent::Text(
                        crate::inference::types::TextKind::Text {
                            text: "new json input".into(),
                        },
                    )],
                }],
            };
            let new_schema =
                json!({"type": "object", "properties": {"result": {"type": "boolean"}}});
            let new_output = json!({"result": true});
            let new_tags = HashMap::from([("json_tag".to_string(), "value".to_string())]);

            let update = UpdateJsonDatapointRequest {
                id: existing.id,
                input: Some(new_input),
                output: Some(Some(new_output.clone())),
                output_schema: Some(new_schema.clone()),
                tags: Some(new_tags.clone()),
                metadata: Some(DatapointMetadataUpdate {
                    name: Some(Some("json_updated".to_string())),
                }),
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
    }
}
