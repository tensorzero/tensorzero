use std::collections::{HashMap, HashSet};

use axum::extract::{Path, State};
use axum::Json;
use chrono::Utc;
use serde::Deserialize;
use serde_json;
use tracing::instrument;
use uuid::Uuid;

use crate::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries, JsonInferenceDatapointInsert,
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
        .get_datapoints(dataset_name, &datapoint_ids, false)
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
    if let Some(new_tool_params) = update.tool_params {
        updated_datapoint.tool_params = new_tool_params;
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
    use crate::config::{ObjectStoreInfo, SchemaData};
    use crate::experimentation::ExperimentationConfig;
    use crate::function::FunctionConfigChat;
    use crate::http::TensorzeroHttpClient;
    use crate::inference::types::storage::{StorageKind, StoragePath};
    use crate::inference::types::StoredInputMessageContent;
    use crate::inference::types::{File, Input, InputMessage, InputMessageContent, Role};
    use crate::tool::ToolChoice;
    use object_store::path::Path as ObjectStorePath;
    use std::collections::{HashMap, HashSet};

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
        let result = convert_input_to_stored_input(Some(input), &fetch_context, &function_config)
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
        let result = convert_input_to_stored_input(Some(input), &fetch_context, &function_config)
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

    #[test]
    fn test_file_object_storage_serializes_with_tagged_format() {
        // Test that File::ObjectStorage uses the new tagged format with flattened structure
        let storage_path = StoragePath {
            kind: StorageKind::Disabled,
            path: ObjectStorePath::parse("test/path/image.png").unwrap(),
        };

        let file = File::ObjectStorage {
            source_url: Some("https://example.com/original.png".parse().unwrap()),
            mime_type: mime::IMAGE_PNG,
            storage_path: storage_path.clone(),
        };

        let serialized = serde_json::to_value(&file).unwrap();

        // Verify tagged format
        assert_eq!(serialized["file_type"], "object_storage");
        assert_eq!(serialized["source_url"], "https://example.com/original.png");
        assert_eq!(serialized["mime_type"], "image/png");
        assert!(serialized.get("storage_path").is_some());

        // Verify flattened structure (no "metadata" field)
        assert!(serialized.get("metadata").is_none());
    }

    #[test]
    fn test_file_url_serializes_with_tagged_format() {
        let file = File::Url {
            url: "https://example.com/image.png".parse().unwrap(),
            mime_type: Some(mime::IMAGE_PNG),
        };

        let serialized = serde_json::to_value(&file).unwrap();

        // Verify tagged format
        assert_eq!(serialized["file_type"], "url");
        assert_eq!(serialized["url"], "https://example.com/image.png");
        assert_eq!(serialized["mime_type"], "image/png");
    }

    #[test]
    fn test_file_base64_serializes_with_tagged_format() {
        let file = File::Base64 {
            mime_type: mime::IMAGE_PNG,
            data: "base64data".to_string(),
        };

        let serialized = serde_json::to_value(&file).unwrap();

        // Verify tagged format
        assert_eq!(serialized["file_type"], "base64");
        assert_eq!(serialized["mime_type"], "image/png");
        assert_eq!(serialized["data"], "base64data");
    }
}
