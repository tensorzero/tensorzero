use axum::extract::{Path, State};
use axum::Json;
use futures::future::try_join_all;
use tracing::instrument;

use crate::config::Config;
use crate::db::datasets::{DatapointInsert, DatasetQueries};
use crate::endpoints::datasets::validate_dataset_name;
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::FetchContext;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{CreateDatapointRequest, CreateDatapointsRequest, CreateDatapointsResponse};

/// Handler for the POST `/v1/datasets/{dataset_id}/datapoints` endpoint.
/// Creates manual datapoints in a dataset.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.create_datapoints", skip(app_state, request))]
pub async fn create_datapoints_handler(
    State(app_state): AppState,
    Path(dataset_name): Path<String>,
    StructuredJson(request): StructuredJson<CreateDatapointsRequest>,
) -> Result<Json<CreateDatapointsResponse>, Error> {
    let response = create_datapoints(
        &app_state.config,
        &app_state.http_client,
        &app_state.clickhouse_connection_info,
        &dataset_name,
        request,
    )
    .await?;
    Ok(Json(response))
}

/// Business logic for creating datapoints manually in a dataset.
/// This function validates the request, converts inputs, validates schemas,
/// and inserts the new datapoints into ClickHouse.
///
/// Returns an error if there are no datapoints, or if validation fails.
pub async fn create_datapoints(
    config: &Config,
    http_client: &TensorzeroHttpClient,
    clickhouse: &impl DatasetQueries,
    dataset_name: &str,
    request: CreateDatapointsRequest,
) -> Result<CreateDatapointsResponse, Error> {
    validate_dataset_name(dataset_name)?;

    if request.datapoints.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "At least one datapoint must be provided".to_string(),
        }));
    }

    let fetch_context = FetchContext {
        client: http_client,
        object_store_info: &config.object_store_info,
    };
    // Convert all datapoints to inserts in parallel (because we may need to store inputs)
    let datapoint_insert_futures = request
        .datapoints
        .into_iter()
        .map(|datapoint_request| async {
            let result: Result<DatapointInsert, Error> = match datapoint_request {
                CreateDatapointRequest::Chat(chat_request) => {
                    let insert = chat_request
                        .into_database_insert(config, &fetch_context, dataset_name)
                        .await
                        .map_err(|e| {
                            Error::new(ErrorDetails::InvalidRequest {
                                message: format!(
                                    "Failed to convert chat datapoint to database insert: {e}"
                                ),
                            })
                        })?;
                    Ok(DatapointInsert::Chat(insert))
                }
                CreateDatapointRequest::Json(json_request) => {
                    let insert = json_request
                        .into_database_insert(config, &fetch_context, dataset_name)
                        .await
                        .map_err(|e| {
                            Error::new(ErrorDetails::InvalidRequest {
                                message: format!(
                                    "Failed to convert json datapoint to database insert: {e}"
                                ),
                            })
                        })?;
                    Ok(DatapointInsert::Json(insert))
                }
            };
            result
        });

    let datapoints_to_insert: Vec<DatapointInsert> = try_join_all(datapoint_insert_futures).await?;
    let ids = datapoints_to_insert
        .iter()
        .map(DatapointInsert::id)
        .collect::<Vec<_>>();

    // Insert all datapoints
    clickhouse.insert_datapoints(&datapoints_to_insert).await?;

    Ok(CreateDatapointsResponse { ids })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, SchemaData};
    use crate::db::clickhouse::MockClickHouseConnectionInfo;
    use crate::endpoints::datasets::v1::types::{
        CreateChatDatapointRequest, CreateDatapointRequest, CreateJsonDatapointRequest,
        JsonDatapointOutputUpdate,
    };
    use crate::experimentation::ExperimentationConfig;
    use crate::function::{FunctionConfig, FunctionConfigChat, FunctionConfigJson};
    use crate::http::TensorzeroHttpClient;
    use crate::inference::types::{
        ContentBlockChatOutput, Input, InputMessage, InputMessageContent, JsonInferenceOutput,
        Role, StoredInput, StoredInputMessage, StoredInputMessageContent, Text,
    };
    use crate::jsonschema_util::StaticJSONSchema;
    use crate::tool::{DynamicToolParams, ToolChoice};
    use serde_json::json;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use uuid::Uuid;

    /// Helper to create a test config with functions registered
    fn create_test_config() -> Config {
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

        config
    }

    /// Helper to create a simple chat input
    fn create_chat_input() -> Input {
        Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "test message".to_string(),
                })],
            }],
        }
    }

    /// Helper to create a simple chat output
    fn create_chat_output() -> Vec<ContentBlockChatOutput> {
        vec![ContentBlockChatOutput::Text(Text {
            text: "test response".to_string(),
        })]
    }

    #[tokio::test]
    async fn test_create_datapoints_success_single_chat_datapoint() {
        let config = create_test_config();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let dataset_name = "test_dataset";

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .withf(move |datapoints| {
                assert_eq!(datapoints.len(), 1, "Should insert exactly 1 datapoint");
                let Some(DatapointInsert::Chat(insert)) = datapoints.first() else {
                    panic!("Expected chat datapoint");
                };

                assert_ne!(insert.id, Uuid::nil());
                assert_eq!(insert.dataset_name, dataset_name);
                assert_eq!(insert.function_name, "test_chat_function");
                assert_eq!(insert.episode_id, None);
                assert_eq!(
                    insert.input,
                    StoredInput {
                        system: None,
                        messages: vec![StoredInputMessage {
                            role: Role::User,
                            content: vec![StoredInputMessageContent::Text(Text {
                                text: "test message".to_string(),
                            })],
                        }],
                    }
                );
                assert_eq!(
                    insert.output,
                    Some(vec![ContentBlockChatOutput::Text(Text {
                        text: "test response".to_string(),
                    })])
                );
                assert_eq!(insert.tool_params, None);
                assert_eq!(insert.tags, None);
                assert_eq!(insert.name, Some("test datapoint".to_string()));
                true
            })
            .times(1)
            .returning(|_| Box::pin(async move { Ok(1) }));

        let request = CreateDatapointsRequest {
            datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                function_name: "test_chat_function".to_string(),
                episode_id: None,
                input: create_chat_input(),
                output: Some(create_chat_output()),
                dynamic_tool_params: DynamicToolParams::default(),
                tags: None,
                name: Some("test datapoint".to_string()),
            })],
        };

        let result = create_datapoints(
            &config,
            &http_client,
            &mock_clickhouse,
            dataset_name,
            request,
        )
        .await
        .unwrap();

        assert_eq!(result.ids.len(), 1);
    }

    #[tokio::test]
    async fn test_create_datapoints_success_single_json_datapoint() {
        let config = create_test_config();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let dataset_name = "test_dataset";

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .withf(move |datapoints| {
                assert_eq!(datapoints.len(), 1, "Should insert exactly 1 datapoint");
                let Some(DatapointInsert::Json(insert)) = datapoints.first() else {
                    panic!("Expected json datapoint");
                };

                assert_ne!(insert.id, Uuid::nil());
                assert_eq!(insert.dataset_name, dataset_name);
                assert_eq!(insert.function_name, "test_json_function");
                assert_eq!(insert.episode_id, None);
                assert_eq!(
                    insert.input,
                    StoredInput {
                        system: None,
                        messages: vec![StoredInputMessage {
                            role: Role::User,
                            content: vec![StoredInputMessageContent::Text(Text {
                                text: "test message".to_string(),
                            })],
                        }],
                    }
                );
                assert_eq!(
                    insert.output,
                    Some(JsonInferenceOutput {
                        raw: Some(r#"{"value": "test"}"#.to_string()),
                        parsed: Some(json!({"value": "test"})),
                    })
                );
                // Output schema is loaded from config
                assert_eq!(insert.output_schema, json!({"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": false}));
                assert_eq!(insert.tags, None);
                assert_eq!(insert.name, Some("test json datapoint".to_string()));
                true
            })
            .times(1)
            .returning(|_| Box::pin(async move { Ok(1) }));

        let request = CreateDatapointsRequest {
            datapoints: vec![CreateDatapointRequest::Json(CreateJsonDatapointRequest {
                function_name: "test_json_function".to_string(),
                episode_id: None,
                input: create_chat_input(),
                output: Some(JsonDatapointOutputUpdate {
                    raw: Some(r#"{"value": "test"}"#.to_string()),
                }),
                output_schema: None,
                tags: None,
                name: Some("test json datapoint".to_string()),
            })],
        };

        let result = create_datapoints(
            &config,
            &http_client,
            &mock_clickhouse,
            dataset_name,
            request,
        )
        .await
        .unwrap();

        assert_eq!(result.ids.len(), 1);
    }

    #[tokio::test]
    async fn test_create_datapoints_success_multiple_datapoints() {
        let config = create_test_config();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let dataset_name = "test_dataset";

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .withf(|datapoints| {
                assert_eq!(datapoints.len(), 3, "Should insert exactly 3 datapoints");

                let DatapointInsert::Chat(insert) = &datapoints[0] else {
                    panic!("Expected chat datapoint");
                };
                assert_eq!(insert.name, Some("chat datapoint 1".to_string()));
                let DatapointInsert::Chat(insert) = &datapoints[1] else {
                    panic!("Expected chat datapoint");
                };
                assert_eq!(insert.name, Some("chat datapoint 2".to_string()));
                let DatapointInsert::Json(insert) = &datapoints[2] else {
                    panic!("Expected json datapoint");
                };
                assert_eq!(insert.name, Some("json datapoint".to_string()));
                true
            })
            .times(1)
            .returning(|_| Box::pin(async move { Ok(3) }));

        let request = CreateDatapointsRequest {
            datapoints: vec![
                CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                    function_name: "test_chat_function".to_string(),
                    episode_id: None,
                    input: create_chat_input(),
                    output: Some(create_chat_output()),
                    dynamic_tool_params: DynamicToolParams::default(),
                    tags: None,
                    name: Some("chat datapoint 1".to_string()),
                }),
                CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                    function_name: "test_chat_function".to_string(),
                    episode_id: None,
                    input: create_chat_input(),
                    output: None,
                    dynamic_tool_params: DynamicToolParams::default(),
                    tags: None,
                    name: Some("chat datapoint 2".to_string()),
                }),
                CreateDatapointRequest::Json(CreateJsonDatapointRequest {
                    function_name: "test_json_function".to_string(),
                    episode_id: None,
                    input: create_chat_input(),
                    output: Some(JsonDatapointOutputUpdate {
                        raw: Some(r#"{"value": "test"}"#.to_string()),
                    }),
                    output_schema: None,
                    tags: None,
                    name: Some("json datapoint".to_string()),
                }),
            ],
        };

        let result = create_datapoints(
            &config,
            &http_client,
            &mock_clickhouse,
            dataset_name,
            request,
        )
        .await
        .unwrap();

        assert_eq!(result.ids.len(), 3);
    }

    #[tokio::test]
    async fn test_create_datapoints_error_empty_datapoints() {
        let config = create_test_config();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let dataset_name = "test_dataset";

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .times(0);

        let request = CreateDatapointsRequest { datapoints: vec![] };

        let result = create_datapoints(
            &config,
            &http_client,
            &mock_clickhouse,
            dataset_name,
            request,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("At least one datapoint must be provided"));
    }

    #[tokio::test]
    async fn test_create_datapoints_error_invalid_dataset_name() {
        let config = create_test_config();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let invalid_dataset_name = "builder"; // Reserved name

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .times(0);

        let request = CreateDatapointsRequest {
            datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                function_name: "test_chat_function".to_string(),
                episode_id: None,
                input: create_chat_input(),
                output: Some(create_chat_output()),
                dynamic_tool_params: DynamicToolParams::default(),
                tags: None,
                name: None,
            })],
        };

        let result = create_datapoints(
            &config,
            &http_client,
            &mock_clickhouse,
            invalid_dataset_name,
            request,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Invalid dataset name"));
    }

    #[tokio::test]
    async fn test_create_datapoints_error_nonexistent_function() {
        let config = create_test_config();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let dataset_name = "test_dataset";

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .times(0);

        let request = CreateDatapointsRequest {
            datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                function_name: "nonexistent_function".to_string(),
                episode_id: None,
                input: create_chat_input(),
                output: Some(create_chat_output()),
                dynamic_tool_params: DynamicToolParams::default(),
                tags: None,
                name: None,
            })],
        };

        let result = create_datapoints(
            &config,
            &http_client,
            &mock_clickhouse,
            dataset_name,
            request,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Failed to convert chat datapoint to database insert"));
    }

    #[tokio::test]
    async fn test_create_datapoints_error_wrong_function_type() {
        let config = create_test_config();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let dataset_name = "test_dataset";

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .times(0);

        // Try to use a JSON function for a chat datapoint request
        let request = CreateDatapointsRequest {
            datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                function_name: "test_json_function".to_string(), // This is a JSON function
                episode_id: None,
                input: create_chat_input(),
                output: Some(create_chat_output()),
                dynamic_tool_params: DynamicToolParams::default(),
                tags: None,
                name: None,
            })],
        };

        let result = create_datapoints(
            &config,
            &http_client,
            &mock_clickhouse,
            dataset_name,
            request,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Failed to convert chat datapoint to database insert"));
    }

    #[tokio::test]
    async fn test_create_datapoints_with_tags_and_episode() {
        let config = create_test_config();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let dataset_name = "test_dataset";
        let episode_id = Uuid::now_v7();

        let mut tags = HashMap::new();
        tags.insert("environment".to_string(), "test".to_string());
        tags.insert("version".to_string(), "1.0".to_string());

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .withf(move |datapoints| {
                assert_eq!(datapoints.len(), 1, "Should insert exactly 1 datapoint");
                // Verify the datapoint has the expected episode_id and tags
                if let Some(crate::db::datasets::DatapointInsert::Chat(insert)) = datapoints.first()
                {
                    assert_eq!(insert.episode_id, Some(episode_id));
                    assert!(insert.tags.is_some());
                    let tags = insert.tags.as_ref().unwrap();
                    assert_eq!(tags.get("environment"), Some(&"test".to_string()));
                    assert_eq!(tags.get("version"), Some(&"1.0".to_string()));
                }
                true
            })
            .times(1)
            .returning(|_| Box::pin(async move { Ok(1) }));

        let request = CreateDatapointsRequest {
            datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                function_name: "test_chat_function".to_string(),
                episode_id: Some(episode_id),
                input: create_chat_input(),
                output: Some(create_chat_output()),
                dynamic_tool_params: DynamicToolParams::default(),
                tags: Some(tags),
                name: Some("tagged datapoint".to_string()),
            })],
        };

        let result = create_datapoints(
            &config,
            &http_client,
            &mock_clickhouse,
            dataset_name,
            request,
        )
        .await
        .unwrap();

        assert_eq!(result.ids.len(), 1);
    }
}
