use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tensorzero::{ClientExt, Role, StoredDatapoint};
use tensorzero_core::endpoints::datasets::v1::types::{
    CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsRequest,
    CreateDatapointsResponse, CreateJsonDatapointRequest, JsonDatapointOutputUpdate,
};
use tensorzero_core::inference::types::{
    Arguments, ContentBlockChatOutput, Input, InputMessage, InputMessageContent,
    JsonInferenceOutput, StoredInput, StoredInputMessage, StoredInputMessageContent, Template,
    Text,
};
use tensorzero_core::tool::{DynamicToolParams, FunctionTool, ToolChoice};
use uuid::Uuid;

use tensorzero_core::config::Config;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::datasets::{DatasetQueries, GetDatapointsParams};

use crate::common::get_gateway_endpoint;

lazy_static::lazy_static! {
    static ref TEST_SETUP: tokio::sync::OnceCell<(ClickHouseConnectionInfo, Arc<Config>)> = tokio::sync::OnceCell::new();
}

async fn get_test_setup() -> &'static (ClickHouseConnectionInfo, Arc<Config>) {
    TEST_SETUP
        .get_or_init(|| async {
            let clickhouse: ClickHouseConnectionInfo = get_clickhouse().await;

            let client = tensorzero::test_helpers::make_embedded_gateway().await;
            let config = client.get_config().unwrap();
            (clickhouse, config)
        })
        .await
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_chat_datapoint_basic() {
    let client = Client::new();
    let (clickhouse, _config) = get_test_setup().await;

    let request = CreateDatapointsRequest {
        datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "write_haiku".to_string(),
            episode_id: None,
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments(
                            json!({"topic": "coding"}).as_object().unwrap().clone(),
                        ),
                    })],
                }],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "Code flows like water\nBugs emerge from the shadows\nRefactor brings peace"
                    .to_string(),
            })]),
            dynamic_tool_params: DynamicToolParams::default(),
            tags: None,
            name: Some("Test Haiku Datapoint".to_string()),
        })],
    };

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_chat_basic/datapoints",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        200,
        "Response: {:?}",
        response.text().await
    );

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_chat_basic/datapoints",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    let result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(result.ids.len(), 1);

    // Verify the datapoint was inserted
    let params = GetDatapointsParams {
        dataset_name: Some("test_chat_basic".to_string()),
        function_name: None,
        ids: Some(result.ids.clone()),
        limit: 10,
        offset: 0,
        allow_stale: false,
        filter: None,
    };

    let datapoints = clickhouse.get_datapoints(&params).await.unwrap();
    assert_eq!(datapoints.len(), 1);
    assert_eq!(datapoints[0].id(), result.ids[0]);

    let StoredDatapoint::Chat(ref chat_datapoint) = datapoints[0] else {
        panic!("Expected chat datapoint");
    };
    assert_eq!(
        chat_datapoint.input,
        StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(json!({"topic": "coding"}).as_object().unwrap().clone()),
                })]
            }],
        }
    );
    assert_eq!(
        chat_datapoint.name,
        Some("Test Haiku Datapoint".to_string())
    );
    assert_eq!(
        chat_datapoint.output,
        Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Code flows like water\nBugs emerge from the shadows\nRefactor brings peace"
                .to_string(),
        })]),
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_json_datapoint_basic() {
    let client = Client::new();
    let (clickhouse, _config) = get_test_setup().await;

    let request = CreateDatapointsRequest {
        datapoints: vec![CreateDatapointRequest::Json(CreateJsonDatapointRequest {
            function_name: "extract_entities".to_string(),
            episode_id: None,
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "+1 Ernie Els ( South Africa ) through 8".to_string(),
                    })],
                }],
            },
            output: Some(JsonDatapointOutputUpdate {
                raw: r#"{"sentiment":"positive","confidence":0.95}"#.to_string(),
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["sentiment", "confidence"]
            })),
            tags: None,
            name: Some("Test Sentiment Datapoint".to_string()),
        })],
    };

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_json_basic/datapoints",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        200,
        "Response: {:?}",
        response.text().await
    );

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_json_basic/datapoints",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    let result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(result.ids.len(), 1);

    // Verify the datapoint was inserted
    let params = GetDatapointsParams {
        dataset_name: Some("test_json_basic".to_string()),
        function_name: None,
        ids: Some(result.ids.clone()),
        limit: 10,
        offset: 0,
        allow_stale: false,
        filter: None,
    };

    let datapoints = clickhouse.get_datapoints(&params).await.unwrap();
    assert_eq!(datapoints.len(), 1);
    assert_eq!(datapoints[0].id(), result.ids[0]);

    // Verify the output is parsed correctly
    let StoredDatapoint::Json(ref json_datapoint) = datapoints[0] else {
        panic!("Expected json datapoint");
    };
    assert_eq!(
        json_datapoint.output,
        Some(JsonInferenceOutput {
            raw: Some(r#"{"sentiment":"positive","confidence":0.95}"#.to_string()),
            parsed: Some(json!({"sentiment": "positive", "confidence": 0.95})),
        })
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_multiple_datapoints() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let request = CreateDatapointsRequest {
        datapoints: vec![
            CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                function_name: "write_haiku".to_string(),
                episode_id: None,
                input: Input {
                    system: None,
                    messages: vec![InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Template(Template {
                            name: "user".to_string(),
                            arguments: Arguments(
                                json!({"topic": "nature"}).as_object().unwrap().clone(),
                            ),
                        })],
                    }],
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "Leaves fall silently\nWhispering ancient secrets\nNature's quiet dance"
                        .to_string(),
                })]),
                dynamic_tool_params: DynamicToolParams::default(),
                tags: None,
                name: None,
            }),
            CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                function_name: "write_haiku".to_string(),
                episode_id: None,
                input: Input {
                    system: None,
                    messages: vec![InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Template(Template {
                            name: "user".to_string(),
                            arguments: Arguments(
                                json!({"topic": "technology"}).as_object().unwrap().clone(),
                            ),
                        })],
                    }],
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "Silicon whispers\nElectrons dance through circuits\nFuture unfolds fast"
                        .to_string(),
                })]),
                dynamic_tool_params: DynamicToolParams::default(),
                tags: None,
                name: None,
            }),
            CreateDatapointRequest::Json(CreateJsonDatapointRequest {
                function_name: "extract_entities".to_string(),
                episode_id: None,
                input: Input {
                    system: None,
                    messages: vec![InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text: "+1 Ernie Els ( South Africa ) through 8".to_string(),
                        })],
                    }],
                },
                output: Some(JsonDatapointOutputUpdate {
                    raw: r#"{"sentiment":"positive","confidence":0.98}"#.to_string(),
                }),
                output_schema: Some(json!({
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["sentiment", "confidence"]
                })),
                tags: None,
                name: None,
            }),
        ],
    };

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_multiple/datapoints",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(result.ids.len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_chat_datapoint_with_tools() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let request = CreateDatapointsRequest {
        datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "write_haiku".to_string(),
            episode_id: None,
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments(
                            json!({"topic": "nature"}).as_object().unwrap().clone(),
                        ),
                    })],
                }],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "A simple haiku".to_string(),
            })]),
            dynamic_tool_params: DynamicToolParams {
                allowed_tools: None,
                additional_tools: Some(vec![FunctionTool {
                    name: "get_weather".to_string(),
                    description: "Get the weather".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }),
                    strict: false,
                }]),
                tool_choice: Some(ToolChoice::Auto),
                parallel_tool_calls: None,
                provider_tools: vec![],
            },
            tags: None,
            name: None,
        })],
    };

    let response = client
        .post(get_gateway_endpoint("/v1/datasets/test_tools/datapoints"))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(result.ids.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_datapoint_with_tags() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let request = CreateDatapointsRequest {
        datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "write_haiku".to_string(),
            episode_id: None,
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments(
                            json!({"topic": "nature"}).as_object().unwrap().clone(),
                        ),
                    })],
                }],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "Tagged haiku here".to_string(),
            })]),
            dynamic_tool_params: DynamicToolParams::default(),
            tags: Some(HashMap::from([
                ("environment".to_string(), "test".to_string()),
                ("version".to_string(), "1.0".to_string()),
                ("quality".to_string(), "high".to_string()),
            ])),
            name: None,
        })],
    };

    let response = client
        .post(get_gateway_endpoint("/v1/datasets/test_tags/datapoints"))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(result.ids.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_datapoint_invalid_function() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let request = json!({
        "datapoints": [{
            "type": "chat",
            "function_name": "nonexistent_function",
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": "Test"}]
                }]
            }
        }]
    });

    let response = client
        .post(get_gateway_endpoint("/v1/datasets/test_invalid/datapoints"))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_datapoint_wrong_function_type() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    // Try to create a JSON datapoint for a chat function
    let request = json!({
        "datapoints": [{
            "type": "json",
            "function_name": "write_haiku",
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"topic": "nature"}}]
                }]
            },
            "output": {
                "raw": r#"{"data":"test"}"#
            },
            "output_schema": {"type": "object"}
        }]
    });

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_wrong_type/datapoints",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_datapoint_empty_list() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let request = json!({
        "datapoints": []
    });

    let response = client
        .post(get_gateway_endpoint("/v1/datasets/test_empty/datapoints"))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_json_datapoint_invalid_schema() {
    let client = Client::new();
    let (clickhouse, _config) = get_test_setup().await;

    let request = CreateDatapointsRequest {
        datapoints: vec![CreateDatapointRequest::Json(CreateJsonDatapointRequest {
            function_name: "extract_entities".to_string(),
            episode_id: None,
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Test".to_string(),
                    })],
                }],
            },
            output: Some(JsonDatapointOutputUpdate {
                raw: r#"{"sentiment":"positive","confidence":"not a number"}"#.to_string(),
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["sentiment", "confidence"]
            })),
            tags: None,
            name: None,
        })],
    };

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_invalid_schema/datapoints",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    // Invalid schema is allowed, but no parsed value is stored.
    assert_eq!(
        response.status(),
        200,
        "Response: {:?}",
        response.text().await
    );

    let result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(result.ids.len(), 1, "Should create exactly 1 datapoint");

    // Verify the datapoint was inserted
    let params = GetDatapointsParams {
        dataset_name: Some("test_invalid_schema".to_string()),
        function_name: None,
        ids: Some(result.ids.clone()),
        limit: 10,
        offset: 0,
        allow_stale: false,
        filter: None,
    };

    let datapoints = clickhouse.get_datapoints(&params).await.unwrap();
    assert_eq!(datapoints.len(), 1);
    let StoredDatapoint::Json(ref json_datapoint) = datapoints[0] else {
        panic!("Expected json datapoint");
    };

    // Raw should be preserved, but parsed is not stored.
    assert_eq!(
        json_datapoint.output,
        Some(JsonInferenceOutput {
            raw: Some(r#"{"sentiment":"positive","confidence":"not a number"}"#.to_string()),
            parsed: None,
        })
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_datapoint_with_episode_id() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let episode_id = Uuid::now_v7();

    let request = CreateDatapointsRequest {
        datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "write_haiku".to_string(),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments(
                            json!({"topic": "nature"}).as_object().unwrap().clone(),
                        ),
                    })],
                }],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "Episode haiku".to_string(),
            })]),
            dynamic_tool_params: DynamicToolParams::default(),
            tags: None,
            name: None,
        })],
    };

    let response = client
        .post(get_gateway_endpoint("/v1/datasets/test_episode/datapoints"))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        200,
        "Response: {:?}",
        response.text().await
    );
    let result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(result.ids.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_datapoint_without_output() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let request = CreateDatapointsRequest {
        datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "write_haiku".to_string(),
            episode_id: None,
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments(
                            json!({"topic": "nature"}).as_object().unwrap().clone(),
                        ),
                    })],
                }],
            },
            output: None,
            dynamic_tool_params: DynamicToolParams::default(),
            tags: None,
            name: None,
        })],
    };

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_no_output/datapoints",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(result.ids.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_json_datapoint_default_schema() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    // Test that output_schema is optional and defaults to function's schema
    let request = CreateDatapointsRequest {
        datapoints: vec![CreateDatapointRequest::Json(CreateJsonDatapointRequest {
            function_name: "extract_entities".to_string(),
            episode_id: None,
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Test".to_string(),
                    })],
                }],
            },
            output: Some(JsonDatapointOutputUpdate {
                raw: r#"{"sentiment":"neutral","confidence":0.5}"#.to_string(),
            }),
            output_schema: None,
            tags: None,
            name: None,
        })],
    };

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_default_schema/datapoints",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(result.ids.len(), 1);
}
