use reqwest::Client;
use serde_json::json;
use std::sync::Arc;
use uuid::Uuid;

use tensorzero_core::config::Config;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::datasets::{DatasetQueries, GetDatapointsParams};
use tensorzero_core::endpoints::datasets::v1::types::CreateDatapointsResponse;

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

#[tokio::test]
async fn test_create_chat_datapoint_basic() {
    let client = Client::new();
    let (clickhouse, _config) = get_test_setup().await;

    let request = json!({
        "datapoints": [{
            "type": "chat",
            "function_name": "write_haiku",
            "input": {
                "system": {"assistant_name": "AI Assistant"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Write a haiku about coding"}]
                }]
            },
            "output": [{
                "type": "text",
                "text": "Code flows like water\nBugs emerge from the shadows\nRefactor brings peace"
            }],
            "name": "Test Haiku Datapoint"
        }]
    });

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
        page_size: 10,
        offset: 0,
        allow_stale: false,
        filter: None,
    };

    let datapoints = clickhouse.get_datapoints(&params).await.unwrap();
    assert_eq!(datapoints.len(), 1);
    assert_eq!(datapoints[0].id(), result.ids[0]);
}

#[tokio::test]
async fn test_create_json_datapoint_basic() {
    let client = Client::new();
    let (clickhouse, _config) = get_test_setup().await;

    let request = json!({
        "datapoints": [{
            "type": "json",
            "function_name": "analyze_sentiment",
            "input": {
                "system": {"assistant_name": "AI Assistant"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "This product is amazing!"}]
                }]
            },
            "output": {
                "sentiment": "positive",
                "confidence": 0.95
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["sentiment", "confidence"]
            },
            "name": "Test Sentiment Datapoint"
        }]
    });

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
        page_size: 10,
        offset: 0,
        allow_stale: false,
        filter: None,
    };

    let datapoints = clickhouse.get_datapoints(&params).await.unwrap();
    assert_eq!(datapoints.len(), 1);
    assert_eq!(datapoints[0].id(), result.ids[0]);
}

#[tokio::test]
async fn test_create_multiple_datapoints() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let request = json!({
        "datapoints": [
            {
                "type": "chat",
                "function_name": "write_haiku",
                "input": {
                    "system": {"assistant_name": "AI Assistant"},
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "text", "value": "Write a haiku about nature"}]
                    }]
                },
                "output": [{
                    "type": "text",
                    "text": "Leaves fall silently\nWhispering ancient secrets\nNature's quiet dance"
                }]
            },
            {
                "type": "chat",
                "function_name": "write_haiku",
                "input": {
                    "system": {"assistant_name": "AI Assistant"},
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "text", "value": "Write a haiku about technology"}]
                    }]
                },
                "output": [{
                    "type": "text",
                    "text": "Silicon whispers\nElectrons dance through circuits\nFuture unfolds fast"
                }]
            },
            {
                "type": "json",
                "function_name": "analyze_sentiment",
                "input": {
                    "system": {"assistant_name": "AI Assistant"},
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "text", "value": "I love this!"}]
                    }]
                },
                "output": {
                    "sentiment": "positive",
                    "confidence": 0.98
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["sentiment", "confidence"]
                }
            }
        ]
    });

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

#[tokio::test]
async fn test_create_chat_datapoint_with_tools() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let request = json!({
        "datapoints": [{
            "type": "chat",
            "function_name": "write_haiku",
            "input": {
                "system": {"assistant_name": "AI Assistant"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Write a haiku"}]
                }]
            },
            "output": [{
                "type": "text",
                "text": "A simple haiku"
            }],
            "tools": [{
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }],
            "tool_choice": "auto"
        }]
    });

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

#[tokio::test]
async fn test_create_datapoint_with_tags() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let request = json!({
        "datapoints": [{
            "type": "chat",
            "function_name": "write_haiku",
            "input": {
                "system": {"assistant_name": "AI Assistant"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Write a haiku"}]
                }]
            },
            "output": [{
                "type": "text",
                "text": "Tagged haiku here"
            }],
            "tags": {
                "environment": "test",
                "version": "1.0",
                "quality": "high"
            }
        }]
    });

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

#[tokio::test]
async fn test_create_datapoint_invalid_function() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let request = json!({
        "datapoints": [{
            "type": "chat",
            "function_name": "nonexistent_function",
            "input": {
                "system": {"assistant_name": "AI Assistant"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Test"}]
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

#[tokio::test]
async fn test_create_datapoint_wrong_function_type() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    // Try to create a JSON datapoint for a chat function
    let request = json!({
        "datapoints": [{
            "type": "json",
            "function_name": "write_haiku",
            "input": {
                "system": {"assistant_name": "AI Assistant"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Test"}]
                }]
            },
            "output": {"data": "test"},
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

#[tokio::test]
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

#[tokio::test]
async fn test_create_json_datapoint_invalid_schema() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let request = json!({
        "datapoints": [{
            "type": "json",
            "function_name": "analyze_sentiment",
            "input": {
                "system": {"assistant_name": "AI Assistant"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Test"}]
                }]
            },
            "output": {
                "sentiment": "positive",
                "confidence": "not a number"
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["sentiment", "confidence"]
            }
        }]
    });

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_invalid_schema/datapoints",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn test_create_datapoint_with_episode_id() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let episode_id = Uuid::now_v7();

    let request = json!({
        "datapoints": [{
            "type": "chat",
            "function_name": "write_haiku",
            "episode_id": episode_id,
            "input": {
                "system": {"assistant_name": "AI Assistant"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Write a haiku"}]
                }]
            },
            "output": [{
                "type": "text",
                "text": "Episode haiku"
            }]
        }]
    });

    let response = client
        .post(get_gateway_endpoint("/v1/datasets/test_episode/datapoints"))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(result.ids.len(), 1);
}

#[tokio::test]
async fn test_create_chat_datapoint_string_output() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    // Test that string output is accepted and parsed correctly
    let request = json!({
        "datapoints": [{
            "type": "chat",
            "function_name": "write_haiku",
            "input": {
                "system": {"assistant_name": "AI Assistant"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Write a haiku"}]
                }]
            },
            "output": "Simple string output"
        }]
    });

    let response = client
        .post(get_gateway_endpoint(
            "/v1/datasets/test_string_output/datapoints",
        ))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CreateDatapointsResponse = response.json().await.unwrap();
    assert_eq!(result.ids.len(), 1);
}

#[tokio::test]
async fn test_create_datapoint_without_output() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    let request = json!({
        "datapoints": [{
            "type": "chat",
            "function_name": "write_haiku",
            "input": {
                "system": {"assistant_name": "AI Assistant"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Write a haiku"}]
                }]
            }
        }]
    });

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

#[tokio::test]
async fn test_create_json_datapoint_default_schema() {
    let client = Client::new();
    let (_clickhouse, _config) = get_test_setup().await;

    // Test that output_schema is optional and defaults to function's schema
    let request = json!({
        "datapoints": [{
            "type": "json",
            "function_name": "analyze_sentiment",
            "input": {
                "system": {"assistant_name": "AI Assistant"},
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "value": "Test"}]
                }]
            },
            "output": {
                "sentiment": "neutral",
                "confidence": 0.5
            }
        }]
    });

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
