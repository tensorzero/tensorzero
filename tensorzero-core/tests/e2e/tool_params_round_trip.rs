/// Tests for tool_params round-trip through database storage
///
/// This test suite verifies that DynamicToolParams correctly converts to/from
/// ToolCallConfigDatabaseInsert when storing and retrieving inferences and datapoints.
///
/// Key conversion behaviors tested:
/// 1. Round-trip: DynamicToolParams → ToolCallConfigDatabaseInsert → DynamicToolParams
/// 2. Tool partitioning: tools_available splits into allowed_tools (static) and additional_tools (dynamic)
/// 3. Lossy conversions: provider_tools and AllowedToolsChoice metadata are NOT persisted
/// 4. Edge cases: empty lists, None values, mixed static/dynamic tools
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use std::time::Duration;
use tensorzero::test_helpers::make_embedded_gateway;
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse,
};
use tensorzero_core::tool::ToolChoice;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Test 1: Full round-trip with all DynamicToolParams fields
///
/// Creates an inference with:
/// - allowed_tools: Some(["get_temperature"]) - static tool from function config
/// - additional_tools: Some([custom_tool]) - dynamic tool added at inference time
/// - tool_choice: Specific("get_temperature")
/// - parallel_tool_calls: Some(false)
///
/// Verifies:
/// - Inference stores correctly in ClickHouse
/// - Retrieved inference correctly reconstructs DynamicToolParams
/// - Tool partitioning works: static tools in allowed_tools, dynamic in additional_tools
#[tokio::test]
async fn test_inference_full_tool_params_round_trip() {
    let episode_id = Uuid::now_v7();

    // Define a dynamic tool to add at inference time
    let additional_tool = json!({
        "name": "custom_weather_tool",
        "description": "A custom tool added dynamically",
        "strict": false,
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["city"],
            "additionalProperties": false
        }
    });

    // Step 1: Create inference with full DynamicToolParams
    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in Brooklyn?"
                }
            ]
        },
        "stream": false,
        // DynamicToolParams
        "allowed_tools": ["get_temperature"],  // Static tool from function config
        "additional_tools": [additional_tool],  // Dynamic tool
        "tool_choice": {"specific": "get_temperature"},  // Correct serde format for ToolChoice::Specific
        "parallel_tool_calls": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    if status != StatusCode::OK {
        let error_text = response.text().await.unwrap();
        panic!("Expected 200 OK, got {status}: {error_text}");
    }
    assert_eq!(status, StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep to allow ClickHouse writes to complete
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Step 2: Retrieve from ClickHouse and verify storage format (ToolCallConfigDatabaseInsert)
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    let tool_params: Value = serde_json::from_str(tool_params).unwrap();

    // In storage, all tools are in tools_available (merged)
    let tools_available = tool_params
        .get("tools_available")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(
        tools_available.len(),
        2,
        "Should have both static and dynamic tools"
    );

    // Verify tool names are present
    let tool_names: Vec<&str> = tools_available
        .iter()
        .map(|t| t.get("name").unwrap().as_str().unwrap())
        .collect();
    assert!(tool_names.contains(&"get_temperature"));
    assert!(tool_names.contains(&"custom_weather_tool"));

    // Verify other fields
    assert_eq!(
        tool_params.get("tool_choice").unwrap(),
        &json!({"specific": "get_temperature"})
    );
    assert_eq!(
        tool_params.get("parallel_tool_calls").unwrap(),
        &json!(false)
    );

    // Step 3: Retrieve via list_inferences API (wire format with DynamicToolParams)
    let client = make_embedded_gateway().await;
    let stored_inferences = client
        .experimental_list_inferences(tensorzero::ListInferencesParams {
            function_name: Some("weather_helper"),
            ids: Some(&[inference_id]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(stored_inferences.len(), 1);
    let tensorzero::StoredInference::Chat(stored_inference) = &stored_inferences[0] else {
        panic!("Expected Chat inference");
    };

    // Step 4: Verify DynamicToolParams correctly reconstructed
    let retrieved_tool_params = &stored_inference.tool_params;

    // Static tools should be in allowed_tools
    let allowed_tools = retrieved_tool_params.allowed_tools.as_ref().unwrap();
    assert_eq!(allowed_tools.len(), 1);
    assert_eq!(allowed_tools[0], "get_temperature");

    // Dynamic tools should be in additional_tools
    let additional_tools = retrieved_tool_params.additional_tools.as_ref().unwrap();
    assert_eq!(additional_tools.len(), 1);
    assert_eq!(additional_tools[0].name, "custom_weather_tool");
    assert_eq!(
        additional_tools[0].description,
        "A custom tool added dynamically"
    );
    assert!(!additional_tools[0].strict);

    // Other fields should match
    assert_eq!(
        retrieved_tool_params.tool_choice,
        Some(ToolChoice::Specific("get_temperature".to_string()))
    );
    assert_eq!(retrieved_tool_params.parallel_tool_calls, Some(false));

    // IMPORTANT: provider_tools is LOSSY - should always be None after round-trip
    // Will fix this in a follow up with databae migrations.
    assert!(
        retrieved_tool_params.provider_tools.is_none(),
        "provider_tools should be None after database round-trip (lossy conversion)"
    );
}

/// Test 2: Only static tools (allowed_tools only)
///
/// Tests the case where only static tools from function config are used,
/// with no additional_tools.
#[tokio::test]
async fn test_inference_only_static_tools() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": "What's the temperature?"
                }
            ]
        },
        "stream": false,
        "allowed_tools": ["get_temperature"],  // Only static tool
        "tool_choice": "auto",
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve via API
    let client = make_embedded_gateway().await;
    let stored_inferences = client
        .experimental_list_inferences(tensorzero::ListInferencesParams {
            function_name: Some("weather_helper"),
            ids: Some(&[inference_id]),
            ..Default::default()
        })
        .await
        .unwrap();

    let tensorzero::StoredInference::Chat(stored_inference) = &stored_inferences[0] else {
        panic!("Expected Chat inference");
    };

    let retrieved_tool_params = &stored_inference.tool_params;

    // Should have allowed_tools
    assert_eq!(
        retrieved_tool_params.allowed_tools.as_ref().unwrap(),
        &vec!["get_temperature".to_string()]
    );

    // Should NOT have additional_tools (or should be None/empty)
    assert!(
        retrieved_tool_params.additional_tools.is_none()
            || retrieved_tool_params
                .additional_tools
                .as_ref()
                .unwrap()
                .is_empty(),
        "additional_tools should be None or empty when only static tools are used"
    );

    assert_eq!(retrieved_tool_params.tool_choice, Some(ToolChoice::Auto));
}

/// Test 3: Only dynamic tools (additional_tools only)
///
/// Tests the case where only dynamic tools are provided at inference time,
/// with no allowed_tools restriction.
#[tokio::test]
async fn test_inference_only_dynamic_tools() {
    let episode_id = Uuid::now_v7();

    let dynamic_tool = json!({
        "name": "runtime_tool",
        "description": "A tool only available at runtime",
        "strict": true,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
            "additionalProperties": false
        }
    });

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": "Check something"
                }
            ]
        },
        "stream": false,
        "additional_tools": [dynamic_tool],  // Only dynamic tool
        "tool_choice": "auto",
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve via API
    let client = make_embedded_gateway().await;
    let stored_inferences = client
        .experimental_list_inferences(tensorzero::ListInferencesParams {
            function_name: Some("weather_helper"),
            ids: Some(&[inference_id]),
            ..Default::default()
        })
        .await
        .unwrap();

    let tensorzero::StoredInference::Chat(stored_inference) = &stored_inferences[0] else {
        panic!("Expected Chat inference");
    };

    let retrieved_tool_params = &stored_inference.tool_params;

    // When no allowed_tools passed, function config tools become allowed_tools
    // (see database_insert_to_dynamic_tool_params logic)
    let allowed_tools = retrieved_tool_params.allowed_tools.as_ref().unwrap();
    assert_eq!(allowed_tools.len(), 1);
    assert_eq!(allowed_tools[0], "get_temperature"); // From function config

    // Dynamic tool should be in additional_tools
    let additional_tools = retrieved_tool_params.additional_tools.as_ref().unwrap();
    assert_eq!(additional_tools.len(), 1);
    assert_eq!(additional_tools[0].name, "runtime_tool");
    assert!(additional_tools[0].strict);
}

/// Test 4: Empty tool params (None/default behavior)
///
/// Tests what happens when no tool_params are provided - should use function config defaults.
#[tokio::test]
async fn test_inference_no_tool_params() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather?"
                }
            ]
        },
        "stream": false,
        // No tool params provided - should use function config defaults
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve from ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    let tool_params: Value = serde_json::from_str(tool_params).unwrap();

    // Should have function config tools
    let tools_available = tool_params
        .get("tools_available")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(tools_available.len(), 1);
    assert_eq!(
        tools_available[0].get("name").unwrap().as_str().unwrap(),
        "get_temperature"
    );

    // Should have function config tool_choice
    assert_eq!(
        tool_params.get("tool_choice").unwrap().as_str().unwrap(),
        "auto"
    );
}

/// Test 5: Provider tools are LOSSY
///
/// Documents that provider_tools are NOT persisted to database.
/// This is a known limitation of the current implementation.
#[tokio::test]
async fn test_provider_tools_not_persisted() {
    let episode_id = Uuid::now_v7();

    // Attempt to provide provider_tools (this field exists in DynamicToolParams)
    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather?"
                }
            ]
        },
        "stream": false,
        "allowed_tools": ["get_temperature"],
        "provider_tools": [{  // This will be lost (for now)
            "tool":
            {"type": "computer_20241022",
            "name": "computer",
            "display_width_px": 1024,
            "display_height_px": 768,
            "display_number": 1
        }}],
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve via API
    let client = make_embedded_gateway().await;
    let stored_inferences = client
        .experimental_list_inferences(tensorzero::ListInferencesParams {
            function_name: Some("weather_helper"),
            ids: Some(&[inference_id]),
            ..Default::default()
        })
        .await
        .unwrap();

    let tensorzero::StoredInference::Chat(stored_inference) = &stored_inferences[0] else {
        panic!("Expected Chat inference");
    };

    // VERIFY: provider_tools should be None after round-trip
    assert!(
        stored_inference.tool_params.provider_tools.is_none(),
        "LOSSY CONVERSION: provider_tools are not persisted to database"
    );
}

/// Test 6: Tool strictness is preserved
///
/// Verifies that the `strict` field on tools survives round-trip.
#[tokio::test]
async fn test_tool_strict_flag_preserved() {
    let episode_id = Uuid::now_v7();

    let strict_tool = json!({
        "name": "strict_tool",
        "description": "A strictly validated tool",
        "strict": true,
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "string"}
            },
            "required": ["value"],
            "additionalProperties": false
        }
    });

    let non_strict_tool = json!({
        "name": "non_strict_tool",
        "description": "A loosely validated tool",
        "strict": false,
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "string"}
            },
            "additionalProperties": false
        }
    });

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": "Test"
                }
            ]
        },
        "stream": false,
        "additional_tools": [strict_tool, non_strict_tool],
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve via API
    let client = make_embedded_gateway().await;
    let stored_inferences = client
        .experimental_list_inferences(tensorzero::ListInferencesParams {
            function_name: Some("weather_helper"),
            ids: Some(&[inference_id]),
            ..Default::default()
        })
        .await
        .unwrap();

    let tensorzero::StoredInference::Chat(stored_inference) = &stored_inferences[0] else {
        panic!("Expected Chat inference");
    };

    let additional_tools = stored_inference
        .tool_params
        .additional_tools
        .as_ref()
        .unwrap();

    // Find the tools (order might not be preserved)
    let strict_tool = additional_tools
        .iter()
        .find(|t| t.name == "strict_tool")
        .expect("Should find strict_tool");
    let non_strict_tool = additional_tools
        .iter()
        .find(|t| t.name == "non_strict_tool")
        .expect("Should find non_strict_tool");

    assert!(strict_tool.strict, "strict flag should be true");
    assert!(!non_strict_tool.strict, "strict flag should be false");
}

/// Test 7: Multiple static tools with allowed_tools restriction
///
/// Tests that allowed_tools can restrict which static tools are available.
#[tokio::test]
async fn test_allowed_tools_restriction() {
    let episode_id = Uuid::now_v7();

    // weather_helper_parallel has two static tools: get_temperature and get_humidity
    // We'll restrict to only get_temperature
    let payload = json!({
        "function_name": "weather_helper_parallel",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": "What's the temperature?"
                }
            ]
        },
        "stream": false,
        "allowed_tools": ["get_temperature"],  // Only one of the two static tools
        "parallel_tool_calls": true,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve from ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    let tool_params: Value = serde_json::from_str(tool_params).unwrap();

    // Should only have get_temperature in storage
    let tools_available = tool_params
        .get("tools_available")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(tools_available.len(), 1);
    assert_eq!(
        tools_available[0].get("name").unwrap().as_str().unwrap(),
        "get_temperature"
    );

    // Retrieve via API
    let client = make_embedded_gateway().await;
    let stored_inferences = client
        .experimental_list_inferences(tensorzero::ListInferencesParams {
            function_name: Some("weather_helper_parallel"),
            ids: Some(&[inference_id]),
            ..Default::default()
        })
        .await
        .unwrap();

    let tensorzero::StoredInference::Chat(stored_inference) = &stored_inferences[0] else {
        panic!("Expected Chat inference");
    };

    // Should have only get_temperature in allowed_tools
    let allowed_tools = stored_inference.tool_params.allowed_tools.as_ref().unwrap();
    assert_eq!(allowed_tools.len(), 1);
    assert_eq!(allowed_tools[0], "get_temperature");

    // Should have no additional_tools
    assert!(
        stored_inference.tool_params.additional_tools.is_none()
            || stored_inference
                .tool_params
                .additional_tools
                .as_ref()
                .unwrap()
                .is_empty()
    );
}

// ============================================================================
// DATASET ROUND-TRIP TESTS
// ============================================================================
//
// The following tests verify that DynamicToolParams correctly round-trips through
// the datapoint create/update/retrieve flow. Key differences from inference tests:
// 1. Datapoint updates create NEW IDs and stale old datapoints
// 2. Tests use direct ClickHouse insertion then HTTP retrieval
// 3. Tests cover both get_datapoints (by ID) and list_datapoints (pagination)
// 4. Update API uses Option<Option<DynamicToolParams>> for omit/null/value

use tensorzero_core::db::clickhouse::test_helpers::clickhouse_flush_async_insert;
use tensorzero_core::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries,
};
use tensorzero_core::inference::types::{
    Arguments, ContentBlockChatOutput, Role, StoredInput, StoredInputMessage,
    StoredInputMessageContent, System, Text,
};
use tensorzero_core::tool::ToolCallConfigDatabaseInsert;

/// Test 5.1: Full datapoint tool params round-trip
///
/// Creates a datapoint via ClickHouse with full DynamicToolParams, then retrieves
/// it via the get_datapoints HTTP API to verify tool partitioning works correctly.
#[tokio::test]
async fn test_datapoint_full_tool_params_round_trip() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-dp-tools-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Define custom dynamic tool (same as inference tests for consistency)
    let custom_tool = tensorzero_core::tool::Tool {
        name: "custom_weather_tool".to_string(),
        description: "A custom tool added dynamically".to_string(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"],
            "additionalProperties": false
        }))
        .unwrap(),
        strict: false,
    };

    // Get the static tool from function config to create proper ToolCallConfigDatabaseInsert
    let get_temp_tool = tensorzero_core::tool::Tool {
        name: "get_temperature".to_string(),
        description: "Get the current temperature in a given location".to_string(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["fahrenheit", "celsius"]}
            },
            "required": ["location"]
        }))
        .unwrap(),
        strict: false,
    };

    // Create tool_params in storage format (ToolCallConfigDatabaseInsert)
    // This has ALL tools in tools_available (merged static + dynamic)
    let tool_params = Some(ToolCallConfigDatabaseInsert {
        tools_available: vec![get_temp_tool, custom_tool],
        tool_choice: ToolChoice::Specific("get_temperature".to_string()),
        parallel_tool_calls: Some(false),
    });

    // Create datapoint via ClickHouse
    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "weather_helper".to_string(),
        name: Some("Test Tool Params Round-Trip".to_string()),
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "WeatherBot"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "What's the weather in Brooklyn?".to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Response".to_string(),
        })]),
        tool_params,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Retrieve via get_datapoints HTTP API
    let resp = http_client
        .post(get_gateway_endpoint("/v1/datasets/get_datapoints"))
        .json(&json!({
            "ids": [datapoint_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "get_datapoints request failed"
    );
    let resp_json: Value = resp.json().await.unwrap();

    // Verify response structure
    let datapoints = resp_json["datapoints"].as_array().unwrap();
    assert_eq!(datapoints.len(), 1);
    let dp = &datapoints[0];

    // Verify basic fields
    assert_eq!(dp["id"], datapoint_id.to_string());
    assert_eq!(dp["dataset_name"], dataset_name);
    assert_eq!(dp["function_name"], "weather_helper");

    // Verify DynamicToolParams structure: tools should be partitioned
    let tool_params = &dp["tool_params"];
    assert!(
        !tool_params.is_null(),
        "tool_params should not be null after round-trip"
    );

    // Static tool (from function config) should be in allowed_tools
    let allowed_tools = tool_params["allowed_tools"].as_array().unwrap();
    assert_eq!(allowed_tools.len(), 1);
    assert_eq!(allowed_tools[0], "get_temperature");

    // Dynamic tool should be in additional_tools
    let additional_tools = tool_params["additional_tools"].as_array().unwrap();
    assert_eq!(additional_tools.len(), 1);
    assert_eq!(additional_tools[0]["name"], "custom_weather_tool");
    assert_eq!(
        additional_tools[0]["description"],
        "A custom tool added dynamically"
    );
    assert_eq!(additional_tools[0]["strict"], false);

    // Other fields should be preserved
    assert_eq!(
        tool_params["tool_choice"],
        json!({"specific": "get_temperature"})
    );
    assert_eq!(tool_params["parallel_tool_calls"], false);

    // provider_tools should be None (lossy conversion)
    assert!(
        tool_params["provider_tools"].is_null(),
        "provider_tools should be null after round-trip (lossy conversion)"
    );
}

/// Test 5.2: Update datapoint tool params
///
/// Verifies that updating a datapoint with new tool_params creates a new ID,
/// stales the old datapoint, and correctly converts the new tool_params.
#[tokio::test]
async fn test_datapoint_update_tool_params() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-dp-update-{}", Uuid::now_v7());
    let original_id = Uuid::now_v7();

    // Create original datapoint with initial tool_params
    let get_temp_tool = tensorzero_core::tool::Tool {
        name: "get_temperature".to_string(),
        description: "Get the current temperature in a given location".to_string(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }))
        .unwrap(),
        strict: false,
    };

    let original_tool_params = Some(ToolCallConfigDatabaseInsert {
        tools_available: vec![get_temp_tool],
        tool_choice: ToolChoice::Auto,
        parallel_tool_calls: Some(false),
    });

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "weather_helper".to_string(),
        name: Some("Update Test".to_string()),
        id: original_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Original message".to_string(),
                })],
            }],
        },
        output: None,
        tool_params: original_tool_params,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Update with new tool_params via HTTP
    let updated_tool = json!({
        "name": "updated_tool",
        "description": "A new tool for the update",
        "strict": true,
        "parameters": {
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            },
            "required": ["param"],
            "additionalProperties": false
        }
    });

    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "id": original_id.to_string(),
                "tool_params": {
                    "allowed_tools": ["get_temperature"],
                    "additional_tools": [updated_tool],
                    "tool_choice": {"specific": "updated_tool"},
                    "parallel_tool_calls": true
                }
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "Update request failed");
    let resp_json: Value = resp.json().await.unwrap();
    let new_ids = resp_json["ids"].as_array().unwrap();
    assert_eq!(new_ids.len(), 1);
    let new_id: Uuid = new_ids[0].as_str().unwrap().parse().unwrap();
    assert_ne!(new_id, original_id, "Should create a new datapoint ID");

    // Wait for writes
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Verify old datapoint is staled
    let old_datapoint = clickhouse
        .get_datapoint(&tensorzero::GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: original_id,
            allow_stale: Some(true),
        })
        .await
        .unwrap();
    let tensorzero::StoredDatapoint::Chat(old_dp) = old_datapoint else {
        panic!("Expected chat datapoint");
    };
    assert!(old_dp.staled_at.is_some(), "Old datapoint should be staled");

    // Retrieve new datapoint via HTTP and verify updated tool_params
    let resp = http_client
        .post(get_gateway_endpoint("/v1/datasets/get_datapoints"))
        .json(&json!({
            "ids": [new_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let resp_json: Value = resp.json().await.unwrap();
    let datapoints = resp_json["datapoints"].as_array().unwrap();
    assert_eq!(datapoints.len(), 1);

    let dp = &datapoints[0];
    let tool_params = &dp["tool_params"];

    // Verify updated tool_params
    assert_eq!(
        tool_params["allowed_tools"],
        json!(["get_temperature"]),
        "allowed_tools should be updated"
    );

    let additional_tools = tool_params["additional_tools"].as_array().unwrap();
    assert_eq!(additional_tools.len(), 1);
    assert_eq!(additional_tools[0]["name"], "updated_tool");
    assert_eq!(additional_tools[0]["strict"], true);

    assert_eq!(
        tool_params["tool_choice"],
        json!({"specific": "updated_tool"})
    );
    assert_eq!(tool_params["parallel_tool_calls"], true);
}

/// Test 5.3: List datapoints with tool params
///
/// Creates multiple datapoints with different tool configurations and verifies
/// they all appear correctly in the list_datapoints endpoint response.
#[tokio::test]
async fn test_list_datapoints_with_tool_params() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-list-tools-{}", Uuid::now_v7());

    // Create 3 datapoints with different tool configs
    let dp1_id = Uuid::now_v7();
    let dp2_id = Uuid::now_v7();
    let dp3_id = Uuid::now_v7();

    let base_tool = tensorzero_core::tool::Tool {
        name: "get_temperature".to_string(),
        description: "Get temperature".to_string(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }))
        .unwrap(),
        strict: false,
    };

    let custom_tool_1 = tensorzero_core::tool::Tool {
        name: "tool_1".to_string(),
        description: "First tool".to_string(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {"a": {"type": "string"}}
        }))
        .unwrap(),
        strict: false,
    };

    let custom_tool_2 = tensorzero_core::tool::Tool {
        name: "tool_2".to_string(),
        description: "Second tool".to_string(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {"b": {"type": "string"}}
        }))
        .unwrap(),
        strict: true,
    };

    // Datapoint 1: Only static tool
    let dp1 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "weather_helper".to_string(),
        name: Some("DP1".to_string()),
        id: dp1_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test 1".to_string(),
                })],
            }],
        },
        output: None,
        tool_params: Some(ToolCallConfigDatabaseInsert {
            tools_available: vec![base_tool.clone()],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        }),
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    // Datapoint 2: Static + one dynamic tool
    let dp2 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "weather_helper".to_string(),
        name: Some("DP2".to_string()),
        id: dp2_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test 2".to_string(),
                })],
            }],
        },
        output: None,
        tool_params: Some(ToolCallConfigDatabaseInsert {
            tools_available: vec![base_tool.clone(), custom_tool_1],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: Some(false),
        }),
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    // Datapoint 3: Static + different dynamic tool with strict
    let dp3 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "weather_helper".to_string(),
        name: Some("DP3".to_string()),
        id: dp3_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test 3".to_string(),
                })],
            }],
        },
        output: None,
        tool_params: Some(ToolCallConfigDatabaseInsert {
            tools_available: vec![base_tool, custom_tool_2],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: Some(true),
        }),
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[dp1, dp2, dp3])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // List datapoints via HTTP
    let resp = http_client
        .post(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/list_datapoints"
        )))
        .json(&json!({
            "function_name": "weather_helper",
            "page_size": 10
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let resp_json: Value = resp.json().await.unwrap();
    let datapoints = resp_json["datapoints"].as_array().unwrap();
    assert_eq!(datapoints.len(), 3, "Should have 3 datapoints");

    // Helper to find datapoint by ID
    let find_dp = |id: &Uuid| -> &Value {
        datapoints
            .iter()
            .find(|dp| dp["id"] == id.to_string())
            .expect(&format!("Should find datapoint {}", id))
    };

    // Verify DP1: Only static tool
    let dp1_json = find_dp(&dp1_id);
    let tp1 = &dp1_json["tool_params"];
    assert_eq!(tp1["allowed_tools"], json!(["get_temperature"]));
    assert!(
        tp1["additional_tools"].is_null() || tp1["additional_tools"].as_array().unwrap().is_empty()
    );
    assert_eq!(tp1["tool_choice"], "auto");

    // Verify DP2: Static + one dynamic
    let dp2_json = find_dp(&dp2_id);
    let tp2 = &dp2_json["tool_params"];
    assert_eq!(tp2["allowed_tools"], json!(["get_temperature"]));
    let add_tools_2 = tp2["additional_tools"].as_array().unwrap();
    assert_eq!(add_tools_2.len(), 1);
    assert_eq!(add_tools_2[0]["name"], "tool_1");
    assert_eq!(tp2["tool_choice"], "required");
    assert_eq!(tp2["parallel_tool_calls"], false);

    // Verify DP3: Static + different dynamic with strict
    let dp3_json = find_dp(&dp3_id);
    let tp3 = &dp3_json["tool_params"];
    assert_eq!(tp3["allowed_tools"], json!(["get_temperature"]));
    let add_tools_3 = tp3["additional_tools"].as_array().unwrap();
    assert_eq!(add_tools_3.len(), 1);
    assert_eq!(add_tools_3[0]["name"], "tool_2");
    assert_eq!(add_tools_3[0]["strict"], true);
    assert_eq!(tp3["tool_choice"], "none");
    assert_eq!(tp3["parallel_tool_calls"], true);
}

/// Test 5.4: Datapoint with only static tools
///
/// Mirrors test_inference_only_static_tools but for datapoints.
#[tokio::test]
async fn test_datapoint_only_static_tools() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-dp-static-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let static_tool = tensorzero_core::tool::Tool {
        name: "get_temperature".to_string(),
        description: "Get temperature".to_string(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }))
        .unwrap(),
        strict: false,
    };

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "weather_helper".to_string(),
        name: None,
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test".to_string(),
                })],
            }],
        },
        output: None,
        tool_params: Some(ToolCallConfigDatabaseInsert {
            tools_available: vec![static_tool],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        }),
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Retrieve via HTTP
    let resp = http_client
        .post(get_gateway_endpoint("/v1/datasets/get_datapoints"))
        .json(&json!({
            "ids": [datapoint_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let resp_json: Value = resp.json().await.unwrap();
    let dp = &resp_json["datapoints"][0];
    let tool_params = &dp["tool_params"];

    // Should have allowed_tools
    assert_eq!(tool_params["allowed_tools"], json!(["get_temperature"]));

    // Should NOT have additional_tools (or should be null/empty)
    assert!(
        tool_params["additional_tools"].is_null()
            || tool_params["additional_tools"].as_array().unwrap().is_empty(),
        "additional_tools should be null or empty when only static tools are used"
    );

    assert_eq!(tool_params["tool_choice"], "auto");
}

/// Test 5.5: Datapoint with only dynamic tools
///
/// Mirrors test_inference_only_dynamic_tools but for datapoints.
#[tokio::test]
async fn test_datapoint_only_dynamic_tools() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-dp-dynamic-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Include both static tool from config AND dynamic tool in storage
    let static_tool = tensorzero_core::tool::Tool {
        name: "get_temperature".to_string(),
        description: "Get temperature".to_string(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }))
        .unwrap(),
        strict: false,
    };

    let dynamic_tool = tensorzero_core::tool::Tool {
        name: "runtime_tool".to_string(),
        description: "A tool only available at runtime".to_string(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }))
        .unwrap(),
        strict: true,
    };

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "weather_helper".to_string(),
        name: None,
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test".to_string(),
                })],
            }],
        },
        output: None,
        tool_params: Some(ToolCallConfigDatabaseInsert {
            // In storage, both tools are in tools_available
            tools_available: vec![static_tool, dynamic_tool],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        }),
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Retrieve via HTTP
    let resp = http_client
        .post(get_gateway_endpoint("/v1/datasets/get_datapoints"))
        .json(&json!({
            "ids": [datapoint_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let resp_json: Value = resp.json().await.unwrap();
    let dp = &resp_json["datapoints"][0];
    let tool_params = &dp["tool_params"];

    // Static tool from function config should be in allowed_tools
    let allowed_tools = tool_params["allowed_tools"].as_array().unwrap();
    assert_eq!(allowed_tools.len(), 1);
    assert_eq!(allowed_tools[0], "get_temperature");

    // Dynamic tool should be in additional_tools
    let additional_tools = tool_params["additional_tools"].as_array().unwrap();
    assert_eq!(additional_tools.len(), 1);
    assert_eq!(additional_tools[0]["name"], "runtime_tool");
    assert_eq!(additional_tools[0]["strict"], true);
}

/// Test 5.6: Datapoint tool params null vs omitted
///
/// Tests the Option<Option<DynamicToolParams>> triple-option pattern for updates.
/// - Omit field: no change
/// - Set to null: removes tool_params
/// - Set to value: updates tool_params
#[tokio::test]
async fn test_datapoint_tool_params_three_states() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-dp-three-states-{}", Uuid::now_v7());

    // Create datapoint with tool_params
    let original_id = Uuid::now_v7();
    let tool = tensorzero_core::tool::Tool {
        name: "get_temperature".to_string(),
        description: "Get temperature".to_string(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }))
        .unwrap(),
        strict: false,
    };

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "weather_helper".to_string(),
        name: None,
        id: original_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Original".to_string(),
                })],
            }],
        },
        output: None,
        tool_params: Some(ToolCallConfigDatabaseInsert {
            tools_available: vec![tool],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        }),
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Test Case 1: Omit tool_params field -> no change to tool_params
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "id": original_id.to_string(),
                "tags": {"updated": "true"}
                // tool_params OMITTED - should not change
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let new_id_1: Uuid = resp.json::<Value>().await.unwrap()["ids"][0]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let resp1 = http_client
        .post(get_gateway_endpoint("/v1/datasets/get_datapoints"))
        .json(&json!({"ids": [new_id_1.to_string()]}))
        .send()
        .await
        .unwrap();

    let dp1 = &resp1.json::<Value>().await.unwrap()["datapoints"][0];
    assert!(
        !dp1["tool_params"].is_null(),
        "tool_params should still exist when field is omitted"
    );
    assert_eq!(
        dp1["tool_params"]["allowed_tools"],
        json!(["get_temperature"])
    );

    // Test Case 2: Set tool_params to null -> removes tool_params
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "id": new_id_1.to_string(),
                "tool_params": null  // Explicitly null - should remove
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let new_id_2: Uuid = resp.json::<Value>().await.unwrap()["ids"][0]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let resp2 = http_client
        .post(get_gateway_endpoint("/v1/datasets/get_datapoints"))
        .json(&json!({"ids": [new_id_2.to_string()]}))
        .send()
        .await
        .unwrap();

    let dp2 = &resp2.json::<Value>().await.unwrap()["datapoints"][0];
    assert!(
        dp2["tool_params"].is_null(),
        "tool_params should be null when explicitly set to null"
    );

    // Test Case 3: Set tool_params to new value -> updates tool_params
    let new_tool = json!({
        "name": "new_tool",
        "description": "New tool",
        "strict": true,
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string"}}
        }
    });

    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "id": new_id_2.to_string(),
                "tool_params": {
                    "additional_tools": [new_tool],
                    "tool_choice": "required"
                }
            }]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let new_id_3: Uuid = resp.json::<Value>().await.unwrap()["ids"][0]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let resp3 = http_client
        .post(get_gateway_endpoint("/v1/datasets/get_datapoints"))
        .json(&json!({"ids": [new_id_3.to_string()]}))
        .send()
        .await
        .unwrap();

    let dp3 = &resp3.json::<Value>().await.unwrap()["datapoints"][0];
    let tp3 = &dp3["tool_params"];
    assert!(!tp3.is_null(), "tool_params should be set to new value");
    assert_eq!(tp3["tool_choice"], "required");

    // When only additional_tools provided, function config tools go into allowed_tools
    assert_eq!(tp3["allowed_tools"], json!(["get_temperature"]));

    let add_tools = tp3["additional_tools"].as_array().unwrap();
    assert_eq!(add_tools.len(), 1);
    assert_eq!(add_tools[0]["name"], "new_tool");
    assert_eq!(add_tools[0]["strict"], true);
}

/// Test 5.7: Datapoint with no tool params
///
/// Verifies handling when a datapoint has no tool_params at all.
#[tokio::test]
async fn test_datapoint_no_tool_params() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-dp-no-tools-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "weather_helper".to_string(),
        name: None,
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Test".to_string(),
                })],
            }],
        },
        output: None,
        tool_params: None,  // No tool params
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Retrieve via HTTP
    let resp = http_client
        .post(get_gateway_endpoint("/v1/datasets/get_datapoints"))
        .json(&json!({
            "ids": [datapoint_id.to_string()]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let resp_json: Value = resp.json().await.unwrap();
    let dp = &resp_json["datapoints"][0];

    // tool_params should be null or not present when None
    assert!(
        dp["tool_params"].is_null(),
        "tool_params should be null when not provided"
    );
}
