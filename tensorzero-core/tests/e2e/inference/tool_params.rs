//! Tests for inference tool_params round-trip through database storage
//!
//! This test suite verifies that DynamicToolParams correctly converts to/from
//! ToolCallConfigDatabaseInsert when storing and retrieving inferences.
//!
//! Key conversion behaviors tested:
//! 1. Round-trip: DynamicToolParams → ToolCallConfigDatabaseInsert → DynamicToolParams
//! 2. Tool partitioning: tools_available splits into allowed_tools (static) and additional_tools (dynamic)
//! 3. Lossy conversions: provider_tools and AllowedToolsChoice metadata are NOT persisted
//! 4. Edge cases: empty lists, None values, mixed static/dynamic tools

use std::collections::HashSet;

use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tensorzero::test_helpers::make_embedded_gateway;
use tensorzero::ClientExt;
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse,
};
use tensorzero_core::tool::{ProviderToolScope, ToolChoice};
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
#[tokio::test(flavor = "multi_thread")]
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

    // Step 3: Retrieve via get_inferences API (wire format with DynamicToolParams)
    let client = make_embedded_gateway().await;
    let response = client
        .get_inferences(
            vec![inference_id],
            Some("weather_helper".to_string()),
            tensorzero::InferenceOutputSource::Inference,
        )
        .await
        .unwrap();

    assert_eq!(response.inferences.len(), 1);
    let tensorzero::StoredInference::Chat(stored_inference) = &response.inferences[0] else {
        panic!("Expected Chat inference");
    };

    // Step 4: Verify DynamicToolParams correctly reconstructed
    let retrieved_tool_params = &stored_inference.tool_params;

    // Only explicitly specified tools should be in allowed tools
    let allowed_tools = retrieved_tool_params.allowed_tools.as_ref().unwrap();
    assert_eq!(allowed_tools.len(), 1);
    assert_eq!(allowed_tools[0], "get_temperature");

    // Dynamic tools should be in additional_tools
    let additional_tools = retrieved_tool_params.additional_tools.as_ref().unwrap();
    assert_eq!(additional_tools.len(), 1);
    let tool = &additional_tools[0];
    if let tensorzero_core::tool::Tool::Function(func) = tool {
        assert_eq!(func.name, "custom_weather_tool");
        assert_eq!(func.description, "A custom tool added dynamically");
        assert!(!func.strict);
    } else {
        panic!("Expected Function tool");
    }

    // Other fields should match
    assert_eq!(
        retrieved_tool_params.tool_choice,
        Some(ToolChoice::Specific("get_temperature".to_string()))
    );
    assert_eq!(retrieved_tool_params.parallel_tool_calls, Some(false));

    // IMPORTANT: provider_tools is LOSSY - should always be empty after round-trip
    // Will fix this in a follow up with databae migrations.
    assert!(
        retrieved_tool_params.provider_tools.is_empty(),
        "provider_tools should be empty after database round-trip (lossy conversion)"
    );
}

/// Test 2: Only static tools (allowed_tools only)
///
/// Tests the case where only static tools from function config are used,
/// with no additional_tools.
#[tokio::test(flavor = "multi_thread")]
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
    println!("Inference ID: {inference_id}");

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve via API
    let client = make_embedded_gateway().await;
    let response = client
        .get_inferences(
            vec![inference_id],
            Some("weather_helper".to_string()),
            tensorzero::InferenceOutputSource::Inference,
        )
        .await
        .unwrap();

    let tensorzero::StoredInference::Chat(stored_inference) = &response.inferences[0] else {
        panic!("Expected Chat inference");
    };

    println!("stored inference: {stored_inference:?}");
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
#[tokio::test(flavor = "multi_thread")]
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
    let response = client
        .get_inferences(
            vec![inference_id],
            Some("weather_helper".to_string()),
            tensorzero::InferenceOutputSource::Inference,
        )
        .await
        .unwrap();

    let tensorzero::StoredInference::Chat(stored_inference) = &response.inferences[0] else {
        panic!("Expected Chat inference");
    };

    let retrieved_tool_params = &stored_inference.tool_params;

    // If we don't specify allowed tools on the way in we don't get allowed tools on the way out.
    assert!(retrieved_tool_params.allowed_tools.is_none());

    // Dynamic tool should be in additional_tools
    let additional_tools = retrieved_tool_params.additional_tools.as_ref().unwrap();
    assert_eq!(additional_tools.len(), 1);
    let tool = &additional_tools[0];
    if let tensorzero_core::tool::Tool::Function(func) = tool {
        assert_eq!(func.name, "runtime_tool");
        assert!(func.strict);
    } else {
        panic!("Expected Function tool");
    }
}

/// Test 4: Empty tool params (None/default behavior)
///
/// Tests what happens when no tool_params are provided - should use function config defaults.
#[tokio::test(flavor = "multi_thread")]
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
#[tokio::test(flavor = "multi_thread")]
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
    let response = client
        .get_inferences(
            vec![inference_id],
            Some("weather_helper".to_string()),
            tensorzero::InferenceOutputSource::Inference,
        )
        .await
        .unwrap();

    let tensorzero::StoredInference::Chat(stored_inference) = &response.inferences[0] else {
        panic!("Expected Chat inference");
    };

    // VERIFY: provider_tools should be present after round-trip
    println!("{:?}", stored_inference.tool_params.provider_tools);
    let stored_provider_tools = &stored_inference.tool_params.provider_tools;
    assert_eq!(stored_provider_tools.len(), 1);
    let first_tool = stored_provider_tools.first().unwrap();
    assert_eq!(first_tool.scope, ProviderToolScope::Unscoped);
    assert_eq!(
        first_tool.tool,
        json!({"type": "computer_20241022",
            "name": "computer",
            "display_width_px": 1024,
            "display_height_px": 768,
            "display_number": 1
        })
    );
}

/// Test 6: Tool strictness is preserved
///
/// Verifies that the `strict` field on tools survives round-trip.
#[tokio::test(flavor = "multi_thread")]
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
    let response = client
        .get_inferences(
            vec![inference_id],
            Some("weather_helper".to_string()),
            tensorzero::InferenceOutputSource::Inference,
        )
        .await
        .unwrap();

    let tensorzero::StoredInference::Chat(stored_inference) = &response.inferences[0] else {
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
        .find(|dt| {
            if let tensorzero_core::tool::Tool::Function(func) = &dt {
                func.name == "strict_tool"
            } else {
                false
            }
        })
        .expect("Should find strict_tool");
    let non_strict_tool = additional_tools
        .iter()
        .find(|dt| {
            if let tensorzero_core::tool::Tool::Function(func) = &dt {
                func.name == "non_strict_tool"
            } else {
                false
            }
        })
        .expect("Should find non_strict_tool");

    if let tensorzero_core::tool::Tool::Function(func) = &strict_tool {
        assert!(func.strict, "strict flag should be true");
    }
    if let tensorzero_core::tool::Tool::Function(func) = &non_strict_tool {
        assert!(!func.strict, "strict flag should be false");
    }
}

/// Test 7: Multiple static tools with allowed_tools restriction
///
/// Tests that allowed_tools can restrict which static tools are available.
#[tokio::test(flavor = "multi_thread")]
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

    // We still send all tools as available
    let tools_available = tool_params
        .get("tools_available")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(tools_available.len(), 2);

    // Verify both tools are present in some order
    let tool_names: HashSet<&str> = tools_available
        .iter()
        .map(|t| t.get("name").unwrap().as_str().unwrap())
        .collect();
    assert!(tool_names.contains(&"get_temperature"));
    assert!(tool_names.contains(&"get_humidity"));

    // Retrieve via API
    let client = make_embedded_gateway().await;
    let response = client
        .get_inferences(
            vec![inference_id],
            Some("weather_helper_parallel".to_string()),
            tensorzero::InferenceOutputSource::Inference,
        )
        .await
        .unwrap();

    let tensorzero::StoredInference::Chat(stored_inference) = &response.inferences[0] else {
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
