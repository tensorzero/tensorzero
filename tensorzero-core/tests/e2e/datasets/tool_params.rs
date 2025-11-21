//! Tests for datapoint tool_params round-trip through database storage
//!
//! This test suite verifies that DynamicToolParams correctly round-trips through
//! the datapoint create/update/retrieve flow. Key differences from inference tests:
//! 1. Datapoint updates create NEW IDs and stale old datapoints
//! 2. Tests use direct ClickHouse insertion then HTTP retrieval
//! 3. Tests cover both get_datapoints (by ID) and list_datapoints (pagination)
//! 4. Update API uses Option<Option<DynamicToolParams>> for omit/null/value

use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use std::time::Duration;
use tensorzero_core::db::clickhouse::test_helpers::{
    clickhouse_flush_async_insert, get_clickhouse,
};
use tensorzero_core::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries,
};
use tensorzero_core::inference::types::{
    Arguments, ContentBlockChatOutput, Role, StoredInput, StoredInputMessage,
    StoredInputMessageContent, System, Text,
};
use tensorzero_core::tool::{
    AllowedTools, AllowedToolsChoice, ProviderTool, ProviderToolScope, Tool,
    ToolCallConfigDatabaseInsert, ToolChoice,
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Test 5.1: Full datapoint tool params round-trip
///
/// Creates a datapoint via ClickHouse with full DynamicToolParams, then retrieves
/// it via the get_datapoints HTTP API to verify tool partitioning works correctly.
#[tokio::test(flavor = "multi_thread")]
async fn test_datapoint_full_tool_params_round_trip() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-dp-tools-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Define custom dynamic tool (same as inference tests for consistency)
    let custom_tool = tensorzero_core::tool::FunctionTool {
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
    let get_temp_tool = tensorzero_core::tool::FunctionTool {
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
    // This has ALL tools in dynamic_tools (merged static + dynamic)
    let tool_params = Some(ToolCallConfigDatabaseInsert::new_for_test(
        vec![Tool::Function(custom_tool.clone())],
        vec![ProviderTool {
            scope: ProviderToolScope::Unscoped,
            tool: json!({"foo": "bar"}),
        }],
        AllowedTools {
            tools: [get_temp_tool.name.clone(), custom_tool.name.clone()]
                .into_iter()
                .collect(),
            choice: AllowedToolsChoice::Explicit,
        },
        ToolChoice::Specific("get_temperature".to_string()),
        Some(false),
    ));

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
    // With flatten, tool params are at the top level, not nested under "tool_params"

    // Static tool (from function config) should be in allowed_tools
    let allowed_tools = dp["allowed_tools"].as_array().unwrap();
    assert_eq!(allowed_tools.len(), 2);
    assert_eq!(allowed_tools[0], "get_temperature");
    assert_eq!(allowed_tools[1], "custom_weather_tool");

    // Dynamic tool should be in additional_tools
    let additional_tools = dp["additional_tools"].as_array().unwrap();
    assert_eq!(additional_tools.len(), 1);
    assert_eq!(additional_tools[0]["name"], "custom_weather_tool");
    assert_eq!(
        additional_tools[0]["description"],
        "A custom tool added dynamically"
    );
    assert_eq!(additional_tools[0]["strict"], false);

    // Other fields should be preserved
    assert_eq!(dp["tool_choice"], json!({"specific": "get_temperature"}));
    assert_eq!(dp["parallel_tool_calls"], false);

    // provider_tools should now be preserved with scope and nested tool field
    let provider_tools = dp["provider_tools"].as_array().unwrap();
    assert_eq!(provider_tools.len(), 1);
    assert!(provider_tools[0]["scope"].is_null());
    assert_eq!(provider_tools[0]["tool"], json!({"foo": "bar"}));
}

/// Test 5.2: Update datapoint tool params
///
/// Verifies that updating a datapoint with new tool_params creates a new ID,
/// stales the old datapoint, and correctly converts the new tool_params.
#[tokio::test(flavor = "multi_thread")]
async fn test_datapoint_update_tool_params() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-dp-update-{}", Uuid::now_v7());
    let original_id = Uuid::now_v7();

    // Create original datapoint with initial tool_params
    let get_temp_tool = tensorzero_core::tool::FunctionTool {
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

    let original_tool_params = Some(ToolCallConfigDatabaseInsert::new_for_test(
        vec![Tool::Function(get_temp_tool.clone())],
        vec![],
        AllowedTools {
            tools: vec![get_temp_tool.name.clone()],
            choice: AllowedToolsChoice::Explicit,
        },
        ToolChoice::Auto,
        Some(false),
    ));

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
                "allowed_tools": ["get_temperature"],
                "additional_tools": [updated_tool],
                "tool_choice": {"specific": "updated_tool"},
                "parallel_tool_calls": true
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

    // Verify updated tool_params (flattened at top level)
    assert_eq!(dp["allowed_tools"], json!(["get_temperature"]),);

    let additional_tools = dp["additional_tools"].as_array().unwrap();
    assert_eq!(additional_tools.len(), 1);
    assert_eq!(additional_tools[0]["name"], "updated_tool");
    assert_eq!(additional_tools[0]["strict"], true);

    assert_eq!(dp["tool_choice"], json!({"specific": "updated_tool"}));
    assert_eq!(dp["parallel_tool_calls"], true);
}

/// Test 5.3: List datapoints with tool params
///
/// Creates multiple datapoints with different tool configurations and verifies
/// they all appear correctly in the list_datapoints endpoint response.
#[tokio::test(flavor = "multi_thread")]
async fn test_list_datapoints_with_tool_params() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-list-tools-{}", Uuid::now_v7());

    // Create 3 datapoints with different tool configs
    let dp1_id = Uuid::now_v7();
    let dp2_id = Uuid::now_v7();
    let dp3_id = Uuid::now_v7();

    let base_tool = tensorzero_core::tool::FunctionTool {
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

    let custom_tool_1 = tensorzero_core::tool::FunctionTool {
        name: "tool_1".to_string(),
        description: "First tool".to_string(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {"a": {"type": "string"}}
        }))
        .unwrap(),
        strict: false,
    };

    let custom_tool_2 = tensorzero_core::tool::FunctionTool {
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
        tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
            vec![],
            vec![],
            AllowedTools {
                tools: vec![base_tool.name.clone()],
                choice: AllowedToolsChoice::Explicit,
            },
            ToolChoice::Auto,
            None,
        )),
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
        tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
            vec![Tool::Function(custom_tool_1.clone())],
            vec![],
            AllowedTools {
                tools: vec![base_tool.name.clone(), custom_tool_1.name.clone()],
                choice: AllowedToolsChoice::Explicit,
            },
            ToolChoice::Required,
            Some(false),
        )),
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
        tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
            vec![Tool::Function(custom_tool_2.clone())],
            vec![],
            AllowedTools {
                tools: vec![custom_tool_2.name.clone()],
                choice: AllowedToolsChoice::Explicit,
            },
            ToolChoice::None,
            Some(true),
        )),
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
            "limit": 10
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
            .unwrap_or_else(|| panic!("Should find datapoint {id}"))
    };

    // Verify DP1: Only static tool (flattened)
    let dp1_json = find_dp(&dp1_id);
    assert_eq!(dp1_json["allowed_tools"], json!(["get_temperature"]));
    assert!(
        dp1_json["additional_tools"].is_null()
            || dp1_json["additional_tools"].as_array().unwrap().is_empty()
    );
    assert_eq!(dp1_json["tool_choice"], "auto");

    // Verify DP2: Static + one dynamic (flattened)
    let dp2_json = find_dp(&dp2_id);
    assert_eq!(
        dp2_json["allowed_tools"],
        json!(["get_temperature", "tool_1"])
    );
    let add_tools_2 = dp2_json["additional_tools"].as_array().unwrap();
    assert_eq!(add_tools_2.len(), 1);
    assert_eq!(add_tools_2[0]["name"], "tool_1");
    assert_eq!(dp2_json["tool_choice"], "required");
    assert_eq!(dp2_json["parallel_tool_calls"], false);

    // Verify DP3: Static + different dynamic with strict (flattened)
    let dp3_json = find_dp(&dp3_id);
    assert_eq!(dp3_json["allowed_tools"], json!(["tool_2"]));
    let add_tools_3 = dp3_json["additional_tools"].as_array().unwrap();
    assert_eq!(add_tools_3.len(), 1);
    assert_eq!(add_tools_3[0]["name"], "tool_2");
    assert_eq!(add_tools_3[0]["strict"], true);
    assert_eq!(dp3_json["tool_choice"], "none");
    assert_eq!(dp3_json["parallel_tool_calls"], true);
}

/// Test 5.4: Datapoint with only static tools
///
/// Mirrors test_inference_only_static_tools but for datapoints.
#[tokio::test(flavor = "multi_thread")]
async fn test_datapoint_only_static_tools() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-dp-static-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let static_tool = tensorzero_core::tool::FunctionTool {
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
        tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
            vec![],
            vec![],
            AllowedTools {
                tools: [static_tool.name.clone()].into_iter().collect(),
                choice: AllowedToolsChoice::FunctionDefault,
            },
            ToolChoice::Auto,
            None,
        )),
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

    // Should have allowed_tools (flattened)
    assert!(dp["allowed_tools"].is_null());

    // Should NOT have additional_tools (or should be null/empty)
    assert!(
        dp["additional_tools"].is_null() || dp["additional_tools"].as_array().unwrap().is_empty(),
        "additional_tools should be null or empty when only static tools are used"
    );

    assert_eq!(dp["tool_choice"], "auto");
}

/// Test 5.5: Datapoint with only dynamic tools
///
/// Mirrors test_inference_only_dynamic_tools but for datapoints.
#[tokio::test(flavor = "multi_thread")]
async fn test_datapoint_only_dynamic_tools() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-dp-dynamic-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Include both static tool from config AND dynamic tool in storage
    let static_tool = tensorzero_core::tool::FunctionTool {
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

    let dynamic_tool = tensorzero_core::tool::FunctionTool {
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
        tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
            vec![Tool::Function(dynamic_tool.clone())],
            vec![],
            AllowedTools {
                tools: vec![static_tool.name.clone(), dynamic_tool.name.clone()],
                choice: AllowedToolsChoice::Explicit,
            },
            ToolChoice::Auto,
            None,
        )),
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

    // Static + dynamic tool should be in allowed_tools
    let allowed_tools = dp["allowed_tools"].as_array().unwrap();
    assert_eq!(allowed_tools.len(), 2);
    let allowed_tools_set: std::collections::HashSet<&str> =
        allowed_tools.iter().map(|v| v.as_str().unwrap()).collect();
    assert!(allowed_tools_set.contains("runtime_tool"));
    assert!(allowed_tools_set.contains("get_temperature"));

    // Dynamic tool should be in additional_tools (flattened)
    let additional_tools = dp["additional_tools"].as_array().unwrap();
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
#[tokio::test(flavor = "multi_thread")]
async fn test_datapoint_tool_params_three_states() {
    let clickhouse = get_clickhouse().await;
    let http_client = Client::new();
    let dataset_name = format!("test-dp-three-states-{}", Uuid::now_v7());

    // Create datapoint with tool_params
    let original_id = Uuid::now_v7();
    let tool = tensorzero_core::tool::FunctionTool {
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
        tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
            vec![Tool::Function(tool.clone())],
            vec![],
            AllowedTools {
                tools: vec![tool.name.clone()],
                choice: AllowedToolsChoice::Explicit,
            },
            ToolChoice::Auto,
            None,
        )),
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
    // With flatten, tool params fields should still exist at top level when omitted
    assert_eq!(
        dp1["allowed_tools"],
        json!(["get_temperature"]),
        "tool_params fields should still exist when field is omitted"
    );

    // Test Case 2: Clear tool params fields
    let resp = http_client
        .patch(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "id": new_id_1.to_string(),
                "allowed_tools": [],
                "additional_tools": [],
                "tool_choice": null,
                "parallel_tool_calls": null,
                "provider_tools": []
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

    assert_eq!(
        dp2["allowed_tools"],
        json!([]),
        "allowed_tools should be set to empty array"
    );
    assert_eq!(
        dp2["additional_tools"],
        json!(null),
        "additional_tools should be cleared"
    );
    assert_eq!(
        dp2["provider_tools"],
        json!([]),
        "provider_tools should be set to empty array"
    );

    // tool_choice and parallel_tool_calls now take on the function's value
    // See tensorzero-core/tests/e2e/tensorzero.toml
    assert_eq!(
        dp2["tool_choice"],
        json!("auto"),
        "tool_choice should be set to function's tool choice"
    );
    assert_eq!(
        dp2["parallel_tool_calls"],
        json!(null),
        "parallel_tool_calls should be cleared"
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
                "allowed_tools": json!(null),
                "additional_tools": [new_tool],
                "tool_choice": "required"
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
    // With flatten, tool params fields should be at top level
    assert_eq!(
        dp3["tool_choice"], "required",
        "tool_choice should be set to new value"
    );

    // When only additional_tools provided without explicit allowed_tools, the database stores
    // AllowedToolsChoice::FunctionDefault which deserializes as None/null on read.
    // This is the expected behavior - None means "use function defaults" without materializing them.
    // Note: The actual allowed tools during inference would be the function's default tools,
    // but we don't materialize them in the API response.
    assert!(dp3["allowed_tools"].is_null());

    let add_tools = dp3["additional_tools"].as_array().unwrap();
    assert_eq!(add_tools.len(), 1);
    assert_eq!(add_tools[0]["name"], "new_tool");
    assert_eq!(add_tools[0]["strict"], true);
}

/// Test 5.7: Datapoint with no tool params
///
/// Verifies handling when a datapoint has no tool_params at all.
#[tokio::test(flavor = "multi_thread")]
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
        tool_params: None, // No tool params
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

    // With flatten, tool_params fields should be null or not present when None
    assert!(
        dp["allowed_tools"].is_null()
            && dp["additional_tools"].is_null()
            && dp["tool_choice"].is_null(),
        "tool_params fields should be null when not provided"
    );
}
