#![expect(clippy::print_stdout)]

use googletest::prelude::*;
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::inferences::{InferenceQueries, ListInferencesParams};
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::inference::types::StoredModelInference;
use tensorzero_core::stored_inference::{StoredChatInferenceDatabase, StoredInferenceDatabase};
use tensorzero_core::test_helpers::get_e2e_config;

/// Test that OpenAI Responses API accepts and uses a custom tool with text format
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_responses_api_custom_tool_text_format() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "openai::responses::gpt-5-codex",
        "episode_id": episode_id,
        "input": {
            "messages": [{
                "role": "user",
                "content": "Generate Python code to print 'Hello, World!' using the code_generator tool."
            }]
        },
        "additional_tools": [{
            "type": "openai_custom",
            "name": "code_generator",
            "description": "Generates Python code snippets based on the given description",
            "format": {
                "type": "text"
            }
        }],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("response: {response_json:#}");

    // Check that we got tool calls
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content_blocks.is_empty());

    // Find a tool_call block (there should be at least one)
    let tool_call_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "tool_call")
        .collect();
    assert!(
        !tool_call_blocks.is_empty(),
        "Should have at least one tool call block"
    );

    // Check that one of the tool calls is to code_generator
    let code_generator_calls: Vec<&Value> = tool_call_blocks
        .into_iter()
        .filter(|block| block.get("raw_name").unwrap().as_str().unwrap() == "code_generator")
        .collect();
    assert_eq!(code_generator_calls.len(), 1);

    let tool_call = code_generator_calls.first().unwrap();
    let tool_call_id = tool_call.get("id").unwrap().as_str().unwrap();
    let raw_arguments = tool_call.get("raw_arguments").unwrap().as_str().unwrap();
    let name = tool_call.get("name").unwrap().as_str().unwrap();
    assert!(!raw_arguments.is_empty());
    assert!(!tool_call_id.is_empty());
    assert_eq!(name, "code_generator");

    // Check inference_id
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Wait for data to be written
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

    // Check inference table
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_that!(inferences, len(eq(1)));
    let chat_inf = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    expect_that!(
        chat_inf,
        matches_pattern!(StoredChatInferenceDatabase {
            inference_id: eq(&inference_id),
            function_name: eq("tensorzero::default"),
            variant_name: eq("openai::responses::gpt-5-codex"),
            episode_id: eq(&episode_id),
            ..
        })
    );

    // Check that dynamic_tools contains the custom tool
    let tool_params = chat_inf
        .tool_params
        .as_ref()
        .expect("tool_params should be present");
    assert_eq!(
        tool_params.dynamic_tools.len(),
        1,
        "Should have exactly one custom tool"
    );

    // Verify the tool structure by serializing to JSON
    let tool_json = serde_json::to_value(&tool_params.dynamic_tools[0]).unwrap();
    assert_eq!(
        tool_json.get("type").unwrap().as_str().unwrap(),
        "openai_custom"
    );
    assert_eq!(
        tool_json.get("name").unwrap().as_str().unwrap(),
        "code_generator"
    );
    assert_eq!(
        tool_json.get("description").unwrap().as_str().unwrap(),
        "Generates Python code snippets based on the given description"
    );

    // Verify the format is text
    let format = tool_json.get("format").unwrap();
    assert_eq!(format.get("type").unwrap().as_str().unwrap(), "text");
}

/// Test that OpenAI accepts and uses a custom tool with text format
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_custom_tool_text_format() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "openai::gpt-5-mini",
        "episode_id": episode_id,
        "input": {
            "messages": [{
                "role": "user",
                "content": "Generate Python code to print 'Hello, World!' using the code_generator tool."
            }],
        },
        "additional_tools": [
            {
                "type": "openai_custom",
                "name": "code_generator",
                "description": "Generates Python code snippets based on requirements",
                "format": {
                    "type": "text"
                }
            }
        ],
        "allowed_tools": ["code_generator"],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json
        .get("inference_id")
        .and_then(|v| v.as_str())
        .expect("inference_id should be present in response");

    println!("Response: {response_json:#?}");

    // Check that we got tool calls in the response
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content_blocks.is_empty());

    // Find a tool_call block (there should be at least one)
    let tool_call_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "tool_call")
        .collect();
    assert!(
        !tool_call_blocks.is_empty(),
        "Should have at least one tool call block"
    );

    // Check that one of the tool calls is to code_generator
    let code_generator_calls: Vec<&Value> = tool_call_blocks
        .into_iter()
        .filter(|block| block.get("raw_name").unwrap().as_str().unwrap() == "code_generator")
        .collect();
    assert_eq!(code_generator_calls.len(), 1);

    let tool_call = code_generator_calls.first().unwrap();
    let tool_call_id = tool_call.get("id").unwrap().as_str().unwrap();
    let raw_arguments = tool_call.get("raw_arguments").unwrap().as_str().unwrap();
    let name = tool_call.get("name").unwrap().as_str().unwrap();
    assert!(!raw_arguments.is_empty());
    assert!(!tool_call_id.is_empty());
    assert_eq!(name, "code_generator");

    let inference_id: Uuid = inference_id.parse().unwrap();

    // Wait for data to be written
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

    // Verify the tool was stored in ModelInference
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];

    let raw_request = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");

    println!("Raw request: {raw_request}");

    // Verify the custom tool appears in the raw request
    assert!(
        raw_request.contains("code_generator"),
        "Expected custom tool `code_generator` in raw_request"
    );
    assert!(
        raw_request.contains("\"type\":\"custom\"") || raw_request.contains("\"type\": \"custom\""),
        "Expected custom tool type in raw_request, got: {raw_request}"
    );

    expect_that!(
        mi,
        matches_pattern!(StoredModelInference {
            inference_id: eq(&inference_id),
            ..
        })
    );

    // Query ChatInference table to verify dynamic_tools
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_that!(inferences, len(eq(1)));
    let chat_inf = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };

    // Check that dynamic_tools contains the custom tool
    let tool_params = chat_inf
        .tool_params
        .as_ref()
        .expect("tool_params should be present");
    assert_eq!(
        tool_params.dynamic_tools.len(),
        1,
        "Should have exactly one custom tool"
    );

    // Verify the tool structure by serializing to JSON
    let tool_json = serde_json::to_value(&tool_params.dynamic_tools[0]).unwrap();
    assert_eq!(
        tool_json.get("type").unwrap().as_str().unwrap(),
        "openai_custom"
    );
    assert_eq!(
        tool_json.get("name").unwrap().as_str().unwrap(),
        "code_generator"
    );
    assert_eq!(
        tool_json.get("description").unwrap().as_str().unwrap(),
        "Generates Python code snippets based on requirements"
    );

    // Verify the format is text
    let format = tool_json.get("format").unwrap();
    assert_eq!(format.get("type").unwrap().as_str().unwrap(), "text");
}

/// Test that OpenAI accepts and uses a custom tool with Lark grammar format
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_custom_tool_grammar_lark() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Simple arithmetic grammar in Lark format
    let lark_grammar = r#"
start: expr

expr: term ((ADD | SUB) term)*
term: factor ((MUL | DIV) factor)*
factor: NUMBER
      | "(" expr ")"

ADD: "+"
SUB: "-"
MUL: "*"
DIV: "/"

NUMBER: /\d+(\.\d+)?/

%import common.WS
%ignore WS
"#;

    let payload = json!({
        "model_name": "openai::gpt-5-mini",
        "episode_id": episode_id,
        "input": {
            "messages": [{
                "role": "user",
                "content": "Use the calculator tool to compute 5 + 3 * 2"
            }],
        },
        "additional_tools": [
            {
                "type": "openai_custom",
                "name": "calculator",
                "description": "Evaluates arithmetic expressions",
                "format": {
                    "type": "grammar",
                    "grammar": {
                    "syntax": "lark",
                    "definition": lark_grammar
                }}
            }
        ],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json
        .get("inference_id")
        .and_then(|v| v.as_str())
        .expect("inference_id should be present in response");

    println!("Response: {response_json:#?}");

    // Check that we got tool calls in the response
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content_blocks.is_empty());

    // Find a tool_call block (there should be at least one)
    let tool_call_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "tool_call")
        .collect();
    assert!(
        !tool_call_blocks.is_empty(),
        "Should have at least one tool call block"
    );

    // Check that one of the tool calls is to calculator
    let calculator_calls: Vec<&Value> = tool_call_blocks
        .into_iter()
        .filter(|block| block.get("raw_name").unwrap().as_str().unwrap() == "calculator")
        .collect();
    assert_eq!(calculator_calls.len(), 1);

    let tool_call = calculator_calls.first().unwrap();
    let tool_call_id = tool_call.get("id").unwrap().as_str().unwrap();
    let raw_arguments = tool_call.get("raw_arguments").unwrap().as_str().unwrap();
    let name = tool_call.get("name").unwrap().as_str().unwrap();
    assert!(!raw_arguments.is_empty());
    assert!(!tool_call_id.is_empty());
    assert_eq!(name, "calculator");

    let inference_id: Uuid = inference_id.parse().unwrap();

    // Wait for data to be written
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

    // Verify the tool was stored in ModelInference
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];

    let raw_request = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");

    // Verify the custom tool with grammar appears in the raw request
    assert!(
        raw_request.contains("calculator"),
        "Expected custom tool `calculator` in raw_request"
    );
    assert!(
        raw_request.contains("\"type\":\"custom\"") || raw_request.contains("\"type\": \"custom\""),
        "Expected custom tool type in raw_request"
    );
    assert!(
        raw_request.contains("lark"),
        "Expected lark grammar syntax in raw_request"
    );

    // Query ChatInference table to verify dynamic_tools
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_that!(inferences, len(eq(1)));
    let chat_inf = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };

    // Check that dynamic_tools contains the custom tool
    let tool_params = chat_inf
        .tool_params
        .as_ref()
        .expect("tool_params should be present");
    assert_eq!(
        tool_params.dynamic_tools.len(),
        1,
        "Should have exactly one custom tool"
    );

    // Verify the tool structure by serializing to JSON
    let tool_json = serde_json::to_value(&tool_params.dynamic_tools[0]).unwrap();
    assert_eq!(
        tool_json.get("type").unwrap().as_str().unwrap(),
        "openai_custom"
    );
    assert_eq!(
        tool_json.get("name").unwrap().as_str().unwrap(),
        "calculator"
    );
    assert_eq!(
        tool_json.get("description").unwrap().as_str().unwrap(),
        "Evaluates arithmetic expressions"
    );

    // Verify the format is grammar with lark syntax
    let format = tool_json.get("format").unwrap();
    assert_eq!(format.get("type").unwrap().as_str().unwrap(), "grammar");
    let grammar = format.get("grammar").unwrap();
    assert_eq!(grammar.get("syntax").unwrap().as_str().unwrap(), "lark");
    assert!(
        grammar
            .get("definition")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("start: expr")
    );
}

/// Test that OpenAI accepts and uses a custom tool with Regex grammar format
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_custom_tool_grammar_regex() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Regex pattern for phone numbers in format XXX-XXX-XXXX
    let regex_pattern = r"^\d{3}-\d{3}-\d{4}$";

    let payload = json!({
        "model_name": "openai::gpt-5-mini",
        "episode_id": episode_id,
        "input": {
            "messages": [{
                "role": "user",
                "content": "Use the phone_formatter tool to format the phone number 415-555-0123"
            }],
        },
        "additional_tools": [
            {
                "type": "openai_custom",
                "name": "phone_formatter",
                "description": "Formats phone numbers in the standard XXX-XXX-XXXX format",
                "format": {
                    "type": "grammar",
                    "grammar": {
                    "syntax": "regex",
                    "definition": regex_pattern
                }}
            }
        ],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json
        .get("inference_id")
        .and_then(|v| v.as_str())
        .expect("inference_id should be present in response");

    println!("Response: {response_json:#?}");

    // Check that we got tool calls in the response
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content_blocks.is_empty());

    // Find a tool_call block (there should be at least one)
    let tool_call_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "tool_call")
        .collect();
    assert!(
        !tool_call_blocks.is_empty(),
        "Should have at least one tool call block"
    );

    // Check that one of the tool calls is to phone_formatter
    let phone_formatter_calls: Vec<&Value> = tool_call_blocks
        .into_iter()
        .filter(|block| block.get("raw_name").unwrap().as_str().unwrap() == "phone_formatter")
        .collect();
    assert_eq!(phone_formatter_calls.len(), 1);

    let tool_call = phone_formatter_calls.first().unwrap();
    let tool_call_id = tool_call.get("id").unwrap().as_str().unwrap();
    let raw_arguments = tool_call.get("raw_arguments").unwrap().as_str().unwrap();
    let name = tool_call.get("name").unwrap().as_str().unwrap();
    assert!(!raw_arguments.is_empty());
    assert!(!tool_call_id.is_empty());
    assert_eq!(name, "phone_formatter");

    let inference_id: Uuid = inference_id.parse().unwrap();

    // Wait for data to be written
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

    // Verify the tool was stored in ModelInference
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];

    let raw_request = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");

    // Verify the custom tool with regex grammar appears in the raw request
    assert!(
        raw_request.contains("phone_formatter"),
        "Expected custom tool `phone_formatter` in raw_request"
    );
    assert!(
        raw_request.contains("\"type\":\"custom\"") || raw_request.contains("\"type\": \"custom\""),
        "Expected custom tool type in raw_request"
    );
    assert!(
        raw_request.contains("regex"),
        "Expected regex grammar syntax in raw_request"
    );

    // Query ChatInference table to verify dynamic_tools
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_that!(inferences, len(eq(1)));
    let chat_inf = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };

    // Check that dynamic_tools contains the custom tool
    let tool_params = chat_inf
        .tool_params
        .as_ref()
        .expect("tool_params should be present");
    assert_eq!(
        tool_params.dynamic_tools.len(),
        1,
        "Should have exactly one custom tool"
    );

    // Verify the tool structure by serializing to JSON
    let tool_json = serde_json::to_value(&tool_params.dynamic_tools[0]).unwrap();
    assert_eq!(
        tool_json.get("type").unwrap().as_str().unwrap(),
        "openai_custom"
    );
    assert_eq!(
        tool_json.get("name").unwrap().as_str().unwrap(),
        "phone_formatter"
    );
    assert_eq!(
        tool_json.get("description").unwrap().as_str().unwrap(),
        "Formats phone numbers in the standard XXX-XXX-XXXX format"
    );

    // Verify the format is grammar with regex syntax
    let format = tool_json.get("format").unwrap();
    assert_eq!(format.get("type").unwrap().as_str().unwrap(), "grammar");
    let grammar = format.get("grammar").unwrap();
    assert_eq!(grammar.get("syntax").unwrap().as_str().unwrap(), "regex");
    assert_eq!(
        grammar.get("definition").unwrap().as_str().unwrap(),
        regex_pattern
    );
}
