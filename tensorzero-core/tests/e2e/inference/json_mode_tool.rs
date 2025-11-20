//! E2E tests for `json_mode="tool"` with chat functions
//!
//! This test suite verifies that chat functions can use `json_mode="tool"`
//! when they have NO tools configured, while maintaining existing JSON function behavior.
//!
//! Key behaviors tested:
//! 1. Chat functions with NO tools + output_schema → SUCCESS
//! 2. Chat functions with NO tools but missing output_schema → ERROR
//! 3. Chat functions WITH tools configured → ERROR
//! 4. Chat functions with dynamic tool params → ERROR
//! 5. Direct model calls (model_name) → SUCCESS with output_schema
//! 6. JSON functions (baseline) → SUCCESS (no regression)

use serde_json::{json, Value};
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, InferenceOutput, InferenceResponse,
    Role,
};
use tensorzero_core::endpoints::inference::ChatCompletionInferenceParams;
use tensorzero_core::inference::types::TextKind;
use tensorzero_core::tool::{DynamicToolParams, FunctionTool};
use tensorzero_core::variant::JsonMode;

// Reusable output schema for tests
fn simple_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"]
            },
            "confidence": {
                "type": "number"
            }
        },
        "required": ["sentiment", "confidence"],
        "additionalProperties": false
    })
}

/// Test 1: Chat function with NO tools + output_schema → SUCCESS
///
/// Verifies that a chat function without tools configured can successfully use
/// `json_mode="tool"` when an `output_schema` is provided at inference time.
#[tokio::test(flavor = "multi_thread")]
async fn test_chat_function_no_tools_with_output_schema_succeeds() {
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(
        r#"
[functions.test_chat_no_tools]
type = "chat"

[functions.test_chat_no_tools.variants.baseline]
type = "chat_completion"
model = "dummy::good"
"#,
    )
    .await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("test_chat_no_tools".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![tensorzero::ClientInputMessageContent::Text(
                        TextKind::Text {
                            text: "This is a great product!".to_string(),
                        },
                    )],
                }],
            },
            params: tensorzero::InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    json_mode: Some(JsonMode::Tool),
                    ..Default::default()
                },
            },
            output_schema: Some(simple_output_schema()),
            stream: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();

    // Should succeed - verify we got a non-streaming response with inference_id
    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };
    assert!(!response.inference_id().to_string().is_empty());
}

/// Test 2: Chat function with NO tools but missing output_schema → ERROR
///
/// Verifies that using `json_mode="tool"` without providing `output_schema`
/// results in an error.
#[tokio::test(flavor = "multi_thread")]
async fn test_chat_function_no_tools_without_output_schema_fails() {
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(
        r#"
[functions.test_chat_no_tools]
type = "chat"

[functions.test_chat_no_tools.variants.baseline]
type = "chat_completion"
model = "dummy::good"
"#,
    )
    .await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("test_chat_no_tools".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![tensorzero::ClientInputMessageContent::Text(
                        TextKind::Text {
                            text: "This is a great product!".to_string(),
                        },
                    )],
                }],
            },
            params: tensorzero::InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    json_mode: Some(JsonMode::Tool),
                    ..Default::default()
                },
            },
            output_schema: None, // Missing output_schema
            stream: Some(false),
            ..Default::default()
        })
        .await;

    // Should fail
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("JSON mode `tool` requires `output_schema`"),
        "Expected error about missing output_schema, got: {err}"
    );
}

/// Test 3: Chat function WITH tools configured → ERROR
///
/// Verifies that chat functions with tools configured cannot use `json_mode="tool"`.
#[tokio::test(flavor = "multi_thread")]
async fn test_chat_function_with_tools_fails() {
    // Create temp directory and write tool parameters schema
    let temp_dir = tempfile::tempdir().unwrap();
    let tool_params_path = temp_dir.path().join("get_temperature_params.json");
    std::fs::write(
        &tool_params_path,
        r#"{
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get temperature for"
                }
            },
            "required": ["location"],
            "additionalProperties": false
        }"#,
    )
    .unwrap();

    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(&format!(
        r#"
[functions.test_chat_with_tools]
type = "chat"
tools = ["get_temperature"]

[functions.test_chat_with_tools.variants.baseline]
type = "chat_completion"
model = "dummy::good"

[tools.get_temperature]
description = "Get the temperature"
parameters = "{}"
"#,
        tool_params_path.display()
    ))
    .await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("test_chat_with_tools".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![tensorzero::ClientInputMessageContent::Text(
                        TextKind::Text {
                            text: "What's the temperature?".to_string(),
                        },
                    )],
                }],
            },
            params: tensorzero::InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    json_mode: Some(JsonMode::Tool),
                    ..Default::default()
                },
            },
            output_schema: Some(simple_output_schema()),
            stream: Some(false),
            ..Default::default()
        })
        .await;

    // Should fail
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("JSON mode `tool` is not supported with other tools configured"),
        "Expected error about tools configured, got: {err}"
    );
}

/// Test 4: Chat function with additional_tools → ERROR
///
/// Verifies that dynamic tool params are rejected when using `json_mode="tool"`.
#[tokio::test(flavor = "multi_thread")]
async fn test_chat_function_with_additional_tools_fails() {
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(
        r#"
[functions.test_chat_no_tools]
type = "chat"

[functions.test_chat_no_tools.variants.baseline]
type = "chat_completion"
model = "dummy::good"
"#,
    )
    .await;

    let additional_tool = FunctionTool {
        name: "custom_tool".to_string(),
        description: "A custom tool".to_string(),
        strict: false,
        parameters: json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string"
                }
            },
            "required": ["input"],
            "additionalProperties": false
        }),
    };

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("test_chat_no_tools".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![tensorzero::ClientInputMessageContent::Text(
                        TextKind::Text {
                            text: "Test message".to_string(),
                        },
                    )],
                }],
            },
            params: tensorzero::InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    json_mode: Some(JsonMode::Tool),
                    ..Default::default()
                },
            },
            output_schema: Some(simple_output_schema()),
            dynamic_tool_params: DynamicToolParams {
                additional_tools: Some(vec![additional_tool]),
                ..Default::default()
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    // Should fail
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Cannot pass `additional_tools` when using JSON mode `tool`"),
        "Expected error about additional_tools, got: {err}"
    );
}

/// Test 5: Direct model call (model_name) with output_schema → SUCCESS
///
/// Verifies that calling a model directly using `model_name` (which creates a default
/// chat function) works with `json_mode="tool"` when output_schema is provided.
#[tokio::test(flavor = "multi_thread")]
async fn test_model_name_with_output_schema_succeeds() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;

    let response = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![tensorzero::ClientInputMessageContent::Text(
                        TextKind::Text {
                            text: "This is a great product!".to_string(),
                        },
                    )],
                }],
            },
            params: tensorzero::InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    json_mode: Some(JsonMode::Tool),
                    ..Default::default()
                },
            },
            output_schema: Some(simple_output_schema()),
            stream: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();

    // Should succeed - verify we got a non-streaming response with inference_id
    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };
    assert!(!response.inference_id().to_string().is_empty());
}

/// Test 6: Direct model call (model_name) without output_schema → ERROR
#[tokio::test(flavor = "multi_thread")]
async fn test_model_name_without_output_schema_fails() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![tensorzero::ClientInputMessageContent::Text(
                        TextKind::Text {
                            text: "This is a great product!".to_string(),
                        },
                    )],
                }],
            },
            params: tensorzero::InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    json_mode: Some(JsonMode::Tool),
                    ..Default::default()
                },
            },
            output_schema: None, // Missing output_schema
            stream: Some(false),
            ..Default::default()
        })
        .await;

    // Should fail
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("JSON mode `tool` requires `output_schema`"),
        "Expected error about missing output_schema, got: {err}"
    );
}

/// Test 7: Chat function with json_mode="tool" - E2E verification
///
/// Verifies the full flow: tool call sent to provider, tool response received,
/// and converted back to JSON text for the user.
#[tokio::test(flavor = "multi_thread")]
async fn test_chat_function_tool_mode_e2e() {
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(
        r#"
[functions.test_chat_tool_mode]
type = "chat"

[functions.test_chat_tool_mode.variants.baseline]
type = "chat_completion"
model = "dummy::good_tool"
"#,
    )
    .await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("test_chat_tool_mode".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![tensorzero::ClientInputMessageContent::Text(
                        TextKind::Text {
                            text: "Analyze sentiment".to_string(),
                        },
                    )],
                }],
            },
            params: tensorzero::InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    json_mode: Some(JsonMode::Tool),
                    ..Default::default()
                },
            },
            output_schema: Some(simple_output_schema()),
            stream: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();

    // Should succeed - verify we got a non-streaming response
    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    // Verify response is InferenceResponse::Chat
    let InferenceResponse::Chat(chat_response) = &response else {
        panic!("Expected chat response");
    };

    // Critical verification: The response should be TEXT, not a tool call
    // This proves that json_mode="tool" converted the tool call back to JSON text
    assert!(!chat_response.content.is_empty(), "Should have content");
    assert_eq!(
        chat_response.content.len(),
        1,
        "Should have exactly one content block"
    );

    // Serialize to JSON to inspect the structure
    let content_json =
        serde_json::to_value(&chat_response.content).expect("Should be able to serialize content");

    // Verify it's a text block (not a tool_call block)
    let content_block = &content_json[0];
    assert_eq!(
        content_block["type"], "text",
        "Content block should be type 'text', not 'tool_call'"
    );

    // Extract the text content
    let text_content = content_block["text"]
        .as_str()
        .expect("Should have text field");

    // Verify the text is valid JSON matching our output schema
    let parsed_json: Value =
        serde_json::from_str(text_content).expect("Response text should be valid JSON");

    // Verify schema structure
    assert!(
        parsed_json.get("sentiment").is_some(),
        "Should have 'sentiment' field"
    );
    assert!(
        parsed_json.get("confidence").is_some(),
        "Should have 'confidence' field"
    );

    // Verify the actual values from dummy provider
    assert_eq!(parsed_json["sentiment"], "positive");
    assert_eq!(parsed_json["confidence"], 0.95);
}

/// Test 8: JSON function baseline → SUCCESS (no regression)
///
/// Verifies that existing JSON function behavior with `json_mode="tool"` still works.
#[tokio::test(flavor = "multi_thread")]
async fn test_json_function_json_mode_tool_baseline() {
    // Create temp directory and write output schema
    let temp_dir = tempfile::tempdir().unwrap();
    let output_schema_path = temp_dir.path().join("output_schema.json");
    std::fs::write(
        &output_schema_path,
        r#"{
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Person's name"
                },
                "email": {
                    "type": "string",
                    "description": "Person's email"
                }
            },
            "required": ["name", "email"],
            "additionalProperties": false
        }"#,
    )
    .unwrap();

    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(&format!(
        r#"
[functions.test_json]
type = "json"
output_schema = "{}"

[functions.test_json.variants.baseline]
type = "chat_completion"
model = "dummy::good"
json_mode = "tool"
"#,
        output_schema_path.display()
    ))
    .await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("test_json".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![tensorzero::ClientInputMessageContent::Text(
                        TextKind::Text {
                            text: "Extract: John Doe, john@example.com".to_string(),
                        },
                    )],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();

    // Should succeed - verify we got a non-streaming response with inference_id
    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };
    assert!(!response.inference_id().to_string().is_empty());
}

/// Test 9: Real OpenAI model with json_mode="tool" - Schema enforcement
///
/// Verifies that output_schema takes precedence over prompt instructions.
/// The prompt asks for name/age/occupation but output_schema specifies sentiment/confidence.
#[tokio::test(flavor = "multi_thread")]
async fn test_model_name_openai_schema_enforcement() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;

    // Prompt asks for a DIFFERENT schema (person info)
    let conflicting_prompt = r#"Please analyze the following text and return a JSON object with these exact fields:
- "name": the persons name (string)
- "age": the persons age (number)
- "occupation": the persons job (string)

Text: \"My name is Megumin. I love my job as a archmage.\""#;

    // But output_schema specifies sentiment + confidence
    let output_schema = simple_output_schema(); // {"sentiment": string, "confidence": number}

    let response = client
        .inference(ClientInferenceParams {
            model_name: Some("openai::gpt-4o-mini".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![tensorzero::ClientInputMessageContent::Text(
                        TextKind::Text {
                            text: conflicting_prompt.to_string(),
                        },
                    )],
                }],
            },
            params: tensorzero::InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    json_mode: Some(JsonMode::Tool),
                    ..Default::default()
                },
            },
            output_schema: Some(output_schema),
            stream: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();

    // Verify response structure
    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    let InferenceResponse::Chat(chat_response) = &response else {
        panic!("Expected chat response");
    };

    // Response should be TEXT, not tool_call
    assert!(!chat_response.content.is_empty(), "Should have content");

    // Serialize to JSON to inspect
    let content_json =
        serde_json::to_value(&chat_response.content).expect("Should be able to serialize content");

    let content_block = &content_json[0];
    assert_eq!(
        content_block["type"], "text",
        "Should be text block, not tool_call"
    );

    // Extract and parse JSON
    let text_content = content_block["text"]
        .as_str()
        .expect("Should have text field");

    let parsed_json: Value =
        serde_json::from_str(text_content).expect("Response should be valid JSON");

    // CRITICAL: Verify model followed output_schema, NOT the prompt
    assert!(
        parsed_json.get("sentiment").is_some(),
        "Should have sentiment field from output_schema"
    );
    assert!(
        parsed_json.get("confidence").is_some(),
        "Should have confidence field from output_schema"
    );

    // Verify model did NOT follow prompts requested schema
    assert!(
        parsed_json.get("name").is_none(),
        "Should NOT have name field from prompt"
    );
    assert!(
        parsed_json.get("age").is_none(),
        "Should NOT have age field from prompt"
    );
    assert!(
        parsed_json.get("occupation").is_none(),
        "Should NOT have occupation field from prompt"
    );

    // Verify types are correct
    assert!(
        parsed_json["sentiment"].is_string(),
        "sentiment should be string"
    );
    assert!(
        parsed_json["confidence"].is_number(),
        "confidence should be number"
    );

    // Verify sentiment value is one of the allowed enum values
    let sentiment = parsed_json["sentiment"].as_str().unwrap();
    assert!(
        ["positive", "negative", "neutral"].contains(&sentiment),
        "sentiment should be one of: positive, negative, neutral. Got: {sentiment}"
    );
}
