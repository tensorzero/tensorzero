#![expect(clippy::print_stdout)]
use jsonschema::draft202012;
use jsonschema::Validator;
use reqwest::Client;
use serde_json::json;
use serde_json::Value;
use std::borrow::Cow;
use tokio::sync::OnceCell;

use crate::common::get_gateway_endpoint;

static OPENAI_SPEC: OnceCell<Value> = OnceCell::const_new();

async fn get_openai_spec() -> &'static Value {
    OPENAI_SPEC
        .get_or_init(|| async { download_openapi_spec().await })
        .await
}

async fn download_openapi_spec() -> Value {
    let url = "https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml";
    let client = Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .expect("Failed to download OpenAI spec");

    let body = response.text().await.expect("Failed to read response text");
    let sanitized_text = fix_yaml_parse_issues(&body);

    let mut yaml_value: serde_yml::Value =
        serde_yml::from_str(&sanitized_text).expect("Failed to parse YAML");
    sanitize_openapi_spec(&mut yaml_value);

    serde_json::to_value(yaml_value).expect("Failed to convert YAML to JSON")
}

/// Spec yaml publishes `seed.minimum` with a value smaller than i64::MIN.
/// We clamp the literal so serde_yml can parse the document
fn fix_yaml_parse_issues(raw: &str) -> Cow<'_, str> {
    const OPENAI_SEED_MIN: &str = "-9223372036854776000";
    if raw.contains(OPENAI_SEED_MIN) {
        Cow::Owned(raw.replace(OPENAI_SEED_MIN, "-9223372036854775808"))
    } else {
        Cow::Borrowed(raw)
    }
}

// OpenAI's schema uses Draft 4 boolean exclusiveMinimum/exclusiveMaximum, which need conversion to Draft 2020-12 numeric form.
// Also removes `$recursiveAnchor` (Draft 2019-09 keyword not recognized in Draft 2020-12).
// Removes duplicate entries in `required` arrays which violate uniqueItems constraint.
// TODO: These are temporary fixes; ideally we'd contact the OpenAI spec maintainers to have these issues corrected.
// Theseb were corrected for now to enable schema validation.
fn sanitize_openapi_spec(value: &mut serde_yml::Value) {
    use serde_yml::Value::*;

    match value {
        Mapping(map) => {
            // Convert Draft 4 boolean exclusiveMinimum/exclusiveMaximum to Draft 2020-12 numeric form
            let mut exclusive_min_value = None;
            let mut exclusive_max_value = None;

            // First pass: detect Draft 4 boolean pattern and get the min/max values
            if let Some(true) = map
                .get(&String("exclusiveMinimum".into()))
                .and_then(|v| match v {
                    Bool(b) => Some(*b),
                    _ => None,
                })
            {
                if let Some(min_val) = map.get(&String("minimum".into())).cloned() {
                    if matches!(min_val, Number(_)) {
                        exclusive_min_value = Some(min_val);
                    }
                }
            }

            if let Some(true) = map
                .get(&String("exclusiveMaximum".into()))
                .and_then(|v| match v {
                    Bool(b) => Some(*b),
                    _ => None,
                })
            {
                if let Some(max_val) = map.get(&String("maximum".into())).cloned() {
                    if matches!(max_val, Number(_)) {
                        exclusive_max_value = Some(max_val);
                    }
                }
            }

            // Second pass: apply conversions
            if let Some(val) = exclusive_min_value {
                map.remove(&String("minimum".into()));
                map.insert(String("exclusiveMinimum".into()), val);
            }

            if let Some(val) = exclusive_max_value {
                map.remove(&String("maximum".into()));
                map.insert(String("exclusiveMaximum".into()), val);
            }

            // Remove $recursiveAnchor (Draft 2019-09 keyword, not recognized in Draft 2020-12)
            let keys_to_remove: Vec<_> = map
                .iter()
                .filter_map(|(key, _val)| {
                    if matches!(key, String(s) if s == "$recursiveAnchor") {
                        Some(key.clone())
                    } else {
                        None
                    }
                })
                .collect();

            for key in keys_to_remove {
                map.remove(&key);
            }

            for (key, val) in map.iter_mut() {
                // Todo: Make a PR to OpenAI OpenAPI spec to fix duplicate in required for ContainerResource
                if matches!(key, String(s) if s == "required") {
                    if let Sequence(items) = val {
                        let mut seen = std::collections::HashSet::new();
                        items.retain(|item| {
                            if let String(s) = item {
                                seen.insert(s.clone())
                            } else {
                                true
                            }
                        });
                    }
                }
                sanitize_openapi_spec(val);
            }
        }
        Sequence(seq) => {
            for item in seq {
                sanitize_openapi_spec(item);
            }
        }
        _ => {}
    }
}

async fn get_component_schema(component_name: &str) -> Option<Value> {
    let spec = get_openai_spec().await;
    let components = spec
        .pointer("/components/schemas")
        .unwrap_or_else(|| panic!("No /components/schemas in OpenAPI spec"));

    if components.get(component_name).is_none() {
        return None;
    }

    let components = components.clone();
    Some(json!({
        "$defs": components,
        "components": { "schemas": components },
        "$ref": format!("#/components/schemas/{}", component_name)
    }))
}

/// Try multiple possible component names (handles schema variations)
async fn get_component_schema_fallback(names: &[&str]) -> Option<Value> {
    for name in names {
        if let Some(schema) = get_component_schema(name).await {
            return Some(schema);
        }
    }
    None
}

fn compile_schema(schema: &Value) -> Validator {
    draft202012::options()
        .build(schema)
        .expect("Failed to compile JSON schema")
}

/// Validate instance against schema and panic with detailed error on failure
fn assert_valid_schema(schema: &Validator, instance: &Value, context: &str) {
    if schema.validate(instance).is_err() {
        let error_msgs: Vec<String> = schema
            .iter_errors(instance)
            .map(|e| e.to_string())
            .collect();
        panic!(
            "\n❌ OpenAPI Spec Validation Failed: {}\n\nErrors:\n{}\n\nResponse JSON:\n{}\n",
            context,
            error_msgs
                .iter()
                .map(|e| format!("  - {}", e))
                .collect::<Vec<_>>()
                .join("\n"),
            serde_json::to_string_pretty(instance).unwrap()
        );
    }
}

async fn get_error_schema() -> Validator {
    let schema = get_component_schema_fallback(&["ErrorResponse", "Error"])
        .await
        .expect("Error schema not found in OpenAPI spec");
    compile_schema(&schema)
}

async fn get_chat_completion_response_schema() -> Option<Validator> {
    get_component_schema_fallback(&[
        "ChatCompletion",
        "CreateChatCompletionResponse",
        "ChatCompletionResponse",
    ])
    .await
    .map(|s| compile_schema(&s))
}

async fn get_embeddings_response_schema() -> Option<Validator> {
    get_component_schema_fallback(&[
        "CreateEmbeddingResponse",
        "EmbeddingsResponse",
        "EmbeddingList",
    ])
    .await
    .map(|s| compile_schema(&s))
}

#[tokio::test]
#[ignore]
async fn test_spec_error_response_400_bad_request() {
    let client = Client::new();

    // Missing required 'model' field
    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&json!({
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 400, "Expected 400 Bad Request");

    let response_json: Value = response.json().await.unwrap();
    println!("400 error response: {response_json:?}");
    let error_schema = get_error_schema().await;

    assert_valid_schema(
        &error_schema,
        &response_json,
        "400 Bad Request error response",
    );
}

#[tokio::test]
#[ignore]
async fn test_spec_error_response_404_not_found() {
    let client = Client::new();

    // Request with invalid model name
    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&json!({
            "model": "tensorzero::model_name::nonexistent_model",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    // Should return an error (likely 400 or 404)
    assert!(
        response.status().is_client_error(),
        "Expected 4xx error for invalid model"
    );

    let response_json: Value = response.json().await.unwrap();
    println!("404 error response: {response_json:?}");
    let error_schema = get_error_schema().await;

    assert_valid_schema(
        &error_schema,
        &response_json,
        "404 Not Found error response",
    );
}

#[tokio::test]
#[ignore]
async fn test_spec_chat_completion_request() {
    let client = Client::new();
    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&json!({
            "model": "tensorzero::model_name::openai::gpt-4o-mini",
            "messages": [{"role": "user", "content": "Say hello world!"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "Expected 200 OK");

    let response_json: Value = response.json().await.unwrap();
    println!("Chat completion response: {response_json:?}");
    if let Some(schema) = get_chat_completion_response_schema().await {
        assert_valid_schema(
            &schema,
            &response_json,
            "Chat completion minimal request response",
        );
    } else {
        eprintln!(
            "⚠️  Warning: ChatCompletion response schema not found in spec, skipping validation"
        );
    }
}

#[tokio::test]
#[ignore]
async fn test_spec_chat_completion_with_tool_calls() {
    let client = Client::new();

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&json!({
            "model": "tensorzero::model_name::openai::gpt-4o-mini",
            "messages": [{"role": "user", "content": "What's the weather in Boston?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200, "Expected 200 OK");

    let response_json: Value = response.json().await.unwrap();

    if let Some(schema) = get_chat_completion_response_schema().await {
        assert_valid_schema(
            &schema,
            &response_json,
            "Chat completion with tool calls response",
        );
    }
}

#[tokio::test]
#[ignore]
async fn test_spec_chat_completion_finish_reasons() {
    // Test different finish_reason values are valid per spec
    let test_cases = vec![
        ("stop", json!({"max_tokens": 1000})),
        ("length", json!({"max_tokens": 1})),
    ];

    for (finish_type, params) in test_cases {
        let client = Client::new();

        let mut request = json!({
            "model": "tensorzero::model_name::openai::gpt-4o-mini",
            "messages": [{"role": "user", "content": "Say hello"}]
        });

        // Merge additional params
        if let Some(obj) = request.as_object_mut() {
            if let Some(params_obj) = params.as_object() {
                obj.extend(params_obj.clone());
            }
        }

        let response = client
            .post(get_gateway_endpoint("/openai/v1/chat/completions"))
            .json(&request)
            .send()
            .await
            .unwrap();

        let response_json: Value = response.json().await.unwrap();
        if let Some(schema) = get_chat_completion_response_schema().await {
            assert_valid_schema(
                &schema,
                &response_json,
                &format!("Chat completion with finish_reason={}", finish_type),
            );
        }
    }
}

#[tokio::test]
#[ignore]
async fn test_spec_embeddings_request() {
    let client = Client::new();

    let response = client
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&json!({
            "model": "tensorzero::model_name::openai::text-embedding-3-small",
            "input": "hello world"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200, "Expected 200 OK");

    let response_json: Value = response.json().await.unwrap();

    if let Some(schema) = get_embeddings_response_schema().await {
        assert_valid_schema(
            &schema,
            &response_json,
            "Embeddings minimal request response",
        );
    } else {
        eprintln!("⚠️  Warning: Embeddings response schema not found in spec, skipping validation");
    }
}

#[tokio::test]
#[ignore]
async fn test_spec_embeddings_array_input() {
    let client = Client::new();

    let response = client
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&json!({
            "model": "tensorzero::model_name::openai::text-embedding-3-small",
            "input": ["hello", "world", "test"]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200, "Expected 200 OK");

    let response_json: Value = response.json().await.unwrap();

    if let Some(schema) = get_embeddings_response_schema().await {
        assert_valid_schema(&schema, &response_json, "Embeddings array input response");
    }
}

// // ============================================================================
// // STREAMING RESPONSE TESTS
// // ============================================================================

// async fn get_streaming_chunk_schema() -> Option<Validator> {
//    get_component_schema_fallback(&[
//        "ChatCompletionChunk",
//        "CreateChatCompletionStreamResponse",
//        "ChatCompletionStreamResponse",
//    ])
//    .await
//    .map(|s| compile_schema(&s))
// }

// #[tokio::test]
// #[ignore]
// async fn test_spec_streaming_response() {
//    use futures::StreamExt;
//    use reqwest_eventsource::{Event, RequestBuilderExt};

//    let client = Client::new();

//    let mut response = client
//        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
//        .json(&json!({
//            "model": "tensorzero::model_name::openai::gpt-4o-mini",
//            "messages": [{"role": "user", "content": "Say hello world!"}],
//            "stream": true
//        }))
//        .eventsource()
//        .unwrap();

//    let mut chunks = vec![];
//    while let Some(event) = response.next().await {
//        let event = event.unwrap();
//        match event {
//            Event::Open => continue,
//            Event::Message(message) => {
//                if message.data == "[DONE]" {
//                    break;
//                }
//                chunks.push(message.data);
//            }
//        }
//    }

//    assert!(!chunks.is_empty(), "Should receive at least one streaming chunk");

//    if let Some(schema) = get_streaming_chunk_schema().await {
//        for (i, chunk_str) in chunks.iter().enumerate() {
//            let chunk: Value = serde_json::from_str(chunk_str)
//                .unwrap_or_else(|e| panic!("Failed to parse chunk {}: {}", i, e));

//            assert_valid_schema(
//                &schema,
//                &chunk,
//                &format!("Streaming chunk #{}", i),
//            );
//        }
//    } else {
//        eprintln!("⚠️  Warning: Streaming chunk schema not found in spec, skipping validation");
//    }
// }
