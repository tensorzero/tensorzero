#![expect(clippy::print_stdout)]
use jsonschema::draft202012;
use jsonschema::Validator;
use reqwest::Client;
use serde_json::json;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashSet;
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
// These were corrected for now to enable schema validation.
fn sanitize_openapi_spec(value: &mut serde_yml::Value) {
    use serde_yml::Value::*;

    match value {
        Mapping(map) => {
            convert_boolean_bounds(map, "exclusiveMinimum", "minimum");
            convert_boolean_bounds(map, "exclusiveMaximum", "maximum");
            remove_recursive_anchor(map);
            dedupe_required_entries(map);
            map.iter_mut()
                .for_each(|(_, value)| sanitize_openapi_spec(value));
        }
        Sequence(seq) => seq.iter_mut().for_each(sanitize_openapi_spec),
        _ => {}
    }
}

fn convert_boolean_bounds(map: &mut serde_yml::Mapping, exclusive_key: &str, bound_key: &str) {
    use serde_yml::Value::{Bool, Number, String as YamlString};

    let flag_key = YamlString(exclusive_key.to_owned());
    let bound_key_value = YamlString(bound_key.to_owned());

    let Some(Bool(true)) = map.get(&flag_key) else {
        return;
    };

    let Some(bound_value) = map.get(&bound_key_value).cloned() else {
        return;
    };

    if !matches!(bound_value, Number(_)) {
        return;
    }

    map.remove(&bound_key_value);
    map.insert(flag_key, bound_value);
}

fn remove_recursive_anchor(map: &mut serde_yml::Mapping) {
    use serde_yml::Value::String as YamlString;

    let key = YamlString("$recursiveAnchor".into());
    while map.remove(&key).is_some() {}
}

fn dedupe_required_entries(map: &mut serde_yml::Mapping) {
    use serde_yml::Value::Sequence;

    for (key, value) in map.iter_mut() {
        if let (Some("required"), Sequence(items)) = (key.as_str(), value) {
            let mut seen = HashSet::new();
            items.retain(|item| {
                item.as_str()
                    .map(|s| seen.insert(s.to_owned()))
                    .unwrap_or(true)
            });
        }
    }
}

async fn get_component_schema(component_name: &str) -> Option<Value> {
    let spec = get_openai_spec().await;
    let components = spec
        .pointer("/components/schemas")
        .unwrap_or_else(|| panic!("No /components/schemas in OpenAPI spec"));

    components.get(component_name)?;

    // Build schema with components referenced once to avoid unnecessary clones
    let mut schema = json!({
        "$ref": format!("#/components/schemas/{}", component_name)
    });
    
    schema.as_object_mut().unwrap().insert(
        "$defs".to_string(),
        components.clone(),
    );
    schema.as_object_mut().unwrap().insert(
        "components".to_string(),
        json!({ "schemas": components }),
    );
    
    Some(schema)
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

fn assert_valid_schema(schema: &Validator, instance: &Value, context: &str) {
    if schema.validate(instance).is_ok() {
        return;
    }

    let error_msgs: Vec<String> = schema
        .iter_errors(instance)
        .map(|e| e.to_string())
        .collect();
    panic!(
        "\nOpenAPI Spec Validation Failed: {}\n\nErrors:\n{}\n\nResponse JSON:\n{}\n",
        context,
        error_msgs
            .iter()
            .map(|e| format!("  - {e}"))
            .collect::<Vec<_>>()
            .join("\n"),
        serde_json::to_string_pretty(instance).unwrap()
    );
}

fn warn_schema_missing(schema_name: &str) {
    println!("{schema_name} schema not found in spec, skipping validation");
}

fn extend_json_object(target: &mut Value, additions: &Value) {
    let (Some(target_map), Some(additions_map)) = (target.as_object_mut(), additions.as_object())
    else {
        return;
    };

    target_map.extend(additions_map.clone());
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

async fn validate_chat_completion_response(instance: &Value, context: &str) {
    get_chat_completion_response_schema().await.map_or_else(
        || warn_schema_missing("ChatCompletion response"),
        |schema| assert_valid_schema(&schema, instance, context),
    );
}

async fn validate_embeddings_response(instance: &Value, context: &str) {
    get_embeddings_response_schema().await.map_or_else(
        || warn_schema_missing("Embeddings response"),
        |schema| assert_valid_schema(&schema, instance, context),
    );
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
    validate_chat_completion_response(&response_json, "Chat completion minimal request response")
        .await;
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

    validate_chat_completion_response(&response_json, "Chat completion with tool calls response")
        .await;
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

        extend_json_object(&mut request, &params);

        let response = client
            .post(get_gateway_endpoint("/openai/v1/chat/completions"))
            .json(&request)
            .send()
            .await
            .unwrap();

        let response_json: Value = response.json().await.unwrap();
        let context = format!("Chat completion with finish_reason={finish_type}");
        validate_chat_completion_response(&response_json, &context).await;
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

    validate_embeddings_response(&response_json, "Embeddings minimal request response").await;
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

    validate_embeddings_response(&response_json, "Embeddings array input response").await;
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
