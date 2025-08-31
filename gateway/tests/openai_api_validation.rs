use jsonschema::{Draft, JSONSchema};
use reqwest::blocking::Client;
use serde_json::{json, Value as JsonValue};


fn load_openai_spec() -> JsonValue {
    let raw = include_str!(concat!(env!("OUT_DIR"), "/openapi.json"));
    serde_json::from_str(raw).expect("failed to parse OUT_DIR/openapi.json")
}

/// Recursively rewrite `$ref` values so that references to OpenAPI components become local:
fn rewrite_component_refs_in_place(node: &mut JsonValue) {
    match node {
        JsonValue::Object(map) => {
            if let Some(JsonValue::String(s)) = map.get_mut("$ref") {
                if let Some(tail) = s.strip_prefix("json-schema:///#/components/schemas/") {
                    *s = format!("#/$defs/{}", tail);
                } else if let Some(tail) = s.strip_prefix("#/components/schemas/") {
                    *s = format!("#/$defs/{}", tail);
                }
            }
            for v in map.values_mut() {
                rewrite_component_refs_in_place(v);
            }
        }
        JsonValue::Array(arr) => {
            for v in arr {
                rewrite_component_refs_in_place(v);
            }
        }
        _ => {}
    }
}


fn schema_for_component(spec: &JsonValue, name: &str) -> Option<JsonValue> {
    let defs = spec.pointer("/components/schemas")?.clone();
    let mut wrapper = json!({
        "$defs": defs,
        "$ref": format!("#/$defs/{}", name),
    });
    rewrite_component_refs_in_place(&mut wrapper);
    Some(wrapper)
}

/// Try several component names (the documented schema sometimes renames these).
fn schema_for_first(spec: &JsonValue, names: &[&str]) -> Option<JsonValue> {
    for n in names {
        if let Some(defs) = spec.pointer(&format!("/components/schemas/{}", n)) {
            if defs.is_object() || defs.is_array() {
                return schema_for_component(spec, n);
            }
        }
    }
    None
}

fn compile(schema: &JsonValue) -> JSONSchema {
    JSONSchema::options()
        .with_draft(Draft::Draft7)
        .compile(schema)
        .expect("failed to compile schema")
}

/// Validate and, on failure, collect errors and panic with a readable message.
fn assert_valid(compiled: &JSONSchema, instance: &JsonValue, context: &str) {
    if let Err(it) = compiled.validate(instance) {
        let details: Vec<String> = it.map(|e| e.to_string()).collect();
        panic!("{}:\n  - {}\nJSON: {}", context, details.join("\n  - "), instance);
    }
}

// Convenience accessors for common components.
fn error_schema(spec: &JsonValue) -> JSONSchema {
    let schema = schema_for_first(spec, &["ErrorResponse", "Error"])
        .expect("Error schema not found in OpenAI spec");
    compile(&schema)
}

fn chat_completions_request_schema(spec: &JsonValue) -> Option<JSONSchema> {
    schema_for_first(
        spec,
        &[
            "CreateChatCompletionRequest",
            "CreateChatCompletionRequestBody",
            "ChatCompletionRequest",
        ],
    )
    .map(|s| compile(&s))
}

fn chat_completions_response_schema(spec: &JsonValue) -> Option<JSONSchema> {
    schema_for_first(
        spec,
        &[
            "ChatCompletion",
            "CreateChatCompletionResponse",
            "ChatCompletionResponse",
        ],
    )
    .map(|s| compile(&s))
}

fn embeddings_request_schema(spec: &JsonValue) -> Option<JSONSchema> {
    schema_for_first(
        spec,
        &[
            "CreateEmbeddingRequest",
            "CreateEmbeddingsRequest",
            "EmbeddingsRequest",
        ],
    )
    .map(|s| compile(&s))
}

fn embeddings_response_schema(spec: &JsonValue) -> Option<JSONSchema> {
    schema_for_first(
        spec,
        &[
            "CreateEmbeddingResponse",
            "EmbeddingsResponse",
            "EmbeddingList",
        ],
    )
    .map(|s| compile(&s))
}

fn chat_completion_supports_content_array(spec: &JsonValue) -> bool {
    let node = spec
        .pointer("/components/schemas/ChatCompletion/properties/choices/items/properties/message/properties/content")
        .or_else(|| spec.pointer("/components/schemas/CreateChatCompletionResponse/properties/choices/items/properties/message/properties/content"))
        .or_else(|| spec.pointer("/components/schemas/ChatCompletionResponse/properties/choices/items/properties/message/properties/content"));

    let Some(content_schema) = node else { return false };

    fn contains_array_type(v: &JsonValue) -> bool {
        match v {
            JsonValue::String(s) => s == "array",
            JsonValue::Array(a) => a.iter().any(contains_array_type),
            JsonValue::Object(m) => m
                .get("type")
                .map(contains_array_type)
                .unwrap_or_else(|| m.values().any(contains_array_type)),
            _ => false,
        }
    }

    if contains_array_type(content_schema) {
        return true;
    }
    for key in ["oneOf", "anyOf", "allOf"] {
        if let Some(JsonValue::Array(parts)) = content_schema.get(key) {
            if parts.iter().any(contains_array_type) {
                return true;
            }
        }
    }
    false
}


#[test]
fn validate_400_bad_request_error() {
    let spec = load_openai_spec();
    let schema = error_schema(&spec);

    let mock = json!({
        "error": {
            "message": "You must provide a model parameter",
            "type": "invalid_request_error",
            "param": "model",
            "code": "invalid_request_error"
        }
    });

    assert_valid(&schema, &mock, "400 Error schema mismatch");
}

#[test]
fn validate_401_unauthorized_error() {
    let spec = load_openai_spec();
    let schema = error_schema(&spec);

    // NOTE: The OpenAI Error schema requires "param" to be present, even if null.
    let mock = serde_json::json!({
        "error": {
            "message": "Incorrect API key provided: sk-...",
            "type": "invalid_request_error",
            "param": null,
            "code": "invalid_api_key"
        }
    });

    assert_valid(&schema, &mock, "401 Error schema mismatch");
}


#[test]
fn validate_chat_completion_request_schema_minimal() {
    let spec = load_openai_spec();
    let Some(schema) = chat_completions_request_schema(&spec) else {
        eprintln!("No ChatCompletion request schema in spec; skipping.");
        return;
    };

    let req = json!({
        "model": "gpt-4o-mini",
        "messages": [{"role":"user","content":"hi"}]
    });

    assert_valid(&schema, &req, "ChatCompletion request (minimal) schema mismatch");
}

#[test]
fn validate_embeddings_request_schema_minimal() {
    let spec = load_openai_spec();
    let Some(schema) = embeddings_request_schema(&spec) else {
        eprintln!("No Embeddings request schema in spec; skipping.");
        return;
    };

    let req = json!({
        "model": "text-embedding-3-small",
        "input": "hello"
    });

    assert_valid(&schema, &req, "Embeddings request (minimal) schema mismatch");
}

#[test]
fn validate_embeddings_request_variants() {
    let spec = load_openai_spec();
    let Some(schema) = embeddings_request_schema(&spec) else {
        eprintln!("No Embeddings request schema in spec; skipping.");
        return;
    };

    let req1 = json!({"model":"text-embedding-3-small","input":["a","b","c"]});
    assert_valid(&schema, &req1, "Embeddings request (array of strings) mismatch");

    let req2 = json!({"model":"text-embedding-3-small","input":"token"});
    assert_valid(&schema, &req2, "Embeddings request (single string) mismatch");
}

#[test]
fn validate_embeddings_mock_response() {
    let spec = load_openai_spec();
    let Some(schema) = embeddings_response_schema(&spec) else {
        eprintln!("No Embeddings response schema in spec; skipping.");
        return;
    };

    let mock = json!({
        "object": "list",
        "data": [{
            "object": "embedding",
            "index": 0,
            "embedding": [0.01, 0.02, 0.03]
        }],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 3,
            "total_tokens": 3
        }
    });

    assert_valid(&schema, &mock, "Embeddings response schema mismatch");
}

#[test]
fn validate_chat_completion_logprobs_and_refusal_null() {
    let spec = load_openai_spec();
    let Some(schema) = chat_completions_response_schema(&spec) else {
        eprintln!("No ChatCompletion response schema in spec; skipping.");
        return;
    };

    let mock = json!({
        "id":"chatcmpl-xyz",
        "object":"chat.completion",
        "created":1677652288,
        "model":"gpt-4o-mini",
        "choices":[{
            "index":0,
            "message":{
                "role":"assistant",
                "content":"Hello there!",
                "refusal": null,
                "tool_calls": []
            },
            "logprobs": null,
            "finish_reason":"stop"
        }],
        "usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12},
        "system_fingerprint":"fp_mock"
    });

    assert_valid(&schema, &mock, "ChatCompletion response (null logprobs/refusal) mismatch");
}

#[test]
fn validate_chat_completion_tool_calls_mock() {
    let spec = load_openai_spec();
    let Some(schema) = chat_completions_response_schema(&spec) else {
        eprintln!("No ChatCompletion response schema in spec; skipping.");
        return;
    };

    // The spec requires "refusal" to be present (nullable).
    let mock = serde_json::json!({
        "id":"chatcmpl-2",
        "object":"chat.completion",
        "created":1677652288,
        "model":"gpt-4o-mini",
        "choices":[{
            "index":0,
            "message":{
                "role":"assistant",
                "content": null,
                "refusal": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "get_weather", "arguments": "{\"city\":\"SF\"}" }
                }]
            },
            "logprobs": null,
            "finish_reason":"tool_calls"
        }],
        "usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13},
        "system_fingerprint":"fp_mock"
    });

    assert_valid(&schema, &mock, "ChatCompletion response (tool_calls) mismatch");
}


#[test]
fn validate_chat_completion_multipart_content_mock() {
    let spec = load_openai_spec();
    let Some(schema) = chat_completions_response_schema(&spec) else {
        eprintln!("No ChatCompletion response schema in spec; skipping.");
        return;
    };
    if !chat_completion_supports_content_array(&spec) {
        eprintln!("Spec does not allow content as an array; skipping multipart content test.");
        return;
    }

    let mock = json!({
        "id":"chatcmpl-multipart-1",
        "object":"chat.completion",
        "created":1677652288,
        "model":"gpt-4o-mini",
        "choices":[{
            "index":0,
            "message":{
                "role":"assistant",
                "content":[ { "type":"text", "text":"Hello from a content-part array." } ],
                "refusal": "",
                "tool_calls": []
            },
            "logprobs":{"content":[], "refusal":[]},
            "finish_reason":"stop"
        }],
        "usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12},
        "system_fingerprint":"fp_mock"
    });

    assert_valid(&schema, &mock, "ChatCompletion response (multipart content) mismatch");
}

#[test]
fn validate_finish_reason_values() {
    let spec = load_openai_spec();
    let Some(schema) = chat_completions_response_schema(&spec) else {
        eprintln!("No ChatCompletion response schema in spec; skipping.");
        return;
    };

    for reason in ["stop", "length", "tool_calls", "content_filter"] {
        let mock = json!({
            "id":"chatcmpl-finish-1",
            "object":"chat.completion",
            "created":1677652288,
            "model":"gpt-4o-mini",
            "choices":[{
                "index":0,
                "message":{"role":"assistant","content":"ok","refusal":null,"tool_calls":[]},
                "logprobs": null,
                "finish_reason": reason
            }],
            "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},
            "system_fingerprint":"fp_mock"
        });
        assert_valid(&schema, &mock, &format!("finish_reason={} mismatch", reason));
    }
}

// --- Live endpoint probes (real HTTP calls to the gateway) -------------------


fn live_tests_enabled() -> bool {
    match std::env::var("DISABLE_LIVE_OPENAI_TESTS") {
        Ok(v) => !(v == "1" || v.eq_ignore_ascii_case("true")),
        Err(_) => true,
    }
}

#[test]
fn test_live_chat_completion_api_call() {
    if !live_tests_enabled() {
        eprintln!("Skipping live test per DISABLE_LIVE_OPENAI_TESTS");
        return;
    }

    let spec = load_openai_spec();
    let err_schema = error_schema(&spec);
    let ok_schema = chat_completions_response_schema(&spec);

    // Intentionally plain model name to trigger an error without provider calls.
    let req_body = json!({
        "model": "gpt-4o-mini",
        "messages": [{ "role": "user", "content": "ping" }]
    });

    let base = std::env::var("GATEWAY_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());
    let url = format!("{}/v1/openai/v1/chat/completions", base);

    let resp = Client::new().post(&url).json(&req_body).send().expect("request failed");
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    let parsed: JsonValue = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("response not JSON ({}): {}", status, e));

    if status.is_success() {
        if let Some(schema) = ok_schema {
            assert_valid(&schema, &parsed, "200 OK but ChatCompletion schema mismatch");
        }
        return;
    }

    if err_schema.validate(&parsed).is_ok() {
        return;
    }
    if parsed.get("error").and_then(|v| v.as_str()).is_some() {
        return;
    }
    panic!("Unexpected error response.\nStatus: {}\nBody: {}", status, parsed);
}

#[test]
fn test_live_embeddings_api_call() {
    if !live_tests_enabled() {
        eprintln!("Skipping live test per DISABLE_LIVE_OPENAI_TESTS");
        return;
    }

    let spec = load_openai_spec();
    let err_schema = error_schema(&spec);
    let ok_schema = embeddings_response_schema(&spec);

    // Intentionally plain model name to trigger an error without provider calls.
    let req_body = json!({
        "model": "text-embedding-3-small",
        "input": "hello"
    });

    let base = std::env::var("GATEWAY_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());
    let url = format!("{}/v1/openai/v1/embeddings", base);

    let resp = Client::new().post(&url).json(&req_body).send().expect("request failed");
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    let parsed: JsonValue = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("response not JSON ({}): {}", status, e));

    if status.is_success() {
        if let Some(schema) = ok_schema {
            assert_valid(&schema, &parsed, "200 OK but Embeddings schema mismatch");
        }
        return;
    }

    if err_schema.validate(&parsed).is_ok() {
        return;
    }
    if parsed.get("error").and_then(|v| v.as_str()).is_some() {
        return;
    }
    panic!("Unexpected error response.\nStatus: {}\nBody: {}", status, parsed);
}
