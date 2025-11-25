#![expect(clippy::print_stdout)]
use jsonschema::{Draft, JSONSchema};
use lazy_static::lazy_static;
use reqwest::Client;
use serde_json::{json, Value};

use crate::common::get_gateway_endpoint;

lazy_static! {
   static ref OPENAI_SPEC: Value = {
       let rt = tokio::runtime::Runtime::new().unwrap();
       rt.block_on(async { download_openapi_spec().await })
   };
}

async fn download_openapi_spec() -> Value {
   let url = "https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml";
   let client = Client::new();
   let response = client
       .get(url)
       .send()
       .await
       .expect("Failed to download OpenAI spec");


   let yaml_value: serde_yml::Value =
       serde_yml::from_str(&response.text().await.expect("Failed to read response text"))
           .expect("Failed to parse YAML");


   let json_value: Value =
       serde_json::to_value(yaml_value).expect("Failed to convert YAML to JSON");


   json_value
}


fn get_component_schema(component_name: &str) -> Value {
   let components = OPENAI_SPEC
       .pointer("/components/schemas")
       .unwrap_or_else(|| panic!("No /components/schemas in OpenAPI spec"))
       .clone();


   // Include both $defs and components so refs like "#/components/..." still resolve
   json!({
       "$defs": components,
       "components": { "schemas": components },
       "$ref": format!("#/components/schemas/{}", component_name)
   })
}


/// Try multiple possible component names (handles schema variations)
fn get_component_schema_fallback(names: &[&str]) -> Option<Value> {
   for name in names {
       if let Some(schema) = OPENAI_SPEC.pointer(&format!("/components/schemas/{}", name)) {
           if schema.is_object() || schema.is_array() {
               return Some(get_component_schema(name));
           }
       }
   }
   None
}


fn compile_schema(schema: &Value) -> JSONSchema {
   JSONSchema::options()
       .with_draft(Draft::Draft202012)
       .compile(schema)
       .expect("Failed to compile JSON schema")
}


/// Validate instance against schema and panic with detailed error on failure
fn assert_valid_schema(schema: &JSONSchema, instance: &Value, context: &str) {
   if let Err(errors) = schema.validate(instance) {
       let error_msgs: Vec<String> = errors.map(|e| e.to_string()).collect();
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


fn get_error_schema() -> JSONSchema {
   let schema = get_component_schema_fallback(&["ErrorResponse", "Error"])
       .expect("Error schema not found in OpenAPI spec");
   compile_schema(&schema)
}


fn get_chat_completion_response_schema() -> Option<JSONSchema> {
   get_component_schema_fallback(&[
       "ChatCompletion",
       "CreateChatCompletionResponse",
       "ChatCompletionResponse",
   ])
   .map(|s| compile_schema(&s))
}


fn get_embeddings_response_schema() -> Option<JSONSchema> {
   get_component_schema_fallback(&[
       "CreateEmbeddingResponse",
       "EmbeddingsResponse",
       "EmbeddingList",
   ])
   .map(|s| compile_schema(&s))
}


// ============================================================================
// ERROR RESPONSE TESTS
// ============================================================================


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
   let error_schema = get_error_schema();


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
   let error_schema = get_error_schema();


   assert_valid_schema(
       &error_schema,
       &response_json,
       "404 Not Found error response",
   );
}


// ============================================================================
// CHAT COMPLETIONS TESTS
// ============================================================================


#[tokio::test]
#[ignore]
async fn test_spec_chat_completion_minimal_request() {
   let client = Client::new();
   let response = client
       .post(get_gateway_endpoint("/openai/v1/chat/completions"))
       .json(&json!({
           "model": "tensorzero::model_name::openai::gpt-4o-mini",
           "messages": [{"role": "user", "content": "Say hello"}]
       }))
       .send()
       .await
       .unwrap();
   assert_eq!(response.status(), 200, "Expected 200 OK");


   let response_json: Value = response.json().await.unwrap();
   println!("Chat completion response: {response_json:?}");
   if let Some(schema) = get_chat_completion_response_schema() {
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
async fn test_spec_chat_completion_required_fields() {
   let client = Client::new();


   let response = client
       .post(get_gateway_endpoint("/openai/v1/chat/completions"))
       .json(&json!({
           "model": "tensorzero::model_name::openai::gpt-4o-mini",
           "messages": [{"role": "user", "content": "hi"}]
       }))
       .send()
       .await
       .unwrap();


   let response_json: Value = response.json().await.unwrap();


   // Check required top-level fields
   assert!(
       response_json.get("id").is_some(),
       "Missing required field: id"
   );
   assert!(
       response_json.get("object").is_some(),
       "Missing required field: object"
   );
   assert!(
       response_json.get("created").is_some(),
       "Missing required field: created"
   );
   assert!(
       response_json.get("model").is_some(),
       "Missing required field: model"
   );
   assert!(
       response_json.get("choices").is_some(),
       "Missing required field: choices"
   );


   // Check choices array structure
   let choices = response_json["choices"].as_array().unwrap();
   assert!(!choices.is_empty(), "choices array should not be empty");


   let first_choice = &choices[0];
   assert!(
       first_choice.get("index").is_some(),
       "Missing required field: choices[0].index"
   );
   assert!(
       first_choice.get("message").is_some(),
       "Missing required field: choices[0].message"
   );
   assert!(
       first_choice.get("finish_reason").is_some(),
       "Missing required field: choices[0].finish_reason"
   );


   // Check message structure
   let message = &first_choice["message"];
   assert!(
       message.get("role").is_some(),
       "Missing required field: choices[0].message.role"
   );
   assert!(
       message.get("content").is_some(),
       "Missing required field: choices[0].message.content"
   );
}


#[tokio::test]
#[ignore] // Non-blocking: Remove once stable
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


   if let Some(schema) = get_chat_completion_response_schema() {
       assert_valid_schema(
           &schema,
           &response_json,
           "Chat completion with tool calls response",
       );
   }
}


#[tokio::test]
#[ignore] // Non-blocking: Remove once stable
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


       if let Some(schema) = get_chat_completion_response_schema() {
           assert_valid_schema(
               &schema,
               &response_json,
               &format!("Chat completion with finish_reason={}", finish_type),
           );
       }
   }
}


#[tokio::test]
#[ignore] // Non-blocking: Remove once stable
async fn test_spec_chat_completion_null_fields() {
   // Test that null fields (logprobs, refusal) are handled correctly
   let client = Client::new();


   let response = client
       .post(get_gateway_endpoint("/openai/v1/chat/completions"))
       .json(&json!({
           "model": "tensorzero::model_name::openai::gpt-4o-mini",
           "messages": [{"role": "user", "content": "hi"}]
       }))
       .send()
       .await
       .unwrap();


   let response_json: Value = response.json().await.unwrap();


   // Check that nullable fields are either null or properly typed
   let first_choice = &response_json["choices"][0];


   if let Some(logprobs) = first_choice.get("logprobs") {
       assert!(
           logprobs.is_null() || logprobs.is_object(),
           "logprobs must be null or object"
       );
   }


   if let Some(refusal) = first_choice["message"].get("refusal") {
       assert!(
           refusal.is_null() || refusal.is_string(),
           "refusal must be null or string"
       );
   }


   if let Some(schema) = get_chat_completion_response_schema() {
       assert_valid_schema(&schema, &response_json, "Chat completion with null fields");
   }
}


// ============================================================================
// EMBEDDINGS TESTS
// ============================================================================


#[tokio::test]
#[ignore] // Non-blocking: Remove once stable
async fn test_spec_embeddings_minimal_request() {
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


   if let Some(schema) = get_embeddings_response_schema() {
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
#[ignore] // Non-blocking: Remove once stable
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


   if let Some(schema) = get_embeddings_response_schema() {
       assert_valid_schema(&schema, &response_json, "Embeddings array input response");
   }
}


#[tokio::test]
#[ignore] // Non-blocking: Remove once stable
async fn test_spec_embeddings_required_fields() {
   let client = Client::new();


   let response = client
       .post(get_gateway_endpoint("/openai/v1/embeddings"))
       .json(&json!({
           "model": "tensorzero::model_name::openai::text-embedding-3-small",
           "input": "test"
       }))
       .send()
       .await
       .unwrap();


   let response_json: Value = response.json().await.unwrap();


   // Check required top-level fields
   assert!(
       response_json.get("object").is_some(),
       "Missing required field: object"
   );
   assert!(
       response_json.get("data").is_some(),
       "Missing required field: data"
   );
   assert!(
       response_json.get("model").is_some(),
       "Missing required field: model"
   );
   assert!(
       response_json.get("usage").is_some(),
       "Missing required field: usage"
   );


   // Check data array structure
   let data = response_json["data"].as_array().unwrap();
   assert!(!data.is_empty(), "data array should not be empty");


   let first_embedding = &data[0];
   assert!(
       first_embedding.get("object").is_some(),
       "Missing required field: data[0].object"
   );
   assert!(
       first_embedding.get("index").is_some(),
       "Missing required field: data[0].index"
   );
   assert!(
       first_embedding.get("embedding").is_some(),
       "Missing required field: data[0].embedding"
   );


   // Check usage structure
   let usage = &response_json["usage"];
   assert!(
       usage.get("prompt_tokens").is_some(),
       "Missing required field: usage.prompt_tokens"
   );
   assert!(
       usage.get("total_tokens").is_some(),
       "Missing required field: usage.total_tokens"
   );
}


// ============================================================================
// FIELD TYPE VALIDATION TESTS (Catches bugs like #2379)
// ============================================================================


#[tokio::test]
#[ignore] // Non-blocking: Remove once stable
async fn test_spec_service_tier_field_type() {
   let client = Client::new();


   let response = client
       .post(get_gateway_endpoint("/openai/v1/chat/completions"))
       .json(&json!({
           "model": "tensorzero::model_name::openai::gpt-4o-mini",
           "messages": [{"role": "user", "content": "hi"}]
       }))
       .send()
       .await
       .unwrap();


   let response_json: Value = response.json().await.unwrap();


   // Validate service_tier field if present
   if let Some(service_tier) = response_json.get("service_tier") {
       if let Some(s) = service_tier.as_str() {
           assert!(
               !s.is_empty(),
               "Bug #2379: service_tier should not be empty string, use null instead"
           );
       }
       // If it's null, that's fine per spec
   }


   // Also validate full schema
   if let Some(schema) = get_chat_completion_response_schema() {
       assert_valid_schema(
           &schema,
           &response_json,
           "service_tier field type validation",
       );
   }
}


#[tokio::test]
#[ignore] // Non-blocking: Remove once stable
async fn test_spec_system_fingerprint_field_type() {
   let client = Client::new();


   let response = client
       .post(get_gateway_endpoint("/openai/v1/chat/completions"))
       .json(&json!({
           "model": "tensorzero::model_name::openai::gpt-4o-mini",
           "messages": [{"role": "user", "content": "hi"}]
       }))
       .send()
       .await
       .unwrap();


   let response_json: Value = response.json().await.unwrap();


   // Validate system_fingerprint field if present
   if let Some(fingerprint) = response_json.get("system_fingerprint") {
       assert!(
           fingerprint.is_null() || fingerprint.is_string(),
           "system_fingerprint must be null or string, not: {:?}",
           fingerprint
       );


       if let Some(s) = fingerprint.as_str() {
           assert!(
               !s.is_empty(),
               "system_fingerprint should not be empty string, use null instead"
           );
       }
   }
}


#[tokio::test]
#[ignore] // Non-blocking: Remove once stable
async fn test_spec_timestamp_format() {
   let client = Client::new();


   let response = client
       .post(get_gateway_endpoint("/openai/v1/chat/completions"))
       .json(&json!({
           "model": "tensorzero::model_name::openai::gpt-4o-mini",
           "messages": [{"role": "user", "content": "hi"}]
       }))
       .send()
       .await
       .unwrap();


   let response_json: Value = response.json().await.unwrap();


   // created field should be Unix timestamp (integer seconds)
   if let Some(created) = response_json.get("created") {
       assert!(
           created.is_i64() || created.is_u64(),
           "created field must be integer Unix timestamp, not: {:?}",
           created
       );
   }
}


// ============================================================================
// STREAMING RESPONSE TESTS
// ============================================================================


#[tokio::test]
#[ignore] // Non-blocking: Remove once stable
async fn test_spec_streaming_chunk_structure() {
   use futures::StreamExt;
   use reqwest_eventsource::{Event, RequestBuilderExt};


   let client = Client::new();


   let mut response = client
       .post(get_gateway_endpoint("/openai/v1/chat/completions"))
       .json(&json!({
           "model": "tensorzero::model_name::openai::gpt-4o-mini",
           "messages": [{"role": "user", "content": "Say hello"}],
           "stream": true
       }))
       .eventsource()
       .unwrap();


   let mut chunks = vec![];
   while let Some(event) = response.next().await {
       let event = event.unwrap();
       match event {
           Event::Open => continue,
           Event::Message(message) => {
               if message.data == "[DONE]" {
                   break;
               }
               chunks.push(message.data);
           }
       }
   }


   assert!(!chunks.is_empty(), "Should receive at least one chunk");


   // Validate first chunk has role
   let first_chunk: Value = serde_json::from_str(&chunks[0]).unwrap();
   assert!(
       first_chunk["choices"][0]["delta"]["role"].is_string(),
       "First chunk should have role in delta"
   );


   // All chunks should have required fields
   for chunk_str in &chunks {
       let chunk: Value = serde_json::from_str(chunk_str).unwrap();
       assert!(chunk.get("id").is_some(), "Chunk missing id field");
       assert!(chunk.get("object").is_some(), "Chunk missing object field");
       assert!(
           chunk.get("created").is_some(),
           "Chunk missing created field"
       );
       assert!(chunk.get("model").is_some(), "Chunk missing model field");
       assert!(
           chunk.get("choices").is_some(),
           "Chunk missing choices field"
       );
   }
}



