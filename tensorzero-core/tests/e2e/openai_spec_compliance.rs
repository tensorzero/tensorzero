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
