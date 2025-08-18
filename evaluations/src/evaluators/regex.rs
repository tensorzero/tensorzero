use anyhow::{bail, Result};
use regex::Regex;
use serde_json::Value;
use tensorzero::InferenceResponse;
use tensorzero_core::endpoints::datasets::Datapoint;
use tensorzero_core::endpoints::inference::{ChatInferenceResponse, JsonInferenceResponse};
use tensorzero_core::inference::types::ContentBlockChatOutput;
use tracing::{debug, instrument, warn};

#[instrument(skip(inference_response, datapoint), fields(datapoint_id = %datapoint.id()))]
pub(super) fn run_regex_evaluator(
    inference_response: &InferenceResponse,
    datapoint: &Datapoint,
    regex_pattern: &str,
) -> Result<Option<Value>> {
    // Compile the regex pattern
    let regex = Regex::new(regex_pattern).map_err(|e| {
        anyhow::anyhow!("Invalid regex pattern '{}': {}", regex_pattern, e)
    })?;

    match (inference_response, datapoint) {
        (InferenceResponse::Chat(response), Datapoint::Chat(_datapoint)) => {
            debug!("Running regex evaluation for chat response");
            
            // Extract text content from the response
            let response_text = extract_text_from_chat_response(response);
            
            // Check if the regex pattern matches
            let matches = regex.is_match(&response_text);
            debug!(matches = %matches, pattern = %regex_pattern, "Chat regex evaluation completed");
            Ok(Some(Value::Bool(matches)))
        }
        (InferenceResponse::Json(json_completion), Datapoint::Json(_json_inference)) => {
            debug!("Running regex evaluation for JSON response");
            
            // Extract text content from the JSON response
            let response_text = extract_text_from_json_response(json_completion);
            
            // Check if the regex pattern matches
            let matches = regex.is_match(&response_text);
            debug!(matches = %matches, pattern = %regex_pattern, "JSON regex evaluation completed");
            Ok(Some(Value::Bool(matches)))
        }
        _ => {
            let datapoint_type = match datapoint {
                Datapoint::Chat(_) => "Chat",
                Datapoint::Json(_) => "Json",
            };
            let response_type = match inference_response {
                InferenceResponse::Chat(_) => "Chat",
                InferenceResponse::Json(_) => "Json",
            };
            warn!(
                datapoint_type = %datapoint_type,
                response_type = %response_type,
                "Datapoint and inference response types do not match"
            );
            bail!("Datapoint and inference response types do not match")
        }
    }
}

/// Extracts text content from a chat response
fn extract_text_from_chat_response(response: &ChatInferenceResponse) -> String {
    let mut text_parts = Vec::new();
    
    for content_block in &response.content {
        match content_block {
            ContentBlockChatOutput::Text(text_block) => {
                text_parts.push(text_block.text.clone());
            }
            ContentBlockChatOutput::ToolCall(tool_call) => {
                // Include tool call arguments as text
                text_parts.push(tool_call.raw_arguments.clone());
            }
            ContentBlockChatOutput::Thought(thought) => {
                // Include thought text if available
                if let Some(text) = &thought.text {
                    text_parts.push(text.clone());
                }
            }
            ContentBlockChatOutput::Unknown { data, .. } => {
                // Include unknown data as text if it's a string
                if let Some(text) = data.as_str() {
                    text_parts.push(text.to_string());
                }
            }
        }
    }
    
    text_parts.join(" ")
}

/// Extracts text content from a JSON response
fn extract_text_from_json_response(response: &JsonInferenceResponse) -> String {
    // For JSON responses, we'll use the raw output as text
    // This allows regex to match against the JSON structure
    response.output.raw.clone().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tensorzero::Role;
    use tensorzero_core::{
        endpoints::{
            datasets::{ChatInferenceDatapoint, JsonInferenceDatapoint},
            inference::{ChatInferenceResponse, JsonInferenceResponse},
        },
        inference::types::{
            ContentBlockChatOutput, JsonInferenceOutput, ResolvedInput, ResolvedInputMessage,
            ResolvedInputMessageContent, Text, Usage,
        },
    };
    use uuid::Uuid;

    #[test]
    fn test_regex_evaluator_chat_matches() {
        let datapoint = Datapoint::Chat(ChatInferenceDatapoint {
            id: Uuid::now_v7(),
            input: ResolvedInput {
                system: None,
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("Hello, world!"),
                    }],
                }],
            },
            dataset_name: "test".to_string(),
            function_name: "test".to_string(),
            episode_id: Some(Uuid::now_v7()),
            output: None,
            tool_params: None,
            tags: None,
            auxiliary: "".to_string(),
            is_deleted: false,
            is_custom: false,
            source_inference_id: None,
            staled_at: None,
        });

        let response = InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            content: vec![ContentBlockChatOutput::Text(Text {
                text: "The answer is 42".to_string(),
            })],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
            original_response: None,
            finish_reason: None,
        });

        // Test regex that should match
        let result = run_regex_evaluator(&response, &datapoint, r"\d+").unwrap();
        assert_eq!(result, Some(Value::Bool(true)));

        // Test regex that should not match
        let result = run_regex_evaluator(&response, &datapoint, r"^[A-Z]+$").unwrap();
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_regex_evaluator_json_matches() {
        let datapoint = Datapoint::Json(JsonInferenceDatapoint {
            id: Uuid::now_v7(),
            input: ResolvedInput {
                system: None,
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("Extract the number"),
                    }],
                }],
            },
            dataset_name: "test".to_string(),
            function_name: "test".to_string(),
            episode_id: Some(Uuid::now_v7()),
            output: None,
            output_schema: json!({}),
            tags: None,
            auxiliary: "".to_string(),
            is_deleted: false,
            is_custom: false,
            source_inference_id: None,
            staled_at: None,
        });

        let response = InferenceResponse::Json(JsonInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            output: JsonInferenceOutput {
                raw: Some(r#"{"number": 42, "text": "hello"}"#.to_string()),
                parsed: Some(json!({"number": 42, "text": "hello"})),
            },
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
            original_response: None,
            finish_reason: None,
        });

        // Test regex that should match the JSON structure
        let result = run_regex_evaluator(&response, &datapoint, r#""number":\s*\d+"#).unwrap();
        assert_eq!(result, Some(Value::Bool(true)));

        // Test regex that should not match
        let result = run_regex_evaluator(&response, &datapoint, r#""missing":\s*\d+"#).unwrap();
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_regex_evaluator_invalid_pattern() {
        let datapoint = Datapoint::Chat(ChatInferenceDatapoint {
            id: Uuid::now_v7(),
            input: ResolvedInput {
                system: None,
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("Hello"),
                    }],
                }],
            },
            dataset_name: "test".to_string(),
            function_name: "test".to_string(),
            episode_id: Some(Uuid::now_v7()),
            output: None,
            tool_params: None,
            tags: None,
            auxiliary: "".to_string(),
            is_deleted: false,
            is_custom: false,
            source_inference_id: None,
            staled_at: None,
        });

        let response = InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            content: vec![ContentBlockChatOutput::Text(Text {
                text: "Hello world".to_string(),
            })],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
            original_response: None,
            finish_reason: None,
        });

        // Test invalid regex pattern
        let result = run_regex_evaluator(&response, &datapoint, r"[invalid");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid regex pattern"));
    }

    #[test]
    fn test_regex_evaluator_type_mismatch() {
        let datapoint = Datapoint::Chat(ChatInferenceDatapoint {
            id: Uuid::now_v7(),
            input: ResolvedInput {
                system: None,
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("Hello"),
                    }],
                }],
            },
            dataset_name: "test".to_string(),
            function_name: "test".to_string(),
            episode_id: Some(Uuid::now_v7()),
            output: None,
            tool_params: None,
            tags: None,
            auxiliary: "".to_string(),
            is_deleted: false,
            is_custom: false,
            source_inference_id: None,
            staled_at: None,
        });

        let response = InferenceResponse::Json(JsonInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            output: JsonInferenceOutput {
                raw: Some(r#"{"result": "test"}"#.to_string()),
                parsed: Some(json!({"result": "test"})),
            },
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
            original_response: None,
            finish_reason: None,
        });

        // Test type mismatch
        let result = run_regex_evaluator(&response, &datapoint, r"test");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Datapoint and inference response types do not match"));
    }

    #[test]
    fn test_extract_text_from_chat_response() {
        let response = ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            content: vec![
                ContentBlockChatOutput::Text(Text {
                    text: "Hello".to_string(),
                }),
                ContentBlockChatOutput::Text(Text {
                    text: "world".to_string(),
                }),
            ],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
            original_response: None,
            finish_reason: None,
        };

        let text = extract_text_from_chat_response(&response);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_extract_text_from_json_response() {
        let response = JsonInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            output: JsonInferenceOutput {
                raw: Some(r#"{"key": "value"}"#.to_string()),
                parsed: Some(json!({"key": "value"})),
            },
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
            original_response: None,
            finish_reason: None,
        };

        let text = extract_text_from_json_response(&response);
        assert_eq!(text, r#"{"key": "value"}"#);
    }
}
