use anyhow::{bail, Result};
use serde_json::Value;
use tensorzero::InferenceResponse;
use tensorzero_internal::endpoints::datasets::Datapoint;
use tracing::{debug, instrument, warn};

#[instrument(skip(inference_response, datapoint), fields(datapoint_id = %datapoint.id()))]
pub(super) fn run_exact_match_evaluator(
    inference_response: &InferenceResponse,
    datapoint: &Datapoint,
) -> Result<Option<Value>> {
    match (inference_response, datapoint) {
        (InferenceResponse::Chat(response), Datapoint::Chat(datapoint)) => {
            debug!("Running exact match evaluation for chat response");
            match &datapoint.output {
                // Right now this is order-sensitive, but we may consider relaxing this in the future
                Some(output) => {
                    let matches = output == &response.content;
                    debug!(matches = %matches, "Chat exact match comparison completed");
                    Ok(Some(Value::Bool(matches)))
                }
                None => {
                    debug!("No reference output available for chat comparison");
                    Ok(None)
                }
            }
        }
        (InferenceResponse::Json(json_completion), Datapoint::Json(json_inference)) => {
            debug!("Running exact match evaluation for JSON response");
            match &json_inference.output {
                Some(output) => {
                    // `output.parsed` is an Option<Value> but it should always be Some here
                    if output.parsed.is_none() {
                        warn!("Datapoint {} has no parsed output", json_inference.id);
                        return Ok(None);
                    }
                    let matches = output.parsed == json_completion.output.parsed;
                    debug!(matches = %matches, "JSON exact match comparison completed");
                    Ok(Some(Value::Bool(matches)))
                }
                None => {
                    debug!("No reference output available for JSON comparison");
                    Ok(None)
                }
            }
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

#[cfg(test)]
mod tests {

    use super::*;
    use serde_json::json;
    use tensorzero::Role;
    use tensorzero_internal::{
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
    fn test_exact_match_evaluator_chat() {
        // Test a match
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
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "hello world".to_string(),
            })]),
            tool_params: None,
            tags: None,
            auxiliary: "".to_string(),
            is_deleted: false,
            source_inference_id: None,
            staled_at: None,
        });
        let inference_response = InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            content: vec![ContentBlockChatOutput::Text(Text {
                text: "hello world".to_string(),
            })],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 10,
            },
            original_response: None,
            finish_reason: None,
        });
        let result = run_exact_match_evaluator(&inference_response, &datapoint).unwrap();
        assert_eq!(result, Some(Value::Bool(true)));

        // Test a mismatch
        let inference_response = InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            content: vec![ContentBlockChatOutput::Text(Text {
                text: "hello, world!".to_string(),
            })],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 10,
            },
            original_response: None,
            finish_reason: None,
        });
        let result = run_exact_match_evaluator(&inference_response, &datapoint).unwrap();
        assert_eq!(result, Some(Value::Bool(false)));

        // Test with missing output (should be None)
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
            source_inference_id: None,
            staled_at: None,
        });
        let result = run_exact_match_evaluator(&inference_response, &datapoint).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_exact_match_evaluator_json() {
        // Test a match
        let datapoint = Datapoint::Json(JsonInferenceDatapoint {
            id: Uuid::now_v7(),
            input: ResolvedInput {
                system: None,
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!({"foo": "bar"}),
                    }],
                }],
            },
            dataset_name: "test".to_string(),
            function_name: "test".to_string(),
            output_schema: json!({
                "type": "object",
                "properties": {
                    "foo": {
                        "type": "string"
                    }
                }
            }),
            episode_id: Some(Uuid::now_v7()),
            output: Some(JsonInferenceOutput {
                parsed: Some(json!({"foo": "bar"})),
                raw: Some(r#"{"foo": "bar"}"#.to_string()),
            }),
            tags: None,
            auxiliary: "".to_string(),
            is_deleted: false,
            source_inference_id: None,
            staled_at: None,
        });
        let inference_response = InferenceResponse::Json(JsonInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            output: JsonInferenceOutput {
                parsed: Some(json!({"foo": "bar"})),
                raw: Some(r#"{"foo": "bar"}"#.to_string()),
            },
            usage: Usage {
                input_tokens: 10,
                output_tokens: 10,
            },
            original_response: None,
            finish_reason: None,
        });
        let result = run_exact_match_evaluator(&inference_response, &datapoint).unwrap();
        assert_eq!(result, Some(Value::Bool(true)));

        // Test a mismatch
        let inference_response = InferenceResponse::Json(JsonInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            output: JsonInferenceOutput {
                parsed: Some(json!({"foo": "baz"})),
                raw: Some(r#"{"foo": "baz"}"#.to_string()),
            },
            usage: Usage {
                input_tokens: 10,
                output_tokens: 10,
            },
            original_response: None,
            finish_reason: None,
        });
        let result = run_exact_match_evaluator(&inference_response, &datapoint).unwrap();
        assert_eq!(result, Some(Value::Bool(false)));

        // Test with missing output (should be None)
        let datapoint = Datapoint::Json(JsonInferenceDatapoint {
            id: Uuid::now_v7(),
            input: ResolvedInput {
                system: None,
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!({"foo": "bar"}),
                    }],
                }],
            },
            dataset_name: "test".to_string(),
            function_name: "test".to_string(),
            episode_id: Some(Uuid::now_v7()),
            output: None,
            output_schema: json!({
                "type": "object",
                "properties": {
                    "foo": {
                        "type": "string"
                    }
                }
            }),
            tags: None,
            auxiliary: "".to_string(),
            is_deleted: false,
            source_inference_id: None,
            staled_at: None,
        });
        let result = run_exact_match_evaluator(&inference_response, &datapoint).unwrap();
        assert_eq!(result, None);

        // Test with datapoint with malformed output schema (should be None)
        let datapoint = Datapoint::Json(JsonInferenceDatapoint {
            id: Uuid::now_v7(),
            input: ResolvedInput {
                system: None,
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!({"foo": "bar"}),
                    }],
                }],
            },
            dataset_name: "test".to_string(),
            function_name: "test".to_string(),
            episode_id: Some(Uuid::now_v7()),
            output: Some(JsonInferenceOutput {
                parsed: None,
                raw: Some(r#"{"foo": "bar"}"#.to_string()),
            }),
            output_schema: json!({
                "type": "object",
                "properties": {
                    "foo": {
                        "type": "string"
                    }
                }
            }),
            tags: None,
            auxiliary: "".to_string(),
            is_deleted: false,
            source_inference_id: None,
            staled_at: None,
        });
        let result = run_exact_match_evaluator(&inference_response, &datapoint).unwrap();
        assert_eq!(result, None);
    }
}
