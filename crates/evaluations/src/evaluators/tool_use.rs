use std::collections::HashMap;

use anyhow::{Result, bail};
use serde_json::Value;
use tensorzero_core::client::InferenceResponse;
use tensorzero_core::evaluations::ToolUseConfig;
use tensorzero_core::inference::types::ContentBlockChatOutput;
use tracing::{debug, instrument};

#[instrument(skip_all)]
pub(super) fn run_tool_use_evaluator(
    inference_response: &InferenceResponse,
    config: &ToolUseConfig,
) -> Result<Option<Value>> {
    let InferenceResponse::Chat(response) = inference_response else {
        bail!(
            "Tool use evaluator does not support JSON inferences (tool calls only exist in chat inferences)"
        )
    };

    let mut called_tool_counts: HashMap<&str, usize> = HashMap::new();
    for block in &response.content {
        if let ContentBlockChatOutput::ToolCall(tc) = block {
            *called_tool_counts.entry(tc.raw_name.as_str()).or_default() += 1;
        }
    }

    debug!(called_tool_counts = ?called_tool_counts, behavior = %config, "Evaluating tool use");

    let result = match config {
        ToolUseConfig::None => called_tool_counts.is_empty(),
        ToolUseConfig::Any => !called_tool_counts.is_empty(),
        ToolUseConfig::NoneOf { tools } => tools
            .iter()
            .all(|t| !called_tool_counts.contains_key(t.as_str())),
        ToolUseConfig::AnyOf { tools } => tools
            .iter()
            .any(|t| called_tool_counts.contains_key(t.as_str())),
        ToolUseConfig::AllOf { tools } => {
            let mut required_counts: HashMap<&str, usize> = HashMap::new();
            for t in tools {
                *required_counts.entry(t.as_str()).or_default() += 1;
            }
            required_counts.iter().all(|(tool, &required)| {
                called_tool_counts.get(tool).copied().unwrap_or(0) >= required
            })
        }
    };

    debug!(result = %result, "Tool use evaluation completed");
    Ok(Some(Value::Bool(result)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorzero_core::endpoints::inference::ChatInferenceResponse;
    use tensorzero_core::inference::types::{
        ContentBlockChatOutput, JsonInferenceOutput, Text, Usage,
    };
    use tensorzero_core::tool::InferenceResponseToolCall;
    use uuid::Uuid;

    /// Always includes a leading text block to exercise mixed-content filtering.
    fn make_chat_response(tool_names: &[&str]) -> InferenceResponse {
        let mut content: Vec<ContentBlockChatOutput> = vec![ContentBlockChatOutput::Text(Text {
            text: "Here is my response.".to_string(),
        })];
        content.extend(tool_names.iter().map(|name| {
            ContentBlockChatOutput::ToolCall(InferenceResponseToolCall {
                id: Uuid::now_v7().to_string(),
                raw_name: name.to_string(),
                raw_arguments: "{}".to_string(),
                name: Some(name.to_string()),
                arguments: Some(serde_json::json!({})),
            })
        }));
        InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            content,
            usage: Usage {
                input_tokens: Some(10),
                output_tokens: Some(10),
                cost: None,
            },
            raw_usage: None,
            original_response: None,
            raw_response: None,
            finish_reason: None,
        })
    }

    fn make_json_response() -> InferenceResponse {
        InferenceResponse::Json(
            tensorzero_core::endpoints::inference::JsonInferenceResponse {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
                variant_name: "test".to_string(),
                output: JsonInferenceOutput {
                    parsed: Some(serde_json::json!({"key": "value"})),
                    raw: Some(r#"{"key": "value"}"#.to_string()),
                },
                usage: Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(10),
                    cost: None,
                },
                raw_usage: None,
                original_response: None,
                raw_response: None,
                finish_reason: None,
            },
        )
    }

    #[test]
    fn test_behavior_none_no_tools() {
        let response = make_chat_response(&[]);
        let config = ToolUseConfig::None;
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for chat response with no tools");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "should pass when no tools are called"
        );
    }

    #[test]
    fn test_behavior_none_with_tools() {
        let response = make_chat_response(&["search"]);
        let config = ToolUseConfig::None;
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for chat response with tools");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "should fail when tools are called"
        );
    }

    #[test]
    fn test_behavior_any_with_tools() {
        let response = make_chat_response(&["search"]);
        let config = ToolUseConfig::Any;
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for chat response with tools");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "should pass when at least one tool is called"
        );
    }

    #[test]
    fn test_behavior_any_no_tools() {
        let response = make_chat_response(&[]);
        let config = ToolUseConfig::Any;
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for chat response with no tools");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "should fail when no tools are called"
        );
    }

    #[test]
    fn test_behavior_none_of_no_match() {
        let response = make_chat_response(&["search", "weather"]);
        let config = ToolUseConfig::NoneOf {
            tools: vec!["calculator".to_string(), "email".to_string()],
        };
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for none_of with no forbidden tools called");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "should pass when called tools are not in the forbidden list"
        );
    }

    #[test]
    fn test_behavior_none_of_with_match() {
        let response = make_chat_response(&["search", "calculator"]);
        let config = ToolUseConfig::NoneOf {
            tools: vec!["calculator".to_string(), "email".to_string()],
        };
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for none_of with forbidden tool called");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "should fail when a called tool is in the forbidden list"
        );
    }

    #[test]
    fn test_behavior_any_of_with_match() {
        let response = make_chat_response(&["search", "calculator"]);
        let config = ToolUseConfig::AnyOf {
            tools: vec!["calculator".to_string(), "email".to_string()],
        };
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for any_of with matching tool");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "should pass when at least one called tool is in the required list"
        );
    }

    #[test]
    fn test_behavior_any_of_no_match() {
        let response = make_chat_response(&["search", "weather"]);
        let config = ToolUseConfig::AnyOf {
            tools: vec!["calculator".to_string(), "email".to_string()],
        };
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for any_of with no matching tools");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "should fail when no called tool is in the required list"
        );
    }

    #[test]
    fn test_behavior_all_of_all_present() {
        let response = make_chat_response(&["search", "calculator", "email"]);
        let config = ToolUseConfig::AllOf {
            tools: vec!["calculator".to_string(), "search".to_string()],
        };
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for all_of with all tools present");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "should pass when all required tools are called (extra tools are ok)"
        );
    }

    #[test]
    fn test_behavior_all_of_partial() {
        let response = make_chat_response(&["search"]);
        let config = ToolUseConfig::AllOf {
            tools: vec!["calculator".to_string(), "search".to_string()],
        };
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for all_of with partial tools");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "should fail when not all required tools are called"
        );
    }

    #[test]
    fn test_json_inference_error() {
        let response = make_json_response();
        let config = ToolUseConfig::Any;
        let result = run_tool_use_evaluator(&response, &config);
        assert!(result.is_err(), "should error for JSON inferences");
    }

    #[test]
    fn test_duplicate_tool_calls_sufficient() {
        let response = make_chat_response(&["search", "search", "search"]);
        let config = ToolUseConfig::AllOf {
            tools: vec!["search".to_string()],
        };
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for all_of with duplicate tool calls");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "should pass when tool is called more times than required"
        );
    }

    #[test]
    fn test_all_of_duplicate_required_tools_satisfied() {
        let response = make_chat_response(&["bash", "bash", "search"]);
        let config = ToolUseConfig::AllOf {
            tools: vec!["bash".to_string(), "bash".to_string()],
        };
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for all_of with duplicate required tools");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "should pass when tool is called at least as many times as required"
        );
    }

    #[test]
    fn test_all_of_duplicate_required_tools_insufficient() {
        let response = make_chat_response(&["bash", "search"]);
        let config = ToolUseConfig::AllOf {
            tools: vec!["bash".to_string(), "bash".to_string()],
        };
        let result = run_tool_use_evaluator(&response, &config)
            .expect("evaluator should succeed for all_of with insufficient duplicate calls");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "should fail when tool is called fewer times than required"
        );
    }
}
