use anyhow::Result;
use regex::Regex;
use serde_json::Value;
use tensorzero_core::client::InferenceResponse;
use tensorzero_core::evaluations::RegexConfig;
use tensorzero_core::inference::types::ContentBlockChatOutput;
use tracing::{debug, instrument};

/// Extracts text content from an inference response.
///
/// For Chat responses, concatenates all `Text` content blocks.
/// For Json responses, uses the `output.raw` string.
fn extract_text(inference_response: &InferenceResponse) -> Option<String> {
    match inference_response {
        InferenceResponse::Chat(response) => {
            let text: String = response
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlockChatOutput::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            Some(text)
        }
        InferenceResponse::Json(response) => response.output.raw.clone(),
    }
}

#[instrument(skip_all)]
pub(super) fn run_regex_evaluator(
    inference_response: &InferenceResponse,
    config: &RegexConfig,
) -> Result<Option<Value>> {
    // TODO(#6584): Precompile regexes instead of rebuilding on every inference for performance.
    let Some(text) = extract_text(inference_response) else {
        debug!("No text content found in inference response");

        // If there's no "must match", then "no output" should be treated as success.
        let result = config.must_match.is_none();
        return Ok(Some(Value::Bool(result)));
    };

    let must_match_ok = match &config.must_match {
        Some(pattern) => {
            let re = Regex::new(pattern)?;
            re.is_match(&text)
        }
        None => true,
    };

    let must_not_match_ok = match &config.must_not_match {
        Some(pattern) => {
            let re = Regex::new(pattern)?;
            !re.is_match(&text)
        }
        None => true,
    };

    let result = must_match_ok && must_not_match_ok;
    debug!(result = %result, "Regex evaluation completed");
    Ok(Some(Value::Bool(result)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorzero_core::endpoints::inference::{ChatInferenceResponse, JsonInferenceResponse};
    use tensorzero_core::inference::types::{JsonInferenceOutput, Text, Usage};
    use tensorzero_core::tool::InferenceResponseToolCall;
    use uuid::Uuid;

    fn make_chat_response(text: &str) -> InferenceResponse {
        InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            content: vec![ContentBlockChatOutput::Text(Text {
                text: text.to_string(),
            })],
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

    fn make_json_response(raw: Option<&str>) -> InferenceResponse {
        InferenceResponse::Json(JsonInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test".to_string(),
            output: JsonInferenceOutput {
                parsed: raw
                    .map(|r| serde_json::from_str(r).unwrap_or(Value::String(r.to_string()))),
                raw: raw.map(|r| r.to_string()),
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
        })
    }

    fn make_mixed_chat_response(texts: &[&str], tool_names: &[&str]) -> InferenceResponse {
        let mut content: Vec<ContentBlockChatOutput> = texts
            .iter()
            .map(|t| {
                ContentBlockChatOutput::Text(Text {
                    text: t.to_string(),
                })
            })
            .collect();
        for name in tool_names {
            content.push(ContentBlockChatOutput::ToolCall(
                InferenceResponseToolCall {
                    id: Uuid::now_v7().to_string(),
                    raw_name: name.to_string(),
                    raw_arguments: "{}".to_string(),
                    name: Some(name.to_string()),
                    arguments: Some(serde_json::json!({})),
                },
            ));
        }
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

    // --- must_match only ---

    #[test]
    fn test_must_match_matches() {
        let response = make_chat_response("Could you please help me?");
        let config = RegexConfig {
            must_match: Some("(?i)please".to_string()),
            must_not_match: None,
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "should pass when must_match pattern is found"
        );
    }

    #[test]
    fn test_must_match_does_not_match() {
        let response = make_chat_response("Help me now!");
        let config = RegexConfig {
            must_match: Some("(?i)please".to_string()),
            must_not_match: None,
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "should fail when must_match pattern is not found"
        );
    }

    // --- must_not_match only ---

    #[test]
    fn test_must_not_match_no_match() {
        let response = make_chat_response("This is a polite response.");
        let config = RegexConfig {
            must_match: None,
            must_not_match: Some("(?i)badword".to_string()),
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "should pass when must_not_match pattern is not found"
        );
    }

    #[test]
    fn test_must_not_match_matches() {
        let response = make_chat_response("This contains badword in it.");
        let config = RegexConfig {
            must_match: None,
            must_not_match: Some("(?i)badword".to_string()),
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "should fail when must_not_match pattern is found"
        );
    }

    // --- both specified ---

    #[test]
    fn test_both_conditions_met() {
        let response = make_chat_response("Could you please help me?");
        let config = RegexConfig {
            must_match: Some("(?i)please".to_string()),
            must_not_match: Some("(?i)badword".to_string()),
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "should pass when must_match matches and must_not_match doesn't"
        );
    }

    #[test]
    fn test_only_must_match_met() {
        let response = make_chat_response("Please don't say badword.");
        let config = RegexConfig {
            must_match: Some("(?i)please".to_string()),
            must_not_match: Some("(?i)badword".to_string()),
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "should fail when must_match matches but must_not_match also matches"
        );
    }

    #[test]
    fn test_only_must_not_match_met() {
        let response = make_chat_response("Here's your answer.");
        let config = RegexConfig {
            must_match: Some("(?i)please".to_string()),
            must_not_match: Some("(?i)badword".to_string()),
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "should fail when must_match doesn't match (even though must_not_match is ok)"
        );
    }

    #[test]
    fn test_neither_condition_met() {
        let response = make_chat_response("badword without politeness.");
        let config = RegexConfig {
            must_match: Some("(?i)please".to_string()),
            must_not_match: Some("(?i)badword".to_string()),
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "should fail when neither condition is met"
        );
    }

    // --- case insensitive ---

    #[test]
    fn test_case_insensitive_match() {
        let response = make_chat_response("PLEASE help me");
        let config = RegexConfig {
            must_match: Some("(?i)please".to_string()),
            must_not_match: None,
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "case-insensitive flag should match uppercase text"
        );
    }

    // --- mixed content blocks ---

    #[test]
    fn test_chat_mixed_content_blocks() {
        let response =
            make_mixed_chat_response(&["Let me search for ", "please wait."], &["search_tool"]);
        let config = RegexConfig {
            must_match: Some("please".to_string()),
            must_not_match: None,
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "should concatenate text blocks and find pattern across them"
        );
    }

    // --- JSON response ---

    #[test]
    fn test_json_response_raw() {
        let response = make_json_response(Some(r#"{"message": "please help"}"#));
        let config = RegexConfig {
            must_match: Some("please".to_string()),
            must_not_match: None,
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "should match against JSON raw output"
        );
    }

    #[test]
    fn test_json_response_none_raw() {
        let response = make_json_response(None);
        let config = RegexConfig {
            must_match: Some("please".to_string()),
            must_not_match: None,
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "should return false when JSON response has no raw output but must_match is set"
        );
    }

    // --- no text with no must_match ---

    #[test]
    fn test_no_text_no_must_match_succeeds() {
        let response = make_json_response(None);
        let config = RegexConfig {
            must_match: None,
            must_not_match: None,
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "no output with no must_match should be treated as success"
        );
    }

    #[test]
    fn test_no_text_no_must_match_with_must_not_match_succeeds() {
        let response = make_json_response(None);
        let config = RegexConfig {
            must_match: None,
            must_not_match: Some("(?i)badword".to_string()),
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(true)),
            "no output with only must_not_match should be treated as success"
        );
    }

    #[test]
    fn test_no_text_with_must_match_fails() {
        let response = make_json_response(None);
        let config = RegexConfig {
            must_match: Some("required_pattern".to_string()),
            must_not_match: Some("(?i)badword".to_string()),
        };
        let result = run_regex_evaluator(&response, &config).expect("evaluator should succeed");
        assert_eq!(
            result,
            Some(Value::Bool(false)),
            "no output with must_match set should fail regardless of must_not_match"
        );
    }
}
