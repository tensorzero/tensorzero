use crate::error::{Error, ErrorDetails};

pub const THINK_TAG: &str = "<think>";
pub const THINK_TAG_LEN: usize = THINK_TAG.len();
pub const END_THINK_TAG: &str = "</think>";
pub const END_THINK_TAG_LEN: usize = END_THINK_TAG.len();

/// Processes the thinking blocks from a span of text.
/// If parsing is disabled, the text is returned as is.
/// If parsing is enabled, the text is checked for a single thinking block.
/// If there is one, the text is cleaned of the thinking block and the reasoning is returned.
/// If there is no thinking block, the text is returned as is and the reasoning is None.
/// If there is more than one thinking block, an error is returned.
///
/// The function also validates that tags are properly matched - an error is returned
/// if there are mismatched opening/closing tags.
///
/// Returns a tuple of (cleaned_text, optional_reasoning).
/// The reasoning, if present, will have leading/trailing whitespace trimmed.
pub fn process_think_blocks(
    text: &str,
    parse: bool,
    provider_type: &str,
) -> Result<(String, Option<String>), Error> {
    if !parse {
        return Ok((text.to_string(), None));
    }
    let think_count = text.matches(THINK_TAG).count();
    if think_count > 1 {
        return Err(Error::new(ErrorDetails::InferenceServer {
            message: "Multiple thinking blocks found".to_string(),
            raw_request: None,
            raw_response: None,
            provider_type: provider_type.to_string(),
        }));
    }

    if think_count != text.matches(END_THINK_TAG).count() {
        Err(Error::new(ErrorDetails::InferenceServer {
            message: "Mismatched thinking tags".to_string(),
            raw_request: None,
            raw_response: None,
            provider_type: provider_type.to_string(),
        }))
    } else if let (Some(start), Some(end)) = (text.find(THINK_TAG), text.find(END_THINK_TAG)) {
        let reasoning = text[start + THINK_TAG_LEN..end].to_string();
        let cleaned = format!("{}{}", &text[..start], &text[end + END_THINK_TAG_LEN..]);
        Ok((cleaned, Some(reasoning)))
    } else {
        Ok((text.to_string(), None))
    }
}

/// State machine for tracking thinking tag transitions during streaming
pub enum ThinkingState {
    Normal,
    Thinking,
    Finished,
}

pub const THINK_CHUNK_ID: u64 = 1;

impl ThinkingState {
    pub fn get_id(&self) -> String {
        match self {
            ThinkingState::Normal => "0".to_string(),
            ThinkingState::Thinking => "1".to_string(),
            ThinkingState::Finished => "2".to_string(),
        }
    }

    fn make_error(msg: &str, provider_type: &str) -> Error {
        Error::new(ErrorDetails::InferenceServer {
            message: msg.to_string(),
            raw_request: None,
            raw_response: None,
            provider_type: provider_type.to_string(),
        })
    }

    /// Returns true if an update was made to the thinking state
    /// Returns false if the text is not a thinking block
    /// Returns an error if the thinking state is invalid
    pub fn update(&mut self, text: &str, provider_type: &str) -> Result<bool, Error> {
        // Early return for empty strings, whitespace-only strings, or just newlines
        if text.trim().is_empty() {
            return Ok(false);
        }

        let has_open = text.contains("<think>");
        let has_close = text.contains("</think>");

        match self {
            ThinkingState::Normal => match (has_open, has_close) {
                (true, false) => {
                    *self = ThinkingState::Thinking;
                    Ok(true)
                }
                (false, true) => Err(Self::make_error(
                    "Found </think> while not thinking",
                    provider_type,
                )),
                _ => Ok(false),
            },
            ThinkingState::Thinking => match (has_open, has_close) {
                (false, true) => {
                    *self = ThinkingState::Finished;
                    Ok(true)
                }
                (true, false) => Err(Self::make_error(
                    "Found <think> while already thinking",
                    provider_type,
                )),
                _ => Ok(false),
            },
            ThinkingState::Finished => {
                if has_open || has_close {
                    Err(Self::make_error(
                        "Found thinking tags after thinking finished",
                        provider_type,
                    ))
                } else {
                    Ok(false)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_think_blocks_single() {
        let text = "Hello <think>this is thinking</think> world";
        let provider_type = "test_provider";

        // With parsing enabled
        let (cleaned_text, reasoning) = process_think_blocks(text, true, provider_type).unwrap();
        assert_eq!(cleaned_text, "Hello  world");
        assert_eq!(reasoning, Some("this is thinking".to_string()));

        // With parsing disabled
        let (cleaned_text, reasoning) = process_think_blocks(text, false, provider_type).unwrap();
        assert_eq!(cleaned_text, text);
        assert_eq!(reasoning, None);
    }

    #[test]
    fn test_process_think_blocks_multiple() {
        let text = "Hello <think>thinking 1</think> middle <think>thinking 2</think>";
        let provider_type = "test_provider";

        // With parsing enabled - should error on multiple blocks
        let result = process_think_blocks(text, true, provider_type);
        assert!(result.is_err());
        if let Err(err) = result {
            if let ErrorDetails::InferenceServer { message, .. } = err.get_details() {
                assert_eq!(message, "Multiple thinking blocks found");
            }
        }

        // With parsing disabled
        let (cleaned_text, reasoning) = process_think_blocks(text, false, provider_type).unwrap();
        assert_eq!(cleaned_text, text);
        assert_eq!(reasoning, None);
    }

    #[test]
    fn test_process_think_blocks_mismatched_tags() {
        let provider_type = "test_provider";

        // Extra closing tag
        let text = "Hello <think>Extra closing tag</think></think> world";
        let result = process_think_blocks(text, true, provider_type);
        assert!(result.is_err());
        if let Err(err) = result {
            if let ErrorDetails::InferenceServer { message, .. } = err.get_details() {
                assert_eq!(message, "Mismatched thinking tags");
            }
        }

        // Missing closing tag
        let text = "Hello <think>thinking without end tag";
        let result = process_think_blocks(text, true, provider_type);
        assert!(result.is_err());
        if let Err(err) = result {
            if let ErrorDetails::InferenceServer { message, .. } = err.get_details() {
                assert_eq!(message, "Mismatched thinking tags");
            }
        }
    }

    #[test]
    fn test_thinking_state_transitions() {
        let provider_type = "test_provider";

        // Normal state tests
        let mut state = ThinkingState::Normal;
        assert_eq!(state.get_id(), "0");

        // Valid transition: Normal -> Thinking
        assert!(state.update("<think>", provider_type).unwrap());
        assert!(matches!(state, ThinkingState::Thinking));
        assert_eq!(state.get_id(), "1");

        // Valid transition: Thinking -> Finished
        assert!(state.update("</think>", provider_type).unwrap());
        assert!(matches!(state, ThinkingState::Finished));
        assert_eq!(state.get_id(), "2");

        // Invalid transitions
        let mut state = ThinkingState::Normal;
        assert!(state.update("</think>", provider_type).is_err());

        let mut state = ThinkingState::Thinking;
        assert!(state.update("<think>", provider_type).is_err());

        let mut state = ThinkingState::Finished;
        assert!(state.update("<think>", provider_type).is_err());
        assert!(state.update("</think>", provider_type).is_err());

        // Non-tag text shouldn't change state
        let mut state = ThinkingState::Normal;
        assert!(!state.update("random text", provider_type).unwrap());
        assert!(matches!(state, ThinkingState::Normal));

        let mut state = ThinkingState::Thinking;
        assert!(!state.update("random text", provider_type).unwrap());
        assert!(matches!(state, ThinkingState::Thinking));
    }

    #[test]
    fn test_thinking_state_with_newline_and_whitespace() {
        let provider_type = "test_provider";

        // Test with newline character - issue reported for Together/Fireworks
        let mut state = ThinkingState::Normal;

        // This should not change state since it's just a newline
        assert!(!state.update("\n", provider_type).unwrap());
        assert!(matches!(state, ThinkingState::Normal));

        // Empty string should also not change state
        assert!(!state.update("", provider_type).unwrap());
        assert!(matches!(state, ThinkingState::Normal));

        // Space characters should not change state
        assert!(!state.update("   ", provider_type).unwrap());
        assert!(matches!(state, ThinkingState::Normal));
    }

    #[test]
    fn test_thinking_state_update_with_newline() {
        // This test verifies that the ThinkingState.update function correctly handles
        // newlines, which was causing issues in the Fireworks/Together implementation
        let provider_type = "fireworks";

        // Test that a newline doesn't change the state
        let mut state = ThinkingState::Normal;
        let result = state.update("\n", provider_type);

        // Should succeed without error
        assert!(result.is_ok());

        // Should return false indicating no change to state
        assert!(!result.unwrap());

        // State should remain Normal
        assert!(matches!(state, ThinkingState::Normal));
    }

    #[test]
    fn test_empty_string_handling() {
        // This simulates the issue reported with the Together/Fireworks providers
        // where a newline chunk was causing errors
        let provider_type = "together";

        // Start with Normal state
        let mut state = ThinkingState::Normal;

        // Test empty string
        let result = state.update("", provider_type);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should not change state
        assert!(matches!(state, ThinkingState::Normal));

        // Test whitespace string
        let result = state.update("   ", provider_type);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should not change state
        assert!(matches!(state, ThinkingState::Normal));

        // Test newline
        let result = state.update("\n", provider_type);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should not change state
        assert!(matches!(state, ThinkingState::Normal));

        // Test multiple newlines
        let result = state.update("\n\n\n", provider_type);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should not change state
        assert!(matches!(state, ThinkingState::Normal));
    }

    #[test]
    fn test_thinking_tag_with_surrounding_whitespace() {
        // Test that tags with surrounding whitespace are still recognized
        let provider_type = "together";

        // Start with Normal state
        let mut state = ThinkingState::Normal;

        // Open tag with leading/trailing whitespace
        let result = state.update("  <think>  ", provider_type);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should change state
        assert!(matches!(state, ThinkingState::Thinking));

        // Close tag with leading/trailing whitespace
        let result = state.update(" </think> ", provider_type);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should change state
        assert!(matches!(state, ThinkingState::Finished));
    }

    #[test]
    fn test_thinking_tag_in_multiline_text() {
        // Test that tags embedded in multiline text are recognized
        let provider_type = "together";

        // Start with Normal state
        let mut state = ThinkingState::Normal;

        // Tag in multiline text
        let result = state.update("Some text\n<think>\nMore text", provider_type);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should change state
        assert!(matches!(state, ThinkingState::Thinking));

        // Close tag in multiline text
        let result = state.update("Some text\n</think>\nMore text", provider_type);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should change state
        assert!(matches!(state, ThinkingState::Finished));
    }

    #[test]
    fn test_fireworks_deepseek_reported_issue() {
        // Test the specific issue reported with fireworks-deepseek
        let provider_type = "together";

        // Create the thinking state tracking
        let mut thinking_state = ThinkingState::Normal;

        // After a newline response that reportedly triggered the issue
        let text = "\n";
        let result = thinking_state.update(text, provider_type);

        // Should succeed without error
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should not change state
        assert!(matches!(thinking_state, ThinkingState::Normal));
    }
}
