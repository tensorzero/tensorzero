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

impl ThinkingState {
    pub fn get_id(&self) -> String {
        match self {
            ThinkingState::Normal => "0".to_string(),
            ThinkingState::Thinking => "1".to_string(),
            ThinkingState::Finished => "2".to_string(),
        }
    }

    /// Returns true if an update was made to the thinking state
    /// Returns false if the text is not a thinking block
    /// Returns an error if the thinking state is invalid
    pub fn update(&mut self, text: &str, provider_type: &str) -> Result<bool, Error> {
        match (&mut *self, text) {
            (ThinkingState::Normal, "<think>") => {
                *self = ThinkingState::Thinking;
                Ok(true)
            }
            (ThinkingState::Normal, "</think>") => Err(Error::new(ErrorDetails::InferenceServer {
                message: "Found </think> while not thinking".to_string(),
                raw_request: None,
                raw_response: None,
                provider_type: provider_type.to_string(),
            })),
            (ThinkingState::Thinking, "</think>") => {
                *self = ThinkingState::Finished;
                Ok(true)
            }
            (ThinkingState::Thinking, "<think>") => {
                Err(Error::new(ErrorDetails::InferenceServer {
                    message: "Found <think> while already thinking".to_string(),
                    raw_request: None,
                    raw_response: None,
                    provider_type: provider_type.to_string(),
                }))
            }
            (ThinkingState::Finished, "<think>") | (ThinkingState::Finished, "</think>") => {
                Err(Error::new(ErrorDetails::InferenceServer {
                    message: "Found thinking tags after thinking finished".to_string(),
                    raw_request: None,
                    raw_response: None,
                    provider_type: provider_type.to_string(),
                }))
            }
            _ => Ok(false),
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
            if let ErrorDetails::InferenceServer { message, .. } = err.get_owned_details() {
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
            if let ErrorDetails::InferenceServer { message, .. } = err.get_owned_details() {
                assert_eq!(message, "Mismatched thinking tags");
            }
        }

        // Missing closing tag
        let text = "Hello <think>thinking without end tag";
        let result = process_think_blocks(text, true, provider_type);
        assert!(result.is_err());
        if let Err(err) = result {
            if let ErrorDetails::InferenceServer { message, .. } = err.get_owned_details() {
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
}
