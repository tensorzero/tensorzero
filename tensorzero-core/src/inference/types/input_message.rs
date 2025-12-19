//! Custom deserializer for `InputMessage`.
//!
//! This module provides a custom deserializer that handles the string content shorthand:
//! - `"content": "text"` → `vec![Text { text }]`
//! - `"content": [...]` → standard `Vec<InputMessageContent>` deserialization

use super::{InputMessageContent, Text};
use serde::Deserializer;
use serde_untagged::UntaggedEnumVisitor;

/// Deserializes content as either a string shorthand or an array.
pub fn deserialize_input_message_content<'de, D>(
    deserializer: D,
) -> Result<Vec<InputMessageContent>, D::Error>
where
    D: Deserializer<'de>,
{
    UntaggedEnumVisitor::new()
        .expecting(format_args!("a string or array"))
        .string(|text| {
            Ok(vec![InputMessageContent::Text(Text {
                text: text.to_owned(),
            })])
        })
        .seq(|seq| seq.deserialize())
        .deserialize(deserializer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::{Input, Role};
    use serde_json::Deserializer as JsonDeserializer;

    fn deserialize_input(
        json: &str,
    ) -> Result<Input, serde_path_to_error::Error<serde_json::Error>> {
        let mut deserializer = JsonDeserializer::from_str(json);
        serde_path_to_error::deserialize(&mut deserializer)
    }

    #[test]
    fn string_content_becomes_text() {
        let json = r#"{
            "messages": [{
                "role": "user",
                "content": "Hello world"
            }]
        }"#;

        let input = deserialize_input(json).expect("input should deserialize");
        assert_eq!(input.messages.len(), 1, "expected 1 message");
        let message = &input.messages[0];
        assert_eq!(message.role, Role::User);
        match message.content.as_slice() {
            [InputMessageContent::Text(Text { text })] => {
                assert_eq!(text, "Hello world", "text content should match");
            }
            other => panic!("unexpected content: {other:?}"),
        }
    }

    #[test]
    fn array_content_deserializes_normally() {
        let json = r#"{
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello world"}]
            }]
        }"#;

        let input = deserialize_input(json).expect("input should deserialize");
        assert_eq!(input.messages.len(), 1, "expected 1 message");
        let message = &input.messages[0];
        assert_eq!(message.role, Role::User);
        match message.content.as_slice() {
            [InputMessageContent::Text(Text { text })] => {
                assert_eq!(text, "Hello world", "text content should match");
            }
            other => panic!("unexpected content: {other:?}"),
        }
    }

    #[test]
    fn preserves_error_path_for_invalid_content_type() {
        let json = r#"{
            "messages": [{
                "role": "user",
                "content": 123
            }]
        }"#;

        let err = deserialize_input(json).unwrap_err();
        assert!(
            err.to_string().contains("messages[0].content"),
            "error should include path: {err}"
        );
    }
}
