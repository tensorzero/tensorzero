//! Custom deserializer for `InputMessage`.
//!
//! This module isolates the legacy compatibility logic needed to deserialize
//! inbound requests while keeping `serde_path_to_error`'s path tracking intact.

use super::{Arguments, InputContentBlock, InputMessage, Role, Template, Text, TextKind};
use serde::de::Error as DeserializerError;
use serde::de::SeqAccess;
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};
use serde_untagged::UntaggedEnumVisitor;

// A helper struct with a custom deserialize impl that handles legacy formats
struct MessageContent {
    inner: Vec<IntermediaryInputContentBlock>,
}

impl<'de> Deserialize<'de> for MessageContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let inner: Vec<IntermediaryInputContentBlock> = UntaggedEnumVisitor::new()
            .expecting(format_args!("a string, object, or array"))
            .string(|text| {
                Ok(vec![IntermediaryInputContentBlock::Final(Box::new(
                    InputContentBlock::Text(Text {
                        text: text.to_owned(),
                    }),
                ))])
            })
            .map(|map| {
                crate::utils::deprecation_warning("passing in an object for `content` is deprecated. Please use an array of content blocks instead.");
                let object: Map<String, Value> = map.deserialize()?;
                Ok(vec![IntermediaryInputContentBlock::TemplateFromArguments(
                    Arguments(object),
                )])
            })
            .seq(|mut seq| {
                let mut contents = Vec::new();
                // Deserialize each element with its own seed so that serde_path_to_error
                // tracks the array index and we can capture legacy shapes before they are
                // converted into modern content blocks.
                while let Some(content) = seq.next_element()? {
                    contents.push(content);
                }
                Ok(contents)
            })
            .deserialize(deserializer)?;
        Ok(MessageContent { inner })
    }
}

/// Custom deserializer for `InputMessage` that handles legacy formats:
/// - `"content": "text"` → `vec![Text { text }]`
/// - `"content": {...}` → `vec![Template { name: role, arguments }]`
/// - `"content": [{"type": "text", "arguments": {...}}]` → `vec![Template { name: role, arguments }]`
impl<'de> Deserialize<'de> for InputMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Helper {
            role: Role,
            content: MessageContent,
        }

        let helper = Helper::deserialize(deserializer)?;
        Ok(InputMessage {
            role: helper.role,
            content: finalize_intermediary_content(helper.content.inner, helper.role),
        })
    }
}

// We first deserialize into these intermediary variants so that we can keep the
// original `serde_path_to_error` context provided by `StructuredJson`. Any data
// that already maps to a modern `InputContentBlock` is boxed immediately,
// while legacy `"content": {...}` shapes are postponed until we have access to
// the message role (needed to synthesize the implicit template name).
enum IntermediaryInputContentBlock {
    Final(Box<InputContentBlock>),
    TemplateFromArguments(Arguments),
}

impl<'de> Deserialize<'de> for IntermediaryInputContentBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut value = Value::deserialize(deserializer)?;

        if let Value::Object(ref mut obj) = value
            && obj.get("type").and_then(|v| v.as_str()) == Some("text")
        {
            // Legacy text blocks used `"type": "text"` with additional fields in the same map.
            // We peel off the legacy fields here, keeping the original path tracking intact,
            // and delegate to `TextKind` so we reuse the existing validation logic.
            let mut text_fields = obj.clone();
            text_fields.remove("type");
            let text_kind_value = Value::Object(text_fields);
            let text_kind: TextKind = serde_json::from_value(text_kind_value)
                .map_err(|err| DeserializerError::custom(err.to_string()))?;
            return match text_kind {
                TextKind::Text { text } => Ok(IntermediaryInputContentBlock::Final(Box::new(
                    InputContentBlock::Text(Text { text }),
                ))),
                TextKind::Arguments { arguments } => Ok(
                    IntermediaryInputContentBlock::TemplateFromArguments(arguments),
                ),
            };
        }

        let content: InputContentBlock = serde_json::from_value(value)
            .map_err(|err| DeserializerError::custom(err.to_string()))?;
        Ok(IntermediaryInputContentBlock::Final(Box::new(content)))
    }
}

fn finalize_intermediary_content(
    intermediaries: Vec<IntermediaryInputContentBlock>,
    role: Role,
) -> Vec<InputContentBlock> {
    // At this point we have already preserved serde's path information. Now we
    // fold the intermediary variants into the final enum, inserting implicit
    // templates for legacy shapes that did not include a template name.
    intermediaries
        .into_iter()
        .map(|content| match content {
            IntermediaryInputContentBlock::Final(content) => *content,
            IntermediaryInputContentBlock::TemplateFromArguments(arguments) => {
                InputContentBlock::Template(Template {
                    name: role.implicit_template_name().to_string(),
                    arguments,
                })
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::Input;
    use serde_json::{Deserializer as JsonDeserializer, json};

    fn deserialize_input(
        json: &str,
    ) -> Result<Input, serde_path_to_error::Error<serde_json::Error>> {
        let mut deserializer = JsonDeserializer::from_str(json);
        serde_path_to_error::deserialize(&mut deserializer)
    }

    #[test]
    fn legacy_string_content_becomes_text() {
        let json = r#"{
            "messages": [{
                "role": "user",
                "content": "Hello world"
            }]
        }"#;

        let input = deserialize_input(json).expect("input should deserialize");
        assert_eq!(input.messages.len(), 1);
        let message = &input.messages[0];
        assert_eq!(message.role, Role::User);
        match message.content.as_slice() {
            [InputContentBlock::Text(Text { text })] => assert_eq!(text, "Hello world"),
            other => panic!("unexpected content: {other:?}"),
        }
    }

    #[test]
    fn legacy_object_content_becomes_template() {
        let json = r#"{
            "messages": [{
                "role": "user",
                "content": {"foo": "bar"}
            }]
        }"#;

        let input = deserialize_input(json).expect("input should deserialize");
        let message = &input.messages[0];
        match message.content.as_slice() {
            [InputContentBlock::Template(template)] => {
                assert_eq!(template.name, "user");
                assert_eq!(template.arguments.0.get("foo"), Some(&json!("bar")));
            }
            other => panic!("unexpected content: {other:?}"),
        }
    }

    #[test]
    fn legacy_text_arguments_array_becomes_template() {
        let json = r#"{
            "messages": [{
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "arguments": {"answer": "42"}
                }]
            }]
        }"#;

        let input = deserialize_input(json).expect("input should deserialize");
        let message = &input.messages[0];
        match message.content.as_slice() {
            [InputContentBlock::Template(template)] => {
                assert_eq!(template.name, "assistant");
                assert_eq!(template.arguments.0.get("answer"), Some(&json!("42")));
            }
            other => panic!("unexpected content: {other:?}"),
        }
    }

    #[test]
    fn preserves_error_path_for_non_legacy_array_content() {
        let json = r#"{
            "messages": [{
                "role": "user",
                "content": 123
            }]
        }"#;

        let err = deserialize_input(json).unwrap_err();
        assert_eq!(
            err.to_string(),
            "messages[0].content: invalid type: integer `123`, expected a string, object, or array at line 4 column 30"
        );
    }

    #[test]
    fn preserves_error_path_for_unknown_text_key() {
        let json = r#"{
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "bad_field": "Blah"
                }]
            }]
        }"#;

        let err = deserialize_input(json).unwrap_err();
        assert_eq!(
            err.to_string(),
            "messages[0].content[0]: Unknown key `bad_field` in text content at line 7 column 18"
        );
    }

    #[test]
    fn preserves_error_path_for_missing_text_payload() {
        let json = r#"{
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text"
                }]
            }]
        }"#;

        let err = deserialize_input(json).unwrap_err();
        assert_eq!(
            err.to_string(),
            "messages[0].content[0]: Expected exactly one other key in text content, found 0 other keys at line 6 column 18"
        );
    }
}
