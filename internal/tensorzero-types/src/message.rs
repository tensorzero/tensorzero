//! Input message types for the TensorZero API.
//!
//! This module contains types for representing messages in inference requests.

use crate::content::{Arguments, RawText, System, Template, Text, Thought, Unknown};
use crate::file::File;
use crate::role::Role;
use crate::tool::{ToolCallWrapper, ToolCallWrapperJsonSchema, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tensorzero_derive::{TensorZeroDeserialize, export_schema};

/// InputMessage and Role are our representation of the input sent by the client
/// prior to any processing into LLM representations below.
/// `InputMessage` has a custom deserializer that addresses legacy data formats that we used to support (see input_message.rs).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize, PartialEq, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct InputMessage {
    pub role: Role,
    pub content: Vec<InputMessageContent>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, JsonSchema, PartialEq, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(
    feature = "ts-bindings",
    ts(export, tag = "type", rename_all = "snake_case")
)]
#[export_schema]
pub enum InputMessageContent {
    #[schemars(title = "InputMessageContentText")]
    Text(Text),
    #[schemars(title = "InputMessageContentTemplate")]
    Template(Template),
    // `ToolCallWrapper` is `serde(untagged)` so no need to name it.
    #[schemars(with = "ToolCallWrapperJsonSchema")]
    ToolCall(ToolCallWrapper),
    #[schemars(title = "InputMessageContentToolResult")]
    ToolResult(ToolResult),
    #[schemars(title = "InputMessageContentRawText")]
    RawText(RawText),
    #[schemars(title = "InputMessageContentThought")]
    Thought(Thought),
    #[serde(alias = "image")]
    #[schemars(title = "InputMessageContentFile")]
    File(File),
    /// An unknown content block type, used to allow passing provider-specific
    /// content blocks (e.g. Anthropic's `redacted_thinking`) in and out
    /// of TensorZero.
    /// The `data` field holds the original content block from the provider,
    /// without any validation or transformation by TensorZero.
    #[schemars(title = "InputMessageContentUnknown")]
    Unknown(Unknown),
}

/// API representation of an input to a model.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default, JsonSchema)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[export_schema]
pub struct Input {
    /// System prompt of the input.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub system: Option<System>,

    /// Messages in the input.
    #[serde(default)]
    pub messages: Vec<InputMessage>,
}

// =============================================================================
// Custom Deserialize for InputMessage (handles legacy formats)
// =============================================================================

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(untagged, deny_unknown_fields)]
pub enum TextKind {
    Text { text: String },
    Arguments { arguments: Arguments },
}

impl<'de> Deserialize<'de> for TextKind {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let object: Map<String, Value> = Map::deserialize(de)?;
        // Expect exactly one key
        if object.keys().len() != 1 {
            return Err(serde::de::Error::custom(format!(
                "Expected exactly one other key in text content, found {} other keys",
                object.keys().len()
            )));
        }
        let (key, value) = object.into_iter().next().ok_or_else(|| {
            serde::de::Error::custom(
                "Internal error: Failed to get key/value after checking length",
            )
        })?;
        match key.as_str() {
            "text" => Ok(TextKind::Text {
                text: serde_json::from_value(value).map_err(|e| {
                    serde::de::Error::custom(format!("Error deserializing `text`: {e}"))
                })?,
            }),
            "arguments" => Ok(TextKind::Arguments {
                arguments: Arguments(serde_json::from_value(value).map_err(|e| {
                    serde::de::Error::custom(format!("Error deserializing `arguments`: {e}"))
                })?),
            }),
            _ => Err(serde::de::Error::custom(format!(
                "Unknown key `{key}` in text content"
            ))),
        }
    }
}

// A helper struct with a custom deserialize impl that handles legacy formats
struct MessageContent {
    inner: Vec<IntermediaryInputMessageContent>,
}

impl<'de> Deserialize<'de> for MessageContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::SeqAccess;

        struct MessageContentVisitor;

        impl<'de> serde::de::Visitor<'de> for MessageContentVisitor {
            type Value = Vec<IntermediaryInputMessageContent>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a string, object, or array")
            }

            fn visit_str<E>(self, text: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(vec![IntermediaryInputMessageContent::Final(Box::new(
                    InputMessageContent::Text(Text {
                        text: text.to_owned(),
                    }),
                ))])
            }

            fn visit_string<E>(self, text: String) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(vec![IntermediaryInputMessageContent::Final(Box::new(
                    InputMessageContent::Text(Text { text }),
                ))])
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                crate::deprecation_warning(
                    "passing in an object for `content` is deprecated. Please use an array of content blocks instead.",
                );
                let object: Map<String, Value> =
                    Deserialize::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
                Ok(vec![
                    IntermediaryInputMessageContent::TemplateFromArguments(Arguments(object)),
                ])
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut contents = Vec::new();
                while let Some(content) = seq.next_element()? {
                    contents.push(content);
                }
                Ok(contents)
            }
        }

        let inner = deserializer.deserialize_any(MessageContentVisitor)?;
        Ok(MessageContent { inner })
    }
}

// We first deserialize into these intermediary variants so that we can keep the
// original `serde_path_to_error` context. Any data that already maps to a modern
// `InputMessageContent` is boxed immediately, while legacy shapes are postponed
// until we have access to the message role.
enum IntermediaryInputMessageContent {
    Final(Box<InputMessageContent>),
    TemplateFromArguments(Arguments),
}

impl<'de> Deserialize<'de> for IntermediaryInputMessageContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as DeserializerError;

        let mut value = Value::deserialize(deserializer)?;

        if let Value::Object(ref mut obj) = value
            && obj.get("type").and_then(|v| v.as_str()) == Some("text")
        {
            // Legacy text blocks used `"type": "text"` with additional fields in the same map.
            let mut text_fields = obj.clone();
            text_fields.remove("type");
            let text_kind_value = Value::Object(text_fields);
            let text_kind: TextKind = serde_json::from_value(text_kind_value)
                .map_err(|err| DeserializerError::custom(err.to_string()))?;
            return match text_kind {
                TextKind::Text { text } => Ok(IntermediaryInputMessageContent::Final(Box::new(
                    InputMessageContent::Text(Text { text }),
                ))),
                TextKind::Arguments { arguments } => Ok(
                    IntermediaryInputMessageContent::TemplateFromArguments(arguments),
                ),
            };
        }

        let content: InputMessageContent = serde_json::from_value(value)
            .map_err(|err| DeserializerError::custom(err.to_string()))?;
        Ok(IntermediaryInputMessageContent::Final(Box::new(content)))
    }
}

fn finalize_intermediary_content(
    intermediaries: Vec<IntermediaryInputMessageContent>,
    role: Role,
) -> Vec<InputMessageContent> {
    intermediaries
        .into_iter()
        .map(|content| match content {
            IntermediaryInputMessageContent::Final(content) => *content,
            IntermediaryInputMessageContent::TemplateFromArguments(arguments) => {
                InputMessageContent::Template(Template {
                    name: role.implicit_template_name().to_string(),
                    arguments,
                })
            }
        })
        .collect()
}

/// Custom deserializer for `InputMessage` that handles legacy formats:
/// - `"content": "text"` → `vec![Text { text }]`
/// - `"content": {...}` → `vec![Template { name: role, arguments }]`
/// - `"content": [{"type": "text", "arguments": {...}}]` → `vec![Template { name: role, arguments }]`
impl<'de> Deserialize<'de> for InputMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
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

impl InputMessageContent {
    /// Test-only helper; do not use in production code.
    pub fn test_only_from_string(text: String) -> Self {
        InputMessageContent::Text(Text { text })
    }
}
