use crate::{
    error::Error,
    inference::types::{
        File, InputMessageContent, RawText, Role, System, Template, Text, TextKind, Thought,
        Unknown,
    },
    tool::{ToolCallWrapper, ToolResult},
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_untagged::UntaggedEnumVisitor;
use tensorzero_derive::TensorZeroDeserialize;

// Like the normal `Input` type, but with `ClientInputMessage` instead of `InputMessage`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
#[derive(ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct ClientInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub system: Option<System>,
    #[serde(default)]
    pub messages: Vec<ClientInputMessage>,
}

// Like the normal `InputMessage` type, but with `ClientInputMessageContent` instead of `InputMessageContent`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[derive(ts_rs::TS)]
#[ts(export)]
pub struct ClientInputMessage {
    pub role: Role,
    #[serde(deserialize_with = "deserialize_content")]
    pub content: Vec<ClientInputMessageContent>,
}

#[derive(Clone, Debug, TensorZeroDeserialize, Serialize, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[derive(ts_rs::TS)]
#[ts(export)]
pub enum ClientInputMessageContent {
    Text(TextKind),
    Template(Template),
    ToolCall(ToolCallWrapper),
    ToolResult(ToolResult),
    RawText(RawText),
    Thought(Thought),
    #[serde(alias = "image")]
    File(File),
    /// An unknown content block type, used to allow passing provider-specific
    /// content blocks (e.g. Anthropic's "redacted_thinking") in and out
    /// of TensorZero.
    /// The `data` field holds the original content block from the provider,
    /// without any validation or transformation by TensorZero.
    Unknown(Unknown),
}

impl ClientInputMessageContent {
    pub fn to_input_message_content(self, role: &Role) -> Result<InputMessageContent, Error> {
        use crate::inference::types::Text;

        Ok(match self {
            ClientInputMessageContent::Text(TextKind::Text { text }) => {
                InputMessageContent::Text(Text { text })
            }
            ClientInputMessageContent::Text(TextKind::Arguments { arguments }) => {
                InputMessageContent::Template(Template {
                    name: role.implicit_template_name().to_string(),
                    arguments,
                })
            }
            ClientInputMessageContent::Template(template) => {
                InputMessageContent::Template(template)
            }
            ClientInputMessageContent::ToolCall(tool_call) => {
                InputMessageContent::ToolCall(tool_call)
            }
            ClientInputMessageContent::ToolResult(tool_result) => {
                InputMessageContent::ToolResult(tool_result)
            }
            ClientInputMessageContent::RawText(raw_text) => InputMessageContent::RawText(raw_text),
            ClientInputMessageContent::Thought(thought) => InputMessageContent::Thought(thought),
            ClientInputMessageContent::File(image) => InputMessageContent::File(image),
            ClientInputMessageContent::Unknown(unknown) => InputMessageContent::Unknown(unknown),
        })
    }
}

pub fn deserialize_content<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<ClientInputMessageContent>, D::Error> {
    UntaggedEnumVisitor::new()
        .string(|text| {
            Ok(vec![ClientInputMessageContent::Text(TextKind::Text {
                text: text.to_string(),
            })])
        })
        .map(|object| {
            crate::utils::deprecation_warning("passing in an object for `content` is deprecated. Please use an array of content blocks instead.");
            Ok(vec![ClientInputMessageContent::Text(TextKind::Arguments {
                arguments: object.deserialize()?,
            })])
        })
        .seq(|seq| seq.deserialize())
        .deserialize(deserializer)
}

// Helper function to make sure that our `Input` and `ClientInput` types match up
// as expected. This is never actually called - we just care that it compiles
pub(super) fn test_client_input_to_input(
    client_input: ClientInput,
) -> crate::inference::types::Input {
    crate::inference::types::Input {
        system: client_input.system,
        messages: client_input
            .messages
            .into_iter()
            .map(|message| {
                let ClientInputMessage { role, content } = message;
                crate::inference::types::InputMessage {
                    role,
                    content: content
                        .into_iter()
                        .map(|content| test_client_to_message_content(role, content))
                        .collect(),
                }
            })
            .collect(),
    }
}

pub(super) fn test_client_to_message_content(
    role: Role,
    content: ClientInputMessageContent,
) -> InputMessageContent {
    match content {
        ClientInputMessageContent::Text(TextKind::Text { text }) => {
            InputMessageContent::Text(Text { text })
        }
        ClientInputMessageContent::Text(TextKind::Arguments { arguments }) => {
            InputMessageContent::Template(Template {
                name: role.to_string(),
                arguments,
            })
        }
        ClientInputMessageContent::Template(template) => InputMessageContent::Template(template),
        ClientInputMessageContent::ToolCall(tool_call) => InputMessageContent::ToolCall(tool_call),
        ClientInputMessageContent::ToolResult(tool_result) => {
            InputMessageContent::ToolResult(tool_result)
        }
        ClientInputMessageContent::RawText(raw_text) => InputMessageContent::RawText(raw_text),
        ClientInputMessageContent::Thought(thought) => InputMessageContent::Thought(thought),
        ClientInputMessageContent::File(image) => InputMessageContent::File(image),
        ClientInputMessageContent::Unknown(unknown) => InputMessageContent::Unknown(unknown),
    }
}
