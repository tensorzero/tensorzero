use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use serde_untagged::UntaggedEnumVisitor;
use tensorzero_core::{
    error::Error,
    inference::types::{File, InputMessageContent, Role, TemplateInput, TextKind, Thought},
    tool::{ToolCallInput, ToolResult},
};
use tensorzero_derive::TensorZeroDeserialize;

// Like the normal `Input` type, but with `ClientInputMessage` instead of `InputMessage`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
#[derive(ts_rs::TS)]
#[ts(export)]
pub struct ClientInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Value>,
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
    Template(TemplateInput),
    ToolCall(ToolCallInput),
    ToolResult(ToolResult),
    RawText {
        value: String,
    },
    Thought(Thought),
    #[serde(alias = "image")]
    File(File),
    /// An unknown content block type, used to allow passing provider-specific
    /// content blocks (e.g. Anthropic's "redacted_thinking") in and out
    /// of TensorZero.
    /// The 'data' field hold the original content block from the provider,
    /// without any validation or transformation by TensorZero.
    Unknown {
        data: Value,
        model_provider_name: Option<String>,
    },
    // We may extend this in the future to include other types of content
}

impl TryFrom<ClientInputMessageContent> for InputMessageContent {
    type Error = Error;
    fn try_from(this: ClientInputMessageContent) -> Result<Self, Error> {
        Ok(match this {
            ClientInputMessageContent::Text(text) => InputMessageContent::Text(text),
            ClientInputMessageContent::Template(template) => {
                InputMessageContent::Template(template)
            }
            ClientInputMessageContent::ToolCall(tool_call) => {
                InputMessageContent::ToolCall(tool_call)
            }
            ClientInputMessageContent::ToolResult(tool_result) => {
                InputMessageContent::ToolResult(tool_result)
            }
            ClientInputMessageContent::RawText { value } => InputMessageContent::RawText { value },
            ClientInputMessageContent::Thought(thought) => InputMessageContent::Thought(thought),
            ClientInputMessageContent::File(image) => InputMessageContent::File(image),
            ClientInputMessageContent::Unknown {
                data,
                model_provider_name,
            } => InputMessageContent::Unknown {
                data,
                model_provider_name,
            },
        })
    }
}

pub fn deserialize_content<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<ClientInputMessageContent>, D::Error> {
    #[expect(clippy::redundant_closure_for_method_calls)]
    UntaggedEnumVisitor::new()
        .string(|text| {
            Ok(vec![ClientInputMessageContent::Text(TextKind::Text {
                text: text.to_string(),
            })])
        })
        .map(|object| {
            tracing::warn!("Deprecation Warning: passing in an object for `content` is deprecated. Please use an array of content blocks instead.");
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
) -> tensorzero_core::inference::types::Input {
    tensorzero_core::inference::types::Input {
        system: client_input.system,
        messages: client_input
            .messages
            .into_iter()
            .map(|message| {
                let ClientInputMessage { role, content } = message;
                tensorzero_core::inference::types::InputMessage {
                    role,
                    content: content
                        .into_iter()
                        .map(test_client_to_message_content)
                        .collect(),
                }
            })
            .collect(),
    }
}

pub(super) fn test_client_to_message_content(
    content: ClientInputMessageContent,
) -> InputMessageContent {
    match content {
        ClientInputMessageContent::Text(text) => InputMessageContent::Text(text),
        ClientInputMessageContent::Template(template) => InputMessageContent::Template(template),
        ClientInputMessageContent::ToolCall(ToolCallInput {
            id,
            name,
            raw_name,
            arguments,
            raw_arguments,
        }) => InputMessageContent::ToolCall(ToolCallInput {
            id,
            name,
            raw_name,
            raw_arguments,
            arguments,
        }),
        ClientInputMessageContent::ToolResult(tool_result) => {
            InputMessageContent::ToolResult(tool_result)
        }
        ClientInputMessageContent::RawText { value } => InputMessageContent::RawText { value },
        ClientInputMessageContent::Thought(thought) => InputMessageContent::Thought(thought),
        ClientInputMessageContent::File(image) => InputMessageContent::File(image),
        ClientInputMessageContent::Unknown {
            data,
            model_provider_name,
        } => InputMessageContent::Unknown {
            data,
            model_provider_name,
        },
    }
}
