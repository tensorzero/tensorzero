use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use serde_untagged::UntaggedEnumVisitor;
use tensorzero_internal::{
    error::Error,
    inference::types::{Image, InputMessageContent, Role, TextKind, Thought},
    tool::{ToolCall, ToolCallInput, ToolResult},
};

// Like the normal `Input` type, but with `ClientInputMessage` instead of `InputMessage`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct ClientInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Value>,
    #[serde(default)]
    pub messages: Vec<ClientInputMessage>,
}

// Like the normal `InputMessage` type, but with `ClientInputMessageContent` instead of `InputMessageContent`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ClientInputMessage {
    pub role: Role,
    #[serde(deserialize_with = "deserialize_content")]
    pub content: Vec<ClientInputMessageContent>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientInputMessageContent {
    Text(TextKind),
    ToolCall(ToolCallInput),
    ToolResult(ToolResult),
    RawText {
        value: String,
    },
    Thought(Thought),
    Image(Image),
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
            ClientInputMessageContent::ToolCall(tool_call) => {
                InputMessageContent::ToolCall(tool_call.try_into()?)
            }
            ClientInputMessageContent::ToolResult(tool_result) => {
                InputMessageContent::ToolResult(tool_result)
            }
            ClientInputMessageContent::RawText { value } => InputMessageContent::RawText { value },
            ClientInputMessageContent::Thought(thought) => InputMessageContent::Thought(thought),
            ClientInputMessageContent::Image(image) => InputMessageContent::Image(image),
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
    UntaggedEnumVisitor::new()
        .string(|text| {
            Ok(vec![ClientInputMessageContent::Text(TextKind::Text {
                text: text.to_string(),
            })])
        })
        .map(|object| {
            tracing::warn!("Deprecation warning - passing in an object for `content` is deprecated. Please use an array of content blocks instead.");
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
) -> tensorzero_internal::inference::types::Input {
    tensorzero_internal::inference::types::Input {
        system: client_input.system,
        messages: client_input
            .messages
            .into_iter()
            .map(|message| {
                let ClientInputMessage { role, content } = message;
                tensorzero_internal::inference::types::InputMessage {
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
        ClientInputMessageContent::ToolCall(ToolCallInput {
            id,
            name,
            raw_name: _,
            arguments,
            raw_arguments: _,
        }) => InputMessageContent::ToolCall(ToolCall {
            id,
            name: name.unwrap_or_default(),
            arguments: arguments.unwrap_or_default().to_string(),
        }),
        ClientInputMessageContent::ToolResult(tool_result) => {
            InputMessageContent::ToolResult(tool_result)
        }
        ClientInputMessageContent::RawText { value } => InputMessageContent::RawText { value },
        ClientInputMessageContent::Thought(thought) => InputMessageContent::Thought(thought),
        ClientInputMessageContent::Image(image) => InputMessageContent::Image(image),
        ClientInputMessageContent::Unknown {
            data,
            model_provider_name,
        } => InputMessageContent::Unknown {
            data,
            model_provider_name,
        },
    }
}
