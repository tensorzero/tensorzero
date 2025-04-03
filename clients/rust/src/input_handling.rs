use futures::future::try_join_all;
use serde_json::Value;

use crate::{Client, TensorZeroError};
use tensorzero_internal::{
    error::ErrorDetails,
    inference::types::{
        storage::StoragePath, Image, Input, InputMessage, InputMessageContent, ResolvedInput,
        ResolvedInputMessage, ResolvedInputMessageContent, TextKind,
    },
};

pub async fn resolved_input_to_input(
    resolved_input: ResolvedInput,
    client: &Client,
) -> Result<Input, TensorZeroError> {
    let ResolvedInput { system, messages } = resolved_input;
    let messages = try_join_all(
        messages
            .into_iter()
            .map(|message| resolved_input_message_to_input_message(message, client)),
    )
    .await?;
    Ok(Input { system, messages })
}

async fn resolved_input_message_to_input_message(
    resolved_input_message: ResolvedInputMessage,
    client: &Client,
) -> Result<InputMessage, TensorZeroError> {
    let ResolvedInputMessage { role, content } = resolved_input_message;
    let content =
        try_join_all(content.into_iter().map(|content| {
            resolved_input_message_content_to_input_message_content(content, client)
        }))
        .await?;
    Ok(InputMessage { role, content })
}

async fn resolved_input_message_content_to_input_message_content(
    resolved_input_message_content: ResolvedInputMessageContent,
    client: &Client,
) -> Result<InputMessageContent, TensorZeroError> {
    match resolved_input_message_content {
        ResolvedInputMessageContent::Text { value } => match value {
            Value::String(s) => Ok(InputMessageContent::Text(TextKind::Text { text: s })),
            Value::Object(o) => Ok(InputMessageContent::Text(TextKind::Arguments {
                arguments: o,
            })),
            _ => Err(TensorZeroError::Other {
                source: tensorzero_internal::error::Error::new(ErrorDetails::Serialization {
                    message: "Text types must be a string or an object".to_string(),
                })
                .into(),
            }),
        },
        ResolvedInputMessageContent::ToolCall(tool_call) => {
            Ok(InputMessageContent::ToolCall(tool_call))
        }
        ResolvedInputMessageContent::ToolResult(tool_result) => {
            Ok(InputMessageContent::ToolResult(tool_result))
        }
        ResolvedInputMessageContent::RawText { value } => {
            Ok(InputMessageContent::RawText { value })
        }
        ResolvedInputMessageContent::Thought(thought) => Ok(InputMessageContent::Thought(thought)),
        ResolvedInputMessageContent::Image(image) => {
            let mime_type = image.image.mime_type;
            let data = match image.image.data {
                Some(data) => data,
                None => {
                    let storage_path = image.storage_path;
                    fetch_image_data(storage_path, client).await?
                }
            };

            Ok(InputMessageContent::Image(Image::Base64 {
                mime_type,
                data,
            }))
        }
        ResolvedInputMessageContent::Unknown {
            data,
            model_provider_name,
        } => Ok(InputMessageContent::Unknown {
            data,
            model_provider_name,
        }),
    }
}

async fn fetch_image_data(
    storage_path: StoragePath,
    client: &Client,
) -> Result<String, TensorZeroError> {
    let response = client.get_object(storage_path).await?;
    Ok(response.data)
}
