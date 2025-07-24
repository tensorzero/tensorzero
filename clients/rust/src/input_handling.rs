use futures::future::try_join_all;
use serde_json::Value;

use crate::{Client, ClientInput, ClientInputMessage, ClientInputMessageContent, TensorZeroError};
use tensorzero_core::tool::{ToolCall, ToolCallInput};
use tensorzero_core::{
    error::ErrorDetails,
    inference::types::{
        storage::StoragePath, File, ResolvedInput, ResolvedInputMessage,
        ResolvedInputMessageContent, TextKind,
    },
};

/// Convert a resolved input to a client input
pub async fn resolved_input_to_client_input(
    resolved_input: ResolvedInput,
    client: &Client,
) -> Result<ClientInput, TensorZeroError> {
    let ResolvedInput { system, messages } = resolved_input;
    let messages = try_join_all(
        messages
            .into_iter()
            .map(|message| resolved_input_message_to_client_input_message(message, client)),
    )
    .await?;
    Ok(ClientInput { system, messages })
}

async fn resolved_input_message_to_client_input_message(
    resolved_input_message: ResolvedInputMessage,
    client: &Client,
) -> Result<ClientInputMessage, TensorZeroError> {
    let ResolvedInputMessage { role, content } = resolved_input_message;
    let content = try_join_all(content.into_iter().map(|content| {
        resolved_input_message_content_to_client_input_message_content(content, client)
    }))
    .await?;
    Ok(ClientInputMessage { role, content })
}

fn convert_tool_call(tool_call: ToolCall) -> ToolCallInput {
    ToolCallInput {
        id: tool_call.id,
        name: Some(tool_call.name),
        arguments: None,
        raw_arguments: Some(tool_call.arguments),
        raw_name: None,
    }
}

async fn resolved_input_message_content_to_client_input_message_content(
    resolved_input_message_content: ResolvedInputMessageContent,
    client: &Client,
) -> Result<ClientInputMessageContent, TensorZeroError> {
    match resolved_input_message_content {
        ResolvedInputMessageContent::Text { value } => match value {
            Value::String(s) => Ok(ClientInputMessageContent::Text(TextKind::Text { text: s })),
            Value::Object(o) => Ok(ClientInputMessageContent::Text(TextKind::Arguments {
                arguments: o,
            })),
            _ => Err(TensorZeroError::Other {
                source: tensorzero_core::error::Error::new(ErrorDetails::Serialization {
                    message: "Text types must be a string or an object".to_string(),
                })
                .into(),
            }),
        },
        ResolvedInputMessageContent::ToolCall(tool_call) => Ok(
            ClientInputMessageContent::ToolCall(convert_tool_call(tool_call)),
        ),
        ResolvedInputMessageContent::ToolResult(tool_result) => {
            Ok(ClientInputMessageContent::ToolResult(tool_result))
        }
        ResolvedInputMessageContent::RawText { value } => {
            Ok(ClientInputMessageContent::RawText { value })
        }
        ResolvedInputMessageContent::Thought(thought) => {
            Ok(ClientInputMessageContent::Thought(thought))
        }
        ResolvedInputMessageContent::File(file) => {
            let mime_type = file.file.mime_type;
            let data = if let Some(data) = file.file.data {
                data
            } else {
                let storage_path = file.storage_path;
                fetch_file_data(storage_path, client).await?
            };

            Ok(ClientInputMessageContent::File(File::Base64 {
                mime_type,
                data,
            }))
        }
        ResolvedInputMessageContent::Unknown {
            data,
            model_provider_name,
        } => Ok(ClientInputMessageContent::Unknown {
            data,
            model_provider_name,
        }),
    }
}

async fn fetch_file_data(
    storage_path: StoragePath,
    client: &Client,
) -> Result<String, TensorZeroError> {
    let response = client.get_object(storage_path).await?;
    Ok(response.data)
}

#[cfg(test)]
mod tests {
    use object_store::path::Path;

    use tensorzero_core::inference::types::{
        resolved_input::FileWithPath, storage::StorageKind, Base64File,
    };
    use url::Url;

    use crate::{ClientBuilder, ClientBuilderMode};

    use super::*;

    #[test]
    fn test_convert_tool_call() {
        let input_tool_call = ToolCall {
            id: "test_id".to_string(),
            name: "test_tool".to_string(),
            arguments: serde_json::json!({
                "param1": "value1",
                "param2": "value2"
            })
            .to_string(),
        };

        let result = convert_tool_call(input_tool_call);

        assert_eq!(result.id, "test_id");
        assert_eq!(result.name, Some("test_tool".to_string()));
        assert_eq!(result.arguments, None);
        assert_eq!(
            result.raw_arguments,
            Some(r#"{"param1":"value1","param2":"value2"}"#.to_string())
        );
        assert_eq!(result.raw_name, None);
    }

    #[tokio::test]
    async fn test_resolved_input_message_content_to_client_message_content_with_image() {
        // Create a mock client that returns a predefined response for get_object
        let mock_client = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
            url: Url::parse("http://notaurl.com").unwrap(),
        })
        .build()
        .await
        .unwrap();

        // Set up the image data
        let image_data = "base64_encoded_image_data";
        let path = Path::parse("test-image.jpg").unwrap();
        let storage_path = StoragePath {
            path,
            kind: StorageKind::S3Compatible {
                bucket_name: Some("test-bucket".to_string()),
                region: Some("test-region".to_string()),
                endpoint: Some("test-endpoint".to_string()),
                allow_http: Some(true),
                #[cfg(feature = "e2e_tests")]
                prefix: "".to_string(),
            },
        };

        // Create the resolved input message content with an image
        let resolved_content = ResolvedInputMessageContent::File(Box::new(FileWithPath {
            file: Base64File {
                url: Some(Url::parse("http://notaurl.com").unwrap()),
                mime_type: mime::IMAGE_JPEG,
                data: Some(image_data.to_string()),
            },
            storage_path: storage_path.clone(),
        }));

        // Call the function under test
        let result = resolved_input_message_content_to_client_input_message_content(
            resolved_content,
            &mock_client,
        )
        .await
        .unwrap();

        // Verify the result
        match result {
            ClientInputMessageContent::File(File::Base64 {
                mime_type: result_mime_type,
                data: result_data,
            }) => {
                assert_eq!(result_mime_type, mime::IMAGE_JPEG);
                assert_eq!(result_data, image_data);
            }
            _ => panic!("Expected ClientInputMessageContent::Image, got something else"),
        }
    }
}
