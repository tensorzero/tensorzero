use futures::future::try_join_all;
use serde_json::Value;
use tensorzero_internal::error::Error;

use crate::{Client, ClientInput, ClientInputMessage, ClientInputMessageContent, TensorZeroError};
use tensorzero_internal::tool::{ToolCall, ToolCallInput};
use tensorzero_internal::{
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
                source: tensorzero_internal::error::Error::new(ErrorDetails::Serialization {
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
            let data = match file.file.data {
                Some(data) => data,
                None => {
                    let storage_path = file.storage_path;
                    fetch_file_data(storage_path, client).await?
                }
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

/// Since we store the input in the database in the form of ResolvedInput but without e.g. images inside,
/// we need to reresolve the input when we retrieve it from the database.
/// Resolves images in place.
pub async fn reresolve_input_for_fine_tuning(
    input: &mut ResolvedInput,
    client: &Client,
) -> Result<(), TensorZeroError> {
    let mut file_fetch_tasks = Vec::new();

    for (message_index, message) in input.messages.iter_mut().enumerate() {
        // First pass: identify files to fetch and collect tasks
        for (content_index, content) in message.content.iter_mut().enumerate() {
            if let ResolvedInputMessageContent::File(file_with_path) = content {
                if file_with_path.file.data.is_none() {
                    let storage_path = file_with_path.storage_path.clone();
                    let fut = async move {
                        let result = fetch_file_data(storage_path, client).await?;
                        Ok((message_index, content_index, result))
                    };
                    file_fetch_tasks.push(fut);
                }
            }
        }
    }

    // Execute fetch tasks concurrently for the current message
    if !file_fetch_tasks.is_empty() {
        let fetched_data_results = try_join_all(file_fetch_tasks).await?;

        // Second pass: update the content with fetched data
        for (message_index, content_index, fetched_data) in fetched_data_results {
            if let Some(message) = input.messages.get_mut(message_index) {
                if let Some(ResolvedInputMessageContent::File(file_with_path)) =
                    message.content.get_mut(content_index)
                {
                    file_with_path.file.data = Some(fetched_data);
                } else {
                    return Err(TensorZeroError::Other {
                        source: Error::new(ErrorDetails::Serialization {
                            message:
                                "Content type changed or index invalid during input reresolution"
                                    .to_string(),
                        })
                        .into(),
                    });
                }
            } else {
                return Err(TensorZeroError::Other {
                    source: Error::new(ErrorDetails::Serialization {
                        message: "Message index invalid during input reresolution".to_string(),
                    })
                    .into(),
                });
            }
        }
    }

    Ok(())
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

    use tensorzero_internal::inference::types::{
        resolved_input::FileWithPath, storage::StorageKind, Base64File, FileKind,
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
        let resolved_content = ResolvedInputMessageContent::File(FileWithPath {
            file: Base64File {
                url: Some(Url::parse("http://notaurl.com").unwrap()),
                mime_type: FileKind::Jpeg,
                data: Some(image_data.to_string()),
            },
            storage_path: storage_path.clone(),
        });

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
                assert_eq!(result_mime_type, FileKind::Jpeg);
                assert_eq!(result_data, image_data);
            }
            _ => panic!("Expected ClientInputMessageContent::Image, got something else"),
        }
    }
}
