use super::{ClientInput, ClientInputMessage, ClientInputMessageContent};
use crate::error::Error;
use crate::inference::types::{
    Base64File, File, ObjectStorageFile, ResolvedInput, ResolvedInputMessage,
    ResolvedInputMessageContent, TextKind,
};
use crate::tool::{ToolCall, ToolCallWrapper};

/// Convert a resolved input to a client input
pub fn resolved_input_to_client_input(resolved_input: ResolvedInput) -> Result<ClientInput, Error> {
    let ResolvedInput { system, messages } = resolved_input;
    let messages = messages
        .into_iter()
        .map(resolved_input_message_to_client_input_message)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ClientInput { system, messages })
}

fn resolved_input_message_to_client_input_message(
    resolved_input_message: ResolvedInputMessage,
) -> Result<ClientInputMessage, Error> {
    let ResolvedInputMessage { role, content } = resolved_input_message;
    let content = content
        .into_iter()
        .map(resolved_input_message_content_to_client_input_message_content)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ClientInputMessage { role, content })
}

fn convert_tool_call(tool_call: ToolCall) -> ToolCallWrapper {
    ToolCallWrapper::ToolCall(tool_call)
}

fn resolved_input_message_content_to_client_input_message_content(
    resolved_input_message_content: ResolvedInputMessageContent,
) -> Result<ClientInputMessageContent, Error> {
    Ok(match resolved_input_message_content {
        ResolvedInputMessageContent::Text(text) => {
            ClientInputMessageContent::Text(TextKind::Text { text: text.text })
        }
        ResolvedInputMessageContent::Template(template) => {
            ClientInputMessageContent::Template(template)
        }
        ResolvedInputMessageContent::ToolCall(tool_call) => {
            ClientInputMessageContent::ToolCall(convert_tool_call(tool_call))
        }
        ResolvedInputMessageContent::ToolResult(tool_result) => {
            ClientInputMessageContent::ToolResult(tool_result)
        }
        ResolvedInputMessageContent::RawText(raw_text) => {
            ClientInputMessageContent::RawText(raw_text)
        }
        ResolvedInputMessageContent::Thought(thought) => {
            ClientInputMessageContent::Thought(thought)
        }
        ResolvedInputMessageContent::File(resolved) => {
            let ObjectStorageFile { file, data } = *resolved;
            let mime_type = file.mime_type;
            let detail = file.detail;
            let filename = file.filename;

            let base64_file = Base64File::new(None, mime_type, data, detail, filename)?;
            ClientInputMessageContent::File(File::Base64(base64_file))
        }
        ResolvedInputMessageContent::Unknown(unknown) => {
            ClientInputMessageContent::Unknown(unknown)
        }
    })
}

#[cfg(test)]
mod tests {
    use object_store::path::Path;

    use crate::inference::types::{
        storage::{StorageKind, StoragePath},
        ObjectStorageFile, ObjectStoragePointer,
    };
    use url::Url;

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

        let result = convert_tool_call(input_tool_call.clone());

        match result {
            ToolCallWrapper::ToolCall(tc) => {
                assert_eq!(tc.id, "test_id");
                assert_eq!(tc.name, "test_tool");
                assert_eq!(tc.arguments, r#"{"param1":"value1","param2":"value2"}"#);
            }
            ToolCallWrapper::InferenceResponseToolCall(_) => {
                panic!("Expected ToolCallWrapper::ToolCall variant")
            }
        }
    }

    #[tokio::test]
    async fn test_resolved_input_message_content_to_client_message_content_with_image() {
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
                prefix: String::new(),
            },
        };

        // Create the resolved input message content with an image
        let resolved_content = ResolvedInputMessageContent::File(Box::new(ObjectStorageFile {
            file: ObjectStoragePointer {
                source_url: Some(Url::parse("http://notaurl.com").unwrap()),
                mime_type: mime::IMAGE_JPEG,
                storage_path: storage_path.clone(),
                detail: None,
                filename: None,
            },
            data: image_data.to_string(),
        }));

        // Call the function under test
        let result =
            resolved_input_message_content_to_client_input_message_content(resolved_content)
                .expect("Should convert successfully");

        // Verify the result
        match result {
            ClientInputMessageContent::File(File::Base64(base64_file)) => {
                assert_eq!(base64_file.mime_type, mime::IMAGE_JPEG);
                assert_eq!(base64_file.data(), image_data);
            }
            _ => panic!("Expected ClientInputMessageContent::Image, got something else"),
        }
    }
}
