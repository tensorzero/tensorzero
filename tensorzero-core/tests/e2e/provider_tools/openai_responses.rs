//! OpenAI Responses API provider tools E2E tests (e.g. web_search)
//!
//! Tests for both static provider tools (configured in model config)
//! and dynamic provider tools (passed via request).

use futures::StreamExt;
use serde_json::json;
use tensorzero::{
    ClientInferenceParams, DynamicToolParams, InferenceOutput, InferenceResponse,
    InferenceResponseChunk, Input, InputMessage, InputMessageContent, Role, ToolCallWrapper,
    test_helpers::make_embedded_gateway_with_config,
};
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_model_inference_clickhouse,
};
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, ContentBlockChunk, Text, UnknownChunk,
};
use tensorzero_core::tool::{ProviderTool, ProviderToolScope, ProviderToolScopeModelProvider};
use uuid::Uuid;

// =============================================================================
// Static Provider Tools Tests (configured in model config)
// =============================================================================

/// Test OpenAI provider tools with web_search (non-streaming)
#[tokio::test(flavor = "multi_thread")]
pub async fn test_openai_provider_tools_web_search_nonstreaming() {
    let config = r#"
gateway.debug = true

[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "openai"
model_name = "gpt-5-nano"
provider_tools = [{type = "web_search"}]
api_type = "responses"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();
    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("test-model".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Tell me some good news that happened today from around the world. Don't ask me any questions, and provide markdown citations in the form [text](url)".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    let response = result.unwrap();
    println!("OpenAI web_search response: {response:?}");

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    let InferenceResponse::Chat(chat_response) = response else {
        panic!("Expected chat inference response");
    };

    // Assert that we have at least one Unknown content block with type "web_search_call"
    let web_search_blocks: Vec<_> = chat_response
        .content
        .iter()
        .filter(|block| {
            if let ContentBlockChatOutput::Unknown(unknown) = block {
                unknown
                    .data
                    .get("type")
                    .and_then(|t| t.as_str())
                    .map(|t| t == "web_search_call")
                    .unwrap_or(false)
            } else {
                false
            }
        })
        .collect();

    assert!(
        !web_search_blocks.is_empty(),
        "Expected at least one Unknown content block with type 'web_search_call', but found none. Content blocks: {:#?}",
        chat_response.content
    );

    // Assert that we have exactly one Text content block
    let text_blocks: Vec<_> = chat_response
        .content
        .iter()
        .filter_map(|block| {
            if let ContentBlockChatOutput::Text(text) = block {
                Some(text)
            } else {
                None
            }
        })
        .collect();

    assert_eq!(
        text_blocks.len(),
        1,
        "Expected exactly one Text content block, but found {}. Content blocks: {:#?}",
        text_blocks.len(),
        chat_response.content
    );

    // Assert that the text block contains citations (markdown links)
    let text_content = &text_blocks[0].text;
    assert!(
        text_content.contains("]("),
        "Expected text content to contain citations in markdown format [text](url), but found none. Text: {text_content}",
    );

    // Round-trip test: Convert output content blocks back to input and make another inference
    let assistant_content: Vec<InputMessageContent> = chat_response
        .content
        .iter()
        .map(|block| match block {
            ContentBlockChatOutput::Text(text) => InputMessageContent::Text(Text {
                text: text.text.clone(),
            }),
            ContentBlockChatOutput::ToolCall(tool_call) => InputMessageContent::ToolCall(
                ToolCallWrapper::InferenceResponseToolCall(tool_call.clone()),
            ),
            ContentBlockChatOutput::Thought(thought) => {
                InputMessageContent::Thought(thought.clone())
            }
            ContentBlockChatOutput::Unknown(unknown) => {
                InputMessageContent::Unknown(unknown.clone())
            }
        })
        .collect();

    // Make a second inference with the assistant's response and a new user question
    let result2 = client
        .inference(ClientInferenceParams {
            model_name: Some("test-model".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![
                    InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text: "Tell me some good news that happened today from around the world. Don't ask me any questions, and provide markdown citations in the form [text](url)".to_string(),
                        })],
                    },
                    InputMessage {
                        role: Role::Assistant,
                        content: assistant_content,
                    },
                    InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text: "Can you summarize what you just told me in one sentence?"
                                .to_string(),
                        })],
                    },
                ],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    // Assert the round-trip inference succeeded
    let response2 = result2.unwrap();
    println!("Round-trip response: {response2:?}");

    let InferenceOutput::NonStreaming(response2) = response2 else {
        panic!("Expected non-streaming inference response for round-trip");
    };

    let InferenceResponse::Chat(chat_response2) = response2 else {
        panic!("Expected chat inference response for round-trip");
    };

    // Assert that the second response has at least one text block
    let has_text = chat_response2
        .content
        .iter()
        .any(|block| matches!(block, ContentBlockChatOutput::Text(_)));
    assert!(
        has_text,
        "Expected at least one text content block in round-trip response. Content blocks: {:#?}",
        chat_response2.content
    );
}

/// Test OpenAI provider tools with web_search (streaming)
#[tokio::test(flavor = "multi_thread")]
pub async fn test_openai_provider_tools_web_search_streaming() {
    let config = r#"
gateway.debug = true

[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "openai"
model_name = "gpt-5-nano"
provider_tools = [{type = "web_search"}]
api_type = "responses"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();
    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("test-model".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Tell me some good news that happened today from around the world. Don't ask me any questions, and provide markdown citations in the form [text](url)".to_string(),
                    })],
                }],
            },
            stream: Some(true),
            ..Default::default()
        })
        .await;

    let response = result.unwrap();
    println!("OpenAI web_search streaming response: {response:?}");

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming inference response");
    };

    // Collect all chunks
    let mut chunks = vec![];
    let mut inference_id: Option<Uuid> = None;
    let mut full_text = String::new();
    let mut unknown_chunks = vec![];

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();

        // Extract inference_id from the first chunk
        if inference_id.is_none()
            && let InferenceResponseChunk::Chat(chat_chunk) = &chunk
        {
            inference_id = Some(chat_chunk.inference_id);
        }

        // Collect text and unknown chunks
        if let InferenceResponseChunk::Chat(chat_chunk) = &chunk {
            for content_block in &chat_chunk.content {
                match content_block {
                    ContentBlockChunk::Text(text_chunk) => {
                        full_text.push_str(&text_chunk.text);
                    }
                    ContentBlockChunk::Unknown(UnknownChunk { id, data, .. }) => {
                        unknown_chunks.push((id.clone(), data.clone()));
                    }
                    _ => {}
                }
            }
        }

        chunks.push(chunk);
    }

    // Assert that we have multiple streaming chunks (indicates streaming is working)
    assert!(
        chunks.len() >= 3,
        "Expected at least 3 streaming chunks, but got {}. Streaming may not be working properly.",
        chunks.len()
    );

    // Assert that all chunks are Chat type
    for chunk in &chunks {
        assert!(
            matches!(chunk, InferenceResponseChunk::Chat(_)),
            "Expected all chunks to be Chat type, but found: {chunk:?}",
        );
    }

    // Assert that the last chunk has usage information
    if let Some(InferenceResponseChunk::Chat(last_chunk)) = chunks.last() {
        assert!(
            last_chunk.usage.is_some(),
            "Expected the last chunk to have usage information, but it was None"
        );
    } else {
        panic!("No chunks received");
    }

    // Assert that the last chunk has a finish_reason
    if let Some(InferenceResponseChunk::Chat(last_chunk)) = chunks.last() {
        assert!(
            last_chunk.finish_reason.is_some(),
            "Expected the last chunk to have a finish_reason, but it was None"
        );
    }

    // Assert that we received Unknown chunks for web_search_call
    assert!(
        !unknown_chunks.is_empty(),
        "Expected at least one Unknown chunk during streaming, but found none"
    );

    // Verify that at least one Unknown chunk contains web_search_call type
    let has_web_search_chunk = unknown_chunks.iter().any(|(_, data)| {
        data.get("type")
            .and_then(|t| t.as_str())
            .map(|t| t == "web_search_call")
            .unwrap_or(false)
    });
    assert!(
        has_web_search_chunk,
        "Expected at least one Unknown chunk with type 'web_search_call', but found none. Unknown chunks: {unknown_chunks:#?}",
    );

    // Assert that the concatenated text contains citations (markdown links)
    assert!(
        full_text.contains("]("),
        "Expected concatenated text to contain citations in markdown format [text](url), but found none. Text length: {}",
        full_text.len()
    );

    // Sleep for 1 second to allow writing to ClickHouse
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let clickhouse = get_clickhouse().await;

    let inference_id = inference_id.expect("Should have extracted inference_id from chunks");

    // Fetch the model inference data from ClickHouse
    let model_inference = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let raw_response = model_inference
        .get("raw_response")
        .unwrap()
        .as_str()
        .unwrap();

    // Assert that raw_response contains web_search_call (confirms web search was used)
    assert!(
        raw_response.contains("web_search_call"),
        "Expected raw_response to contain 'web_search_call', but it was not found"
    );

    // Assert that raw_response contains response.completed event
    assert!(
        raw_response.contains("response.completed"),
        "Expected raw_response to contain 'response.completed' event, but it was not found"
    );
}

// =============================================================================
// Dynamic Provider Tools Tests (passed via request, not config)
// =============================================================================

/// Test OpenAI dynamic provider tools with web_search (non-streaming)
/// This test passes provider_tools via the request instead of the model config
#[tokio::test(flavor = "multi_thread")]
pub async fn test_openai_dynamic_provider_tools_web_search_nonstreaming() {
    // Config WITHOUT provider_tools - they will be passed dynamically
    let config = r#"
gateway.debug = true

[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "openai"
model_name = "gpt-5-nano"
api_type = "responses"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();
    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("test-model".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Tell me some good news that happened today from around the world. Don't ask me any questions, and provide markdown citations in the form [text](url)".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            dynamic_tool_params: DynamicToolParams {
                provider_tools: vec![
                    ProviderTool {
                        scope: ProviderToolScope::Unscoped,
                        tool: json!({"type": "web_search"}),
                    },
                    // This should get filtered out (scoped to wrong model)
                    ProviderTool {
                        scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                            model_name: "garbage".to_string(),
                            provider_name: Some("model".to_string()),
                        }),
                        tool: json!({"type": "garbage"}),
                    },
                ],
                ..Default::default()
            },
            ..Default::default()
        })
        .await;

    let response = result.unwrap();
    println!("OpenAI dynamic web_search response: {response:?}");

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    let InferenceResponse::Chat(chat_response) = response else {
        panic!("Expected chat inference response");
    };

    // Assert that we have at least one Unknown content block with type "web_search_call"
    let web_search_blocks: Vec<_> = chat_response
        .content
        .iter()
        .filter(|block| {
            if let ContentBlockChatOutput::Unknown(unknown) = block {
                unknown
                    .data
                    .get("type")
                    .and_then(|t| t.as_str())
                    .map(|t| t == "web_search_call")
                    .unwrap_or(false)
            } else {
                false
            }
        })
        .collect();

    assert!(
        !web_search_blocks.is_empty(),
        "Expected at least one Unknown content block with type 'web_search_call', but found none. Content blocks: {:#?}",
        chat_response.content
    );

    // Assert that we have exactly one Text content block
    let text_blocks: Vec<_> = chat_response
        .content
        .iter()
        .filter_map(|block| {
            if let ContentBlockChatOutput::Text(text) = block {
                Some(text)
            } else {
                None
            }
        })
        .collect();

    assert_eq!(
        text_blocks.len(),
        1,
        "Expected exactly one Text content block, but found {}. Content blocks: {:#?}",
        text_blocks.len(),
        chat_response.content
    );

    // Assert that the text block contains citations (markdown links)
    let text_content = &text_blocks[0].text;
    assert!(
        text_content.contains("]("),
        "Expected text content to contain citations in markdown format [text](url), but found none. Text: {text_content}",
    );

    // Round-trip test: Convert output content blocks back to input and make another inference
    let assistant_content: Vec<InputMessageContent> = chat_response
        .content
        .iter()
        .map(|block| match block {
            ContentBlockChatOutput::Text(text) => InputMessageContent::Text(Text {
                text: text.text.clone(),
            }),
            ContentBlockChatOutput::ToolCall(tool_call) => InputMessageContent::ToolCall(
                ToolCallWrapper::InferenceResponseToolCall(tool_call.clone()),
            ),
            ContentBlockChatOutput::Thought(thought) => {
                InputMessageContent::Thought(thought.clone())
            }
            ContentBlockChatOutput::Unknown(unknown) => {
                InputMessageContent::Unknown(unknown.clone())
            }
        })
        .collect();

    // Make a second inference with the assistant's response and a new user question
    let result2 = client
        .inference(ClientInferenceParams {
            model_name: Some("test-model".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![
                    InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text: "Tell me some good news that happened today from around the world. Don't ask me any questions, and provide markdown citations in the form [text](url)".to_string(),
                        })],
                    },
                    InputMessage {
                        role: Role::Assistant,
                        content: assistant_content,
                    },
                    InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text: "Can you summarize what you just told me in one sentence?"
                                .to_string(),
                        })],
                    },
                ],
            },
            stream: Some(false),
            dynamic_tool_params: DynamicToolParams {
                provider_tools: vec![ProviderTool {
                    scope: ProviderToolScope::Unscoped,
                    tool: json!({"type": "web_search"}),
                }],
                ..Default::default()
            },
            ..Default::default()
        })
        .await;

    // Assert the round-trip inference succeeded
    let response2 = result2.unwrap();
    println!("Round-trip response: {response2:?}");

    let InferenceOutput::NonStreaming(response2) = response2 else {
        panic!("Expected non-streaming inference response for round-trip");
    };

    let InferenceResponse::Chat(chat_response2) = response2 else {
        panic!("Expected chat inference response for round-trip");
    };

    // Assert that the second response has at least one text block
    let has_text = chat_response2
        .content
        .iter()
        .any(|block| matches!(block, ContentBlockChatOutput::Text(_)));
    assert!(
        has_text,
        "Expected at least one text content block in round-trip response. Content blocks: {:#?}",
        chat_response2.content
    );
}

/// Test OpenAI dynamic provider tools with web_search (streaming)
/// This test passes provider_tools via the request instead of the model config
#[tokio::test(flavor = "multi_thread")]
pub async fn test_openai_dynamic_provider_tools_web_search_streaming() {
    // Config WITHOUT provider_tools - they will be passed dynamically
    let config = r#"
gateway.debug = true

[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "openai"
model_name = "gpt-5-nano"
api_type = "responses"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();

    // Pass provider_tools dynamically via the request
    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("test-model".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Tell me some good news that happened today from around the world. Don't ask me any questions, and provide markdown citations in the form [text](url)".to_string(),
                    })],
                }],
            },
            stream: Some(true),
            dynamic_tool_params: DynamicToolParams {
                provider_tools: vec![ProviderTool {
                    scope: ProviderToolScope::Unscoped,
                    tool: json!({"type": "web_search"}),
                }],
                ..Default::default()
            },
            ..Default::default()
        })
        .await;

    let response = result.unwrap();
    println!("OpenAI dynamic web_search streaming response: {response:?}");

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming inference response");
    };

    let mut chunk_count = 0;
    let mut has_unknown_chunk = false;
    let mut has_text_chunk = false;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();
        chunk_count += 1;

        if let InferenceResponseChunk::Chat(chat_chunk) = &chunk {
            for content_block in &chat_chunk.content {
                match content_block {
                    ContentBlockChunk::Unknown(_) => {
                        has_unknown_chunk = true;
                    }
                    ContentBlockChunk::Text(_) => {
                        has_text_chunk = true;
                    }
                    _ => {}
                }
            }
        }
    }

    assert!(
        chunk_count >= 3,
        "Expected at least 3 streaming chunks, but got {chunk_count}"
    );

    assert!(
        has_unknown_chunk,
        "Expected at least one Unknown chunk from dynamic web_search streaming"
    );

    assert!(has_text_chunk, "Expected at least one Text chunk");
}
