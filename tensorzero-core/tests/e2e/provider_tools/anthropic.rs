//! Anthropic provider tools E2E tests (e.g. web_search, bash)
//!
//! Tests for both static provider tools (configured in model config)
//! and dynamic provider tools (passed via request).

use futures::StreamExt;
use serde_json::json;
use tensorzero::{
    ClientInferenceParams, DynamicToolParams, InferenceOutput, InferenceResponse,
    InferenceResponseChunk, Input, InputMessage, InputMessageContent, Role, Unknown,
    test_helpers::make_embedded_gateway_with_config,
};
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_model_inference_clickhouse,
};
use tensorzero_core::inference::types::{ContentBlockChatOutput, Text};
use tensorzero_core::tool::{ProviderTool, ProviderToolScope, ProviderToolScopeModelProvider};
use uuid::Uuid;

// =============================================================================
// Static Provider Tools Tests (configured in model config)
// =============================================================================

/// Test Anthropic provider tools with web_search (non-streaming, multi-turn)
#[tokio::test(flavor = "multi_thread")]
pub async fn test_anthropic_provider_tools_web_search_nonstreaming() {
    let config = r#"
gateway.debug = true

[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "anthropic"
model_name = "claude-sonnet-4-5"
provider_tools = [{type = "web_search_20250305", name = "web_search", max_uses = 1}]
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();

    // === Turn 1: Ask for current news (triggers web search) ===
    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("test-model".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "What's in the news today? Give me a brief summary.".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    let response = result.unwrap();
    println!("Anthropic web_search turn 1 response: {response:?}");

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    let InferenceResponse::Chat(chat_response) = response else {
        panic!("Expected chat inference response");
    };

    // Assert that we have at least one Unknown content block (server_tool_use or web_search_tool_result)
    let unknown_blocks: Vec<_> = chat_response
        .content
        .iter()
        .filter(|block| matches!(block, ContentBlockChatOutput::Unknown(_)))
        .collect();

    assert!(
        !unknown_blocks.is_empty(),
        "Turn 1: Expected at least one Unknown content block from web_search, but found none. Content blocks: {:#?}",
        chat_response.content
    );

    // Assert that we have at least one Text content block
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

    assert!(
        !text_blocks.is_empty(),
        "Turn 1: Expected at least one Text content block, but found none. Content blocks: {:#?}",
        chat_response.content
    );

    // === Turn 2: Follow-up question about the news ===
    // Convert the assistant's response content to input format for the next turn
    let assistant_content: Vec<InputMessageContent> = chat_response
        .content
        .iter()
        .map(|block| match block {
            ContentBlockChatOutput::Text(text) => InputMessageContent::Text(Text {
                text: text.text.clone(),
            }),
            ContentBlockChatOutput::Unknown(unknown) => {
                InputMessageContent::Unknown(unknown.clone())
            }
            _ => panic!("Unexpected content block type in response"),
        })
        .collect();

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("test-model".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![
                    InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text: "What's in the news today? Give me a brief summary.".to_string(),
                        })],
                    },
                    InputMessage {
                        role: Role::Assistant,
                        content: assistant_content,
                    },
                    InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text:
                                "Thanks! Can you tell me more about the first story you mentioned?"
                                    .to_string(),
                        })],
                    },
                ],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    let response = result.unwrap();
    println!("Anthropic web_search turn 2 response: {response:?}");

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response for turn 2");
    };

    let InferenceResponse::Chat(chat_response) = response else {
        panic!("Expected chat inference response for turn 2");
    };

    // Turn 2 should have a text response (may or may not have web search depending on model)
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

    assert!(
        !text_blocks.is_empty(),
        "Turn 2: Expected at least one Text content block, but found none. Content blocks: {:#?}",
        chat_response.content
    );
}

/// Test Anthropic provider tools with web_search (streaming, multi-turn)
#[tokio::test(flavor = "multi_thread")]
pub async fn test_anthropic_provider_tools_web_search_streaming() {
    let config = r#"
gateway.debug = true

[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "anthropic"
model_name = "claude-sonnet-4-5"
provider_tools = [{type = "web_search_20250305", name = "web_search", max_uses = 1}]
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();

    // === Turn 1: Ask for current news (triggers web search) ===
    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("test-model".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "What's in the news today? Give me a brief summary.".to_string(),
                    })],
                }],
            },
            stream: Some(true),
            ..Default::default()
        })
        .await;

    let response = result.unwrap();
    println!("Anthropic web_search streaming turn 1 response: {response:?}");

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming inference response");
    };

    // Collect the streamed content for turn 2
    let mut collected_text = String::new();
    let mut collected_unknown_blocks: Vec<Unknown> = Vec::new();
    let mut chunk_count = 0;
    let mut has_unknown_chunk = false;
    let mut has_text_chunk = false;
    let mut inference_id: Option<Uuid> = None;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();
        chunk_count += 1;

        if let InferenceResponseChunk::Chat(chat_chunk) = &chunk {
            // Extract inference_id from the first chunk
            if inference_id.is_none() {
                inference_id = Some(chat_chunk.inference_id);
            }

            for content_block in &chat_chunk.content {
                match content_block {
                    tensorzero::ContentBlockChunk::Unknown(unknown_chunk) => {
                        has_unknown_chunk = true;
                        // Filter out citations_delta - it's a streaming-only output type
                        // and not valid for input in multi-turn conversations
                        if unknown_chunk
                            .data
                            .get("type")
                            .and_then(|t| t.as_str())
                            .is_some_and(|t| t == "citations_delta")
                        {
                            continue;
                        }
                        // Convert UnknownChunk to Unknown for input
                        collected_unknown_blocks.push(Unknown {
                            data: unknown_chunk.data.clone(),
                            model_name: unknown_chunk.model_name.clone(),
                            provider_name: unknown_chunk.provider_name.clone(),
                        });
                    }
                    tensorzero::ContentBlockChunk::Text(text) => {
                        has_text_chunk = true;
                        collected_text.push_str(&text.text);
                    }
                    _ => {}
                }
            }
        }
    }

    assert!(
        chunk_count >= 3,
        "Turn 1: Expected at least 3 streaming chunks, but got {chunk_count}"
    );

    assert!(
        has_unknown_chunk,
        "Turn 1: Expected at least one Unknown chunk from web_search streaming"
    );

    assert!(has_text_chunk, "Turn 1: Expected at least one Text chunk");

    // Check ClickHouse persistence for turn 1
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let clickhouse = get_clickhouse().await;
    let inference_id = inference_id.expect("Should have extracted inference_id from chunks");
    let model_inference = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let raw_response = model_inference
        .get("raw_response")
        .unwrap()
        .as_str()
        .unwrap();
    // Assert that raw_response contains server_tool_use (confirms web search was used)
    assert!(
        raw_response.contains("server_tool_use"),
        "Expected raw_response to contain 'server_tool_use', but it was not found"
    );

    // === Turn 2: Follow-up question about the news ===
    // Build assistant content from collected chunks
    let mut assistant_content: Vec<InputMessageContent> = Vec::new();
    for unknown in collected_unknown_blocks {
        assistant_content.push(InputMessageContent::Unknown(unknown));
    }
    if !collected_text.is_empty() {
        assistant_content.push(InputMessageContent::Text(Text {
            text: collected_text,
        }));
    }

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("test-model".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![
                    InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text: "What's in the news today? Give me a brief summary.".to_string(),
                        })],
                    },
                    InputMessage {
                        role: Role::Assistant,
                        content: assistant_content,
                    },
                    InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text:
                                "Thanks! Can you tell me more about the first story you mentioned?"
                                    .to_string(),
                        })],
                    },
                ],
            },
            stream: Some(true),
            ..Default::default()
        })
        .await;

    let response = result.unwrap();
    println!("Anthropic web_search streaming turn 2 response: {response:?}");

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming inference response for turn 2");
    };

    let mut turn2_chunk_count = 0;
    let mut turn2_has_text_chunk = false;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();
        turn2_chunk_count += 1;

        if let tensorzero::InferenceResponseChunk::Chat(chat_chunk) = &chunk {
            for content_block in &chat_chunk.content {
                if matches!(content_block, tensorzero::ContentBlockChunk::Text(_)) {
                    turn2_has_text_chunk = true;
                }
            }
        }
    }

    assert!(
        turn2_chunk_count >= 1,
        "Turn 2: Expected at least 1 streaming chunk, but got {turn2_chunk_count}"
    );

    assert!(
        turn2_has_text_chunk,
        "Turn 2: Expected at least one Text chunk"
    );
}

/// Test Anthropic provider tools with bash tool
#[tokio::test(flavor = "multi_thread")]
pub async fn test_anthropic_provider_tools_bash_nonstreaming() {
    let config = r#"
gateway.debug = true

[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "anthropic"
model_name = "claude-sonnet-4-5"
provider_tools = [{type = "bash_20250124", name = "bash"}]
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
                        text: "List the files in the current directory using the bash tool."
                            .to_string(),
                    })],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    let response = result.unwrap();
    println!("Anthropic bash tool response: {response:?}");

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    let InferenceResponse::Chat(chat_response) = response else {
        panic!("Expected chat inference response");
    };

    // For bash tool, we expect a tool_use block with name "bash"
    let tool_call_blocks: Vec<_> = chat_response
        .content
        .iter()
        .filter_map(|block| {
            if let ContentBlockChatOutput::ToolCall(tool_call) = block {
                Some(tool_call)
            } else {
                None
            }
        })
        .collect();

    assert!(
        !tool_call_blocks.is_empty(),
        "Expected at least one ToolCall content block for bash tool, but found none. Content blocks: {:#?}",
        chat_response.content
    );

    // Verify the tool call is for "bash"
    let bash_tool_call = tool_call_blocks.iter().find(|tc| tc.raw_name == "bash");

    assert!(
        bash_tool_call.is_some(),
        "Expected a tool call with name 'bash', but found: {:?}",
        tool_call_blocks
            .iter()
            .map(|tc| &tc.raw_name)
            .collect::<Vec<_>>()
    );
}

/// Test Anthropic provider tools with bash tool (streaming)
#[tokio::test(flavor = "multi_thread")]
pub async fn test_anthropic_provider_tools_bash_streaming() {
    let config = r#"
gateway.debug = true

[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "anthropic"
model_name = "claude-sonnet-4-5"
provider_tools = [{type = "bash_20250124", name = "bash"}]
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
                        text: "List the files in the current directory using the bash tool."
                            .to_string(),
                    })],
                }],
            },
            stream: Some(true),
            ..Default::default()
        })
        .await;

    let response = result.unwrap();
    println!("Anthropic bash tool streaming response: {response:?}");

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming inference response");
    };

    let mut chunk_count = 0;
    let mut has_tool_call_chunk = false;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();
        chunk_count += 1;

        if let tensorzero::InferenceResponseChunk::Chat(chat_chunk) = &chunk {
            for content_block in &chat_chunk.content {
                if let tensorzero::ContentBlockChunk::ToolCall(tool_call) = content_block
                    && tool_call.raw_name.as_deref() == Some("bash")
                {
                    has_tool_call_chunk = true;
                }
            }
        }
    }

    assert!(
        chunk_count >= 3,
        "Expected at least 3 streaming chunks, but got {chunk_count}"
    );

    assert!(
        has_tool_call_chunk,
        "Expected at least one ToolCall chunk with name 'bash'"
    );
}

// =============================================================================
// Dynamic Provider Tools Tests (passed via request, not config)
// =============================================================================

/// Test Anthropic dynamic provider tools with web_search (non-streaming)
/// This test passes provider_tools via the request instead of the model config
#[tokio::test(flavor = "multi_thread")]
pub async fn test_anthropic_dynamic_provider_tools_web_search_nonstreaming() {
    // Config WITHOUT provider_tools - they will be passed dynamically
    let config = r#"
gateway.debug = true

[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "anthropic"
model_name = "claude-sonnet-4-5"
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
                        text: "What's in the news today? Give me a brief summary.".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            dynamic_tool_params: DynamicToolParams {
                provider_tools: vec![
                    ProviderTool {
                        scope: ProviderToolScope::Unscoped,
                        tool: json!({"type": "web_search_20250305", "name": "web_search", "max_uses": 1}),
                    },
                    // This should get filtered out (scoped to wrong model)
                    ProviderTool {
                        scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                            model_name: "garbage".to_string(),
                            provider_name: Some("garbage".to_string()),
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
    println!("Anthropic dynamic web_search response: {response:?}");

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    let InferenceResponse::Chat(chat_response) = response else {
        panic!("Expected chat inference response");
    };

    // Assert that we have at least one Unknown content block (server_tool_use or web_search_tool_result)
    let unknown_blocks: Vec<_> = chat_response
        .content
        .iter()
        .filter(|block| matches!(block, ContentBlockChatOutput::Unknown(_)))
        .collect();

    assert!(
        !unknown_blocks.is_empty(),
        "Expected at least one Unknown content block from dynamic web_search, but found none. Content blocks: {:#?}",
        chat_response.content
    );

    // Assert that we have at least one Text content block
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

    assert!(
        !text_blocks.is_empty(),
        "Expected at least one Text content block, but found none. Content blocks: {:#?}",
        chat_response.content
    );

    // Round-trip test: Convert output content blocks back to input and make another inference
    let assistant_content: Vec<InputMessageContent> = chat_response
        .content
        .iter()
        .map(|block| match block {
            ContentBlockChatOutput::Text(text) => InputMessageContent::Text(Text {
                text: text.text.clone(),
            }),
            ContentBlockChatOutput::Unknown(unknown) => {
                InputMessageContent::Unknown(unknown.clone())
            }
            _ => panic!("Unexpected content block type in response"),
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
                            text: "What's in the news today? Give me a brief summary.".to_string(),
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
                    tool: json!({"type": "web_search_20250305", "name": "web_search", "max_uses": 1}),
                }],
                ..Default::default()
            },
            ..Default::default()
        })
        .await;

    // Assert the round-trip inference succeeded
    let response2 = result2.unwrap();
    println!("Anthropic dynamic round-trip response: {response2:?}");

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

/// Test Anthropic dynamic provider tools with web_search (streaming)
/// This test passes provider_tools via the request instead of the model config
#[tokio::test(flavor = "multi_thread")]
pub async fn test_anthropic_dynamic_provider_tools_web_search_streaming() {
    // Config WITHOUT provider_tools - they will be passed dynamically
    let config = r#"
gateway.debug = true

[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "anthropic"
model_name = "claude-sonnet-4-5"
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
                        text: "What's in the news today? Give me a brief summary.".to_string(),
                    })],
                }],
            },
            stream: Some(true),
            dynamic_tool_params: DynamicToolParams {
                provider_tools: vec![
                    ProviderTool {
                        scope: ProviderToolScope::Unscoped,
                        tool: json!({"type": "web_search_20250305", "name": "web_search", "max_uses": 1}),
                    },
                    // This should get filtered out (scoped to wrong model)
                    ProviderTool {
                        scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                            model_name: "garbage".to_string(),
                            provider_name: Some("garbage".to_string()),
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
    println!("Anthropic dynamic web_search streaming response: {response:?}");

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming inference response");
    };

    let mut chunk_count = 0;
    let mut has_unknown_chunk = false;
    let mut has_text_chunk = false;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();
        chunk_count += 1;

        if let tensorzero::InferenceResponseChunk::Chat(chat_chunk) = &chunk {
            for content_block in &chat_chunk.content {
                match content_block {
                    tensorzero::ContentBlockChunk::Unknown(_) => {
                        has_unknown_chunk = true;
                    }
                    tensorzero::ContentBlockChunk::Text(_) => {
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

/// Test that provider tools with correct ModelProvider scope are included
/// This test ONLY passes a correctly-scoped tool (no unscoped fallback) to verify scope matching
#[tokio::test(flavor = "multi_thread")]
pub async fn test_anthropic_dynamic_provider_tools_positive_scope_matching() {
    let config = r#"
gateway.debug = true

[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "anthropic"
model_name = "claude-sonnet-4-5"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();

    // Pass ONLY a correctly-scoped tool - no unscoped fallback
    // If scope matching is broken, this test will fail because web_search won't be available
    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("test-model".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "What's in the news today? Give me a brief summary.".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            dynamic_tool_params: DynamicToolParams {
                provider_tools: vec![
                    // Correctly scoped to test-model/test-provider - should be included
                    ProviderTool {
                        scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                            model_name: "test-model".to_string(),
                            provider_name: Some("test-provider".to_string()),
                        }),
                        tool: json!({"type": "web_search_20250305", "name": "web_search", "max_uses": 1}),
                    },
                    // Wrongly scoped - should be filtered out
                    ProviderTool {
                        scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                            model_name: "garbage".to_string(),
                            provider_name: Some("garbage".to_string()),
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
    println!("Anthropic positive scope matching response: {response:?}");

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    let InferenceResponse::Chat(chat_response) = response else {
        panic!("Expected chat inference response");
    };

    // Assert that we have at least one Unknown content block (proves correctly-scoped web_search was included)
    let unknown_blocks: Vec<_> = chat_response
        .content
        .iter()
        .filter(|block| matches!(block, ContentBlockChatOutput::Unknown(_)))
        .collect();

    assert!(
        !unknown_blocks.is_empty(),
        "Expected at least one Unknown content block from correctly-scoped web_search. \
         If this fails, scope matching for ModelProvider may be broken. Content blocks: {:#?}",
        chat_response.content
    );
}
