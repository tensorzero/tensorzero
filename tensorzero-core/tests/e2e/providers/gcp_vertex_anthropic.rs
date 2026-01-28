use std::collections::HashMap;

use futures::StreamExt;
use tensorzero::ClientInferenceParams;
use tensorzero::InferenceOutput;
use tensorzero::InferenceResponse;
use tensorzero::InferenceResponseChunk;
use tensorzero::Input;
use tensorzero::InputMessage;
use tensorzero::InputMessageContent;
use tensorzero::Role;
use tensorzero::Unknown;
use tensorzero::test_helpers::make_embedded_gateway_with_config;
use tensorzero_core::inference::types::{ContentBlockChatOutput, Text};
use uuid::Uuid;

use crate::providers::anthropic::test_redacted_thinking_helper;
use crate::providers::anthropic::test_streaming_thinking_helper;
use crate::providers::anthropic::test_thinking_signature_helper;
use crate::providers::common::{E2ETestProvider, E2ETestProviders};

use super::common::ModelTestProvider;

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-haiku".to_string(),
        model_name: "claude-haiku-4-5-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let image_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-haiku".to_string(),
        model_name: "claude-haiku-4-5-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let pdf_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex_anthropic".to_string(),
        model_name: "gcp_vertex_anthropic::projects/tensorzero-public/locations/global/publishers/anthropic/models/claude-sonnet-4-5@20250929".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-haiku-extra-body".to_string(),
        model_name: "claude-haiku-4-5-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-haiku-extra-headers".to_string(),
        model_name: "claude-haiku-4-5-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-haiku".to_string(),
            model_name: "claude-haiku-4-5-gcp-vertex".into(),
            model_provider_name: "gcp_vertex_anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-haiku-implicit".to_string(),
            model_name: "claude-haiku-4-5-gcp-vertex".into(),
            model_provider_name: "gcp_vertex_anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-haiku-strict".to_string(),
            model_name: "claude-haiku-4-5-gcp-vertex".into(),
            model_provider_name: "gcp_vertex_anthropic".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex_haiku_json_mode_off".to_string(),
        model_name: "claude-haiku-4-5-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex_anthropic_shorthand".to_string(),
        model_name: "gcp_vertex_anthropic::projects/tensorzero-public/locations/us-east5/publishers/anthropic/models/claude-haiku-4-5@20251001".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "gcp_vertex_anthropic".to_string(),
        model_info: HashMap::from([
            (
                "model_id".to_string(),
                "claude-haiku-4-5@20251001".to_string(),
            ),
            ("location".to_string(), "us-east5".to_string()),
            ("project_id".to_string(), "tensorzero-public".to_string()),
        ]),
        use_modal_headers: false,
    }];

    let reasoning_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-anthropic-thinking".to_string(),
        model_name: "gcp-vertex-anthropic-claude-haiku-4-5@20251001-thinking".into(),
        model_provider_name: "gcp_vertex_anthropic_thinking".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        // GCP Vertex Anthropic JSON + reasoning tests use json_mode=off (prompt-based JSON) to avoid prefill conflicts
        // GCP Vertex AI doesn't support Anthropic's structured outputs beta header yet
        reasoning_inference: reasoning_providers.clone(),
        reasoning_usage_inference: reasoning_providers,
        cache_input_tokens_inference: standard_providers.clone(),
        embeddings: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: vec![],
        provider_type_default_credentials: vec![],
        provider_type_default_credentials_shorthand: vec![],
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: image_providers,
        pdf_inference: pdf_providers,
        input_audio: vec![],
        shorthand_inference: shorthand_providers,
        credential_fallbacks,
    }
}

#[tokio::test]
pub async fn test_thinking_signature() {
    test_thinking_signature_helper(
        "gcp-vertex-anthropic-thinking",
        "gcp-vertex-anthropic-claude-haiku-4-5@20251001-thinking",
        "gcp_vertex_anthropic_thinking",
    )
    .await;
}

/// This test checks that streaming inference works as expected.
#[tokio::test]
async fn test_streaming_thinking() {
    test_streaming_thinking_helper(
        "gcp-vertex-anthropic-claude-haiku-4-5@20251001-thinking",
        "gcp_vertex_anthropic",
    )
    .await;
}

#[tokio::test]
async fn test_global_region_non_streaming() {
    let config = r#"
    [models."claude"]
    routing = ["gcp_vertex_anthropic"]

    [models."claude".providers.gcp_vertex_anthropic]
    type = "gcp_vertex_anthropic"
    model_id = "claude-sonnet-4@20250514"
    location = "global"
    project_id = "tensorzero-public"
    "#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();
    let response = client
        .inference(ClientInferenceParams {
            model_name: Some("claude".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Say hello".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    let InferenceResponse::Chat(chat_response) = response else {
        panic!("Expected chat inference response");
    };

    // Just verify we got some content back
    assert!(!chat_response.content.is_empty());
}

#[tokio::test]
async fn test_global_region_streaming() {
    let config = r#"
    [models."claude"]
    routing = ["gcp_vertex_anthropic"]

    [models."claude".providers.gcp_vertex_anthropic]
    type = "gcp_vertex_anthropic"
    model_id = "claude-sonnet-4@20250514"
    location = "global"
    project_id = "tensorzero-public"
    "#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();
    let response = client
        .inference(ClientInferenceParams {
            model_name: Some("claude".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Say hello".to_string(),
                    })],
                }],
            },
            stream: Some(true),
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming inference response");
    };

    // Collect all chunks
    let mut inference_id: Option<Uuid> = None;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();

        // Extract inference_id from the first chunk
        if inference_id.is_none()
            && let InferenceResponseChunk::Chat(chat_chunk) = &chunk
        {
            inference_id = Some(chat_chunk.inference_id);
        }
    }

    // Verify we got an inference_id
    assert!(inference_id.is_some(), "Should have received inference_id");
}

#[tokio::test]
pub async fn test_redacted_thinking() {
    test_redacted_thinking_helper(
        "gcp-vertex-anthropic-claude-haiku-4-5@20251001-thinking",
        "gcp_vertex_anthropic_thinking",
        "gcp_vertex_anthropic",
    )
    .await;
}

// =============================================================================
// Provider Tools Tests
// =============================================================================

const BASH_PROMPT: &str = "List the files in the current directory using the bash tool.";

/// Test GCP Vertex Anthropic provider tools with web_search (non-streaming, multi-turn)
#[tokio::test(flavor = "multi_thread")]
pub async fn test_gcp_vertex_anthropic_provider_tools_web_search_nonstreaming() {
    let config = r#"
gateway.debug = true

[models."test-model"]
routing = ["test-provider"]

[models."test-model".providers.test-provider]
type = "gcp_vertex_anthropic"
model_id = "claude-sonnet-4-5@20250929"
location = "us-east5"
project_id = "tensorzero-public"
provider_tools = [{type = "web_search_20250305", name = "web_search", max_uses = 1}]

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "test-model"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();

    // === Turn 1: Ask for current news (triggers web search) ===
    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
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
    println!("GCP Vertex Anthropic web_search turn 1 response: {response:?}");

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
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
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
    println!("GCP Vertex Anthropic web_search turn 2 response: {response:?}");

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

/// Test GCP Vertex Anthropic provider tools with web_search (streaming, multi-turn)
#[tokio::test(flavor = "multi_thread")]
pub async fn test_gcp_vertex_anthropic_provider_tools_web_search_streaming() {
    let config = r#"
gateway.debug = true

[models."test-model"]
routing = ["test-provider"]

[models."test-model".providers.test-provider]
type = "gcp_vertex_anthropic"
model_id = "claude-sonnet-4-5@20250929"
location = "us-east5"
project_id = "tensorzero-public"
provider_tools = [{type = "web_search_20250305", name = "web_search", max_uses = 1}]

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "test-model"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();

    // === Turn 1: Ask for current news (triggers web search) ===
    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
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
    println!("GCP Vertex Anthropic web_search streaming turn 1 response: {response:?}");

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming inference response");
    };

    // Collect the streamed content for turn 2
    let mut collected_text = String::new();
    let mut collected_unknown_blocks: Vec<Unknown> = Vec::new();
    let mut chunk_count = 0;
    let mut has_unknown_chunk = false;
    let mut has_text_chunk = false;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();
        chunk_count += 1;

        if let tensorzero::InferenceResponseChunk::Chat(chat_chunk) = &chunk {
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
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
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
    println!("GCP Vertex Anthropic web_search streaming turn 2 response: {response:?}");

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

/// Test GCP Vertex Anthropic provider tools with bash tool
#[tokio::test(flavor = "multi_thread")]
pub async fn test_gcp_vertex_anthropic_provider_tools_bash_nonstreaming() {
    let config = r#"
gateway.debug = true

[models."test-model"]
routing = ["test-provider"]

[models."test-model".providers.test-provider]
type = "gcp_vertex_anthropic"
model_id = "claude-sonnet-4-5@20250929"
location = "us-east5"
project_id = "tensorzero-public"
provider_tools = [{type = "bash_20250124", name = "bash"}]

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "test-model"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();
    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: BASH_PROMPT.to_string(),
                    })],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    let response = result.unwrap();
    println!("GCP Vertex Anthropic bash tool response: {response:?}");

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

/// Test GCP Vertex Anthropic provider tools with bash tool (streaming)
#[tokio::test(flavor = "multi_thread")]
pub async fn test_gcp_vertex_anthropic_provider_tools_bash_streaming() {
    let config = r#"
gateway.debug = true

[models."test-model"]
routing = ["test-provider"]

[models."test-model".providers.test-provider]
type = "gcp_vertex_anthropic"
model_id = "claude-sonnet-4-5@20250929"
location = "us-east5"
project_id = "tensorzero-public"
provider_tools = [{type = "bash_20250124", name = "bash"}]

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "test-model"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let episode_id = Uuid::now_v7();
    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: BASH_PROMPT.to_string(),
                    })],
                }],
            },
            stream: Some(true),
            ..Default::default()
        })
        .await;

    let response = result.unwrap();
    println!("GCP Vertex Anthropic bash tool streaming response: {response:?}");

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
