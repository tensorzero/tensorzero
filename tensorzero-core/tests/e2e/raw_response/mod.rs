//! E2E tests for `include_raw_response` parameter.
//!
//! Tests that raw provider-specific response data is correctly returned when requested.

mod cache;
mod embeddings;
mod errors;
mod openai_compatible;
mod retries;

use futures::StreamExt;
use serde_json::{Map, json};
use tensorzero::test_helpers::make_embedded_gateway_e2e_with_unique_db;
use tensorzero::{
    ClientInferenceParams, InferenceOutput, Input, InputMessage, InputMessageContent,
};
use tensorzero_core::inference::types::usage::{ApiType, RawResponseEntry};
use tensorzero_core::inference::types::{Arguments, Role, System, Template, Text};
use uuid::Uuid;

/// Helper to assert raw_response entry structure is valid
fn assert_raw_response_entry(entry: &RawResponseEntry) {
    assert!(
        entry.model_inference_id.is_some(),
        "raw_response entry should have model_inference_id"
    );
    assert!(
        !entry.provider_type.is_empty(),
        "raw_response entry should have provider_type"
    );
    // api_type is always present as it's not an Option

    // Verify api_type is a valid value
    assert!(
        matches!(
            entry.api_type,
            ApiType::ChatCompletions | ApiType::Responses | ApiType::Embeddings | ApiType::Other
        ),
        "api_type should be a valid ApiType variant, got: {:?}",
        entry.api_type
    );

    // Verify data is a non-empty string (raw response from provider)
    assert!(
        !entry.data.is_empty(),
        "data should be a non-empty string (raw response from provider)"
    );
}

// =============================================================================
// Chat Completions API Tests (api_type = "chat_completions")
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_chat_completions_non_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_chat_completions_non_streaming")
            .await;

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("weather_helper".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("WeatherBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: format!("What is the weather in Tokyo? {random_suffix}"),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };

    // Check raw_response exists at response level
    let raw_response = response
        .raw_response()
        .expect("Response should have raw_response when include_raw_response=true");

    assert!(
        !raw_response.is_empty(),
        "raw_response should have at least one entry"
    );

    // Validate first entry structure
    let first_entry = &raw_response[0];
    assert_raw_response_entry(first_entry);

    // For OpenAI chat completions, api_type should be ChatCompletions
    assert_eq!(
        first_entry.api_type,
        ApiType::ChatCompletions,
        "OpenAI chat completions should have api_type ChatCompletions"
    );

    // Provider type should be "openai"
    assert_eq!(
        first_entry.provider_type, "openai",
        "Provider type should be 'openai'"
    );

    // The data field should be a non-empty string (raw response from provider)
    assert!(!first_entry.data.is_empty(), "data should not be empty");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_chat_completions_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_chat_completions_streaming").await;

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("weather_helper".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("WeatherBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: format!("What is the weather in Paris? {random_suffix}"),
                    })],
                }],
            },
            stream: Some(true),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming response");
    };

    let mut found_raw_chunk = false;
    let mut content_chunks_count: usize = 0;
    let mut chunks_with_raw_chunk: usize = 0;
    let mut all_chunks_count: usize = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        all_chunks_count += 1;

        // Check if this is a content chunk and if it has raw_chunk
        let (has_content, has_raw_chunk) = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => {
                (!c.content.is_empty(), c.raw_chunk.is_some())
            }
            tensorzero::InferenceResponseChunk::Json(j) => {
                (!j.raw.is_empty(), j.raw_chunk.is_some())
            }
        };

        if has_content {
            content_chunks_count += 1;
        }

        if has_raw_chunk {
            found_raw_chunk = true;
            chunks_with_raw_chunk += 1;
        }
    }

    assert!(
        found_raw_chunk,
        "Streaming response should include raw_chunk in at least one chunk.\n\
        Total chunks received: {all_chunks_count}\n\
        Content chunks: {content_chunks_count}"
    );

    // Most content chunks should have raw_chunk (allow first/last to not have it)
    assert!(
        chunks_with_raw_chunk >= content_chunks_count.saturating_sub(2),
        "Most content chunks should have raw_chunk: {chunks_with_raw_chunk} of {content_chunks_count}"
    );
}

// =============================================================================
// Responses API Tests (api_type = "responses")
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_responses_api_non_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_responses_api_non_streaming").await;

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("weather_helper".to_string()),
            variant_name: Some("openai-responses".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("WeatherBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: format!("What is the weather in London? {random_suffix}"),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };

    // Check raw_response exists at response level
    let raw_response = response
        .raw_response()
        .expect("Response should have raw_response when include_raw_response=true");

    assert!(
        !raw_response.is_empty(),
        "raw_response should have at least one entry for responses API"
    );

    // For OpenAI Responses API, api_type should be Responses
    let first_entry = &raw_response[0];
    assert_raw_response_entry(first_entry);

    assert_eq!(
        first_entry.api_type,
        ApiType::Responses,
        "OpenAI Responses API should have api_type Responses"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_responses_api_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_responses_api_streaming").await;

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("weather_helper".to_string()),
            variant_name: Some("openai-responses".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("WeatherBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: format!("What is the weather in Berlin? {random_suffix}"),
                    })],
                }],
            },
            stream: Some(true),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming response");
    };

    let mut found_raw_chunk = false;
    let mut all_chunks_count: usize = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        all_chunks_count += 1;

        // Check if this chunk has raw_chunk
        let has_raw_chunk = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => c.raw_chunk.is_some(),
            tensorzero::InferenceResponseChunk::Json(j) => j.raw_chunk.is_some(),
        };
        if has_raw_chunk {
            found_raw_chunk = true;
        }
    }

    assert!(
        found_raw_chunk,
        "Streaming response should include raw_chunk for Responses API.\n\
        Total chunks received: {all_chunks_count}"
    );
}

// =============================================================================
// Raw response NOT requested - should not be included
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_not_requested_non_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_not_requested_non_streaming").await;

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("weather_helper".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("WeatherBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: format!("What is the weather in Sydney? {random_suffix}"),
                    })],
                }],
            },
            stream: Some(false),
            // include_raw_response is NOT set (defaults to false)
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };

    // raw_response should NOT be present at response level when not requested
    assert!(
        response.raw_response().is_none(),
        "raw_response should not be present when include_raw_response is not set"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_not_requested_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_not_requested_streaming").await;

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("weather_helper".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("WeatherBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: format!("What is the weather in Madrid? {random_suffix}"),
                    })],
                }],
            },
            stream: Some(true),
            // include_raw_response is NOT set
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming response");
    };

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();

        // raw_chunk should NOT be present at chunk level when not requested
        let (has_raw_chunk, has_raw_response) = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => {
                (c.raw_chunk.is_some(), c.raw_response.is_some())
            }
            tensorzero::InferenceResponseChunk::Json(j) => {
                (j.raw_chunk.is_some(), j.raw_response.is_some())
            }
        };

        assert!(
            !has_raw_chunk,
            "raw_chunk should not be present in streaming chunks when not requested"
        );
        assert!(
            !has_raw_response,
            "raw_response should not be present in streaming chunks when not requested"
        );
    }
}

// =============================================================================
// Multi-Inference Variant Tests (Best-of-N)
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_best_of_n_non_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_best_of_n_non_streaming").await;

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("best_of_n".to_string()),
            variant_name: Some("best_of_n_variant_openai".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: format!("Hello, what is your name? {random_suffix}"),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };

    // Check raw_response exists at response level
    let raw_response = response
        .raw_response()
        .expect("Response should have raw_response when include_raw_response=true");

    // Best-of-N should have multiple entries:
    // - 2 candidate inferences (openai_variant0 and openai_variant1)
    // - 1 evaluator/judge inference
    // Total: 3 model inferences
    assert!(
        raw_response.len() >= 3,
        "Best-of-N should have at least 3 raw_response entries (2 candidates + 1 judge), got {}",
        raw_response.len()
    );

    // Validate each entry has required fields
    for entry in raw_response {
        assert_raw_response_entry(entry);
    }

    // All entries should have api_type = ChatCompletions for this variant
    for entry in raw_response {
        assert_eq!(
            entry.api_type,
            ApiType::ChatCompletions,
            "All Best-of-N inferences should have api_type ChatCompletions"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_best_of_n_streaming() {
    let client = make_embedded_gateway_e2e_with_unique_db("raw_response_best_of_n_streaming").await;

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("best_of_n".to_string()),
            variant_name: Some("best_of_n_variant_openai".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: format!("What is your favorite color? {random_suffix}"),
                    })],
                }],
            },
            stream: Some(true),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming response");
    };

    let mut found_raw_response = false;
    let mut found_raw_chunk = false;
    let mut raw_response_entries: Vec<RawResponseEntry> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();

        // Check for raw_chunk (current streaming inference)
        let has_raw_chunk = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => c.raw_chunk.is_some(),
            tensorzero::InferenceResponseChunk::Json(j) => j.raw_chunk.is_some(),
        };
        if has_raw_chunk {
            found_raw_chunk = true;
        }

        // Check for raw_response (previous model inferences - candidates + evaluator)
        let entries = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => c.raw_response.as_ref(),
            tensorzero::InferenceResponseChunk::Json(j) => j.raw_response.as_ref(),
        };
        if let Some(entries) = entries {
            found_raw_response = true;
            // Validate each entry has required fields
            for entry in entries {
                assert_raw_response_entry(entry);
            }
            raw_response_entries.extend(entries.iter().cloned());
        }
    }

    // Best-of-N uses fake streaming (non-streaming candidate converted to stream)
    // so raw_chunk should NOT be present (no actual streaming data)
    assert!(
        !found_raw_chunk,
        "Best-of-N streaming should NOT have raw_chunk (fake streaming has no chunk data)"
    );

    assert!(
        found_raw_response,
        "Streaming Best-of-N response should include raw_response array for all model inferences"
    );

    // Best-of-N should have at least the 2 candidates in raw_response
    assert!(
        raw_response_entries.len() >= 2,
        "Best-of-N streaming should have at least 2 raw_response entries (2 candidates), got {}",
        raw_response_entries.len()
    );
}

// =============================================================================
// Mixture-of-N Tests
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_mixture_of_n_non_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_mixture_of_n_non_streaming").await;

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("mixture_of_n".to_string()),
            variant_name: Some("mixture_of_n_variant".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: format!("Please write a short sentence. {random_suffix}"),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };

    // Check raw_response exists at response level
    let raw_response = response
        .raw_response()
        .expect("Response should have raw_response when include_raw_response=true");

    // Mixture-of-N should have multiple entries:
    // - 2 candidate inferences
    // - 1 fuser inference
    // Total: 3 model inferences
    assert!(
        raw_response.len() >= 3,
        "Mixture-of-N should have at least 3 raw_response entries (2 candidates + 1 fuser), got {}",
        raw_response.len()
    );

    // Validate each entry has required fields
    for entry in raw_response {
        assert_raw_response_entry(entry);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_mixture_of_n_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_mixture_of_n_streaming").await;

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("mixture_of_n".to_string()),
            variant_name: Some("mixture_of_n_variant".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: format!("Tell me a fun fact. {random_suffix}"),
                    })],
                }],
            },
            stream: Some(true),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming response");
    };

    let mut found_raw_response = false;
    let mut found_raw_chunk = false;
    let mut raw_response_entries: Vec<RawResponseEntry> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();

        // Check for raw_chunk (current streaming inference)
        let has_raw_chunk = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => c.raw_chunk.is_some(),
            tensorzero::InferenceResponseChunk::Json(j) => j.raw_chunk.is_some(),
        };
        if has_raw_chunk {
            found_raw_chunk = true;
        }

        // Check for raw_response (previous model inferences - candidates)
        let entries = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => c.raw_response.as_ref(),
            tensorzero::InferenceResponseChunk::Json(j) => j.raw_response.as_ref(),
        };
        if let Some(entries) = entries {
            found_raw_response = true;
            // Validate each entry has required fields
            for entry in entries {
                assert_raw_response_entry(entry);
            }
            raw_response_entries.extend(entries.iter().cloned());
        }
    }

    // Mixture-of-N with a streaming fuser (like gpt-4o-mini) uses real streaming,
    // so raw_chunk SHOULD be present (contains actual fuser streaming data).
    // Note: If the fuser were non-streaming, raw_chunk would NOT be present.
    assert!(
        found_raw_chunk,
        "Mixture-of-N streaming with streaming fuser should have raw_chunk (real streaming data)"
    );

    assert!(
        found_raw_response,
        "Streaming Mixture-of-N response should include raw_response array for candidate inferences"
    );

    // Mixture-of-N should have at least the 2 candidates in raw_response (fuser is streaming)
    assert!(
        raw_response_entries.len() >= 2,
        "Mixture-of-N streaming should have at least 2 raw_response entries (2 candidates), got {}",
        raw_response_entries.len()
    );
}

// =============================================================================
// DICL Tests (api_type includes "embeddings")
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_dicl_non_streaming() {
    let client = make_embedded_gateway_e2e_with_unique_db("raw_response_dicl_non_streaming").await;

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("dicl".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: format!("What is the capital of France? {random_suffix}"),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };

    // Check raw_response exists at response level
    let raw_response = response
        .raw_response()
        .expect("Response should have raw_response when include_raw_response=true");

    // DICL should have exactly 2 entries:
    // - 1 embedding call (api_type = Embeddings)
    // - 1 chat completion call (api_type = ChatCompletions)
    assert_eq!(
        raw_response.len(),
        2,
        "DICL should have exactly 2 raw_response entries (1 embedding + 1 chat), got {}. raw_response:\n{:#?}",
        raw_response.len(),
        raw_response
    );

    // Validate each entry has required fields
    for entry in raw_response {
        assert_raw_response_entry(entry);
    }

    // Check that we have exactly one of each api_type
    let embedding_count = raw_response
        .iter()
        .filter(|e| e.api_type == ApiType::Embeddings)
        .count();
    let chat_completions_count = raw_response
        .iter()
        .filter(|e| e.api_type == ApiType::ChatCompletions)
        .count();

    assert_eq!(
        embedding_count, 1,
        "DICL should have exactly 1 embedding entry, got {embedding_count}"
    );
    assert_eq!(
        chat_completions_count, 1,
        "DICL should have exactly 1 chat_completions entry, got {chat_completions_count}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_dicl_streaming() {
    let client = make_embedded_gateway_e2e_with_unique_db("raw_response_dicl_streaming").await;

    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("dicl".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: format!("What is the capital of Germany? {random_suffix}"),
                    })],
                }],
            },
            stream: Some(true),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming response");
    };

    let mut found_raw_response = false;
    let mut found_raw_chunk = false;
    let mut api_types: Vec<ApiType> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();

        // Check for raw_chunk
        let has_raw_chunk = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => c.raw_chunk.is_some(),
            tensorzero::InferenceResponseChunk::Json(j) => j.raw_chunk.is_some(),
        };
        if has_raw_chunk {
            found_raw_chunk = true;
        }

        // Check for raw_response (previous model inferences - embedding)
        let entries = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => c.raw_response.as_ref(),
            tensorzero::InferenceResponseChunk::Json(j) => j.raw_response.as_ref(),
        };
        if let Some(entries) = entries {
            found_raw_response = true;
            for entry in entries {
                assert_raw_response_entry(entry);
                api_types.push(entry.api_type);
            }
        }
    }

    assert!(
        found_raw_chunk,
        "Streaming DICL response should include raw_chunk for the chat inference"
    );

    assert!(
        found_raw_response,
        "Streaming DICL response should include raw_response for the embedding inference"
    );

    // DICL streaming should have embeddings in raw_response (chat is streamed via raw_chunk)
    assert!(
        api_types.contains(&ApiType::Embeddings),
        "DICL streaming should have an entry with api_type Embeddings in raw_response, got: {api_types:?}"
    );
}

// =============================================================================
// JSON Function Tests
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_json_function_non_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_json_function_non_streaming").await;

    let episode_id = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("json_success".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("JsonBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments({
                            let mut args = Map::new();
                            args.insert("country".to_string(), json!("Japan"));
                            args
                        }),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };

    // Check raw_response exists at response level
    let raw_response = response
        .raw_response()
        .expect("Response should have raw_response when include_raw_response=true");

    assert!(
        !raw_response.is_empty(),
        "raw_response should have at least one entry for JSON function"
    );

    // Validate entry structure
    let first_entry = &raw_response[0];
    assert_raw_response_entry(first_entry);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_json_function_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_json_function_streaming").await;

    let episode_id = Uuid::now_v7();

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("json_success".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(episode_id),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("JsonBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments({
                            let mut args = Map::new();
                            args.insert("country".to_string(), json!("France"));
                            args
                        }),
                    })],
                }],
            },
            stream: Some(true),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming response");
    };

    let mut found_raw_chunk = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();

        // Check for raw_chunk
        let has_raw_chunk = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => c.raw_chunk.is_some(),
            tensorzero::InferenceResponseChunk::Json(j) => j.raw_chunk.is_some(),
        };
        if has_raw_chunk {
            found_raw_chunk = true;
        }
    }

    assert!(
        found_raw_chunk,
        "Streaming JSON function response should include raw_chunk"
    );
}

// =============================================================================
// Failed Candidate Tests (Best-of-N and Mixture-of-N)
// =============================================================================

/// Test that failed candidates in best-of-n have their raw_response captured
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_best_of_n_failed_candidate() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_bon_failed_candidate").await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("best_of_n_with_failing_candidate".to_string()),
            variant_name: Some("best_of_n".to_string()),
            episode_id: Some(Uuid::now_v7()),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Hello".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };

    // raw_response should include entries from the failed candidate
    let raw_response = response
        .raw_response()
        .expect("Should have raw_response when include_raw_response=true");

    // Should have entries from: working candidate + evaluator + failed candidate
    // The failed candidate's raw_response should be captured
    assert!(
        raw_response.len() >= 2,
        "Should have raw_response entries from successful inferences + failed candidate, got {}",
        raw_response.len()
    );

    // At least one entry should be from the failed candidate (contains error data)
    let has_failed_entry = raw_response.iter().any(|e| e.data.contains("test_error"));
    assert!(
        has_failed_entry,
        "Should include raw_response from failed candidate containing error data. Entries: {:?}",
        raw_response.iter().map(|e| &e.data).collect::<Vec<_>>()
    );
}

/// Test that failed candidates in best-of-n streaming have their raw_response captured
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_best_of_n_failed_candidate_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_bon_failed_candidate_streaming")
            .await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("best_of_n_with_failing_candidate".to_string()),
            variant_name: Some("best_of_n".to_string()),
            episode_id: Some(Uuid::now_v7()),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Hello streaming".to_string(),
                    })],
                }],
            },
            stream: Some(true),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming response");
    };

    let mut raw_response_entries: Vec<RawResponseEntry> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();

        // Check for raw_response (previous model inferences - including failed candidates)
        let entries = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => c.raw_response.as_ref(),
            tensorzero::InferenceResponseChunk::Json(j) => j.raw_response.as_ref(),
        };
        if let Some(entries) = entries {
            raw_response_entries.extend(entries.iter().cloned());
        }
    }

    // Should have entries from: working candidate + evaluator + failed candidate
    assert!(
        raw_response_entries.len() >= 2,
        "Should have raw_response entries from successful inferences + failed candidate, got {}",
        raw_response_entries.len()
    );

    // At least one entry should be from the failed candidate (contains error data)
    let has_failed_entry = raw_response_entries
        .iter()
        .any(|e| e.data.contains("test_error"));
    assert!(
        has_failed_entry,
        "Should include raw_response from failed candidate in streaming. Entries: {:?}",
        raw_response_entries
            .iter()
            .map(|e| &e.data)
            .collect::<Vec<_>>()
    );
}

/// Test that failed candidates in mixture-of-n have their raw_response captured
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_mixture_of_n_failed_candidate() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_mon_failed_candidate").await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("mixture_of_n_with_failing_candidate".to_string()),
            variant_name: Some("mixture_of_n".to_string()),
            episode_id: Some(Uuid::now_v7()),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Hello mixture".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };

    // raw_response should include entries from the failed candidate
    let raw_response = response
        .raw_response()
        .expect("Should have raw_response when include_raw_response=true");

    // Should have entries from: working candidate + fuser + failed candidate
    assert!(
        raw_response.len() >= 2,
        "Should have raw_response entries from successful inferences + failed candidate, got {}",
        raw_response.len()
    );

    // At least one entry should be from the failed candidate (contains error data)
    let has_failed_entry = raw_response.iter().any(|e| e.data.contains("test_error"));
    assert!(
        has_failed_entry,
        "Should include raw_response from failed candidate containing error data. Entries: {:?}",
        raw_response.iter().map(|e| &e.data).collect::<Vec<_>>()
    );
}

/// Test that failed candidates in mixture-of-n streaming have their raw_response captured
#[tokio::test(flavor = "multi_thread")]
async fn test_raw_response_mixture_of_n_failed_candidate_streaming() {
    let client =
        make_embedded_gateway_e2e_with_unique_db("raw_response_mon_failed_candidate_streaming")
            .await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("mixture_of_n_with_failing_candidate".to_string()),
            variant_name: Some("mixture_of_n".to_string()),
            episode_id: Some(Uuid::now_v7()),
            input: Input {
                system: Some(System::Template(Arguments({
                    let mut args = Map::new();
                    args.insert("assistant_name".to_string(), json!("TestBot"));
                    args
                }))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Hello streaming mixture".to_string(),
                    })],
                }],
            },
            stream: Some(true),
            include_raw_response: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::Streaming(mut stream) = response else {
        panic!("Expected streaming response");
    };

    let mut raw_response_entries: Vec<RawResponseEntry> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();

        // Check for raw_response (previous model inferences - including failed candidates)
        let entries = match &chunk {
            tensorzero::InferenceResponseChunk::Chat(c) => c.raw_response.as_ref(),
            tensorzero::InferenceResponseChunk::Json(j) => j.raw_response.as_ref(),
        };
        if let Some(entries) = entries {
            raw_response_entries.extend(entries.iter().cloned());
        }
    }

    // Should have entries from: working candidate + fuser + failed candidate
    assert!(
        raw_response_entries.len() >= 2,
        "Should have raw_response entries from successful inferences + failed candidate, got {}",
        raw_response_entries.len()
    );

    // At least one entry should be from the failed candidate (contains error data)
    let has_failed_entry = raw_response_entries
        .iter()
        .any(|e| e.data.contains("test_error"));
    assert!(
        has_failed_entry,
        "Should include raw_response from failed candidate in streaming. Entries: {:?}",
        raw_response_entries
            .iter()
            .map(|e| &e.data)
            .collect::<Vec<_>>()
    );
}
