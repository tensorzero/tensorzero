#![expect(clippy::print_stdout)]

use futures::StreamExt;
use googletest::prelude::*;
use rand::RngExt;
use reqwest::Client;
use reqwest_sse_stream::Event;
use reqwest_sse_stream::RequestBuilderExt;
use serde_json::Value;
use serde_json::json;
use std::time::Duration;
use tensorzero::CacheParamsOptions;
use tensorzero::ClientExt;
use tensorzero::ClientInferenceParams;
use tensorzero::DynamicToolParams;
use tensorzero::FunctionTool;
use tensorzero::InferenceOutput;
use tensorzero::InferenceResponse;
use tensorzero::Input;
use tensorzero::InputMessage;
use tensorzero::InputMessageContent;
use tensorzero::Tool;
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::inference::types::ContentBlockChatOutput;
use tensorzero_core::inference::types::Text;
use tensorzero_core::tool::InferenceResponseToolCall;
use uuid::Uuid;

use tensorzero_core::inference::types::Role;
use tensorzero_core::inference::types::{StoredContentBlock, StoredRequestMessage};

use crate::common::get_gateway_endpoint;
use crate::utils::skip_for_postgres;
use tensorzero::test_helpers::{
    make_embedded_gateway_e2e_with_unique_db,
    make_embedded_gateway_e2e_with_unique_db_all_backends,
    make_http_gateway_openai_only_with_unique_db,
};
use tensorzero_core::db::clickhouse::test_helpers::select_model_inference_clickhouse;
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::inferences::{InferenceQueries, ListInferencesParams};
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::stored_inference::StoredInferenceDatabase;
use tensorzero_core::test_helpers::get_e2e_config;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CacheBackend {
    Clickhouse,
    Valkey,
}

/// Generates test variants for each cache backend (ClickHouse and Valkey).
macro_rules! make_cache_tests {
    ($test_name:ident) => {
        paste::paste! {
            #[gtest]
            #[tokio::test(flavor = "multi_thread")]
            async fn [<$test_name _clickhouse>]() {
                $test_name(CacheBackend::Clickhouse).await;
            }

            #[gtest]
            #[tokio::test(flavor = "multi_thread")]
            async fn [<$test_name _valkey>]() {
                $test_name(CacheBackend::Valkey).await;
            }
        }
    };
}

/// Creates a gateway configured to use the specified cache backend.
///
/// - `CacheBackend::Clickhouse`: default mode (ClickHouse as primary datastore)
/// - `CacheBackend::Valkey`: uses Postgres as primary datastore + Valkey connections
async fn make_cache_test_gateway(backend: CacheBackend, db_prefix: &str) -> tensorzero::Client {
    match backend {
        CacheBackend::Clickhouse => make_embedded_gateway_e2e_with_unique_db(db_prefix).await,
        CacheBackend::Valkey => {
            make_embedded_gateway_e2e_with_unique_db_all_backends(db_prefix).await
        }
    }
}

fn is_batched_writes_enabled() -> bool {
    match std::env::var("TENSORZERO_CLICKHOUSE_BATCH_WRITES") {
        Ok(value) => value == "true",
        Err(_) => false,
    }
}

make_cache_tests!(test_dont_cache_invalid_tool_call);

async fn test_dont_cache_invalid_tool_call(backend: CacheBackend) {
    skip_for_postgres!();
    let logs_contain = tensorzero_core::utils::testing::capture_logs();
    if is_batched_writes_enabled() {
        // Skip test if batched writes are enabled.
        // The message is logged from the batch writer tokio task, which may run on
        // a different thread when the multi-threaded tokio runtime is used (and fail to be captured).
        // We cannot use the single-threaded tokio runtime here, since we need to call `block_in_place`
        // from `GatewayHandle`.
        return;
    }
    let client = make_cache_test_gateway(backend, "dont_cache_invalid_tool_call").await;
    let randomness = Uuid::now_v7();
    let params = ClientInferenceParams {
        model_name: Some("dummy::invalid_tool_arguments".to_string()),
        input: Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: format!("Test inference: {randomness}"),
                })],
            }],
        },
        cache_options: CacheParamsOptions {
            enabled: CacheEnabledMode::On,
            max_age_s: None,
        },
        ..Default::default()
    };
    client.inference(params.clone()).await.unwrap();

    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    assert!(
        logs_contain("Skipping cache write"),
        "Expected log message about skipping cache write"
    );

    // Run again, and check that we get a cache miss
    let res = client.inference(params).await.unwrap();
    let InferenceOutput::NonStreaming(res) = res else {
        panic!("Expected non-streaming inference response");
    };

    // ClickHouse-specific: verify the `cached` column in model_inference
    if backend == CacheBackend::Clickhouse {
        let clickhouse = client
            .get_app_state_data()
            .unwrap()
            .clickhouse_connection_info();
        let model_inference = select_model_inference_clickhouse(&clickhouse, res.inference_id())
            .await
            .unwrap();
        assert_eq!(
            model_inference.get("cached").unwrap(),
            false,
            "Second inference should not be cached"
        );
    }
}

make_cache_tests!(test_dont_cache_tool_call_schema_error);

async fn test_dont_cache_tool_call_schema_error(backend: CacheBackend) {
    skip_for_postgres!();
    let logs_contain = tensorzero_core::utils::testing::capture_logs();
    if is_batched_writes_enabled() {
        // Skip test if batched writes are enabled (same reason as above)
        return;
    }
    let client = make_cache_test_gateway(backend, "dont_cache_tool_call_schema_error").await;
    let randomness = Uuid::now_v7();
    let params = ClientInferenceParams {
        model_name: Some("dummy::tool".to_string()),
        input: Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: format!("Test inference: {randomness}"),
                })],
            }],
        },
        cache_options: CacheParamsOptions {
            enabled: CacheEnabledMode::On,
            max_age_s: None,
        },
        dynamic_tool_params: DynamicToolParams {
            additional_tools: Some(vec![Tool::Function(FunctionTool {
                name: "get_temperature".to_string(),
                description: "Get the temperature".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "other_param": {"type": "string"},
                    },
                    "required": ["other_param"]
                }),
                strict: true,
            })]),
            ..Default::default()
        },
        ..Default::default()
    };
    let res = client.inference(params.clone()).await.unwrap();
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(res)) = res else {
        panic!("Expected non-streaming chat  inference response");
    };
    assert_eq!(res.content.len(), 1);
    assert_eq!(
        res.content[0],
        ContentBlockChatOutput::ToolCall(InferenceResponseToolCall {
            name: Some("get_temperature".to_string()),
            raw_name: "get_temperature".to_string(),
            arguments: None,
            raw_arguments: "{\"location\":\"Brooklyn\",\"units\":\"celsius\"}".to_string(),
            id: "0".to_string(),
        })
    );

    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    assert!(
        logs_contain("Skipping cache write"),
        "Expected log message about skipping cache write"
    );

    // Run again, and check that we get a cache miss
    let res = client.inference(params).await.unwrap();
    let InferenceOutput::NonStreaming(res) = res else {
        panic!("Expected non-streaming inference response");
    };

    // ClickHouse-specific: verify the `cached` column in model_inference
    if backend == CacheBackend::Clickhouse {
        let clickhouse = client
            .get_app_state_data()
            .unwrap()
            .clickhouse_connection_info();
        let model_inference = select_model_inference_clickhouse(&clickhouse, res.inference_id())
            .await
            .unwrap();
        assert_eq!(
            model_inference.get("cached").unwrap(),
            false,
            "Second inference should not be cached"
        );
    }
}

#[gtest]
#[tokio::test]
pub async fn test_streaming_cache_with_err() {
    skip_for_postgres!();
    let episode_id = Uuid::now_v7();
    // Generate random u32
    let seed = rand::rng().random_range(0..u32::MAX);

    // When the stream includes an error, we should not cache the response (we pass `expect_cached = false`
    // for both calls)
    let original_content = check_test_streaming_cache_with_err(episode_id, seed, true, false).await;
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let cached_content = check_test_streaming_cache_with_err(episode_id, seed, true, false).await;
    expect_that!(original_content, eq(&cached_content));
}

#[gtest]
#[tokio::test]
pub async fn test_streaming_cache_without_err() {
    skip_for_postgres!();
    let episode_id = Uuid::now_v7();
    // Generate random u32
    let seed = rand::rng().random_range(0..u32::MAX);

    // When the stream does not include an error, we should cache the response (we pass `expect_cached = true`
    // for the second call)
    let original_content =
        check_test_streaming_cache_with_err(episode_id, seed, false, false).await;
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    let cached_content = check_test_streaming_cache_with_err(episode_id, seed, false, true).await;
    expect_that!(original_content, eq(&cached_content));
}

pub async fn check_test_streaming_cache_with_err(
    episode_id: Uuid,
    seed: u32,
    inject_err: bool,
    expect_cached: bool,
) -> String {
    let input_variant_name = if inject_err { "err_in_stream" } else { "test" };
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": input_variant_name,
        "episode_id": episode_id,
        "params": {
            "chat_completion": {
                "seed": seed,
            }
        },
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "My test input string"
                }
            ]},
        "stream": true,
        "cache_options": {"enabled": "on", "lookback_s": 10}
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                if serde_json::from_str::<Value>(&message.data)
                    .unwrap()
                    .get("error")
                    .is_some()
                {
                    continue;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id: Option<Uuid> = None;
    let mut full_content = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for chunk in chunks.clone() {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            Some(inference_id) => {
                assert_eq!(inference_id, chunk_inference_id);
            }
            None => {
                inference_id = Some(chunk_inference_id);
            }
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        let content_blocks = chunk_json.get("content").unwrap().as_array().unwrap();
        if !content_blocks.is_empty() {
            let content_block = content_blocks.first().unwrap();
            let content = content_block.get("text").unwrap().as_str().unwrap();
            full_content.push_str(content);
        }

        if let Some(usage) = chunk_json.get("usage") {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    let inference_id = inference_id.unwrap();
    if inject_err {
        // We skip over errors in the loop above, so this will be missing the token "retriever,"
        assert_eq!(
            full_content,
            "Wally, the golden wagged his tail excitedly as he devoured a slice of cheese pizza."
        );
    } else {
        assert_eq!(
            full_content,
            "Wally, the golden retriever, wagged his tail excitedly as he devoured a slice of cheese pizza."
        );
    }

    if expect_cached {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert_eq!(input_tokens, 10);
        assert_eq!(output_tokens, 16);
    }

    // Sleep to allow time for data to be inserted into the database (trailing writes from API)
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let config = get_e2e_config().await;

    // Check ChatInference
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(inferences.len(), 1, "Expected exactly one inference");

    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };

    println!("ChatInference: {chat:#?}");

    assert_eq!(chat.inference_id, inference_id);
    assert_eq!(chat.function_name, payload["function_name"]);
    assert_eq!(chat.variant_name, input_variant_name);
    assert_eq!(chat.episode_id, episode_id);

    let input_value = serde_json::to_value(&chat.input).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "My test input string"}]
            }
        ]
    });
    assert_eq!(input_value, correct_input);

    let output = chat.output.as_ref().expect("Expected output");
    assert_eq!(output.len(), 1);
    let ContentBlockChatOutput::Text(text_block) = &output[0] else {
        panic!("Expected text content block");
    };
    assert_eq!(text_block.text, full_content);

    assert!(chat.tool_params.is_none(), "Expected no tool params");

    let inference_params_value = serde_json::to_value(&chat.inference_params).unwrap();
    let chat_completion = inference_params_value
        .get("chat_completion")
        .expect("Expected chat_completion in inference_params");
    assert_eq!(
        chat_completion.get("seed").unwrap().as_u64().unwrap(),
        seed as u64
    );
    assert_eq!(
        chat_completion.get("max_tokens").unwrap().as_u64().unwrap(),
        100
    );

    let processing_time_ms = chat
        .processing_time_ms
        .expect("Expected processing_time_ms");
    assert!(processing_time_ms > 0);

    // Check ModelInference
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(
        model_inferences.len(),
        1,
        "Expected exactly one model inference"
    );

    let mi = &model_inferences[0];

    println!("ModelInference: {mi:#?}");

    assert_eq!(mi.inference_id, inference_id);
    assert_eq!(mi.cached, expect_cached);
    assert_eq!(mi.model_name, input_variant_name);
    assert_eq!(
        mi.model_provider_name,
        if inject_err { "dummy" } else { "good" }
    );

    let raw_request = mi.raw_request.as_ref().expect("Expected raw_request");
    assert!(raw_request.to_lowercase().contains("raw request"));

    let raw_response = mi.raw_response.as_ref().expect("Expected raw_response");
    for line in raw_response.lines() {
        assert!(
            !line.is_empty(),
            "Unexpected empty line in raw_response: {raw_response:?}"
        );
    }

    assert_eq!(mi.input_tokens, Some(10));
    assert_eq!(mi.output_tokens, Some(16));

    if expect_cached {
        assert_eq!(mi.response_time_ms, Some(0));
    } else {
        assert!(mi.response_time_ms.unwrap() > 0);
    }

    let system = mi.system.as_ref().expect("Expected system");
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named AskJeeves"
    );

    let input_messages = mi.input_messages.as_ref().expect("Expected input_messages");
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "My test input string".to_string(),
        })],
    }];
    assert_eq!(input_messages, &expected_input_messages);

    let output = mi.output.as_ref().expect("Expected model inference output");
    assert_eq!(output.len(), 1);

    full_content
}

make_cache_tests!(test_streaming_cache_usage_only_in_final_chunk_native);

async fn test_streaming_cache_usage_only_in_final_chunk_native(backend: CacheBackend) {
    skip_for_postgres!();
    use serde_json::Map;
    use tensorzero::InferenceResponseChunk;
    use tensorzero_core::inference::types::System;

    let client = make_cache_test_gateway(backend, "cache_usage_final_chunk").await;

    let input = "cache_usage_test: Tell me a story";

    // Helper to make streaming request and count chunks with usage
    async fn make_streaming_request(
        client: &tensorzero::Client,
        input: &str,
    ) -> (usize, usize, u64, u64) {
        let response = client
            .inference(ClientInferenceParams {
                function_name: Some("weather_helper".to_string()),
                variant_name: Some("openai".to_string()),
                episode_id: Some(Uuid::now_v7()),
                input: Input {
                    system: Some(System::Template(
                        tensorzero_core::inference::types::Arguments({
                            let mut args = Map::new();
                            args.insert("assistant_name".to_string(), json!("TestBot"));
                            args
                        }),
                    )),
                    messages: vec![InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text: input.to_string(),
                        })],
                    }],
                },
                stream: Some(true),
                cache_options: CacheParamsOptions {
                    enabled: CacheEnabledMode::On,
                    max_age_s: None,
                },
                ..Default::default()
            })
            .await
            .unwrap();

        let InferenceOutput::Streaming(mut stream) = response else {
            panic!("Expected streaming response");
        };

        let mut chunks_with_usage = 0;
        let mut total_chunks = 0;
        let mut total_input_tokens = 0u64;
        let mut total_output_tokens = 0u64;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            total_chunks += 1;

            let usage = match &chunk {
                InferenceResponseChunk::Chat(c) => &c.usage,
                InferenceResponseChunk::Json(j) => &j.usage,
            };
            if let Some(u) = usage {
                chunks_with_usage += 1;
                total_input_tokens += u.input_tokens.unwrap_or(0) as u64;
                total_output_tokens += u.output_tokens.unwrap_or(0) as u64;
            }
        }
        (
            chunks_with_usage,
            total_chunks,
            total_input_tokens,
            total_output_tokens,
        )
    }

    // First request: cache miss
    let (chunks_with_usage, total_chunks, input_tokens, output_tokens) =
        make_streaming_request(&client, input).await;

    assert!(
        total_chunks > 1,
        "Test expects multiple chunks to verify usage placement, got {total_chunks}"
    );
    assert_eq!(
        chunks_with_usage, 1,
        "Cache miss: only the final chunk should have usage, got {chunks_with_usage} out of {total_chunks}"
    );
    assert!(
        input_tokens > 0 || output_tokens > 0,
        "Cache miss: usage should have non-zero tokens"
    );

    // Wait for cache write
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Second request: cache hit
    let (chunks_with_usage, total_chunks, input_tokens, output_tokens) =
        make_streaming_request(&client, input).await;

    assert!(
        total_chunks > 1,
        "Test expects multiple chunks to verify usage placement, got {total_chunks}"
    );
    assert_eq!(
        chunks_with_usage, 1,
        "Cache hit: only the final chunk should have usage, got {chunks_with_usage} out of {total_chunks}"
    );
    assert_eq!(
        input_tokens, 0,
        "Cache hit: usage should have zero input tokens"
    );
    assert_eq!(
        output_tokens, 0,
        "Cache hit: usage should have zero output tokens"
    );
}

/// Tests that cached streaming responses only have usage on the final chunk (OpenAI-compatible API)
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_cache_usage_only_in_final_chunk_openai() {
    skip_for_postgres!();
    let (base_url, _shutdown_handle) =
        make_http_gateway_openai_only_with_unique_db("cache_usage_final_chunk_openai").await;

    let input = "cache_usage_openai_test: Tell me a story";

    // Helper to make streaming request and count chunks with usage
    async fn make_streaming_request(base_url: &str, input: &str) -> (usize, usize, u64, u64) {
        let payload = json!({
            "model": "tensorzero::function_name::weather_helper",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "tensorzero::arguments": {
                                "assistant_name": "TestBot"
                            }
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": input
                }
            ],
            "stream": true,
            "stream_options": {"include_usage": true},
            "tensorzero::variant_name": "openai",
            "tensorzero::episode_id": Uuid::now_v7(),
            "tensorzero::cache_options": {
                "enabled": "on"
            }
        });

        let url = format!("{base_url}/openai/v1/chat/completions");

        let mut chunks = Client::new()
            .post(&url)
            .json(&payload)
            .eventsource()
            .await
            .unwrap();

        let mut chunks_with_usage = 0;
        let mut total_chunks = 0;
        let mut total_prompt_tokens = 0u64;
        let mut total_completion_tokens = 0u64;

        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.unwrap();
            let Event::Message(chunk) = chunk else {
                continue;
            };
            if chunk.data == "[DONE]" {
                break;
            }

            total_chunks += 1;

            let chunk_json: Value = serde_json::from_str(&chunk.data).unwrap();
            if let Some(usage) = chunk_json.get("usage")
                && !usage.is_null()
            {
                chunks_with_usage += 1;
                total_prompt_tokens += usage
                    .get("prompt_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                total_completion_tokens += usage
                    .get("completion_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
            }
        }

        (
            chunks_with_usage,
            total_chunks,
            total_prompt_tokens,
            total_completion_tokens,
        )
    }

    // First request: cache miss
    let (chunks_with_usage, total_chunks, prompt_tokens, completion_tokens) =
        make_streaming_request(&base_url, input).await;

    expect_that!(
        total_chunks,
        gt(1),
        "Test expects multiple chunks to verify usage placement, got {total_chunks}"
    );
    expect_that!(
        chunks_with_usage,
        eq(1),
        "Cache miss: only the final chunk should have usage, got {chunks_with_usage} out of {total_chunks}"
    );
    expect_that!(
        prompt_tokens > 0 || completion_tokens > 0,
        eq(true),
        "Cache miss: usage should have non-zero tokens"
    );

    // Wait for cache write
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Second request: cache hit
    let (chunks_with_usage, total_chunks, prompt_tokens, completion_tokens) =
        make_streaming_request(&base_url, input).await;

    expect_that!(
        total_chunks,
        gt(1),
        "Test expects multiple chunks to verify usage placement, got {total_chunks}"
    );
    expect_that!(
        chunks_with_usage,
        eq(1),
        "Cache hit: only the final chunk should have usage, got {chunks_with_usage} out of {total_chunks}"
    );
    expect_that!(
        prompt_tokens,
        eq(0),
        "Cache hit: usage should have zero prompt tokens"
    );
    expect_that!(
        completion_tokens,
        eq(0),
        "Cache hit: usage should have zero completion tokens"
    );
}
