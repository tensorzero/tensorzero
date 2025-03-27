#![allow(clippy::print_stdout)]

use futures::StreamExt;
use rand::Rng;
use reqwest::Client;
use reqwest::StatusCode;
use reqwest_eventsource::Event;
use reqwest_eventsource::RequestBuilderExt;
use serde_json::json;
use serde_json::Value;
use std::time::Duration;
use tensorzero::ContentBlockChunk;
use tensorzero_internal::cache::cache_lookup_streaming;
use tensorzero_internal::cache::start_cache_write_streaming;
use tensorzero_internal::cache::NonStreamingCacheData;
use tensorzero_internal::inference::types::ContentBlock;
use tensorzero_internal::inference::types::ContentBlockOutput;
use tensorzero_internal::inference::types::FinishReason;
use tensorzero_internal::inference::types::ProviderInferenceResponseChunk;
use tensorzero_internal::inference::types::Text;
use tensorzero_internal::inference::types::TextChunk;
use uuid::Uuid;

use tensorzero_internal::cache::cache_lookup;
use tensorzero_internal::cache::start_cache_write;
use tensorzero_internal::cache::ModelProviderRequest;
use tensorzero_internal::inference::types::Latency;
use tensorzero_internal::inference::types::RequestMessage;
use tensorzero_internal::inference::types::Role;
use tensorzero_internal::inference::types::Usage;
use tensorzero_internal::inference::types::{
    FunctionType, ModelInferenceRequest, ModelInferenceRequestJsonMode,
};

use crate::common::get_gateway_endpoint;
use tensorzero_internal::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_model_inference_clickhouse,
};

/// This test does a cache read then write then read again to ensure that
/// the cache is working as expected.
/// Then, it reads with a short max age to ensure that the cache is not
/// returning stale data.
#[tokio::test]
async fn test_cache_write_and_read() {
    let clickhouse_connection_info = get_clickhouse().await;
    // Generate a random seed to guarantee a fresh cache key
    let seed = rand::random::<u32>();
    let max_age_s = 10;
    let model_inference_request = ModelInferenceRequest {
        inference_id: Uuid::now_v7(),
        messages: vec![RequestMessage {
            role: Role::User,
            content: vec!["test message".to_string().into()],
        }],
        system: Some("test system".to_string()),
        tool_config: None,
        temperature: None,
        top_p: None,
        presence_penalty: None,
        frequency_penalty: None,
        max_tokens: None,
        seed: Some(seed),
        stream: false,
        json_mode: ModelInferenceRequestJsonMode::Off,
        function_type: FunctionType::Chat,
        output_schema: None,
        extra_body: Default::default(),
        ..Default::default()
    };
    let model_provider_request = ModelProviderRequest {
        request: &model_inference_request,
        model_name: "test_model",
        provider_name: "test_provider",
    };

    // Read (should be None)
    let result = cache_lookup(
        &clickhouse_connection_info,
        model_provider_request,
        Some(max_age_s),
    )
    .await
    .unwrap();
    assert!(result.is_none());

    // Write
    start_cache_write(
        &clickhouse_connection_info,
        model_provider_request.get_cache_key().unwrap(),
        NonStreamingCacheData {
            blocks: vec![ContentBlockOutput::Text(Text {
                text: "my test content".to_string(),
            })],
        },
        "raw request",
        "raw response",
        &Usage {
            input_tokens: 10,
            output_tokens: 16,
        },
        Some(&FinishReason::Stop),
    )
    .unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Read (should be Some)
    let result = cache_lookup(
        &clickhouse_connection_info,
        model_provider_request,
        Some(max_age_s),
    )
    .await
    .unwrap();
    assert!(result.is_some());
    let result = result.unwrap();
    assert_eq!(
        result.output,
        [ContentBlockOutput::Text(Text {
            text: "my test content".to_string(),
        })]
    );
    assert_eq!(result.raw_request, "raw request");
    assert_eq!(result.raw_response, "raw response");
    assert_eq!(
        result.usage,
        Usage {
            input_tokens: 10,
            output_tokens: 16,
        }
    );
    assert_eq!(*result.model_provider_name, *"test_provider");
    assert_eq!(result.system, Some("test system".to_string()));
    assert_eq!(
        result.input_messages,
        vec![RequestMessage {
            role: Role::User,
            content: vec!["test message".to_string().into()],
        }]
    );
    assert_eq!(
        result.usage,
        Usage {
            input_tokens: 10,
            output_tokens: 16
        }
    );
    assert_eq!(
        result.latency,
        Latency::NonStreaming {
            response_time: Duration::from_secs(0)
        }
    );
    assert!(result.cached);

    // Read (should be None)
    tokio::time::sleep(Duration::from_secs(2)).await;
    let result = cache_lookup(&clickhouse_connection_info, model_provider_request, Some(0))
        .await
        .unwrap();
    assert!(result.is_none());
}

/// This test does a cache read then write then read again to ensure that
/// the cache is working as expected.
/// Then, it reads with a short max age to ensure that the cache is not
/// returning stale data.
#[tokio::test]
async fn test_cache_stream_write_and_read() {
    let clickhouse_connection_info = get_clickhouse().await;
    // Generate a random seed to guarantee a fresh cache key
    let seed = rand::random::<u32>();
    let max_age_s = 10;
    let model_inference_request = ModelInferenceRequest {
        inference_id: Uuid::now_v7(),
        messages: vec![RequestMessage {
            role: Role::User,
            content: vec!["test message".to_string().into()],
        }],
        system: Some("test system".to_string()),
        tool_config: None,
        temperature: None,
        top_p: None,
        presence_penalty: None,
        frequency_penalty: None,
        max_tokens: None,
        seed: Some(seed),
        stream: true,
        json_mode: ModelInferenceRequestJsonMode::Off,
        function_type: FunctionType::Chat,
        output_schema: None,
        extra_body: Default::default(),
        ..Default::default()
    };
    let model_provider_request = ModelProviderRequest {
        request: &model_inference_request,
        model_name: "test_model",
        provider_name: "test_provider",
    };

    // Read (should be None)
    let result = cache_lookup_streaming(
        &clickhouse_connection_info,
        model_provider_request,
        Some(max_age_s),
    )
    .await
    .unwrap();
    assert!(result.is_none());

    let initial_chunks = vec![ProviderInferenceResponseChunk {
        content: vec![ContentBlockChunk::Text(TextChunk {
            id: "0".to_string(),
            text: "test content".to_string(),
        })],
        created: 1234,
        usage: Some(Usage {
            input_tokens: 20,
            output_tokens: 40,
        }),
        raw_response: "raw response".to_string(),
        latency: Duration::from_secs(999),
        finish_reason: Some(FinishReason::Stop),
    }];

    // Write
    start_cache_write_streaming(
        &clickhouse_connection_info,
        model_provider_request.get_cache_key().unwrap(),
        initial_chunks.clone(),
        "raw request",
        &Usage {
            input_tokens: 20,
            output_tokens: 40,
        },
    )
    .unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Read (should be Some)
    let result = cache_lookup_streaming(
        &clickhouse_connection_info,
        model_provider_request,
        Some(max_age_s),
    )
    .await
    .unwrap();
    assert!(result.is_some());
    let result = result.unwrap();
    let chunks = result.stream.map(|c| c.unwrap()).collect::<Vec<_>>().await;
    assert_eq!(chunks.len(), 1);
    let ProviderInferenceResponseChunk {
        content,
        created,
        usage,
        raw_response,
        latency,
        finish_reason,
    } = &chunks[0];
    assert_eq!(content, &initial_chunks[0].content);
    // 'created' should be different (current timestamp is different)
    assert_ne!(created, &initial_chunks[0].created);
    assert_eq!(
        usage,
        &Some(Usage {
            input_tokens: 20,
            output_tokens: 40,
        })
    );
    assert_eq!(raw_response, &initial_chunks[0].raw_response);
    assert_eq!(latency, &Duration::from_secs(0));
    assert_eq!(finish_reason, &Some(FinishReason::Stop));
    // Read (should be None)
    tokio::time::sleep(Duration::from_secs(2)).await;
    let result =
        cache_lookup_streaming(&clickhouse_connection_info, model_provider_request, Some(0))
            .await
            .unwrap();
    assert!(result.is_none());
}

#[tokio::test]
pub async fn test_streaming_cache_with_err() {
    let episode_id = Uuid::now_v7();
    // Generate random u32
    let seed = rand::rng().random_range(0..u32::MAX);

    // When the stream includes an error, we should not cache the response (we pass `expect_cached = false`
    // for both calls)
    let original_content = check_test_streaming_cache_with_err(episode_id, seed, true, false).await;
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let cached_content = check_test_streaming_cache_with_err(episode_id, seed, true, false).await;
    assert_eq!(original_content, cached_content);
}

#[tokio::test]
pub async fn test_streaming_cache_without_err() {
    let episode_id = Uuid::now_v7();
    // Generate random u32
    let seed = rand::rng().random_range(0..u32::MAX);

    // When the stream does not include an error, we should cache the response (we pass `expect_cached = true`
    // for the second call)
    let original_content =
        check_test_streaming_cache_with_err(episode_id, seed, false, false).await;
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let cached_content = check_test_streaming_cache_with_err(episode_id, seed, false, true).await;
    assert_eq!(original_content, cached_content);
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

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
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

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, input_variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": format!("AskJeeves")},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "My test input string"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<Value> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let content_block = output.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, full_content);

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    assert!(tool_params.is_empty());

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert_eq!(
        inference_params.get("seed").unwrap().as_u64().unwrap(),
        seed as u64
    );
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check ClickHouse - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let result_cached = result.get("cached").unwrap().as_bool().unwrap();
    assert_eq!(result_cached, expect_cached);

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, input_variant_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(
        model_provider_name,
        if inject_err { "dummy" } else { "good" }
    );

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("raw request"));
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();

    // Check if raw_response is non-empty
    for line in raw_response.lines() {
        assert!(
            !line.is_empty(),
            "Unexpected empty line in raw_response: {raw_response:?}"
        );
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    assert_eq!(input_tokens.as_u64().unwrap(), 10);
    assert_eq!(output_tokens.as_u64().unwrap(), 16);

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    if expect_cached {
        assert_eq!(response_time_ms, 0);
    } else {
        assert!(response_time_ms > 0);
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        format!("You are a helpful and friendly assistant named AskJeeves")
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["My test input string".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);

    full_content
}
