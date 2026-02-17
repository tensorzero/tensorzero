use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use tensorzero::ContentBlockChunk;
use tensorzero_core::cache::CacheData;
use tensorzero_core::cache::CacheValidationInfo;
use tensorzero_core::cache::NonStreamingCacheData;
use tensorzero_core::cache::cache_lookup_streaming;
use tensorzero_core::cache::start_cache_write_streaming;
use tensorzero_core::db::cache::CacheQueries;
use tensorzero_core::inference::types::ContentBlock;
use tensorzero_core::inference::types::ContentBlockOutput;
use tensorzero_core::inference::types::FinishReason;
use tensorzero_core::inference::types::ProviderInferenceResponseChunk;
use tensorzero_core::inference::types::Text;
use tensorzero_core::inference::types::TextChunk;
use uuid::Uuid;

use tensorzero_core::cache::ModelProviderRequest;
use tensorzero_core::cache::cache_lookup;
use tensorzero_core::cache::start_cache_write;
use tensorzero_core::inference::types::Latency;
use tensorzero_core::inference::types::RequestMessage;
use tensorzero_core::inference::types::Role;
use tensorzero_core::inference::types::Usage;
use tensorzero_core::inference::types::{
    FunctionType, ModelInferenceRequest, ModelInferenceRequestJsonMode,
};

// ===== SHARED TEST IMPLEMENTATIONS =====
// These tests use CacheQueries directly and can be parameterized over backends.

/// This test does a cache read then write then read again to ensure that
/// the cache is working as expected.
/// Then, it reads with a short max age to ensure that the cache is not
/// returning stale data.
async fn test_cache_write_and_read(conn: impl CacheQueries + Clone + 'static) {
    // Generate a random seed to guarantee a fresh cache key
    let seed = rand::random::<u32>();
    let max_age_s = 10;
    let model_inference_request = ModelInferenceRequest {
        inference_id: Uuid::now_v7(),
        messages: vec![RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::Text(Text {
                text: "test message".to_string(),
            })],
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
        otlp_config: &Default::default(),
        model_inference_id: Uuid::now_v7(),
    };

    // Read (should be None)
    let result = cache_lookup(
        &conn,
        model_provider_request,
        Some(max_age_s),
        Arc::from("dummy"),
    )
    .await
    .unwrap();
    assert!(result.is_none());

    // Write
    start_cache_write(
        &conn,
        model_provider_request.get_cache_key().unwrap(),
        CacheData {
            output: NonStreamingCacheData {
                blocks: vec![ContentBlockOutput::Text(Text {
                    text: "my test content".to_string(),
                })],
            },
            raw_request: "raw request".to_string(),
            raw_response: "raw response".to_string(),
            input_tokens: Some(10),
            output_tokens: Some(16),
            finish_reason: Some(FinishReason::Stop),
        },
        CacheValidationInfo { tool_config: None },
    )
    .unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Read (should be Some)
    let result = cache_lookup(
        &conn,
        model_provider_request,
        Some(max_age_s),
        Arc::from("dummy"),
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
            input_tokens: Some(10),
            output_tokens: Some(16),
        }
    );
    assert_eq!(*result.model_provider_name, *"test_provider");
    assert_eq!(result.system, Some("test system".to_string()));
    assert_eq!(
        result.input_messages,
        vec![RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::Text(Text {
                text: "test message".to_string(),
            })],
        }]
    );
    assert_eq!(
        result.usage,
        Usage {
            input_tokens: Some(10),
            output_tokens: Some(16),
        }
    );
    assert_eq!(
        result.provider_latency,
        Latency::NonStreaming {
            response_time: Duration::from_secs(0)
        }
    );
    assert!(result.cached);

    // Read (should be None)
    tokio::time::sleep(Duration::from_secs(2)).await;
    let result = cache_lookup(&conn, model_provider_request, Some(0), Arc::from("dummy"))
        .await
        .unwrap();
    assert!(result.is_none());
}
make_clickhouse_only_test!(test_cache_write_and_read);

/// This test does a cache read then write then read again to ensure that
/// the cache is working as expected.
/// Then, it reads with a short max age to ensure that the cache is not
/// returning stale data.
async fn test_cache_stream_write_and_read(conn: impl CacheQueries + Clone + 'static) {
    // Generate a random seed to guarantee a fresh cache key
    let seed = rand::random::<u32>();
    let max_age_s = 10;
    let model_inference_request = ModelInferenceRequest {
        inference_id: Uuid::now_v7(),
        messages: vec![RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::Text(Text {
                text: "test message".to_string(),
            })],
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
        otlp_config: &Default::default(),
        model_inference_id: Uuid::now_v7(),
    };

    // Read (should be None)
    let result = cache_lookup_streaming(
        &conn,
        model_provider_request,
        Some(max_age_s),
        Arc::from("dummy"),
    )
    .await
    .unwrap();
    assert!(result.is_none());

    let initial_chunks = vec![
        ProviderInferenceResponseChunk {
            content: vec![ContentBlockChunk::Text(TextChunk {
                id: "0".to_string(),
                text: "test content".to_string(),
            })],
            usage: Some(Usage {
                input_tokens: Some(20),
                output_tokens: Some(40),
            }),
            raw_usage: None,
            raw_response: "raw response".to_string(),
            provider_latency: Duration::from_secs(999),
            finish_reason: None,
        },
        ProviderInferenceResponseChunk {
            content: vec![ContentBlockChunk::Text(TextChunk {
                id: "1".to_string(),
                text: "test content 2".to_string(),
            })],
            usage: Some(Usage {
                input_tokens: Some(100),
                output_tokens: Some(200),
            }),
            raw_usage: None,
            raw_response: "raw response 2".to_string(),
            provider_latency: Duration::from_secs(999),
            finish_reason: Some(FinishReason::Stop),
        },
    ];

    // Write
    start_cache_write_streaming(
        &conn,
        model_provider_request.get_cache_key().unwrap(),
        initial_chunks.clone(),
        "raw request",
        &Usage {
            input_tokens: Some(1),
            output_tokens: Some(2),
        },
        None,
    )
    .unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Read (should be Some)
    let result = cache_lookup_streaming(
        &conn,
        model_provider_request,
        Some(max_age_s),
        Arc::from("dummy"),
    )
    .await
    .unwrap();
    assert!(result.is_some());
    let result = result.unwrap();
    let chunks = result.stream.map(|c| c.unwrap()).collect::<Vec<_>>().await;
    assert_eq!(chunks.len(), 2);
    for (i, chunk) in chunks.into_iter().enumerate() {
        let ProviderInferenceResponseChunk {
            content,
            usage,
            raw_response,
            provider_latency,
            finish_reason,
            ..
        } = &chunk;
        assert_eq!(content, &initial_chunks[i].content);
        if i == 0 {
            assert_eq!(
                usage,
                &Some(Usage {
                    input_tokens: Some(20),
                    output_tokens: Some(40),
                })
            );
        } else {
            assert_eq!(
                usage,
                &Some(Usage {
                    input_tokens: Some(100),
                    output_tokens: Some(200),
                })
            );
        };
        assert_eq!(raw_response, &initial_chunks[i].raw_response);
        assert_eq!(provider_latency, &Duration::from_secs(0));
        if i == 0 {
            assert_eq!(finish_reason, &None);
        } else {
            assert_eq!(finish_reason, &Some(FinishReason::Stop));
        }
    }

    // Read (should be None)
    tokio::time::sleep(Duration::from_secs(2)).await;
    let result = cache_lookup_streaming(&conn, model_provider_request, Some(0), Arc::from("dummy"))
        .await
        .unwrap();
    assert!(result.is_none());
}
make_clickhouse_only_test!(test_cache_stream_write_and_read);
