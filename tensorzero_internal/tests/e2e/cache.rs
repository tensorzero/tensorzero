use std::time::Duration;

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

use crate::common::get_clickhouse;

/// This test does a cache read then write then read again to ensure that
/// the cache is working as expected.
/// Then, it reads with a short lookback to ensure that the cache is not
/// returning stale data.
#[tokio::test]
async fn test_cache_write_and_read() {
    let clickhouse_connection_info = get_clickhouse().await;
    // Generate a random seed to guarantee a fresh cache key
    let seed = rand::random::<u32>();
    let lookback_s = 10;
    let model_inference_request = ModelInferenceRequest {
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
    };
    let model_provider_request = ModelProviderRequest {
        request: &model_inference_request,
        model_name: "test_model",
        provider_name: "test_provider",
    };

    // Read (should be None)
    let result = cache_lookup(
        &clickhouse_connection_info,
        model_provider_request.clone(),
        Some(lookback_s),
    )
    .await
    .unwrap();
    assert!(result.is_none());

    // Write
    start_cache_write(
        &clickhouse_connection_info,
        model_provider_request.clone(),
        &["test content".to_string().into()],
        "raw request",
        "raw response",
    )
    .unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Read (should be Some)
    let result = cache_lookup(
        &clickhouse_connection_info,
        model_provider_request.clone(),
        Some(lookback_s),
    )
    .await
    .unwrap();
    assert!(result.is_some());
    let result = result.unwrap();
    assert_eq!(result.output, vec!["test content".to_string().into()]);
    assert_eq!(result.raw_request, "raw request");
    assert_eq!(result.raw_response, "raw response");
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
            input_tokens: 0,
            output_tokens: 0
        }
    );
    assert_eq!(
        result.latency,
        Latency::NonStreaming {
            response_time: Duration::from_secs(0)
        }
    );
    assert!(result.cache_hit);

    // Read (should be None)
    tokio::time::sleep(Duration::from_secs(2)).await;
    let result = cache_lookup(&clickhouse_connection_info, model_provider_request, Some(0))
        .await
        .unwrap();
    assert!(result.is_none());
}
