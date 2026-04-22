use std::borrow::Cow;
use std::sync::LazyLock;

use criterion::{Criterion, criterion_group, criterion_main};
use uuid::Uuid;

use tensorzero_core::cache::ModelProviderRequest;
use tensorzero_core::config::OtlpConfig;
use tensorzero_core::inference::types::chat_completion_inference_params::ChatCompletionInferenceParamsV2;
use tensorzero_core::inference::types::{
    ContentBlock, FunctionType, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    RequestMessage, Role,
};

static OUTPUT_SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(
    || serde_json::json!({"type": "object", "properties": {"answer": {"type": "string"}}}),
);

fn make_request() -> ModelInferenceRequest<'static> {
    ModelInferenceRequest {
        inference_id: Uuid::now_v7(),
        messages: vec![
            RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::from(
                    "What is the capital of France?".to_string(),
                )],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec![ContentBlock::from("Paris.".to_string())],
            },
            RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::from(
                    "And what is the population of that city?".to_string(),
                )],
            },
        ],
        system: Some("You are a helpful assistant that answers questions concisely.".to_string()),
        tool_config: None,
        temperature: Some(0.7),
        top_p: Some(0.9),
        presence_penalty: Some(0.1),
        frequency_penalty: Some(0.2),
        max_tokens: Some(512),
        seed: Some(42),
        stream: false,
        json_mode: ModelInferenceRequestJsonMode::Off,
        function_type: FunctionType::Chat,
        output_schema: Some(&OUTPUT_SCHEMA),
        extra_body: Default::default(),
        extra_headers: Default::default(),
        fetch_and_encode_input_files_before_inference: false,
        extra_cache_key: Some("bench-key".to_string()),
        stop_sequences: Some(Cow::Owned(vec!["STOP".to_string(), "END".to_string()])),
        inference_params_v2: ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1000),
            verbosity: Some("verbose".to_string()),
        },
    }
}

fn bench_get_cache_key(c: &mut Criterion) {
    let request = make_request();
    let otlp_config = OtlpConfig::default();

    c.bench_function("get_cache_key", |b| {
        b.iter(|| {
            let provider_request = ModelProviderRequest {
                request: &request,
                model_name: "gpt-4o",
                provider_name: "openai",
                otlp_config: &otlp_config,
                model_inference_id: Uuid::now_v7(),
                function_name: Some("bench_function"),
            };
            provider_request
                .get_cache_key()
                .expect("get_cache_key should not fail")
        })
    });
}

criterion_group!(benches, bench_get_cache_key);
criterion_main!(benches);
