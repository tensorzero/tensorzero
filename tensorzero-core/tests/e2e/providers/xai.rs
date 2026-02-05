use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders, ModelTestProvider};
use tensorzero::{
    ClientInferenceParams, InferenceOutput, InferenceResponse, Input, InputMessage,
    InputMessageContent,
};
use tensorzero_core::inference::types::{Role, Text, Thought};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("XAI_API_KEY") {
        Ok(key) => HashMap::from([("xai_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai".to_string(),
        model_name: "grok_4_1_fast_non_reasoning".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai-extra-body".to_string(),
        model_name: "grok_4_1_fast_non_reasoning".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai-extra-headers".to_string(),
        model_name: "grok_4_1_fast_non_reasoning".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "xai".to_string(),
            model_name: "grok_4_1_fast_non_reasoning".into(),
            model_provider_name: "xai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "xai-strict".to_string(),
            model_name: "grok_4_1_fast_non_reasoning".into(),
            model_provider_name: "xai".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai_json_mode_off".to_string(),
        model_name: "grok_4_1_fast_non_reasoning".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai".to_string(),
        model_name: "grok_4_1_fast_non_reasoning".into(),
        model_provider_name: "xai".into(),
        credentials,
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai-shorthand".to_string(),
        model_name: "xai::grok-4-1-fast-non-reasoning".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai".to_string(),
        model_name: "grok-4-1-fast-non-reasoning".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai-shorthand".to_string(),
        model_name: "xai::grok-4-1-fast-non-reasoning".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "xai".into(),
        model_info: HashMap::from([(
            "model_name".to_string(),
            "grok-4-1-fast-non-reasoning".to_string(),
        )]),
        use_modal_headers: false,
    }];

    let reasoning_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai-reasoning".to_string(),
        model_name: "grok-3-mini".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let reasoning_usage_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "xai-reasoning".to_string(),
            model_name: "grok-3-mini".into(),
            model_provider_name: "xai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "xai-reasoning-usage".to_string(),
            model_name: "grok-4-fast-reasoning".into(),
            model_provider_name: "xai".into(),
            credentials: HashMap::new(),
        },
    ];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: reasoning_providers,
        reasoning_usage_inference: reasoning_usage_providers,
        cache_input_tokens_inference: standard_providers.clone(),
        embeddings: vec![],
        inference_params_inference: inference_params_providers,
        inference_params_dynamic_credentials: vec![],
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand:
            provider_type_default_credentials_shorthand_providers,
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers,
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: vec![],
        shorthand_inference: shorthand_providers.clone(),
        credential_fallbacks,
    }
}

/// Tests that thought blocks with a mismatched `provider_type` are filtered out at the model layer
/// before reaching the xAI provider. This exercises the debug_assert invariant in the provider
/// that it only receives thought blocks with `provider_type` matching "xai" or `None`.
#[tokio::test]
async fn test_mismatched_provider_type_thought_block_filtered() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;

    // Send a request with a thought block that has a mismatched provider_type.
    // This should be filtered out at the model layer before reaching the xAI provider.
    let res = client
        .inference(ClientInferenceParams {
            model_name: Some("xai::grok-4-1-fast-non-reasoning".to_string()),
            input: Input {
                system: None,
                messages: vec![
                    InputMessage {
                        role: Role::Assistant,
                        content: vec![
                            InputMessageContent::Text(Text {
                                text: "Hello!".to_string(),
                            }),
                            InputMessageContent::Thought(Thought {
                                text: Some(
                                    "This thought has a mismatched provider_type".to_string(),
                                ),
                                signature: None,
                                summary: None,
                                // Use a provider_type that doesn't match "xai"
                                provider_type: Some("openai".to_string()),
                                extra_data: None,
                            }),
                        ],
                    },
                    InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text: "What is the capital of Japan?".to_string(),
                        })],
                    },
                ],
            },
            ..Default::default()
        })
        .await;

    // The request should succeed because the mismatched thought block is filtered out
    // at the model layer before reaching the xAI provider.
    // If the debug_assert in the provider were to fire (in debug builds), this test would panic.
    let response = res.expect(
        "Request should succeed - mismatched provider_type thought blocks should be filtered at model layer",
    );

    // Verify we got a valid response with content
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(chat_response)) = response else {
        panic!("Expected non-streaming chat inference response");
    };
    assert!(
        !chat_response.content.is_empty(),
        "Response should have content after filtering mismatched thought block"
    );
}
