use std::collections::HashMap;

use futures::StreamExt;
use tensorzero::test_helpers::make_embedded_gateway_with_config;
use tensorzero::ClientInferenceParams;
use tensorzero::ClientInput;
use tensorzero::ClientInputMessage;
use tensorzero::ClientInputMessageContent;
use tensorzero::InferenceOutput;
use tensorzero::InferenceResponse;
use tensorzero::InferenceResponseChunk;
use tensorzero::Role;
use tensorzero_core::inference::types::TextKind;
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
        model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let image_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-haiku".to_string(),
        model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
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
        model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-haiku-extra-headers".to_string(),
        model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-haiku".to_string(),
            model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
            model_provider_name: "gcp_vertex_anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-haiku-implicit".to_string(),
            model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
            model_provider_name: "gcp_vertex_anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-haiku-strict".to_string(),
            model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
            model_provider_name: "gcp_vertex_anthropic".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex_haiku_json_mode_off".to_string(),
        model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex_anthropic_shorthand".to_string(),
        model_name: "gcp_vertex_anthropic::projects/tensorzero-public/locations/us-central1/publishers/anthropic/models/claude-3-haiku@20240307".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "gcp_vertex_anthropic".to_string(),
        model_info: HashMap::from([
            (
                "model_id".to_string(),
                "claude-3-haiku@20240307".to_string(),
            ),
            ("location".to_string(), "us-central1".to_string()),
            ("project_id".to_string(), "tensorzero-public".to_string()),
        ]),
        use_modal_headers: false,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
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
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
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
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
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
        if inference_id.is_none() {
            if let InferenceResponseChunk::Chat(chat_chunk) = &chunk {
                inference_id = Some(chat_chunk.inference_id);
            }
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
