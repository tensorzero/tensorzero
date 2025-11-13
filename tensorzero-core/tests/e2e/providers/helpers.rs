use std::env;

use tensorzero_core::inference::types::extra_headers::{
    ExtraHeaderKind, InferenceExtraHeader, UnfilteredInferenceExtraHeaders,
};

/// Our self-hosted providers (vllm and sglang) require modal credentials to be
/// passed as extra headers alongside other credentials.
/// Only call this function when testing vllm or sglang providers.
pub fn get_modal_extra_headers() -> UnfilteredInferenceExtraHeaders {
    let mut extra_headers = Vec::new();
    if let Ok(modal_key) = env::var("MODAL_KEY") {
        extra_headers.push(InferenceExtraHeader::ModelProvider {
            model_name: "qwen2.5-0.5b-instruct-vllm".to_string(),
            provider_name: "vllm".to_string(),
            name: "Modal-Key".to_string(),
            kind: ExtraHeaderKind::Value(modal_key.clone()),
        });
        extra_headers.push(InferenceExtraHeader::ModelProvider {
            model_name: "qwen2.5-0.5b-instruct-vllm-dynamic".to_string(),
            provider_name: "vllm".to_string(),
            name: "Modal-Key".to_string(),
            kind: ExtraHeaderKind::Value(modal_key.clone()),
        });
        extra_headers.push(InferenceExtraHeader::ModelProvider {
            model_name: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
            provider_name: "sglang".to_string(),
            name: "Modal-Key".to_string(),
            kind: ExtraHeaderKind::Value(modal_key.clone()),
        });
        extra_headers.push(InferenceExtraHeader::ModelProvider {
            model_name: "gpt-oss-20b-vllm".to_string(),
            provider_name: "vllm".to_string(),
            name: "Modal-Key".to_string(),
            kind: ExtraHeaderKind::Value(modal_key.clone()),
        });
    }
    if let Ok(modal_secret) = env::var("MODAL_SECRET") {
        extra_headers.push(InferenceExtraHeader::ModelProvider {
            model_name: "qwen2.5-0.5b-instruct-vllm".to_string(),
            provider_name: "vllm".to_string(),
            name: "Modal-Secret".to_string(),
            kind: ExtraHeaderKind::Value(modal_secret.clone()),
        });
        extra_headers.push(InferenceExtraHeader::ModelProvider {
            model_name: "qwen2.5-0.5b-instruct-vllm-dynamic".to_string(),
            provider_name: "vllm".to_string(),
            name: "Modal-Secret".to_string(),
            kind: ExtraHeaderKind::Value(modal_secret.clone()),
        });
        extra_headers.push(InferenceExtraHeader::ModelProvider {
            model_name: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
            provider_name: "sglang".to_string(),
            name: "Modal-Secret".to_string(),
            kind: ExtraHeaderKind::Value(modal_secret.clone()),
        });
        extra_headers.push(InferenceExtraHeader::ModelProvider {
            model_name: "gpt-oss-20b-vllm".to_string(),
            provider_name: "vllm".to_string(),
            name: "Modal-Secret".to_string(),
            kind: ExtraHeaderKind::Value(modal_secret.clone()),
        });
    }
    UnfilteredInferenceExtraHeaders { extra_headers }
}

/// Get extra headers for the test-model/test-provider configuration
/// used in credential fallback tests.
pub fn get_test_model_extra_headers() -> UnfilteredInferenceExtraHeaders {
    let mut headers = Vec::new();
    if let Ok(modal_key) = env::var("MODAL_KEY") {
        headers.push(InferenceExtraHeader::ModelProvider {
            model_name: "test-model".to_string(),
            provider_name: "test-provider".to_string(),
            name: "Modal-Key".to_string(),
            kind: ExtraHeaderKind::Value(modal_key),
        });
    }
    if let Ok(modal_secret) = env::var("MODAL_SECRET") {
        headers.push(InferenceExtraHeader::ModelProvider {
            model_name: "test-model".to_string(),
            provider_name: "test-provider".to_string(),
            name: "Modal-Secret".to_string(),
            kind: ExtraHeaderKind::Value(modal_secret),
        });
    }
    UnfilteredInferenceExtraHeaders {
        extra_headers: headers,
    }
}
