use std::env;

use tensorzero_internal::inference::types::extra_headers::{
    InferenceExtraHeader, UnfilteredInferenceExtraHeaders,
};

/// Our self-hosted providers require modal credentials to be
/// passed as extra headers alongside other credentials.
/// We can safely pass these in lots of places since they will be filtered
/// out by model provider name for the places where it's not needed.
pub fn get_extra_headers() -> UnfilteredInferenceExtraHeaders {
    let mut extra_headers = Vec::new();

    if let Ok(modal_key) = env::var("MODAL_KEY") {
        extra_headers.push(InferenceExtraHeader::Provider {
            model_provider_name:
                "tensorzero::model_name::smol-lm-instruct-vllm::provider_name::vllm".to_string(),
            name: "Modal-Key".to_string(),
            value: modal_key.clone(),
        });

        extra_headers.push(InferenceExtraHeader::Provider {
            model_provider_name:
                "tensorzero::model_name::smol-lm-instruct-vllm-dynamic::provider_name::vllm"
                    .to_string(),
            name: "Modal-Key".to_string(),
            value: modal_key.clone(),
        });
        extra_headers.push(InferenceExtraHeader::Provider {
            model_provider_name:
                "tensorzero::model_name::google/gemma-3-1b-it::provider_name::sglang".to_string(),
            name: "Modal-Key".to_string(),
            value: modal_key.clone(),
        });
    }

    if let Ok(modal_secret) = env::var("MODAL_SECRET") {
        extra_headers.push(InferenceExtraHeader::Provider {
            model_provider_name:
                "tensorzero::model_name::smol-lm-instruct-vllm::provider_name::vllm".to_string(),
            name: "Modal-Secret".to_string(),
            value: modal_secret.clone(),
        });
        extra_headers.push(InferenceExtraHeader::Provider {
            model_provider_name:
                "tensorzero::model_name::smol-lm-instruct-vllm-dynamic::provider_name::vllm"
                    .to_string(),
            name: "Modal-Secret".to_string(),
            value: modal_secret.clone(),
        });
        extra_headers.push(InferenceExtraHeader::Provider {
            model_provider_name:
                "tensorzero::model_name::google/gemma-3-1b-it::provider_name::sglang".to_string(),
            name: "Modal-Secret".to_string(),
            value: modal_secret.clone(),
        });
    }

    UnfilteredInferenceExtraHeaders {
        headers: extra_headers,
    }
}
