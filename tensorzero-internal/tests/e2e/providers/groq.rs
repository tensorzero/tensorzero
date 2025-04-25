#![allow(clippy::print_stdout)]
use std::collections::HashMap;

use reqwest::Client;
use reqwest::StatusCode;
use serde_json::json;
use serde_json::Value;
use tensorzero::ClientInput;
use tensorzero::ClientInputMessage;
use tensorzero::ClientInputMessageContent;
use tensorzero_internal::cache::CacheEnabledMode;
use tensorzero_internal::cache::CacheOptions;
use tensorzero_internal::embeddings::EmbeddingModelConfig;
#[allow(unused)]
use tensorzero_internal::embeddings::EmbeddingProvider;
use tensorzero_internal::endpoints::inference::InferenceClients;
use tensorzero_internal::{
    embeddings::{EmbeddingProviderConfig, EmbeddingRequest},
    endpoints::inference::InferenceCredentials,
    inference::types::{Latency, ModelInferenceRequestJsonMode},
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use crate::providers::common::{E2ETestProvider, E2ETestProviders};
use tensorzero_internal::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_model_inference_clickhouse,
};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("GROQ_API_KEY") {
        Ok(key) => HashMap::from([("groq_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq".to_string(),
        model_name: "gpt-4o-mini-groq".into(),
        model_provider_name: "groq".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq-extra-body".to_string(),
        model_name: "gpt-4o-mini-groq".into(),
        model_provider_name: "groq".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq-extra-headers".to_string(),
        model_name: "gpt-4o-mini-groq".into(),
        model_provider_name: "groq".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq".to_string(),
        model_name: "gpt-4o-mini-groq".into(),
        model_provider_name: "groq".into(),
        credentials: credentials.clone(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq-dynamic".to_string(),
        model_name: "gpt-4o-mini-groq-dynamic".into(),
        model_provider_name: "groq".into(),
        credentials,
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "groq".to_string(),
            model_name: "gpt-4o-mini-groq".into(),
            model_provider_name: "groq".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "groq-implicit".to_string(),
            model_name: "gpt-4o-mini-groq".into(),
            model_provider_name: "groq".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "groq-strict".to_string(),
            model_name: "gpt-4o-mini-groq".into(),
            model_provider_name: "groq".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "groq-default".to_string(),
            model_name: "gpt-4o-mini-groq".into(),
            model_provider_name: "groq".into(),
            credentials: HashMap::new(),
        },
    ];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        inference_params_inference: inference_params_providers,
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
        image_inference: vec![],
        shorthand_inference: vec![],
    }
}
