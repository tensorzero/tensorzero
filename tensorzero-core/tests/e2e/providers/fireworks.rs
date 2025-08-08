#![expect(clippy::print_stderr)]
use std::collections::HashMap;

use http::StatusCode;
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use tokio_stream::StreamExt;

use crate::{
    common::get_gateway_endpoint,
    providers::common::{E2ETestProvider, E2ETestProviders},
};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("FIREWORKS_API_KEY") {
        Ok(key) => HashMap::from([("fireworks_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks".to_string(),
        model_name: "fireworks::accounts/fireworks/models/mixtral-8x22b-instruct".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks-extra-body".to_string(),
        model_name: "fireworks::accounts/fireworks/models/deepseek-r1-0528".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks-extra-headers".to_string(),
        model_name: "fireworks::accounts/fireworks/models/deepseek-r1-0528".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks-dynamic".to_string(),
        model_name: "fireworks::accounts/fireworks/models/mixtral-8x22b-instruct".into(),
        model_provider_name: "fireworks".into(),
        credentials,
    }];

    let tool_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks".to_string(),
        model_name: "fireworks::accounts/fireworks/models/mixtral-8x22b-instruct".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "fireworks".to_string(),
            model_name: "fireworks::accounts/fireworks/models/mixtral-8x22b-instruct".into(),
            model_provider_name: "fireworks".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "fireworks-implicit".to_string(),
            model_name: "fireworks::accounts/fireworks/models/mixtral-8x22b-instruct".into(),
            model_provider_name: "fireworks".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "fireworks-strict".to_string(),
            model_name: "fireworks::accounts/fireworks/models/mixtral-8x22b-instruct".into(),
            model_provider_name: "fireworks".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks_json_mode_off".to_string(),
        model_name: "fireworks::accounts/fireworks/models/mixtral-8x22b-instruct".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks-shorthand".to_string(),
        model_name: "fireworks::accounts/fireworks/models/llama-v3p1-8b-instruct".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let thinking_block_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks-deepseek".to_string(),
        model_name: "deepseek-r1".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: thinking_block_providers.clone(),
        embeddings: vec![],
        inference_params_inference: providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers,
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        shorthand_inference: shorthand_providers,
    }
}

#[tokio::test]
async fn test_fireworks_reasoning_content_non_stream() {
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&serde_json::json!({
            "model_name": "gpt-oss-20b-fireworks",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of France? Think before responding."
                    }
                ]
            }
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_body = response.json::<serde_json::Value>().await.unwrap();

    eprintln!("API response: {response_body}");
    let content = response_body["content"].as_array().unwrap();
    // Check that the response contains a thought block
    let thought = content.iter().find(|c| c["type"] == "thought").unwrap();
    assert!(
        !thought["text"].as_str().unwrap().is_empty(),
        "Thought block was empty: {thought:?}",
    );
}

#[tokio::test]
async fn test_fireworks_reasoning_content_stream() {
    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&serde_json::json!({
            "model_name": "gpt-oss-20b-fireworks",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of France? Think before responding."
                    }
                ]
            },
            "stream": true,
        }))
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                eprintln!("API chunk: {message:?}");
                if message.data == "[DONE]" {
                    break;
                }
                chunks.push(message.data.to_string());
            }
        }
    }

    let mut found_thought = false;
    for chunk in chunks {
        let chunk_json: serde_json::Value = serde_json::from_str(&chunk).unwrap();
        eprintln!("Chunk: {chunk_json}");
        if chunk_json["content"][0]["type"] == "thought" {
            found_thought = true;
            break;
        }
    }
    assert!(found_thought, "Thought chunk not found");
}
