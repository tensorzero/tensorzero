use std::{collections::HashMap, time::Duration};

use serde_json::json;
use tempfile::NamedTempFile;
use tensorzero::{
    ClientBuilder, ClientBuilderMode, ClientInferenceParams, ClientInput, ClientInputMessage,
    ClientInputMessageContent, InferenceOutput, InferenceResponse, Role, System,
};
use tensorzero_core::{
    db::clickhouse::test_helpers::{get_clickhouse, select_model_inference_clickhouse},
    inference::types::{ContentBlockChatOutput, Text, TextKind},
};
use tokio::time::sleep;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "tensorzero-relay-dummy".to_string(),
        model_name: "tensorzero-relay-dummy".into(),
        model_provider_name: "tensorzero_relay".into(),
        credentials: HashMap::new(),
    }];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "tensorzero-relay-dummy_json_mode_off".to_string(),
        model_name: "tensorzero-relay-dummy".into(),
        model_provider_name: "tensorzero_relay".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: vec![],
        bad_auth_extra_headers: vec![],
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
        json_mode_inference: standard_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: standard_providers.clone(),
        pdf_inference: standard_providers.clone(),
        input_audio: standard_providers.clone(),
        shorthand_inference: vec![],
        credential_fallbacks: vec![],
    }
}

#[tokio::test]
async fn test_tensorzero_relay_reject_extra_headers() {
    let client = reqwest::Client::new();
    let response = client
        .post(tensorzero::test_helpers::get_gateway_endpoint("/inference"))
        .json(&json!({
            "model_name": "tensorzero-relay-dummy",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, world!"
                    }
                ]
            },
            "extra_headers": [{
                "model_provider_name": "tensorzero::model_name::tensorzero-relay-dummy::provider_name::tensorzero_relay",
                "name": "X-My-Header",
                "value": "my_value"
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), http::StatusCode::BAD_GATEWAY);
    let text = response.text().await.unwrap();
    assert!(
        text.contains("Extra headers and body are not supported for `tensorzero_relay` provider"),
        "Unexpected response text: {text:?}"
    );
}

#[tokio::test]
async fn test_tensorzero_relay_reject_extra_body() {
    let client = reqwest::Client::new();
    let response = client
        .post(tensorzero::test_helpers::get_gateway_endpoint("/inference"))
        .json(&json!({
            "model_name": "tensorzero-relay-dummy",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, world!"
                    }
                ]
            },
            "extra_body": [{
                "model_provider_name": "tensorzero::model_name::tensorzero-relay-dummy::provider_name::tensorzero_relay",
                "pointer": "/my_key",
                "value": "my_value"
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), http::StatusCode::BAD_GATEWAY);
    let text = response.text().await.unwrap();
    assert!(
        text.contains("Extra headers and body are not supported for `tensorzero_relay` provider"),
        "Unexpected response text: {text:?}"
    );
}

#[tokio::test]
async fn test_tensorzero_relay_reject_auth() {
    std::env::set_var("MY_API_KEY", "my_api_key");
    let config = format!(
        r#"
    [models."tensorzero-relay-dummy"]
    routing = ["tensorzero_relay"]

    [models."tensorzero-relay-dummy".providers."tensorzero_relay"]
    type = "tensorzero_relay"
    gateway_base_url = "{}"
    function_name = "basic_test_no_system_schema"
    variant_name = "test"
    api_key_location = "env::MY_API_KEY"
    "#,
        tensorzero::test_helpers::get_gateway_endpoint("/")
    );

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();
    let err = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: None,
        postgres_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("Credentials are not yet supported for `tensorzero_relay` provider"),
        "Unexpected error: {err:?}"
    );
}

#[tokio::test]
async fn test_tensorzero_relay_function_variant() {
    let config = format!(
        r#"
    [models."tensorzero-relay-dummy"]
    routing = ["tensorzero_relay"]

    [models."tensorzero-relay-dummy".providers."tensorzero_relay"]
    type = "tensorzero_relay"
    gateway_base_url = "{}"
    function_name = "basic_test_no_system_schema"
    variant_name = "test"
    "#,
        tensorzero::test_helpers::get_gateway_endpoint("/")
    );

    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(&config).await;

    let response = client
        .inference(ClientInferenceParams {
            model_name: Some("tensorzero-relay-dummy".into()),
            input: ClientInput {
                system: Some(System::Text("My system prompt".to_string())),
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Hello, world!".into(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) = response else {
        panic!("Expected non-streaming chat response");
    };

    assert_eq!(
        response.content,
        vec![ContentBlockChatOutput::Text(Text {
            text: "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.".into()
        })]
    );

    let inference_id = response.inference_id;
    // Sleep to allow time for data to be inserted into ClickHouse
    sleep(Duration::from_secs(1)).await;

    // Check the ModelInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    assert_eq!(result["raw_request"], "{\"function_name\":\"basic_test_no_system_schema\",\"model_name\":null,\"episode_id\":null,\"input\":{\"system\":\"My system prompt\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hello, world!\"}]}]},\"stream\":false,\"params\":{\"chat_completion\":{\"json_mode\":\"off\"}},\"variant_name\":\"test\",\"dryrun\":null,\"internal\":false,\"tags\":{},\"provider_tools\":[],\"output_schema\":null,\"credentials\":{},\"cache_options\":{\"max_age_s\":null,\"enabled\":\"write_only\"},\"include_original_response\":false,\"extra_body\":[],\"extra_headers\":[],\"internal_dynamic_variant_config\":null}");
}
