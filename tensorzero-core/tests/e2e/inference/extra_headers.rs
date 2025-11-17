use serde_json::json;
use tensorzero::{ClientInferenceParams, InferenceOutput, InferenceResponse};
use tensorzero_core::inference::types::{ContentBlockChatOutput, Text};

// Import shared helpers and config from extra_body
use super::extra_body::{create_test_gateway, create_test_input, STANDARD_CONFIG};

// Helper function to parse injected headers from response
fn parse_injected_headers(response: InferenceOutput) -> Vec<(String, String)> {
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(chat_response)) = response else {
        panic!("Expected non-streaming chat response");
    };
    let output_text = match &chat_response.content[0] {
        ContentBlockChatOutput::Text(Text { text }) => text,
        _ => panic!("Expected text block"),
    };
    let injected: serde_json::Value = serde_json::from_str(output_text).unwrap();
    serde_json::from_value(injected["injected_headers"].clone()).unwrap()
}

// ========== Function Name + Dynamic Extra Headers ==========

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_always_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_explicit".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "name": "X-Always-Header",
                "value": "always_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-always-header" && v == "always_works"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_variant_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_explicit".to_string()),
            variant_name: Some("variant_with_extra".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "name": "X-Variant-Header",
                "value": "variant_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-variant-header" && v == "variant_works"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_provider_fully_qualified_config_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_explicit".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "model_provider_name": "tensorzero::model_name::my_echo_injected_data::provider_name::dummy",
                "name": "X-Provider-Header",
                "value": "provider_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-provider-header" && v == "provider_works"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_provider_fully_qualified_shorthand_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_shorthand".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "model_provider_name": "tensorzero::model_name::dummy::echo_injected_data::provider_name::dummy",
                "name": "X-Shorthand-Provider-Header",
                "value": "shorthand_provider_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-shorthand-provider-header" && v == "shorthand_provider_works"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_model_provider_no_shorthand_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_explicit".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "model_name": "my_echo_injected_data",
                "provider_name": "dummy",
                "name": "X-MP-Header",
                "value": "mp_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-mp-header" && v == "mp_works"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_model_provider_shorthand_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_shorthand".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "model_name": "dummy::echo_injected_data",
                "provider_name": "dummy",
                "name": "X-MP-Shorthand-Header",
                "value": "mp_shorthand_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-mp-shorthand-header" && v == "mp_shorthand_works"));
}

// ========== Model Name + Dynamic Extra Headers ==========

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_always_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("my_echo_injected_data".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "name": "X-Always-Header",
                "value": "always_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-always-header" && v == "always_works"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_provider_fully_qualified_config_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("my_echo_injected_data".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "model_provider_name": "tensorzero::model_name::my_echo_injected_data::provider_name::dummy",
                "name": "X-Provider-Header",
                "value": "provider_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-provider-header" && v == "provider_works"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_provider_fully_qualified_shorthand_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::echo_injected_data".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "model_provider_name": "tensorzero::model_name::dummy::echo_injected_data::provider_name::dummy",
                "name": "X-Shorthand-Provider-Header",
                "value": "shorthand_provider_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-shorthand-provider-header" && v == "shorthand_provider_works"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_model_provider_no_shorthand_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("my_echo_injected_data".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "model_name": "my_echo_injected_data",
                "provider_name": "dummy",
                "name": "X-MP-Header",
                "value": "mp_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-mp-header" && v == "mp_works"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_model_provider_shorthand_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::echo_injected_data".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "model_name": "dummy::echo_injected_data",
                "provider_name": "dummy",
                "name": "X-MP-Shorthand-Header",
                "value": "mp_shorthand_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-mp-shorthand-header" && v == "mp_shorthand_works"));
}

// ========== Optional provider_name Tests ==========

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_model_provider_no_provider_name_explicit_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_explicit".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "model_name": "my_echo_injected_data",
                "name": "X-MP-No-Provider-Header",
                "value": "mp_no_provider_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-mp-no-provider-header" && v == "mp_no_provider_works"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_model_provider_no_provider_name_shorthand_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_shorthand".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "model_name": "dummy::echo_injected_data",
                "name": "X-MP-No-Provider-Shorthand-Header",
                "value": "mp_no_provider_shorthand_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-mp-no-provider-shorthand-header"
            && v == "mp_no_provider_shorthand_works"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_model_provider_no_provider_name_explicit_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("my_echo_injected_data".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "model_name": "my_echo_injected_data",
                "name": "X-MP-No-Provider-Header",
                "value": "mp_no_provider_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-mp-no-provider-header" && v == "mp_no_provider_works"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_headers_model_provider_no_provider_name_shorthand_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::echo_injected_data".to_string()),
            input: create_test_input(),
            extra_headers: serde_json::from_value(json!([{
                "model_name": "dummy::echo_injected_data",
                "name": "X-MP-No-Provider-Shorthand-Header",
                "value": "mp_no_provider_shorthand_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let headers = parse_injected_headers(result);
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-mp-no-provider-shorthand-header"
            && v == "mp_no_provider_shorthand_works"));
}
