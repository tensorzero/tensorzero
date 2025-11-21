use serde_json::json;
use tempfile::NamedTempFile;
use tensorzero::{
    Client, ClientBuilder, ClientBuilderMode, ClientInferenceParams, ClientInput,
    ClientInputMessage, ClientInputMessageContent, InferenceOutput, InferenceResponse,
};
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;
use tensorzero_core::inference::types::{ContentBlockChatOutput, Role, Text, TextKind};

// Helper function to create a test gateway with the given config
pub async fn create_test_gateway(config: &str) -> Client {
    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

// Helper function to create standard test input
pub fn create_test_input() -> ClientInput {
    ClientInput {
        messages: vec![ClientInputMessage {
            role: Role::User,
            content: vec![ClientInputMessageContent::Text(TextKind::Text {
                text: "test".to_string(),
            })],
        }],
        system: None,
    }
}

// Helper function to parse injected body from response
fn parse_injected_body(response: InferenceOutput) -> serde_json::Value {
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(chat_response)) = response else {
        panic!("Expected non-streaming chat response");
    };
    let output_text = match &chat_response.content[0] {
        ContentBlockChatOutput::Text(Text { text }) => text,
        _ => panic!("Expected text block"),
    };
    serde_json::from_str(output_text).unwrap()
}

// Standard config used by all tests
pub const STANDARD_CONFIG: &str = r#"
[models.my_echo_injected_data]
routing = ["dummy"]

[models.my_echo_injected_data.providers.dummy]
type = "dummy"
model_name = "echo_injected_data"

[functions.function_echo_injected_data_explicit]
type = "chat"

[functions.function_echo_injected_data_explicit.variants.default]
type = "chat_completion"
model = "my_echo_injected_data"

[functions.function_echo_injected_data_explicit.variants.variant_with_extra]
type = "chat_completion"
model = "my_echo_injected_data"

[functions.function_echo_injected_data_shorthand]
type = "chat"

[functions.function_echo_injected_data_shorthand.variants.default]
type = "chat_completion"
model = "dummy::echo_injected_data"
"#;

// ========== Function Name + Dynamic Extra Body ==========

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_always_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_explicit".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "pointer": "/test_field",
                "value": "always_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(injected["injected_body"]["test_field"], "always_works");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_variant_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_explicit".to_string()),
            variant_name: Some("variant_with_extra".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "pointer": "/variant_field",
                "value": "variant_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(injected["injected_body"]["variant_field"], "variant_works");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_provider_fully_qualified_config_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_explicit".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "model_provider_name": "tensorzero::model_name::my_echo_injected_data::provider_name::dummy",
                "pointer": "/provider_test",
                "value": "provider_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(injected["injected_body"]["provider_test"], "provider_works");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_provider_fully_qualified_shorthand_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_shorthand".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "model_provider_name": "tensorzero::model_name::dummy::echo_injected_data::provider_name::dummy",
                "pointer": "/shorthand_provider_test",
                "value": "shorthand_provider_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(
        injected["injected_body"]["shorthand_provider_test"],
        "shorthand_provider_works"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_model_provider_no_shorthand_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_explicit".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "model_name": "my_echo_injected_data",
                "provider_name": "dummy",
                "pointer": "/mp_test",
                "value": "mp_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(injected["injected_body"]["mp_test"], "mp_works");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_model_provider_shorthand_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_shorthand".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "model_name": "dummy::echo_injected_data",
                "provider_name": "dummy",
                "pointer": "/mp_shorthand_test",
                "value": "mp_shorthand_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(
        injected["injected_body"]["mp_shorthand_test"],
        "mp_shorthand_works"
    );
}

// ========== Model Name + Dynamic Extra Body ==========

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_always_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("my_echo_injected_data".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "pointer": "/test_field",
                "value": "always_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(injected["injected_body"]["test_field"], "always_works");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_provider_fully_qualified_config_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("my_echo_injected_data".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "model_provider_name": "tensorzero::model_name::my_echo_injected_data::provider_name::dummy",
                "pointer": "/provider_test",
                "value": "provider_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(injected["injected_body"]["provider_test"], "provider_works");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_provider_fully_qualified_shorthand_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::echo_injected_data".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "model_provider_name": "tensorzero::model_name::dummy::echo_injected_data::provider_name::dummy",
                "pointer": "/shorthand_provider_test",
                "value": "shorthand_provider_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(
        injected["injected_body"]["shorthand_provider_test"],
        "shorthand_provider_works"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_model_provider_no_shorthand_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("my_echo_injected_data".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "model_name": "my_echo_injected_data",
                "provider_name": "dummy",
                "pointer": "/mp_test",
                "value": "mp_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(injected["injected_body"]["mp_test"], "mp_works");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_model_provider_shorthand_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::echo_injected_data".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "model_name": "dummy::echo_injected_data",
                "provider_name": "dummy",
                "pointer": "/mp_shorthand_test",
                "value": "mp_shorthand_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(
        injected["injected_body"]["mp_shorthand_test"],
        "mp_shorthand_works"
    );
}

// ========== Optional provider_name Tests ==========

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_model_provider_no_provider_name_explicit_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_explicit".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "model_name": "my_echo_injected_data",
                "pointer": "/mp_no_provider_test",
                "value": "mp_no_provider_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(
        injected["injected_body"]["mp_no_provider_test"],
        "mp_no_provider_works"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_model_provider_no_provider_name_shorthand_function() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("function_echo_injected_data_shorthand".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "model_name": "dummy::echo_injected_data",
                "pointer": "/mp_no_provider_shorthand_test",
                "value": "mp_no_provider_shorthand_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(
        injected["injected_body"]["mp_no_provider_shorthand_test"],
        "mp_no_provider_shorthand_works"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_model_provider_no_provider_name_explicit_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("my_echo_injected_data".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "model_name": "my_echo_injected_data",
                "pointer": "/mp_no_provider_test",
                "value": "mp_no_provider_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(
        injected["injected_body"]["mp_no_provider_test"],
        "mp_no_provider_works"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_extra_body_model_provider_no_provider_name_shorthand_model() {
    let client = create_test_gateway(STANDARD_CONFIG).await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::echo_injected_data".to_string()),
            input: create_test_input(),
            extra_body: serde_json::from_value(json!([{
                "model_name": "dummy::echo_injected_data",
                "pointer": "/mp_no_provider_shorthand_test",
                "value": "mp_no_provider_shorthand_works"
            }]))
            .unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();

    let injected = parse_injected_body(result);
    assert_eq!(
        injected["injected_body"]["mp_no_provider_shorthand_test"],
        "mp_no_provider_shorthand_works"
    );
}
