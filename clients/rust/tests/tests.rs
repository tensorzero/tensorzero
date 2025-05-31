#![cfg(feature = "e2e_tests")]
#![expect(clippy::unwrap_used)]
use serde_json::json;
use tensorzero::{
    input_handling::resolved_input_to_client_input, ClientBuilder, ClientBuilderMode,
    ClientInferenceParams, ClientInput, ClientInputMessageContent, File,
};

use reqwest::Url;
use tensorzero_internal::inference::types::{FileKind, ResolvedInput};

lazy_static::lazy_static! {
    static ref GATEWAY_URL: String = std::env::var("GATEWAY_URL").unwrap_or("http://localhost:3000".to_string());
}

pub fn get_gateway_endpoint(endpoint: Option<&str>) -> Url {
    let base_url: Url = GATEWAY_URL.parse().unwrap();
    match endpoint {
        Some(endpoint) => base_url.join(endpoint).unwrap(),
        None => base_url,
    }
}

#[tokio::test]
async fn test_versioning() {
    std::env::set_var("TENSORZERO_E2E_GATEWAY_VERSION_OVERRIDE", "0.1.0");
    let client = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
        url: get_gateway_endpoint(None),
    })
    .build_http()
    .unwrap();
    let version = client.get_gateway_version().await;
    assert!(version.is_none());

    let client = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
        url: get_gateway_endpoint(None),
    })
    .build()
    .await
    .unwrap();
    let version = client.get_gateway_version().await;
    assert_eq!(version.unwrap(), "0.1.0");

    std::env::set_var("TENSORZERO_E2E_GATEWAY_VERSION_OVERRIDE", "0.2.0");
    client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            episode_id: None,
            input: ClientInput {
                system: Some(json!({"assistant_name": "John"})),
                messages: vec![],
            },
            ..Default::default()
        })
        .await
        .unwrap();
    let version = client.get_gateway_version().await;
    assert_eq!(version.unwrap(), "0.2.0");
}

#[tokio::test]
async fn test_conversion() {
    let client = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
        url: get_gateway_endpoint(None),
    })
    .build_http()
    .unwrap();
    // Taken from the database and contains an image
    let input = json!({"messages":[{"role":"user","content":[{"type":"text","value":"What kind of animal is in this image?"},{"type":"image","image":{"url":"https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png","mime_type":"image/png"},"storage_path":{"kind":{"type":"s3_compatible","bucket_name":"tensorzero-e2e-test-images","region":"us-east-1","endpoint":null,"allow_http":null,"prefix":""},"path":"observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png"}}]}]});
    let resolved_input: ResolvedInput = serde_json::from_value(input).unwrap();
    let client_input = resolved_input_to_client_input(resolved_input, &client)
        .await
        .unwrap();
    assert!(client_input.messages.len() == 1);
    assert!(client_input.messages[0].content.len() == 2);
    assert!(matches!(
        client_input.messages[0].content[0],
        ClientInputMessageContent::Text(_)
    ));
    let ClientInputMessageContent::File(File::Base64 { mime_type, data }) =
        &client_input.messages[0].content[1]
    else {
        panic!("Expected file");
    };
    assert_eq!(mime_type, &FileKind::Png);
    assert!(!data.is_empty());
}
