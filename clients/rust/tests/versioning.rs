#![cfg(feature = "e2e_tests")]
#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
use serde_json::json;
use tensorzero::{ClientBuilder, ClientBuilderMode, ClientInferenceParams, ClientInput};

use reqwest::Url;

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
