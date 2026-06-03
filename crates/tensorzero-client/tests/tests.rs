#![cfg(feature = "e2e_tests")]
#![expect(clippy::unwrap_used, clippy::missing_panics_doc)]

use reqwest::Url;
use serde_json::json;
use tempfile::NamedTempFile;
use tensorzero::{
    ClientBuilder, ClientBuilderMode, File, InputMessageContent,
    input_handling::resolved_input_to_client_input,
};
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;
use tensorzero_core::inference::types::StoredInput;

mod test_configs;
mod test_datasets;
mod test_episodes;
mod test_stored_inferences;

lazy_static::lazy_static! {
    static ref GATEWAY_URL: String = std::env::var("TENSORZERO_GATEWAY_URL").unwrap_or_else(|_|"http://localhost:3000".to_string());
}

pub fn get_gateway_endpoint(endpoint: Option<&str>) -> Url {
    let base_url: Url = GATEWAY_URL.parse().unwrap();
    match endpoint {
        Some(endpoint) => base_url.join(endpoint).unwrap(),
        None => base_url,
    }
}

/// `[object_storage]` block pointing at the shared e2e S3 bucket
/// (`tensorzero-e2e-test-images`). The gateway only fetches from its own
/// configured store, so this test builds an embedded gateway whose
/// `[object_storage]` matches the `kind` of the `StoragePath` it resolves
/// rather than relying on the shared long-running gateway (which has
/// `object_storage.type = "disabled"` via the default
/// `object-storage-disabled.gateway.toml` e2e override).
const S3_OBJECT_STORAGE_CONFIG: &str = r#"
[object_storage]
type = "s3_compatible"
bucket_name = "tensorzero-e2e-test-images"
region = "us-east-1"
"#;

#[tokio::test(flavor = "multi_thread")]
async fn test_conversion() {
    // Embedded gateway with the S3 store the fixture lives in. We skip
    // credential verification — the probe write to `.tensorzero-validate`
    // would otherwise require write access to the shared test bucket, but
    // this test only needs to read an existing object.
    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), S3_OBJECT_STORAGE_CONFIG).unwrap();
    let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: None,
        valkey_url: None,
        timeout: None,
        verify_credentials: false,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap();
    // Taken from the database and contains an image
    let input = json!({"messages":[{"role":"user","content":[{"type":"text","value":"What kind of animal is in this image?"},{"type":"image","image":{"url":"https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png","mime_type":"image/png"},"storage_path":{"kind":{"type":"s3_compatible","bucket_name":"tensorzero-e2e-test-images","region":"us-east-1","endpoint":null,"allow_http":null,"prefix":""},"path":"observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png"}}]}]});
    let stored_input: StoredInput = serde_json::from_value(input).unwrap();
    let resolved_input = stored_input.reresolve(&client).await.unwrap();
    let client_input = resolved_input_to_client_input(resolved_input).unwrap();
    assert!(client_input.messages.len() == 1);
    assert!(client_input.messages[0].content.len() == 2);
    assert!(matches!(
        client_input.messages[0].content[0],
        InputMessageContent::Text(_)
    ));
    let InputMessageContent::File(File::Base64(base64_file)) = &client_input.messages[0].content[1]
    else {
        panic!("Expected file");
    };
    assert_eq!(&base64_file.mime_type, &mime::IMAGE_PNG);
    assert!(!base64_file.data().is_empty());
}
