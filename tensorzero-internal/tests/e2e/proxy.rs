use tensorzero_internal::gateway_util::setup_http_client;

/// Tests that the HTTP proxy is used if the TENSORZERO_E2E_PROXY environment variable is set.
#[tokio::test]
async fn test_setup_http_client() {
    let http_client = setup_http_client().unwrap();
    let response = http_client
        .get("https://www.tensorzero.com")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    // We want this test to pass if for some reason the TENSORZERO_E2E_PROXY environment variable is not set.
    if std::env::var("TENSORZERO_E2E_PROXY").is_ok() {
        let headers = response.headers();
        assert!(
            headers.contains_key("x-tensorzero-provider-proxy-cache"),
            "x-tensorzero-provider-proxy-cache header not found"
        );
    }
}

/// Tests that the HTTP proxy is not used if the TENSORZERO_E2E_PROXY environment variable is not set.
#[tokio::test]
async fn test_setup_http_client_no_proxy() {
    std::env::remove_var("TENSORZERO_E2E_PROXY");
    let http_client = setup_http_client().unwrap();
    let response = http_client
        .get("https://www.tensorzero.com")
        .send()
        .await
        .unwrap();
    let headers = response.headers();
    assert_eq!(response.status(), 200);
    assert!(!headers.contains_key("x-tensorzero-provider-proxy-cache"));
}
