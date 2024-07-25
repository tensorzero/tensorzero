use reqwest::{Client, StatusCode};

// TODO: make this endpoint configurable with main.rs
const HEALTH_URL: &str = "http://localhost:3000/health";

#[tokio::test]
async fn test_health_handler() {
    let client = Client::new();
    let response = client.get(HEALTH_URL).send().await;
    assert!(response.is_ok());
    assert_eq!(response.unwrap().status(), StatusCode::OK);
}
