#![allow(clippy::print_stdout)]

mod common;
use serde_json::json;

use crate::common::start_gateway_on_random_port;

#[tokio::test]
async fn test_global_http_timeout() {
    let child_data =
        start_gateway_on_random_port("global_outbound_http_timeout_ms = 1", None).await;
    let inference_response = reqwest::Client::new()
        .post(format!("http://{}/inference", child_data.addr))
        .json(&json!({
            "model_name": "openai::fake-model-name",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, world!",
                    }
                ]
            }
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    println!("API response: {inference_response}");
    assert!(inference_response.contains("source: TimedOut"));
}
