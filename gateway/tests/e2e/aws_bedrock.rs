use gateway::clickhouse::ClickHouseConnectionInfo;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::e2e::common::{select_inference_clickhouse, select_model_inferences_clickhouse};

/// AWS Bedrock E2E tests
///
/// This file contains a set of tests that check that the AWS Bedrock integration works and the appropriate stuff is written to DB.
///
/// Currently we test:
/// - basic inference
///
/// TODOs (#81):
/// - streaming inference
/// - tool calling
/// - JSON mode
/// - other API parameters (temp, max_tokens, etc.)

// TODO (#74): make this endpoint configurable with some kind of env var
const INFERENCE_URL: &str = "http://localhost:3000/inference";
lazy_static::lazy_static! {
    static ref CLICKHOUSE_URL: String = std::env::var("CLICKHOUSE_URL").expect("Environment variable CLICKHOUSE_URL must be set");
}

#[tokio::test]
async fn test_inference_basic() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "aws-bedrock",
        "episode_id": episode_id,
        "input":
            [
                {"role": "system", "content": {"assistant_name": "AskJeeves"}},
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ],
        "stream": false,
    });

    let response = client
        .post(INFERENCE_URL)
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let content = response_json.get("content").unwrap();
    let content = content.as_str().unwrap();
    // Check that created is here
    response_json.get("created").unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content (no output schema)
    let raw_content = response_json.get("raw_content").unwrap();
    assert_eq!(raw_content, content);
    // Check that tool_calls is null
    response_json.get("tool_calls").unwrap();
    // Check that type is "chat"
    let r#type = response_json.get("type").unwrap().as_str().unwrap();
    assert_eq!(r#type, "Chat");

    // Sleep for 0.1 seconds to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse
    let clickhouse =
        ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, "e2e_tensorzero", false, None).unwrap();
    // First, check Inference table
    let result = select_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, payload["input"]);
    let output = result.get("output").unwrap().as_str().unwrap();
    assert_eq!(output, content);
    let output_raw = result.get("raw_output").unwrap().as_str().unwrap();
    assert_eq!(output_raw, content);
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "aws-bedrock");
    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    // Bedrock can't be faster than this
    assert!(processing_time_ms > 100);

    // Next, check ModelInference table
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, payload["input"]);
    let input_tokens: u64 = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens: u64 = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);
    let output = result.get("output").unwrap().as_str().unwrap();
    assert_eq!(output, content);
    let _ = result.get("raw_response").unwrap().as_str().unwrap();
    // This is Debug output, not a nice JSON unfortunately
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 100);
    assert!(result.get("ttft_ms").unwrap().is_null());
}
