use crate::e2e::common::clickhouse_flush_async_insert;
use api::clickhouse::ClickHouseConnectionInfo;
use api::inference::providers::dummy::DUMMY_INFER_RESPONSE_CONTENT;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use uuid::Uuid;

// TODO: make this endpoint configurable with main.rs
const INFERENCE_URL: &str = "http://localhost:3000/inference";
lazy_static::lazy_static! {
    static ref CLICKHOUSE_URL: String = std::env::var("CLICKHOUSE_URL").expect("CLICKHOUSE_URL must be set");
}

#[tokio::test]
async fn e2e_test_inference_basic() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
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
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that created is here
    response_json.get("created").unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let raw_content = response_json.get("raw_content").unwrap();
    assert_eq!(raw_content, content);
    // Check that tool_calls is null
    response_json.get("tool_calls").unwrap();
    // Check that type is "chat"
    let r#type = response_json.get("type").unwrap().as_str().unwrap();
    assert_eq!(r#type, "Chat");

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
    let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
    assert_eq!(prompt_tokens, 10);
    assert_eq!(completion_tokens, 10);
    // Sleep for 0.1 seconds to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse
    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, false, None);
    let result = select_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    println!("result: {}", result);
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, payload["input"]);
    let output = result.get("output").unwrap().as_str().unwrap();
    assert_eq!(output, DUMMY_INFER_RESPONSE_CONTENT);
    let output_raw = result.get("raw_output").unwrap().as_str().unwrap();
    assert_eq!(output_raw, DUMMY_INFER_RESPONSE_CONTENT);
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
}

#[tokio::test]
async fn e2e_test_inference_model_fallback() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "model_fallback_test",
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
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that created is here
    response_json.get("created").unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let raw_content = response_json.get("raw_content").unwrap();
    assert_eq!(raw_content, content);
    // Check that tool_calls is null
    response_json.get("tool_calls").unwrap();
    // Check that type is "chat"
    let r#type = response_json.get("type").unwrap().as_str().unwrap();
    assert_eq!(r#type, "Chat");

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
    let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
    assert_eq!(prompt_tokens, 10);
    assert_eq!(completion_tokens, 10);
    // Sleep for 0.1 seconds to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse
    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, false, None);
    let result = select_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    println!("result: {}", result);
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, payload["input"]);
    let output = result.get("output").unwrap().as_str().unwrap();
    assert_eq!(output, DUMMY_INFER_RESPONSE_CONTENT);
    let output_raw = result.get("raw_output").unwrap().as_str().unwrap();
    assert_eq!(output_raw, DUMMY_INFER_RESPONSE_CONTENT);
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
}

#[tokio::test]
async fn e2e_test_inference_json_fail() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_fail",
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
    assert!(content.is_null());
    // Check that created is here
    response_json.get("created").unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is present
    let raw_content = response_json.get("raw_content").unwrap();
    assert_eq!(raw_content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that tool_calls is null
    response_json.get("tool_calls").unwrap();
    // Check that type is "chat"
    let r#type = response_json.get("type").unwrap().as_str().unwrap();
    assert_eq!(r#type, "Chat");

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
    let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
    assert_eq!(prompt_tokens, 10);
    assert_eq!(completion_tokens, 10);
    // Sleep for 0.1 seconds to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse
    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, false, None);
    let result = select_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    println!("result: {}", result);
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, payload["input"]);
    let output = result.get("output").unwrap().as_str().unwrap();
    // TODO: handle the fact that this should be Optional
    assert_eq!(output, "");
    let output_raw = result.get("raw_output").unwrap().as_str().unwrap();
    assert_eq!(output_raw, DUMMY_INFER_RESPONSE_CONTENT);
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
}

async fn select_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;
    let (url, client) = match clickhouse_connection_info {
        ClickHouseConnectionInfo::Mock { .. } => unimplemented!(),
        ClickHouseConnectionInfo::Production { url, client } => (url.clone(), client),
    };
    println!("url: {}", url);
    println!("inference_id: {}", inference_id);
    let query = format!(
        "SELECT * FROM Inference WHERE id = '{}' FORMAT JSONEachRow",
        inference_id
    );
    let response = client
        .post(url)
        .body(query)
        .send()
        .await
        .expect("Failed to query ClickHouse");
    println!("status: {}", response.status());
    let text = response.text().await.ok()?;
    println!("text: {}", text);
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}
