use futures::StreamExt;
use gateway::inference::providers::dummy::{
    DUMMY_INFER_RESPONSE_CONTENT, DUMMY_STREAMING_RESPONSE,
};
use gateway::{
    clickhouse::ClickHouseConnectionInfo, inference::providers::dummy::DUMMY_JSON_RESPONSE_RAW,
};
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::e2e::common::clickhouse_flush_async_insert;

// TODO (#74): make this endpoint configurable with main.rs
const INFERENCE_URL: &str = "http://localhost:3000/inference";
lazy_static::lazy_static! {
    static ref CLICKHOUSE_URL: String = std::env::var("CLICKHOUSE_URL").expect("Environment variable CLICKHOUSE_URL must be set");
}

#[tokio::test]
async fn e2e_test_inference_basic() {
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

    let response = Client::new()
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
    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, false, None).unwrap();
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
    assert_eq!(output, DUMMY_INFER_RESPONSE_CONTENT);
    let output_raw = result.get("raw_output").unwrap().as_str().unwrap();
    assert_eq!(output_raw, DUMMY_INFER_RESPONSE_CONTENT);
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
}

#[tokio::test]
async fn e2e_test_inference_dryrun() {
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input":
            [
                {"role": "system", "content": {"assistant_name": "AskJeeves"}},
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ],
        "stream": false,
        "dryrun": true,
    });

    let response = Client::new()
        .post(INFERENCE_URL)
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 0.1 seconds to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse
    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, false, None).unwrap();
    let result = select_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_none()); // No inference should be written to ClickHouse when dryrun is true
}

/// This test calls a function which calls a model where the first provider is broken but
/// then the second provider works fine. We expect this request to work despite the first provider
/// being broken.
#[tokio::test]
async fn e2e_test_inference_model_fallback() {
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

    let response = Client::new()
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
    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, false, None).unwrap();
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
    assert_eq!(output, DUMMY_INFER_RESPONSE_CONTENT);
    let output_raw = result.get("raw_output").unwrap().as_str().unwrap();
    assert_eq!(output_raw, DUMMY_INFER_RESPONSE_CONTENT);
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
}

/// This test checks the return type and clickhouse writes for a function with an output schema and
/// a response which does not satisfy the schema.
/// We expect to see a null `content` field in the response and a null `output` field in the table.
#[tokio::test]
async fn e2e_test_inference_json_fail() {
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

    let response = Client::new()
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
    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, false, None).unwrap();
    let result = select_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, payload["input"]);
    let output = result.get("output").unwrap();
    assert!(output.is_null());
    let output_raw = result.get("raw_output").unwrap().as_str().unwrap();
    assert_eq!(output_raw, DUMMY_INFER_RESPONSE_CONTENT);
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
}

/// This test checks the return type and clickhouse writes for a function with an output schema and
/// a response which satisfies the schema.
/// We expect to see a filled-out `content` field in the response and a filled-out `output` field in the table.
#[tokio::test]
async fn e2e_test_inference_json_succeed() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_succeed",
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

    let response = Client::new()
        .post(INFERENCE_URL)
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let content = response_json.get("content").unwrap().as_object().unwrap();
    let answer = content.get("answer").unwrap().as_str().unwrap();
    assert_eq!(answer, "Hello");
    // Check that created is here
    response_json.get("created").unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is present
    let raw_content = response_json.get("raw_content").unwrap();
    assert_eq!(raw_content, DUMMY_JSON_RESPONSE_RAW);
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
    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, false, None).unwrap();
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
    // TODO (#89): handle the fact that this should be Optional
    assert_eq!(output, DUMMY_JSON_RESPONSE_RAW);
    let output_raw = result.get("raw_output").unwrap().as_str().unwrap();
    assert_eq!(output_raw, DUMMY_JSON_RESPONSE_RAW);
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
}

/// The variant_failover function has two variants: good and error, each with weight 0.5
/// We want to make sure that this does not fail despite the error variant failing every time
/// We do this by making several requests and checking that the response is 200 in each, then checking that
/// the response is correct for the last one.
#[tokio::test]
async fn e2e_test_variant_failover() {
    let mut last_response = None;
    let mut last_payload = None;
    let mut last_episode_id = None;
    for _ in 0..50 {
        let episode_id = Uuid::now_v7();

        let payload = json!({
            "function_name": "variant_failover",
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

        let response = Client::new()
            .post(INFERENCE_URL)
            .json(&payload)
            .send()
            .await
            .unwrap();
        // Check Response is OK, then fields in order
        assert_eq!(response.status(), StatusCode::OK);
        last_response = Some(response);
        last_payload = Some(payload);
        last_episode_id = Some(episode_id);
    }
    let response = last_response.unwrap();
    let payload = last_payload.unwrap();
    let episode_id = last_episode_id.unwrap();
    let response_json = response.json::<Value>().await.unwrap();
    // Check Response is OK, then fields in order
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
    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, false, None).unwrap();
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
    // TODO (#89): handle the fact that this should be Optional
    assert_eq!(output, DUMMY_INFER_RESPONSE_CONTENT);
    let output_raw = result.get("raw_output").unwrap().as_str().unwrap();
    assert_eq!(output_raw, DUMMY_INFER_RESPONSE_CONTENT);
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
}

/// This test checks that streaming inference works as expected.
#[tokio::test]
async fn e2e_test_streaming() {
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
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(INFERENCE_URL)
        .json(&payload)
        .eventsource()
        .unwrap();
    let mut chunks = vec![];
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    let mut inference_id = None;
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_json: Value = serde_json::from_str(chunk).unwrap();
        if i < DUMMY_STREAMING_RESPONSE.len() {
            let content = chunk_json.get("content").unwrap().as_str().unwrap();
            assert_eq!(content, DUMMY_STREAMING_RESPONSE[i]);
        } else {
            assert!(chunk_json.get("content").unwrap().is_null());
            let usage = chunk_json.get("usage").unwrap().as_object().unwrap();
            let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
            let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
            assert_eq!(prompt_tokens, 10);
            assert_eq!(completion_tokens, 16);
            inference_id = Some(
                Uuid::parse_str(chunk_json.get("inference_id").unwrap().as_str().unwrap()).unwrap(),
            );
        }
    }
    let inference_id = inference_id.unwrap();
    // Sleep for 0.1 seconds to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse
    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, false, None).unwrap();
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
    assert_eq!(output, DUMMY_STREAMING_RESPONSE.join(""));
    let output_raw = result.get("raw_output").unwrap().as_str().unwrap();
    assert_eq!(output_raw, DUMMY_STREAMING_RESPONSE.join(""));
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
}

/// This test checks that streaming inference works as expected when dryrun is true.
#[tokio::test]
async fn e2e_test_streaming_dryrun() {
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input":
            [
                {"role": "system", "content": {"assistant_name": "AskJeeves"}},
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ],
        "stream": true,
        "dryrun": true,
    });

    let mut event_source = Client::new()
        .post(INFERENCE_URL)
        .json(&payload)
        .eventsource()
        .unwrap();
    let mut chunks = vec![];
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    let mut inference_id = None;
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_json: Value = serde_json::from_str(chunk).unwrap();
        if i < DUMMY_STREAMING_RESPONSE.len() {
            let content = chunk_json.get("content").unwrap().as_str().unwrap();
            assert_eq!(content, DUMMY_STREAMING_RESPONSE[i]);
        } else {
            assert!(chunk_json.get("content").unwrap().is_null());
            let usage = chunk_json.get("usage").unwrap().as_object().unwrap();
            let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
            let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
            assert_eq!(prompt_tokens, 10);
            assert_eq!(completion_tokens, 16);
            inference_id = Some(
                Uuid::parse_str(chunk_json.get("inference_id").unwrap().as_str().unwrap()).unwrap(),
            );
        }
    }
    let inference_id = inference_id.unwrap();

    // Sleep for 0.1 seconds to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse
    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, false, None).unwrap();
    let result = select_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_none()); // No inference should be written to ClickHouse when dryrun is true
}

async fn select_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM Inference WHERE id = '{}' FORMAT JSONEachRow",
        inference_id
    );

    let text = clickhouse_connection_info.run_query(query).await.unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}
