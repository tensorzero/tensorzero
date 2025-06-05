#![allow(clippy::print_stdout)]

use std::time::{Duration, Instant};

use http::StatusCode;
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use tensorzero_internal::clickhouse::test_helpers::{
    get_clickhouse, select_model_inference_clickhouse, select_model_inferences_clickhouse,
};
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[tokio::test]
async fn test_model_provider_timeout_non_streaming() {
    let payload = json!({
        "function_name": "basic_test_timeout",
        "variant_name": "timeout",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let response_json = response.json::<Value>().await.unwrap();
    assert_eq!(
        response_json,
        json!({
            "error":"All variants failed with errors: timeout: All model providers failed to infer with errors: slow: Model provider slow timed out due to configured `non_streaming.total_ms` timeout (400ms)"
        })
    );
}

#[tokio::test]
async fn test_model_provider_timeout_streaming() {
    let payload = json!({
        "function_name": "basic_test_timeout",
        "variant_name": "timeout",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    assert_eq!(
        response_json,
        json!({
            "error":"All variants failed with errors: timeout: All model providers failed to infer with errors: slow: Model provider slow timed out due to configured `streaming.ttft_ms` timeout (500ms)"
        })
    );
    assert_eq!(status, StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn test_model_provider_timeout_slow_second_chunk_streaming() {
    let payload = json!({
        "function_name": "basic_test_timeout",
        "variant_name": "slow_second_chunk",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    });

    let start = Instant::now();
    let mut response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut inference_id = None;

    while let Some(event) = response.next().await {
        let chunk = event.unwrap();
        println!("chunk: {chunk:?}");
        if let Event::Message(event) = chunk {
            if event.data == "[DONE]" {
                break;
            }
            let event = serde_json::from_str::<Value>(&event.data).unwrap();
            inference_id = Some(event["inference_id"].as_str().unwrap().parse().unwrap());
        }
    }

    // The overall stream duration should be at least 2 seconds, becaues we used the 'slow_second_chunk' model
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_millis(2000),
        "slow_second_chunk should take at least 2 seconds, but took {elapsed:?}"
    );

    // Wait 100ms to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let clickhouse = get_clickhouse().await;

    let model_inference = select_model_inference_clickhouse(&clickhouse, inference_id.unwrap())
        .await
        .unwrap();

    // The TTFT should be under 400ms (it should really be instant, but we give some buffer in case CI is overloaded)
    // As a result, the 'streaming.ttft_ms' timeout was not hit.
    let ttft_ms = model_inference["ttft_ms"].as_u64().unwrap();
    assert!(
        ttft_ms <= 400,
        "ttft_ms should be less than 400ms, but was {ttft_ms}"
    );
}

// Test that if a candidate times out, the evaluator can still see the other candidates
#[tokio::test]
async fn test_model_provider_timeout_best_of_n_other_candidate() {
    let payload = json!({
        "function_name": "basic_test_timeout",
        "variant_name": "best_of_n",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();

    let response_json = response.json::<Value>().await.unwrap();
    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(
        content.as_slice(),
        &[serde_json::json!({
            "type": "text",
            "text": "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
        })]
    );
    assert_eq!(status, StatusCode::OK);

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 100ms to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    let model_inferences = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    // One of the candidates timed out, leaving us with 2 candidates and the judge
    assert_eq!(model_inferences.len(), 3);
    assert_eq!(
        model_inferences[0]
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap(),
        inference_id.to_string()
    );

    let mut model_names = Vec::new();

    for model_inference in &model_inferences {
        let model_name = model_inference.get("model_name").unwrap().as_str().unwrap();
        model_names.push(model_name);
    }
    model_names.sort();
    assert_eq!(
        model_names,
        ["dummy::best_of_n_0", "dummy::good", "dummy::reasoner"]
    );

    // We don't check the content beyond the model names, as we already have lots
    // of tests for best_of_n
}

#[tokio::test]
async fn test_model_provider_timeout_best_of_n_judge_timeout() {
    let payload = json!({
        "function_name": "basic_test_timeout",
        "variant_name": "best_of_n_judge_timeout",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();

    // We don't check the content, as it will be a random candidate
    let response_json = response.json::<Value>().await.unwrap();
    assert_eq!(status, StatusCode::OK);

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 100ms to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    let model_inferences = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    // Both of the candidates should succeed, but the judge should time out,
    // leaving us with 2 successful inferences
    assert_eq!(model_inferences.len(), 2);
    assert_eq!(
        model_inferences[0]
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap(),
        inference_id.to_string()
    );

    let mut model_names = Vec::new();

    for model_inference in &model_inferences {
        let model_name = model_inference.get("model_name").unwrap().as_str().unwrap();
        model_names.push(model_name);
    }
    model_names.sort();
    assert_eq!(model_names, ["dummy::good", "dummy::reasoner"]);
}
