#![expect(clippy::print_stdout)]
use crate::common::get_gateway_endpoint;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::clickhouse::test_helpers::select_chat_inference_clickhouse;
use uuid::Uuid;

#[tokio::test]
async fn e2e_test_dynamic_chat_variant() {
    let mut payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "internal_dynamic_variant_config":
            {"type": "chat_completion",
                "weight": 0., "model": "dummy::echo_request_messages", "system_template": {"__tensorzero_remapped_path": "system", "__data": "You are a cranky assistant named {{ assistant_name }}"}},
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // This should 400 since `dryrun` is not set
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    payload["dryrun"] = json!(true);
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("{response_json:#}");

    let content = response_json["content"].as_array().unwrap();
    let text_block = content.first().unwrap();
    let text = text_block["text"].as_str().unwrap();
    assert!(text.contains("You are a cranky assistant named AskJeeves"));

    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_none()); // No inference should be written to ClickHouse when dryrun is true
}

#[tokio::test]
async fn e2e_test_dynamic_mixture_of_n() {
    let mut payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "Alfred"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "internal_dynamic_variant_config": {
            "type": "experimental_mixture_of_n", "weight": 0., "candidates": ["test", "test2"], "fuser": {"weight": 0., "model": "dummy::echo_request_messages", "system_template": {"__tensorzero_remapped_path": "system", "__data":"be mean {{ assistant_name }}" }},
        },
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // This should 400 since `dryrun` is not set
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    payload["dryrun"] = json!(true);
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("{response_json:#}");

    let content = response_json["content"].as_array().unwrap();
    let text_block = content.first().unwrap();
    let text = text_block["text"].as_str().unwrap();
    assert!(text.contains("be mean Alfred"));
    assert!(text.contains("You have been provided with a set of responses"));
    assert!(text.contains("synthesize these responses into"));
    assert!(text.contains("gleefully chanted"));

    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_none()); // No inference should be written to ClickHouse when dryrun is true
}

#[tokio::test]
async fn e2e_test_dynamic_best_of_n() {
    let mut payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "Watson"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "internal_dynamic_variant_config": {
            "type": "experimental_best_of_n_sampling", "weight": 0., "candidates": ["test", "test2"], "evaluator": {"weight": 0., "model": "dummy::echo_request_messages", "system_template": {"__tensorzero_remapped_path": "system", "__data": "be mean {{ assistant_name }}"}}},
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // This should 400 since `dryrun` is not set
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    payload["dryrun"] = json!(true);
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("{response_json:#}");

    let content = response_json["content"].as_array().unwrap();
    let text_block = content.first().unwrap();
    let text = text_block["text"].as_str().unwrap();
    // The best of n thing will always pick a candidate
    assert!(text.contains("gleefully chanted"));

    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_none()); // No inference should be written to ClickHouse when dryrun is true
}
