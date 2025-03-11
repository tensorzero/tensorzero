#![allow(clippy::print_stdout)]

use reqwest::{Client, StatusCode};
use serde_json::Value;
use tensorzero_internal::endpoints::datasets::CLICKHOUSE_DATETIME_FORMAT;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use tensorzero_internal::clickhouse::test_helpers::{
    get_clickhouse, select_chat_datapoint_clickhouse, select_json_datapoint_clickhouse,
};

#[tokio::test]
async fn test_datapoint_insert_output_inherit_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = serde_json::json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints"
        )))
        .json(&serde_json::json!({
            "inference_id": inference_id,
            "output": "inherit"
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        panic!("Bad request: {:?}", resp.text().await.unwrap());
    }

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    let expected = serde_json::json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": inference_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"Hello, world!\"}]}]}",
      "output": "[{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]",
      "tool_params": "",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": false
    });
    assert_eq!(datapoint, expected);

    let resp = client
        .delete(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/function/basic_test/kind/chat/datapoint/{inference_id}",
        )))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_text = resp.text().await.unwrap();
    assert_eq!(status, StatusCode::OK, "Delete failed: {resp_text}");

    // Force deduplication to run
    clickhouse
        .run_query("OPTIMIZE TABLE ChatInferenceDatapoint".to_string(), None)
        .await
        .unwrap();

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let new_updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let new_updated_at = chrono::NaiveDateTime::parse_from_str(
        new_updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(new_updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {new_updated_at:?}"
    );

    let expected = serde_json::json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": inference_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"Hello, world!\"}]}]}",
      "output": "[{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]",
      "tool_params": "",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": true
    });
    assert_eq!(datapoint, expected);
    assert_ne!(
        updated_at, new_updated_at,
        "Deleting datapoint should change updated_at"
    );
}

#[tokio::test]
async fn test_bad_delete_datapoint() {
    let client = Client::new();

    let id = Uuid::now_v7();
    let resp = client
        .delete(get_gateway_endpoint(&format!(
            "/datasets/missing/function/basic_test/kind/chat/datapoint/{id}",
        )))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "Delete should have failed: {resp}"
    );
    assert_eq!(
        resp,
        serde_json::json!({
            "error": format!("Datapoint not found with params DeletePathParams {{ dataset: \"missing\", function: \"basic_test\", kind: Chat, id: {id} }}")
        })
    );
}

#[tokio::test]
async fn test_datapoint_insert_output_none_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = serde_json::json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints"
        )))
        .json(&serde_json::json!({
            "inference_id": inference_id,
            "output": "none"
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        panic!("Bad request: {:?}", resp.text().await.unwrap());
    }

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    let expected = serde_json::json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": inference_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"Hello, world!\"}]}]}",
      "output": null,
      "tool_params": "",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": false
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_output_demonstration_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = serde_json::json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&serde_json::json!({
            "inference_id": inference_id,
            "metric_name": "demonstration",
            "value": [{"type": "text", "text": "My demonstration chat answer"}],
        }))
        .send()
        .await
        .unwrap();
    if !response.status().is_success() {
        panic!("Bad request: {:?}", response.text().await.unwrap());
    }

    // Sleep to allow writing demonstration before making datapoint
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints"
        )))
        .json(&serde_json::json!({
            "inference_id": inference_id,
            "output": "demonstration"
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        panic!("Bad request: {:?}", resp.text().await.unwrap());
    }

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    println!(
        "Datapoint: {}",
        serde_json::to_string_pretty(&datapoint).unwrap()
    );

    let expected = serde_json::json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": inference_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"Hello, world!\"}]}]}",
      "output": "[{\"type\":\"text\",\"text\":\"My demonstration chat answer\"}]",
      "tool_params": "",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": false
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_output_inherit_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = serde_json::json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": {"country": "Japan"}}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints"
        )))
        .json(&serde_json::json!({
            "inference_id": inference_id,
            "output": "inherit"
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        panic!("Bad request: {:?}", resp.text().await.unwrap());
    }

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    let expected = serde_json::json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": inference_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":{\"country\":\"Japan\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"Hello\\\"}\",\"parsed\":{\"answer\":\"Hello\"}}",
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": false,
    });
    assert_eq!(datapoint, expected);

    let resp = client
        .delete(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/function/json_success/kind/json/datapoint/{inference_id}",
        )))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_text = resp.text().await.unwrap();
    assert_eq!(status, StatusCode::OK, "Delete failed: {resp_text}");

    // Force deduplication to run
    clickhouse
        .run_query("OPTIMIZE TABLE JsonInferenceDatapoint".to_string(), None)
        .await
        .unwrap();

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let new_updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let new_updated_at = chrono::NaiveDateTime::parse_from_str(
        new_updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(new_updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {new_updated_at:?}"
    );

    let expected = serde_json::json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": inference_id,
      "episode_id": episode_id,
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":{\"country\":\"Japan\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"Hello\\\"}\",\"parsed\":{\"answer\":\"Hello\"}}",
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": true
    });
    assert_eq!(
        datapoint,
        expected,
        "Unexpected datapoint: {}",
        serde_json::to_string_pretty(&datapoint).unwrap()
    );
    assert_ne!(
        updated_at, new_updated_at,
        "Deleting datapoint should change updated_at"
    );
}

#[tokio::test]
async fn test_datapoint_insert_output_none_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = serde_json::json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": {"country": "Japan"}}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints"
        )))
        .json(&serde_json::json!({
            "inference_id": inference_id,
            "output": "none"
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        panic!("Bad request: {:?}", resp.text().await.unwrap());
    }

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    let expected = serde_json::json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": inference_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":{\"country\":\"Japan\"}}]}]}",
      "output":null,
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": false,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_output_demonstration_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = serde_json::json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": {"country": "Japan"}}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&serde_json::json!({
            "inference_id": inference_id,
            "metric_name": "demonstration",
            "value": {"answer": "My demonstration answer"},
        }))
        .send()
        .await
        .unwrap();
    if !response.status().is_success() {
        panic!("Bad request: {:?}", response.text().await.unwrap());
    }

    // Sleep to allow writing demonstration before making datapoint
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints"
        )))
        .json(&serde_json::json!({
            "inference_id": inference_id,
            "output": "demonstration"
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        panic!("Bad request: {:?}", resp.text().await.unwrap());
    }

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let updated_at = datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();
    let updated_at = chrono::NaiveDateTime::parse_from_str(
        updated_at.as_str().unwrap(),
        CLICKHOUSE_DATETIME_FORMAT,
    )
    .unwrap()
    .and_utc();
    assert!(
        chrono::Utc::now()
            .signed_duration_since(updated_at)
            .num_seconds()
            < 5,
        "Unexpected updated_at: {updated_at:?}"
    );

    println!(
        "Datapoint: {}",
        serde_json::to_string_pretty(&datapoint).unwrap()
    );

    let expected = serde_json::json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": inference_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":{\"country\":\"Japan\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"My demonstration answer\\\"}\",\"parsed\":{\"answer\":\"My demonstration answer\"}}",
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": false,
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_missing_inference_id() {
    let client = Client::new();
    let fake_inference_id = Uuid::now_v7();
    let resp = client
        .post(get_gateway_endpoint("/datasets/dummy-dataset/datapoints"))
        .json(&serde_json::json!({
            "inference_id": fake_inference_id,
            "output": "inherit"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains(&format!("Inference `{fake_inference_id}` not found")),
        "Unexpected response: {body}"
    );
}

#[tokio::test]
async fn test_datapoint_missing_demonstration() {
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = serde_json::json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": {"country": "Japan"}}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints"
        )))
        .json(&serde_json::json!({
            "inference_id": inference_id,
            "output": "demonstration"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains(&format!(
            "No demonstration found for inference `{inference_id}`"
        )),
        "Unexpected response: {body}"
    );
}
