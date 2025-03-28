#![allow(clippy::print_stdout)]

use std::time::Duration;

use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tensorzero_internal::endpoints::datasets::CLICKHOUSE_DATETIME_FORMAT;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use tensorzero_internal::clickhouse::test_helpers::{
    get_clickhouse, select_chat_datapoint_clickhouse, select_json_datapoint_clickhouse,
};

#[tokio::test]
async fn test_datapoint_insert_synthetic_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "text", "value": "My synthetic input"}]}]},
            "output": [{"type": "text", "text": "My synthetic output"}],
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();

    if !status.is_success() {
        panic!("Bad request: {:?}", resp_json);
    }

    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    assert_eq!(id, datapoint_id);

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, id)
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

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": id.to_string(),
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"My synthetic input\"}]}]}",
      "output": "[{\"type\":\"text\",\"text\":\"My synthetic output\"}]",
      "tool_params": "",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_synthetic_chat_with_tools() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Define the tool params once to avoid duplication
    let tool_params = json!({
        "tools_available": [
            {
                "description": "Get the current temperature in a given location",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for (e.g. \"New York\")"
                        },
                        "units": {
                            "type": "string",
                            "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",
                            "enum": ["fahrenheit", "celsius"]
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                },
                "name": "get_temperature",
                "strict": false
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": false
    });

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Dummy"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "value": "My synthetic input"}
                        ]
                    }
                ]
            },
            "output": [
                {"type": "tool_call", "name": "get_humidity", "arguments": {"location": "New York", "units": "fahrenheit"}}
            ],
            "tool_params": tool_params,
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();

    // This should fail because the tool call is not in the tool_params
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let err_msg = resp_json.get("error").unwrap().as_str().unwrap();
    println!("Error: {}", err_msg);
    assert!(
        err_msg.contains("Demonstration contains invalid tool name"),
        "Unexpected error message: {err_msg}"
    );

    // Next we check invalid arguments
    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Dummy"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "value": "My synthetic input"}
                        ]
                    }
                ]
            },
            "output": [
                {"type": "tool_call", "name": "get_temperature", "arguments": {"city": "New York", "units": "fahrenheit"}}
            ],
            "tool_params": tool_params,
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();

    // This request is correct
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let err_msg = resp_json.get("error").unwrap().as_str().unwrap();
    println!("Error: {}", err_msg);
    assert!(
        err_msg.contains("Demonstration contains invalid tool call arguments"),
        "Unexpected error message: {err_msg}"
    );

    let resp = client
    .put(get_gateway_endpoint(&format!(
        "/datasets/{dataset_name}/datapoints/{datapoint_id}",
    )))
    .json(&json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "Dummy"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "value": "My synthetic input"}
                    ]
                }
            ]
        },
        "output": [
            {"type": "tool_call", "name": "get_temperature", "arguments": {"location": "New York", "units": "fahrenheit"}}
        ],
        "tool_params": tool_params,
    }))
    .send()
    .await
    .unwrap();

    let status = resp.status();
    assert!(status.is_success());
    let resp_json = resp.json::<Value>().await.unwrap();

    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    assert_eq!(id, datapoint_id);

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, id)
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
    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": id.to_string(),
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"My synthetic input\"}]}]}",
      "output": "[{\"type\":\"tool_call\",\"arguments\":{\"location\":\"New York\",\"units\":\"fahrenheit\"},\"id\":\"\",\"name\":\"get_temperature\",\"raw_arguments\":\"{\\\"location\\\":\\\"New York\\\",\\\"units\\\":\\\"fahrenheit\\\"}\",\"raw_name\":\"get_temperature\"}]",
      "tool_params": "{\"tools_available\":[{\"description\":\"Get the current temperature in a given location\",\"parameters\":{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the temperature for (e.g. \\\"New York\\\")\"},\"units\":{\"type\":\"string\",\"description\":\"The units to get the temperature in (must be \\\"fahrenheit\\\" or \\\"celsius\\\")\",\"enum\":[\"fahrenheit\",\"celsius\"]}},\"required\":[\"location\"],\"additionalProperties\":false},\"name\":\"get_temperature\",\"strict\":false}],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_synthetic_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "text", "arguments": {"country": "US"}}]}]},
            "output": {"answer": "Hello"},
            "output_schema": {},
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();

    if !status.is_success() {
        panic!("Bad request: {:?}", resp_json);
    }

    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(id, datapoint_id);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, id)
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

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": id,
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":{\"country\":\"US\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"Hello\\\"}\",\"parsed\":{\"answer\":\"Hello\"}}",
      "output_schema": "{}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);

    // Sleep to ensure that we get a different `updated_at` timestamp
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // Now, update the existing datapoint and verify that it changes in clickhouse
    // Test updating with a different output schema (this should fail)
    let new_resp = client
    .put(get_gateway_endpoint(&format!(
        "/datasets/{dataset_name}/datapoints/{datapoint_id}",
    )))
    .json(&json!({
        "function_name": "json_success",
        "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "text", "arguments": {"country": "US"}}]}]},
        "output": {"answer": "New answer"},
        "output_schema": {"type": "object", "properties": {"confidence": {"type": "number"}}, "required": ["confidence"]},
    }))
    .send()
    .await
    .unwrap();
    assert_eq!(new_resp.status(), StatusCode::BAD_REQUEST);
    let resp_json = new_resp.json::<Value>().await.unwrap();
    let err_msg = resp_json.get("error").unwrap().as_str().unwrap();
    assert!(
        err_msg.contains("Demonstration does not fit function output schema"),
        "Unexpected error message: {err_msg}"
    );
    assert!(
        err_msg.contains("\"confidence\" is a required property"),
        "Error should mention the missing required property: {err_msg}"
    );

    // Sleep to ensure that we get a different `updated_at` timestamp
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // Now try with the correct schema
    let new_resp = client
    .put(get_gateway_endpoint(&format!(
        "/datasets/{dataset_name}/datapoints/{datapoint_id}",
    )))
    .json(&json!({
        "function_name": "json_success",
        "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "text", "arguments": {"country": "US"}}]}]},
        "output": {"answer": "New answer"},
        "output_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"],
            "additionalProperties": false
        },
    }))
    .send()
    .await
    .unwrap();

    assert!(
        new_resp.status().is_success(),
        "Bad request: {:?}",
        new_resp.text().await.unwrap()
    );

    // Force deduplication to run
    clickhouse
        .run_query("OPTIMIZE TABLE JsonInferenceDatapoint".to_string(), None)
        .await
        .unwrap();

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, id)
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

    assert!(
        new_updated_at > updated_at,
        "Expected updated_at to change: new_updated_at={new_updated_at:?} updated_at={updated_at:?}"
    );

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": id,
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":{\"country\":\"US\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"New answer\\\"}\",\"parsed\":{\"answer\":\"New answer\"}}",
      "output_schema": "{}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_bad_name() {
    let client = Client::new();
    let dataset_name = "builder";
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "text", "arguments": {"country": "US"}}]}]},
            "output": {"answer": "Hello"},
            "output_schema": {},
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let err_msg = resp_json.get("error").unwrap().as_str().unwrap();
    assert!(
        err_msg.contains("Invalid dataset name"),
        "Unexpected error message: {err_msg}"
    );
}

#[tokio::test]
async fn test_datapoint_insert_invalid_input_synthetic_chat() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "variant_failover",
            "input": {"system": {"assistant_name": "Ferris"}, "messages": [{"role": "user", "content": [{"type": "text", "value": "My synthetic input"}]}]},
            "output": "Not a json object",
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();
    let err_msg = resp_json
        .get("error")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        err_msg.contains("\"My synthetic input\" is not of type \"object\""),
        "Unexpected response: {resp_json}"
    );
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_datapoint_insert_invalid_input_synthetic_json() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Ferris"}, "messages": [{"role": "user", "content": [{"type": "text", "value": "My synthetic input"}]}]},
            "output": "Not a json object",
            "output_schema": {},
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();
    let err_msg = resp_json
        .get("error")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        err_msg.contains("\"My synthetic input\" is not of type \"object\""),
        "Unexpected response: {resp_json}"
    );
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_datapoint_insert_invalid_output_synthetic_json() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Ferris"}, "messages": [{"role": "user", "content": [{"type": "text", "arguments": {"country": "US"}}]}]},
            "output": "Not a json object",
            "output_schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();
    let err_msg = resp_json
        .get("error")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        err_msg.contains("is not of type \"object\""),
        "Unexpected response: {resp_json}"
    );
    assert!(
        err_msg.contains("\"Not a json object\" is not of type \"object\""),
        "Unexpected response: {resp_json}"
    );
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_datapoint_insert_synthetic_bad_uuid() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_uuid_v4 = "1b3e1480-a24a-4a69-942e-da960f9045fc";

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_uuid_v4}",
        )))
        .json(&json!({
            "function_name": "basic_test",
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        resp_json,
        json!({
            "error": "Invalid Datapoint ID: Version must be 7, got 4",
        })
    );
}

#[tokio::test]
async fn test_datapoint_insert_synthetic_bad_params() {
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let resp_json = resp.json::<Value>().await.unwrap();
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        resp_json,
        json!({
            "error": "Failed to deserialize chat datapoint: missing field `input`",
        })
    );
}

#[tokio::test]
async fn test_datapoint_insert_output_inherit_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
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
        .json(&json!({
            "inference_id": inference_id,
            "output": "inherit"
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        panic!("Bad request: {:?}", resp.text().await.unwrap());
    }
    let datapoint_id =
        Uuid::parse_str(resp.json::<Value>().await.unwrap()["id"].as_str().unwrap()).unwrap();
    assert_ne!(datapoint_id, inference_id);

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, datapoint_id)
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

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"Hello, world!\"}]}]}",
      "output": "[{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]",
      "tool_params": "",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);

    let resp = client
        .delete(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/function/basic_test/kind/chat/datapoint/{datapoint_id}",
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

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, datapoint_id)
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

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"Hello, world!\"}]}]}",
      "output": "[{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]",
      "tool_params": "",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": true,
      "staled_at": null
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
        json!({
            "error": format!("Datapoint not found with params DeletePathParams {{ dataset: \"missing\", function: \"basic_test\", kind: Chat, id: {id} }}")
        })
    );
}

#[tokio::test]
async fn test_datapoint_insert_output_none_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
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
        .json(&json!({
            "inference_id": inference_id,
            "output": "none"
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        panic!("Bad request: {:?}", resp.text().await.unwrap());
    }
    let datapoint_id =
        Uuid::parse_str(resp.json::<Value>().await.unwrap()["id"].as_str().unwrap()).unwrap();
    assert_ne!(datapoint_id, inference_id);
    tokio::time::sleep(Duration::from_millis(500)).await;

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, datapoint_id)
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

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"Hello, world!\"}]}]}",
      "output": null,
      "tool_params": "",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_create_bad_name() {
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
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
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let dataset_name = "builder";

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "inference_id": inference_id,
            "output": "none"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let err_msg = resp.text().await.unwrap();
    assert!(
        err_msg.contains("Invalid dataset name"),
        "Unexpected error message: {err_msg}"
    );
}

#[tokio::test]
async fn test_datapoint_insert_output_demonstration_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
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
        .json(&json!({
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

    // Sleep to ensure that we wrote to ClickHouse
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "inference_id": inference_id,
            "output": "demonstration"
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        panic!("Bad request: {:?}", resp.text().await.unwrap());
    }
    let datapoint_id =
        Uuid::parse_str(resp.json::<Value>().await.unwrap()["id"].as_str().unwrap()).unwrap();
    assert_ne!(datapoint_id, inference_id);

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, datapoint_id)
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

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"Hello, world!\"}]}]}",
      "output": "[{\"type\":\"text\",\"text\":\"My demonstration chat answer\"}]",
      "tool_params": "",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_output_inherit_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
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
        .json(&json!({
            "inference_id": inference_id,
            "output": "inherit"
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        panic!("Bad request: {:?}", resp.text().await.unwrap());
    }
    let datapoint_id =
        Uuid::parse_str(resp.json::<Value>().await.unwrap()["id"].as_str().unwrap()).unwrap();
    assert_ne!(datapoint_id, inference_id);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, datapoint_id)
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

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":{\"country\":\"Japan\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"Hello\\\"}\",\"parsed\":{\"answer\":\"Hello\"}}",
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);

    let resp = client
        .delete(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/function/json_success/kind/json/datapoint/{datapoint_id}",
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

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, datapoint_id)
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

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": datapoint_id,
      "episode_id": episode_id,
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":{\"country\":\"Japan\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"Hello\\\"}\",\"parsed\":{\"answer\":\"Hello\"}}",
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": true,
      "staled_at": null
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
    let inference_payload = json!({
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
        .json(&json!({
            "inference_id": inference_id,
            "output": "none"
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        panic!("Bad request: {:?}", resp.text().await.unwrap());
    }
    let datapoint_id =
        Uuid::parse_str(resp.json::<Value>().await.unwrap()["id"].as_str().unwrap()).unwrap();
    assert_ne!(datapoint_id, inference_id);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, datapoint_id)
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

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":{\"country\":\"Japan\"}}]}]}",
      "output":null,
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_output_demonstration_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = json!({
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
        .json(&json!({
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
        .json(&json!({
            "inference_id": inference_id,
            "output": "demonstration"
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        panic!("Bad request: {:?}", resp.text().await.unwrap());
    }
    let datapoint_id =
        Uuid::parse_str(resp.json::<Value>().await.unwrap()["id"].as_str().unwrap()).unwrap();
    assert_ne!(datapoint_id, inference_id);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, datapoint_id)
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

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": datapoint_id.to_string(),
      "episode_id": episode_id.to_string(),
      "input": "{\"system\":{\"assistant_name\":\"Alfred Pennyworth\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":{\"country\":\"Japan\"}}]}]}",
      "output": "{\"raw\":\"{\\\"answer\\\":\\\"My demonstration answer\\\"}\",\"parsed\":{\"answer\":\"My demonstration answer\"}}",
      "output_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}",
      "tags": {},
      "auxiliary": "{}",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_missing_inference_id() {
    let client = Client::new();
    let fake_inference_id = Uuid::now_v7();
    let resp = client
        .post(get_gateway_endpoint("/datasets/dummy-dataset/datapoints"))
        .json(&json!({
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
    let inference_payload = json!({
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
        .json(&json!({
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

#[tokio::test]
async fn test_datapoint_insert_missing_output_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "text", "value": "My synthetic input"}]}]},
            // output field is deliberately omitted
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    assert!(
        status.is_success(),
        "Expected successful response, got: {}",
        status
    );

    let resp_json = resp.json::<Value>().await.unwrap();
    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(id, datapoint_id);

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();

    datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": id.to_string(),
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"My synthetic input\"}]}]}",
      "output": null,
      "tool_params": "",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_null_output_chat() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "text", "value": "My synthetic input"}]}]},
            "output": null, // explicitly null output
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    assert!(
        status.is_success(),
        "Expected successful response, got: {}",
        status
    );

    let resp_json = resp.json::<Value>().await.unwrap();
    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(id, datapoint_id);

    let mut datapoint = select_chat_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();

    datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "basic_test",
      "id": id.to_string(),
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":\"My synthetic input\"}]}]}",
      "output": null,
      "tool_params": "",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_missing_output_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "text", "arguments": {"country": "US"}}]}]},
            "output_schema": {},
            // output field is deliberately omitted
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    assert!(
        status.is_success(),
        "Expected successful response, got: {}",
        status
    );

    let resp_json = resp.json::<Value>().await.unwrap();
    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(id, datapoint_id);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();

    datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": id,
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":{\"country\":\"US\"}}]}]}",
      "output": null,
      "output_schema": "{}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);
}

#[tokio::test]
async fn test_datapoint_insert_null_output_json() {
    let clickhouse = get_clickhouse().await;
    let client = Client::new();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {"system": {"assistant_name": "Dummy"}, "messages": [{"role": "user", "content": [{"type": "text", "arguments": {"country": "US"}}]}]},
            "output": null, // explicitly null output
            "output_schema": {},
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    assert!(
        status.is_success(),
        "Expected successful response, got: {}",
        status
    );

    let resp_json = resp.json::<Value>().await.unwrap();
    let id: Uuid = resp_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(id, datapoint_id);

    let mut datapoint = select_json_datapoint_clickhouse(&clickhouse, id)
        .await
        .unwrap();

    datapoint
        .as_object_mut()
        .unwrap()
        .remove("updated_at")
        .unwrap();

    let expected = json!({
      "dataset_name": dataset_name,
      "function_name": "json_success",
      "id": id,
      "episode_id": null,
      "input": "{\"system\":{\"assistant_name\":\"Dummy\"},\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"value\":{\"country\":\"US\"}}]}]}",
      "output": null,
      "output_schema": "{}",
      "tags": {},
      "auxiliary": "",
      "is_deleted": false,
      "staled_at": null
    });
    assert_eq!(datapoint, expected);
}
