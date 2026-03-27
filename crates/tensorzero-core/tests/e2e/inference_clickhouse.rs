#![expect(clippy::print_stdout, clippy::print_stderr)]

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use tensorzero::{
    ClientExt, ClientInferenceParams, InferenceOutput, Input, InputMessage, InputMessageContent,
};
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, get_clickhouse_replica,
    select_all_model_inferences_by_chat_episode_id_clickhouse, select_chat_inference_clickhouse,
    select_chat_inferences_clickhouse,
};
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::inference::types::{Role, Text};
use tokio::task::JoinSet;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use crate::utils::poll_for_result::poll_for_result;
use crate::utils::skip_for_postgres;

#[tokio::test]
async fn test_dummy_only_replicated_clickhouse() {
    skip_for_postgres!();

    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let response = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::my-model".to_string()),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "What is the name of the capital city of Japan?".to_string(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap();
    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming response");
    };

    // Poll until the trailing write from the API lands in ClickHouse
    let clickhouse = get_clickhouse().await;
    let inference_id = response.inference_id();
    let result = poll_for_result(
        || async {
            Ok::<_, String>(select_chat_inference_clickhouse(&clickhouse, inference_id).await)
        },
        |r| r.is_some(),
        "Timed out waiting for ChatInference to appear in ClickHouse",
    )
    .await
    .unwrap();

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, response.inference_id());

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "tensorzero::default");

    // Check if ClickHouse replica is ok - ChatInference Table on replica
    let clickhouse_replica = get_clickhouse_replica().await;
    if let Some(clickhouse_replica) = clickhouse_replica {
        println!("ClickHouse replica is ok");
        let result = select_chat_inference_clickhouse(&clickhouse_replica, response.inference_id())
            .await
            .unwrap();
        let id_str = result.get("id").unwrap().as_str().unwrap();
        let id = Uuid::parse_str(id_str).unwrap();
        assert_eq!(id, response.inference_id());
        let episode_id_str = result.get("episode_id").unwrap().as_str().unwrap();

        let function_name = result.get("function_name").unwrap().as_str().unwrap();
        assert_eq!(function_name, "tensorzero::default");

        // Let's also check that the data is in InferenceById to make sure that the data is replicated to materialize views too
        let result = clickhouse_replica.run_query_synchronous(
            "SELECT * FROM InferenceById WHERE id_uint = toUInt128({id:UUID}) FORMAT JSONEachRow".to_string(),
            &HashMap::from([("id", id_str)]),
        ).await.unwrap();
        let result: Value = serde_json::from_str(result.response.trim()).unwrap();
        let episode_id_str_mv = result.get("episode_id").unwrap().as_str().unwrap();
        assert_eq!(episode_id_str_mv, episode_id_str);
    }
}

// We don't use the word 'batch' in the test name, since we already
// group those tests as 'batch inference' tests
#[tokio::test(flavor = "multi_thread")]
async fn test_clickhouse_bulk_insert_off_default() {
    skip_for_postgres!();

    let client = Arc::new(
        tensorzero::test_helpers::make_embedded_gateway_with_config(
            "
    ",
        )
        .await,
    );

    assert!(
        !client
            .get_app_state_data()
            .unwrap()
            .clickhouse_connection_info
            .is_batching_enabled(),
        "Batching is enabled, but should be disabled with default config!"
    );
}

// We don't use the word 'batch' in the test name, since we already
// group those tests as 'batch inference' tests
#[tokio::test(flavor = "multi_thread")]
async fn test_clickhouse_bulk_insert() {
    skip_for_postgres!();

    let client = Arc::new(
        tensorzero::test_helpers::make_embedded_gateway_with_config(
            "
    [gateway.observability]
    enabled = true
    batch_writes = { enabled = true }
    ",
        )
        .await,
    );

    assert!(
        client
            .get_app_state_data()
            .unwrap()
            .clickhouse_connection_info
            .is_batching_enabled(),
        "Batching should be enabled with config, but is disabled!"
    );

    let mut join_set = JoinSet::new();
    let episode_id = Uuid::now_v7();
    let inference_count = 10_000;
    for _ in 0..inference_count {
        let client = client.clone();
        join_set.spawn(async move {
            client
                .inference(ClientInferenceParams {
                    episode_id: Some(episode_id),
                    model_name: Some("dummy::my-model".to_string()),
                    input: Input {
                        system: None,
                        messages: vec![InputMessage {
                            role: Role::User,
                            content: vec![InputMessageContent::Text(Text {
                                text: "What is the name of the capital city of Japan?".to_string(),
                            })],
                        }],
                    },
                    ..Default::default()
                })
                .await
                .unwrap()
        });
    }

    let mut expected_inference_ids = HashSet::new();
    while let Some(result) = join_set.join_next().await {
        let result = result.unwrap();
        let InferenceOutput::NonStreaming(response) = result else {
            panic!("Expected non-streaming response");
        };
        expected_inference_ids.insert(response.inference_id());
    }
    assert_eq!(expected_inference_ids.len(), inference_count);

    assert_eq!(Arc::strong_count(&client), 1);
    eprintln!("Dropping client");
    // Drop the last client, which will drop all of our `ClickhouseConnectionInfo`s
    // and allow the batch writer to shut down.
    drop(client);
    eprintln!("Dropped client");

    // Poll until all batch writes have been flushed to ClickHouse
    let clickhouse_client = get_clickhouse().await;
    let inferences = poll_for_result(
        || async {
            Ok::<_, String>(select_chat_inferences_clickhouse(&clickhouse_client, episode_id).await)
        },
        |rows| rows.as_ref().is_some_and(|r| r.len() == inference_count),
        "Timed out waiting for all batch-written inferences to appear in ClickHouse",
    )
    .await
    .unwrap();
    let actual_inference_ids = inferences
        .iter()
        .map(|i| {
            i.get("id")
                .unwrap()
                .as_str()
                .unwrap()
                .parse::<Uuid>()
                .unwrap()
        })
        .collect::<HashSet<_>>();

    assert_eq!(actual_inference_ids.len(), inference_count);
    assert_eq!(actual_inference_ids, expected_inference_ids);

    let model_inferences =
        select_all_model_inferences_by_chat_episode_id_clickhouse(episode_id, &clickhouse_client)
            .await
            .unwrap();

    let actual_model_inference_ids = model_inferences
        .iter()
        .map(|i| {
            i.get("inference_id")
                .unwrap()
                .as_str()
                .unwrap()
                .parse::<Uuid>()
                .unwrap()
        })
        .collect::<HashSet<_>>();
    assert_eq!(actual_model_inference_ids.len(), inference_count);
    assert_eq!(actual_model_inference_ids, expected_inference_ids);
}

/// Tests that ClickHouse materialized views (InferenceById, InferenceByEpisodeId,
/// InferenceTag, TagInference) contain the `snapshot_hash` column after an inference
/// with tags is written.
#[tokio::test]
async fn test_materialized_views_have_snapshot_hash() {
    skip_for_postgres!();

    let payload = json!({
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
        "stream": false,
        "tags": {
            "test_tag_key": "test_tag_value"
        }
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();

    // Poll until the trailing write lands in the InferenceById materialized view
    let clickhouse = get_clickhouse().await;

    // Assert InferenceById materialized view has snapshot_hash
    let view_result = poll_for_result(
        || {
            let query = format!(
                "SELECT snapshot_hash FROM InferenceById WHERE id_uint = toUInt128(toUUID('{inference_id}')) FORMAT JSONEachRow"
            );
            let clickhouse = &clickhouse;
            async move {
                clickhouse.flush_pending_writes().await;
                let response = clickhouse
                    .run_query_synchronous_no_params(query)
                    .await
                    .map_err(|e| e.to_string())?;
                serde_json::from_str::<serde_json::Value>(&response.response)
                    .map_err(|e| e.to_string())
            }
        },
        |v| !v["snapshot_hash"].is_null(),
        "Timed out waiting for InferenceById snapshot_hash",
    )
    .await;
    assert!(
        !view_result["snapshot_hash"].is_null(),
        "InferenceById should have snapshot_hash"
    );

    // Assert InferenceByEpisodeId materialized view has snapshot_hash
    let query = format!(
        "SELECT snapshot_hash FROM InferenceByEpisodeId WHERE episode_id_uint = toUInt128(toUUID('{episode_id}')) AND id_uint = toUInt128(toUUID('{inference_id}')) FORMAT JSONEachRow"
    );
    let response = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let view_result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
    assert!(
        !view_result["snapshot_hash"].is_null(),
        "InferenceByEpisodeId should have snapshot_hash"
    );

    // Assert InferenceTag materialized view has snapshot_hash
    let query = format!(
        "SELECT snapshot_hash FROM InferenceTag WHERE inference_id = '{inference_id}' AND key = 'test_tag_key' FORMAT JSONEachRow"
    );
    let response = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let view_result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
    assert!(
        !view_result["snapshot_hash"].is_null(),
        "InferenceTag should have snapshot_hash"
    );

    // Assert TagInference materialized view has snapshot_hash
    let query = format!(
        "SELECT snapshot_hash FROM TagInference WHERE key = 'test_tag_key' AND value = 'test_tag_value' AND inference_id = '{inference_id}' FORMAT JSONEachRow"
    );
    let response = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let view_result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
    assert!(
        !view_result["snapshot_hash"].is_null(),
        "TagInference should have snapshot_hash"
    );
}
