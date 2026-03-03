use std::{
    collections::HashMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use googletest::prelude::*;
use serde_json::json;
use tensorzero::{
    ClientExt, ClientInferenceParams, FeedbackParams, InferenceOutput, Input, InputMessage,
    InputMessageContent, Role, WorkflowEvaluationRunParams,
};
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::delegating_connection::{DelegatingDatabaseConnection, PrimaryDatastore};
use tensorzero_core::db::inferences::{InferenceQueries, ListInferencesParams};
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::db::workflow_evaluation_queries::WorkflowEvaluationQueries;
use tensorzero_core::endpoints::workflow_evaluation_run::WorkflowEvaluationRunEpisodeParams;
use tensorzero_core::inference::types::{Arguments, System, Text};
use tensorzero_core::stored_inference::StoredInferenceDatabase;
use tensorzero_core::test_helpers::get_e2e_config;
use uuid::{Timestamp, Uuid};

#[gtest]
#[tokio::test]
async fn test_workflow_evaluation() {
    let client = tensorzero::test_helpers::make_http_gateway().await;
    let params = WorkflowEvaluationRunParams {
        internal: false,
        variants: HashMap::from([("basic_test".to_string(), "test2".to_string())]),
        tags: HashMap::from([
            ("foo".to_string(), "bar".to_string()),
            ("baz".to_string(), "bat".to_string()),
        ]),
        project_name: Some("test_project".to_string()),
        display_name: Some("test_display_name".to_string()),
    };
    let workflow_evaluation_info = client.workflow_evaluation_run(params).await.unwrap();
    let run_id = workflow_evaluation_info.run_id;
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let runs = conn
        .get_workflow_evaluation_runs(&[run_id], None)
        .await
        .unwrap();
    let run_row = &runs[0];
    expect_that!(&run_row.project_name, eq(&Some("test_project".to_string())));
    expect_that!(&run_row.name, eq(&Some("test_display_name".to_string())));

    // Assert DynamicEvaluationRun and materialized views have snapshot_hash (ClickHouse only)
    if PrimaryDatastore::from_test_env() != PrimaryDatastore::Postgres {
        let clickhouse = get_clickhouse().await;

        let query = format!(
            "SELECT snapshot_hash FROM DynamicEvaluationRun WHERE run_id_uint = toUInt128(toUUID('{run_id}')) FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let run_result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        expect_that!(
            run_result["snapshot_hash"].is_null(),
            eq(false),
            "DynamicEvaluationRun should have snapshot_hash"
        );

        let query = format!(
            "SELECT snapshot_hash FROM DynamicEvaluationRunByProjectName WHERE run_id_uint = toUInt128(toUUID('{run_id}')) FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let view_result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        expect_that!(
            view_result["snapshot_hash"].is_null(),
            eq(false),
            "DynamicEvaluationRunByProjectName should have snapshot_hash"
        );
    }

    for i in 0..2 {
        // Get the episode_id from the workflow_evaluation_run_episode endpoint
        let episode_id = client
            .workflow_evaluation_run_episode(
                run_id,
                WorkflowEvaluationRunEpisodeParams {
                    task_name: Some(format!("test_datapoint_{i}")),
                    tags: HashMap::from([
                        ("baz".to_string(), format!("baz_{i}")),
                        ("zoo".to_string(), format!("zoo_{i}")),
                    ]),
                },
            )
            .await
            .unwrap()
            .episode_id;
        // Run an inference with the episode_id given
        let inference_params = ClientInferenceParams {
            episode_id: Some(episode_id),
            function_name: Some("basic_test".to_string()),
            input: Input {
                system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                    "assistant_name".to_string(),
                    "AskJeeves".into(),
                )])))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Please write me a sentence about Megumin making an explosion."
                            .into(),
                    })],
                }],
            },
            tags: HashMap::from([
                ("bop".to_string(), format!("bop_{i}")),
                ("zoo".to_string(), format!("boo_{i}")),
            ]),
            ..Default::default()
        };
        let response = if let InferenceOutput::NonStreaming(response) =
            client.inference(inference_params).await.unwrap()
        {
            response
        } else {
            panic!("Expected a non-streaming response");
        };
        conn.flush_pending_writes().await;
        conn.sleep_for_writes_to_be_visible().await;

        let config = get_e2e_config().await;
        let inferences = conn
            .list_inferences(
                &config,
                &ListInferencesParams {
                    ids: Some(&[response.inference_id()]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let chat = match &inferences[0] {
            StoredInferenceDatabase::Chat(c) => c,
            StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
        };
        println!("ChatInference: {chat:#?}");
        expect_that!(chat.variant_name, eq("test2"));
        // Verify tags are correctly applied with the following precedence:
        // 1. Tags from the inference request (highest priority)
        // 2. Tags from the episode creation
        // 3. Tags from the workflow evaluation run (lowest priority)
        // When tags have the same key, the higher priority source overwrites the lower priority one.
        // In this case:
        // - "foo" comes from the workflow evaluation run
        // - "baz" comes from the episode creation
        // - "zoo" is in both episode creation and inference request, so inference request wins
        // - "bop" comes from the inference request
        let tags = &chat.tags;
        expect_that!(tags.get("foo"), some(eq(&"bar".to_string())));
        expect_that!(tags.get("baz"), some(eq(&format!("baz_{i}"))));
        expect_that!(tags.get("zoo"), some(eq(&format!("boo_{i}"))));
        expect_that!(tags.get("bop"), some(eq(&format!("bop_{i}"))));
        // Verify both old and new tag names are present (double-write for backward compatibility)
        expect_that!(
            tags.get("tensorzero::dynamic_evaluation_run_id"),
            some(eq(&run_id.to_string())),
            "Old tag name should be present for backward compatibility"
        );
        expect_that!(
            tags.get("tensorzero::workflow_evaluation_run_id"),
            some(eq(&run_id.to_string())),
            "New tag name should be present for future migration"
        );
        // Check for some git tags too
        expect_that!(
            tags.contains_key("tensorzero::git_commit_hash"),
            eq(true),
            "Missing git_commit_hash tag"
        );
        expect_that!(
            tags.contains_key("tensorzero::git_branch"),
            eq(true),
            "Missing git_branch tag"
        );

        let episodes = conn
            .get_workflow_evaluation_run_episodes_with_feedback(run_id, 100, 0)
            .await
            .unwrap();
        let episode_row = episodes
            .iter()
            .find(|e| e.episode_id == episode_id)
            .expect("should find episode");
        println!("WorkflowEvaluationRunEpisode: {episode_row:#?}");
        // variant_pins is on the run, not the episode; verify via the run row
        expect_that!(
            &run_row.variant_pins,
            eq(&HashMap::from([(
                "basic_test".to_string(),
                "test2".to_string()
            )]))
        );
        expect_that!(
            &episode_row.task_name,
            eq(&Some(format!("test_datapoint_{i}")))
        );
        let expected_tags = HashMap::from([
            ("foo".to_string(), "bar".to_string()),
            ("baz".to_string(), format!("baz_{i}")),
            ("zoo".to_string(), format!("zoo_{i}")),
            (
                "tensorzero::dynamic_evaluation_run_id".to_string(),
                run_id.to_string(),
            ),
            (
                "tensorzero::workflow_evaluation_run_id".to_string(),
                run_id.to_string(),
            ),
        ]);
        for (k, v) in &expected_tags {
            expect_that!(
                episode_row.tags.get(k),
                some(eq(v)),
                "Tag {k:?} missing or incorrect"
            );
        }

        // Assert DynamicEvaluationRunEpisode and materialized views have snapshot_hash (ClickHouse only)
        if PrimaryDatastore::from_test_env() != PrimaryDatastore::Postgres {
            let clickhouse = get_clickhouse().await;

            let query = format!(
                "SELECT snapshot_hash FROM DynamicEvaluationRunEpisode WHERE run_id = '{run_id}' AND episode_id_uint = toUInt128(toUUID('{episode_id}')) FORMAT JSONEachRow"
            );
            let response = clickhouse
                .run_query_synchronous_no_params(query)
                .await
                .unwrap();
            let episode_result: serde_json::Value =
                serde_json::from_str(&response.response).unwrap();
            expect_that!(
                episode_result["snapshot_hash"].is_null(),
                eq(false),
                "DynamicEvaluationRunEpisode should have snapshot_hash"
            );

            let query = format!(
                "SELECT snapshot_hash FROM DynamicEvaluationRunEpisodeByRunId WHERE run_id_uint = toUInt128(toUUID('{run_id}')) AND episode_id_uint = toUInt128(toUUID('{episode_id}')) FORMAT JSONEachRow"
            );
            let response = clickhouse
                .run_query_synchronous_no_params(query)
                .await
                .unwrap();
            let view_result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
            expect_that!(
                view_result["snapshot_hash"].is_null(),
                eq(false),
                "DynamicEvaluationRunEpisodeByRunId should have snapshot_hash"
            );
        }

        // Send feedback for the workflow evaluation run episode
        let feedback_params = FeedbackParams {
            episode_id: Some(episode_id),
            metric_name: "goal_achieved".to_string(),
            value: json!(true),
            inference_id: None,
            internal: false,
            tags: HashMap::new(),
            dryrun: None,
        };
        // We just want to make sure this doesn't error
        // Feedback is thoroughly tested elsewhere
        client.feedback(feedback_params).await.unwrap();
    }
}

#[gtest]
#[tokio::test]
async fn test_workflow_evaluation_nonexistent_function() {
    let client = tensorzero::test_helpers::make_http_gateway().await;
    let params = WorkflowEvaluationRunParams {
        variants: HashMap::from([("nonexistent_function".to_string(), "test2".to_string())]),
        tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
        project_name: None,
        display_name: None,
        internal: false,
    };
    let result = client.workflow_evaluation_run(params).await.unwrap_err();
    println!("Result: {result:#?}");
    expect_that!(
        result.to_string(),
        contains_substring("Unknown function: nonexistent_function")
    );
}

/// Test that the variant behavior is default if we use a different function name
/// But the tags are applied
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_workflow_evaluation_other_function() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let params = WorkflowEvaluationRunParams {
        variants: HashMap::from([("dynamic_json".to_string(), "gcp-vertex-haiku".to_string())]),
        tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
        project_name: None,
        display_name: None,
        internal: false,
    };
    let result = client.workflow_evaluation_run(params).await.unwrap();
    let run_id = result.run_id;
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let runs = conn
        .get_workflow_evaluation_runs(&[run_id], None)
        .await
        .unwrap();
    let run_row = &runs[0];
    expect_that!(run_row.project_name, none());
    expect_that!(run_row.name, none());
    let episode_id = client
        .workflow_evaluation_run_episode(
            run_id,
            WorkflowEvaluationRunEpisodeParams {
                task_name: None,
                tags: HashMap::new(),
            },
        )
        .await
        .unwrap()
        .episode_id;
    // Run an inference with the episode_id given
    let inference_params = ClientInferenceParams {
        episode_id: Some(episode_id),
        function_name: Some("basic_test".to_string()),
        input: Input {
            system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                "assistant_name".to_string(),
                "AskJeeves".into(),
            )])))),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Please write me a sentence about Megumin making an explosion.".into(),
                })],
            }],
        },
        ..Default::default()
    };
    let response = if let InferenceOutput::NonStreaming(response) =
        client.inference(inference_params).await.unwrap()
    {
        response
    } else {
        panic!("Expected a non-streaming response");
    };
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[response.inference_id()]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    println!("ChatInference: {chat:#?}");
    expect_that!(chat.variant_name, eq("test"));
    expect_that!(chat.tags.get("foo"), some(eq(&"bar".to_string())));
}

/// Test that the variant does not fall back in a workflow evaluation run
/// This should error
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_workflow_evaluation_variant_error() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let params = WorkflowEvaluationRunParams {
        variants: HashMap::from([("basic_test".to_string(), "error".to_string())]),
        tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
        project_name: None,
        display_name: None,
        internal: false,
    };
    let result = client.workflow_evaluation_run(params).await.unwrap();
    let run_id = result.run_id;
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let runs = conn
        .get_workflow_evaluation_runs(&[run_id], None)
        .await
        .unwrap();
    let run_row = &runs[0];
    expect_that!(run_row.project_name, none());
    expect_that!(run_row.name, none());
    let episode_id = client
        .workflow_evaluation_run_episode(
            run_id,
            WorkflowEvaluationRunEpisodeParams {
                task_name: None,
                tags: HashMap::new(),
            },
        )
        .await
        .unwrap()
        .episode_id;
    // Run an inference with the episode_id given
    let inference_params = ClientInferenceParams {
        episode_id: Some(episode_id),
        function_name: Some("basic_test".to_string()),
        input: Input {
            system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                "assistant_name".to_string(),
                "AskJeeves".into(),
            )])))),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Please write me a sentence about Megumin making an explosion.".into(),
                })],
            }],
        },
        ..Default::default()
    };
    let response = client.inference(inference_params).await.unwrap_err();
    println!("Response: {response:#?}");
    expect_that!(
        response.to_string(),
        contains_substring("All model providers failed")
    );
}

/// Test that the variant behavior is default if we pin a different variant name
/// But the tags are applied
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_workflow_evaluation_override_variant_tags() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let params = WorkflowEvaluationRunParams {
        internal: false,
        variants: HashMap::from([("basic_test".to_string(), "error".to_string())]),
        tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
        project_name: None,
        display_name: None,
    };
    let result = client.workflow_evaluation_run(params).await.unwrap();
    let run_id = result.run_id;
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let runs = conn
        .get_workflow_evaluation_runs(&[run_id], None)
        .await
        .unwrap();
    let run_row = &runs[0];
    expect_that!(run_row.project_name, none());
    expect_that!(run_row.name, none());
    let episode_id = client
        .workflow_evaluation_run_episode(
            run_id,
            WorkflowEvaluationRunEpisodeParams {
                task_name: None,
                tags: HashMap::new(),
            },
        )
        .await
        .unwrap()
        .episode_id;
    let inference_params = ClientInferenceParams {
        episode_id: Some(episode_id),
        function_name: Some("basic_test".to_string()),
        input: Input {
            system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                "assistant_name".to_string(),
                "AskJeeves".into(),
            )])))),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Please write me a sentence about Megumin making an explosion.".into(),
                })],
            }],
        },
        variant_name: Some("test2".to_string()),
        tags: HashMap::from([("foo".to_string(), "baz".to_string())]),
        ..Default::default()
    };
    let response = if let InferenceOutput::NonStreaming(response) =
        client.inference(inference_params).await.unwrap()
    {
        response
    } else {
        panic!("Expected a non-streaming response");
    };
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[response.inference_id()]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    println!("ChatInference: {chat:#?}");
    // Test that inference time settings override the workflow evaluation run settings
    expect_that!(chat.variant_name, eq("test2"));
    expect_that!(chat.tags.get("foo"), some(eq(&"baz".to_string())));
}

#[gtest]
#[tokio::test]
async fn test_bad_workflow_evaluation_run() {
    let client = tensorzero::test_helpers::make_http_gateway().await;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let now_plus_offset = now + Duration::from_secs(100_000_000_000);
    let timestamp = Timestamp::from_unix_time(
        now_plus_offset.as_secs(),
        now_plus_offset.subsec_nanos(),
        0, // counter
        0, // usable_counter_bits
    );
    let episode_id = Uuid::new_v7(timestamp);
    // Run an inference with the episode_id given
    let inference_params = ClientInferenceParams {
        episode_id: Some(episode_id),
        function_name: Some("basic_test".to_string()),
        input: Input {
            system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                "assistant_name".to_string(),
                "AskJeeves".into(),
            )])))),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Please write me a sentence about Megumin making an explosion.".into(),
                })],
            }],
        },
        ..Default::default()
    };
    let response = client.inference(inference_params).await.unwrap_err();
    println!("Response: {response:#?}");
    expect_that!(
        response.to_string(),
        contains_substring("Workflow evaluation run not found")
    );
}

#[gtest]
#[tokio::test]
async fn test_workflow_evaluation_tag_validation() {
    let client = tensorzero::test_helpers::make_http_gateway().await;
    let params = WorkflowEvaluationRunParams {
        internal: false,
        variants: HashMap::from([("basic_test".to_string(), "test2".to_string())]),
        tags: HashMap::from([("tensorzero::foo".to_string(), "bar".to_string())]),
        project_name: Some("test_project".to_string()),
        display_name: Some("test_display_name".to_string()),
    };
    let workflow_evaluation_info = client.workflow_evaluation_run(params).await.unwrap_err();
    expect_that!(
        workflow_evaluation_info.to_string(),
        contains_substring("Tag name cannot start with 'tensorzero::'")
    );
}
