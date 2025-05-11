#![allow(clippy::print_stdout)]
use std::{
    collections::HashMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::providers::common::{make_embedded_gateway, make_http_gateway};
use serde_json::json;
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    DynamicEvaluationRunParams, FeedbackParams, InferenceOutput, Role,
};
use tensorzero_internal::{
    clickhouse::test_helpers::{
        get_clickhouse, select_chat_inference_clickhouse, select_dynamic_evaluation_run_clickhouse,
        select_dynamic_evaluation_run_episode_clickhouse,
    },
    endpoints::dynamic_evaluation_run::DynamicEvaluationRunEpisodeParams,
    inference::types::TextKind,
};
use uuid::{Timestamp, Uuid};

#[tokio::test]
async fn test_dynamic_evaluation() {
    let client = make_http_gateway().await;
    let params = DynamicEvaluationRunParams {
        internal: false,
        variants: HashMap::from([("basic_test".to_string(), "test2".to_string())]),
        tags: HashMap::from([
            ("foo".to_string(), "bar".to_string()),
            ("baz".to_string(), "bat".to_string()),
        ]),
        project_name: Some("test_project".to_string()),
        display_name: Some("test_display_name".to_string()),
    };
    let dynamic_evaluation_info = client.dynamic_evaluation_run(params).await.unwrap();
    let run_id = dynamic_evaluation_info.run_id;
    let clickhouse = get_clickhouse().await;
    let run_row = select_dynamic_evaluation_run_clickhouse(&clickhouse, run_id)
        .await
        .unwrap();
    assert_eq!(run_row.project_name, Some("test_project".to_string()));
    assert_eq!(
        run_row.run_display_name,
        Some("test_display_name".to_string())
    );
    for i in 0..2 {
        // Get the episode_id from the dynamic_evaluation_run_episode endpoint
        let episode_id = client
            .dynamic_evaluation_run_episode(
                run_id,
                DynamicEvaluationRunEpisodeParams {
                    task_name: Some(format!("test_datapoint_{i}")),
                    datapoint_name: None,
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
            input: ClientInput {
                system: Some(json!({
                    "assistant_name": "AskJeeves",
                })),
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
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
        // We won't test the output here but will grab from ClickHouse so we can check the variant name
        // and tags
        let clickhouse = get_clickhouse().await;
        let result = select_chat_inference_clickhouse(&clickhouse, response.inference_id())
            .await
            .unwrap();

        println!("ClickHouse - ChatInference: {result:#?}");
        let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
        assert_eq!(variant_name, "test2");
        let tags = result.get("tags").unwrap().as_object().unwrap();
        // Verify tags are correctly applied with the following precedence:
        // 1. Tags from the inference request (highest priority)
        // 2. Tags from the episode creation
        // 3. Tags from the dynamic evaluation run (lowest priority)
        // When tags have the same key, the higher priority source overwrites the lower priority one.
        // In this case:
        // - "foo" comes from the dynamic evaluation run
        // - "baz" comes from the episode creation
        // - "zoo" is in both episode creation and inference request, so inference request wins
        // - "bop" comes from the inference request

        assert_eq!(tags.get("foo").unwrap().as_str().unwrap(), "bar");
        assert_eq!(
            tags.get("baz").unwrap().as_str().unwrap(),
            format!("baz_{i}")
        );
        assert_eq!(
            tags.get("zoo").unwrap().as_str().unwrap(),
            format!("boo_{i}")
        );
        assert_eq!(
            tags.get("bop").unwrap().as_str().unwrap(),
            format!("bop_{i}")
        );
        assert_eq!(
            tags.get("tensorzero::dynamic_evaluation_run_id")
                .unwrap()
                .as_str()
                .unwrap(),
            run_id.to_string()
        );
        // Check for some git tags too
        tags.get("tensorzero::git_commit_hash")
            .unwrap()
            .as_str()
            .unwrap();
        tags.get("tensorzero::git_branch")
            .unwrap()
            .as_str()
            .unwrap();
        let episode_row =
            select_dynamic_evaluation_run_episode_clickhouse(&clickhouse, run_id, episode_id)
                .await
                .unwrap();
        println!("ClickHouse - DynamicEvaluationRunEpisode: {episode_row:#?}");
        assert_eq!(
            episode_row.variant_pins,
            HashMap::from([("basic_test".to_string(), "test2".to_string())])
        );
        assert_eq!(
            episode_row.datapoint_name,
            Some(format!("test_datapoint_{i}"))
        );
        let expected_tags = HashMap::from([
            ("foo".to_string(), "bar".to_string()),
            ("baz".to_string(), format!("baz_{i}")),
            ("zoo".to_string(), format!("zoo_{i}")),
            (
                "tensorzero::dynamic_evaluation_run_id".to_string(),
                run_id.to_string(),
            ),
        ]);
        for (k, v) in &expected_tags {
            assert_eq!(
                episode_row.tags.get(k),
                Some(v),
                "Tag {k:?} missing or incorrect"
            );
        }
        // Send feedback for the dynamic evaluation run episode
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

#[tokio::test]
async fn test_dynamic_evaluation_nonexistent_function() {
    let client = make_http_gateway().await;
    let params = DynamicEvaluationRunParams {
        variants: HashMap::from([("nonexistent_function".to_string(), "test2".to_string())]),
        tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
        project_name: None,
        display_name: None,
        internal: false,
    };
    let result = client.dynamic_evaluation_run(params).await.unwrap_err();
    println!("Result: {result:#?}");
    assert!(result
        .to_string()
        .contains("Unknown function: nonexistent_function"));
}

/// Test that the variant behavior is default if we use a different function name
/// But the tags are applied
#[tokio::test(flavor = "multi_thread")]
async fn test_dynamic_evaluation_other_function() {
    let client = make_embedded_gateway().await;
    let params = DynamicEvaluationRunParams {
        variants: HashMap::from([("dynamic_json".to_string(), "gcp-vertex-haiku".to_string())]),
        tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
        project_name: None,
        display_name: None,
        internal: false,
    };
    let result = client.dynamic_evaluation_run(params).await.unwrap();
    let run_id = result.run_id;
    let clickhouse = get_clickhouse().await;
    let run_row = select_dynamic_evaluation_run_clickhouse(&clickhouse, run_id)
        .await
        .unwrap();
    assert_eq!(run_row.project_name, None);
    assert_eq!(run_row.run_display_name, None);
    let episode_id = client
        .dynamic_evaluation_run_episode(
            run_id,
            DynamicEvaluationRunEpisodeParams {
                task_name: None,
                datapoint_name: None,
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
        input: ClientInput {
            system: Some(json!({
                "assistant_name": "AskJeeves",
            })),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
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
    // We won't test the output here but will grab from ClickHouse so we can check the variant name
    // and tags
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, response.inference_id())
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test");
    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.get("foo").unwrap().as_str().unwrap(), "bar");
}

/// Test that the variant does not fall back in a dynamic evaluation run
/// This should error
#[tokio::test(flavor = "multi_thread")]
async fn test_dynamic_evaluation_variant_error() {
    let client = make_embedded_gateway().await;
    let params = DynamicEvaluationRunParams {
        variants: HashMap::from([("basic_test".to_string(), "error".to_string())]),
        tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
        project_name: None,
        display_name: None,
        internal: false,
    };
    let result = client.dynamic_evaluation_run(params).await.unwrap();
    let run_id = result.run_id;
    let clickhouse = get_clickhouse().await;
    let run_row = select_dynamic_evaluation_run_clickhouse(&clickhouse, run_id)
        .await
        .unwrap();
    assert_eq!(run_row.project_name, None);
    assert_eq!(run_row.run_display_name, None);
    let episode_id = client
        .dynamic_evaluation_run_episode(
            run_id,
            DynamicEvaluationRunEpisodeParams {
                task_name: None,
                datapoint_name: None,
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
        input: ClientInput {
            system: Some(json!({
                "assistant_name": "AskJeeves",
            })),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Please write me a sentence about Megumin making an explosion.".into(),
                })],
            }],
        },
        ..Default::default()
    };
    let response = client.inference(inference_params).await.unwrap_err();
    println!("Response: {response:#?}");
    assert!(response.to_string().contains("All variants failed"));
}

/// Test that the variant behavior is default if we pin a different variant name
/// But the tags are applied
#[tokio::test(flavor = "multi_thread")]
async fn test_dynamic_evaluation_override_variant_tags() {
    let client = make_embedded_gateway().await;
    let params = DynamicEvaluationRunParams {
        internal: false,
        variants: HashMap::from([("basic_test".to_string(), "error".to_string())]),
        tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
        project_name: None,
        display_name: None,
    };
    let result = client.dynamic_evaluation_run(params).await.unwrap();
    let run_id = result.run_id;
    let clickhouse = get_clickhouse().await;
    let run_row = select_dynamic_evaluation_run_clickhouse(&clickhouse, run_id)
        .await
        .unwrap();
    assert_eq!(run_row.project_name, None);
    assert_eq!(run_row.run_display_name, None);
    let episode_id = client
        .dynamic_evaluation_run_episode(
            run_id,
            DynamicEvaluationRunEpisodeParams {
                task_name: None,
                datapoint_name: None,
                tags: HashMap::new(),
            },
        )
        .await
        .unwrap()
        .episode_id;
    let inference_params = ClientInferenceParams {
        episode_id: Some(episode_id),
        function_name: Some("basic_test".to_string()),
        input: ClientInput {
            system: Some(json!({
                "assistant_name": "AskJeeves",
            })),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
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
    // We won't test the output here but will grab from ClickHouse so we can check the variant name
    // and tags
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, response.inference_id())
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");
    // Test that inference time settings override the dynamic evaluation run settings
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test2");
    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.get("foo").unwrap().as_str().unwrap(), "baz");
}

#[tokio::test]
async fn test_bad_dynamic_evaluation_run() {
    let client = make_http_gateway().await;
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
        input: ClientInput {
            system: Some(json!({
                "assistant_name": "AskJeeves",
            })),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Please write me a sentence about Megumin making an explosion.".into(),
                })],
            }],
        },
        ..Default::default()
    };
    let response = client.inference(inference_params).await.unwrap_err();
    println!("Response: {response:#?}");
    assert!(response
        .to_string()
        .contains("Dynamic evaluation run not found"));
}

#[tokio::test]
async fn test_dynamic_evaluation_tag_validation() {
    let client = make_http_gateway().await;
    let params = DynamicEvaluationRunParams {
        internal: false,
        variants: HashMap::from([("basic_test".to_string(), "test2".to_string())]),
        tags: HashMap::from([("tensorzero::foo".to_string(), "bar".to_string())]),
        project_name: Some("test_project".to_string()),
        display_name: Some("test_display_name".to_string()),
    };
    let dynamic_evaluation_info = client.dynamic_evaluation_run(params).await.unwrap_err();
    assert!(dynamic_evaluation_info
        .to_string()
        .contains("Tag name cannot start with 'tensorzero::'"));
}
