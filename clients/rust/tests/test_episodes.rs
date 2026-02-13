#![cfg(feature = "e2e_tests")]
#![expect(clippy::unwrap_used)]
#![expect(clippy::unreachable)]
use serde_json::json;
use tensorzero::{
    BooleanMetricFilter, Client, ClientExt, ClientInferenceParams, FeedbackParams,
    FloatComparisonOperator, FloatMetricFilter, InferenceFilter, InferenceOutput, Input,
    InputMessage, InputMessageContent, ListEpisodesRequest,
};
use tensorzero_core::inference::types::{Arguments, Text};
use uuid::Uuid;

// Helper function to create test inferences using the client
async fn create_test_inference(client: &Client) -> Uuid {
    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            input: Input {
                system: Some(tensorzero::System::Template(Arguments(
                    json!({"assistant_name": "Assistant"})
                        .as_object()
                        .unwrap()
                        .clone(),
                ))),
                messages: vec![InputMessage {
                    role: tensorzero::Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Hello".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();

    match response {
        InferenceOutput::NonStreaming(r) => r.inference_id(),
        InferenceOutput::Streaming(_) => unreachable!("Expected non-streaming response"),
    }
}

// ============================================================================
// Episode Tests
// ============================================================================

/// Test listing episodes
async fn test_list_episodes(client: Client) {
    // Create some test inferences (each gets its own episode)
    for _ in 0..3 {
        let _ = create_test_inference(&client).await;
    }

    // Wait for async writes to be visible
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let response = client
        .list_episodes(ListEpisodesRequest {
            limit: 10,
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(
        !response.episodes.is_empty(),
        "Expected at least some episodes"
    );

    for episode in &response.episodes {
        assert!(
            episode.count > 0,
            "Each episode should have at least one inference"
        );
        assert!(
            episode.start_time <= episode.end_time,
            "start_time should be <= end_time"
        );
    }
}

tensorzero::make_gateway_test_functions!(test_list_episodes);

/// Test listing episodes with pagination
async fn test_list_episodes_pagination(client: Client) {
    // Create some test inferences (each gets its own episode)
    for _ in 0..3 {
        let _ = create_test_inference(&client).await;
    }

    // Wait for async writes to be visible
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // List episodes with limit of 1
    let first_page = client
        .list_episodes(ListEpisodesRequest {
            limit: 1,
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        first_page.episodes.len(),
        1,
        "Expected exactly one episode with limit=1"
    );

    let first_episode_id = first_page.episodes[0].episode_id;

    // Use that episode's ID as the `before` param to get the next page
    let second_page = client
        .list_episodes(ListEpisodesRequest {
            limit: 1,
            before: Some(first_episode_id),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(
        !second_page.episodes.is_empty(),
        "Expected at least one episode on the second page"
    );

    // The second page should not contain the first episode
    for episode in &second_page.episodes {
        assert_ne!(
            episode.episode_id, first_episode_id,
            "Second page should not contain the first episode"
        );
    }
}

tensorzero::make_gateway_test_functions!(test_list_episodes_pagination);

/// Test listing episodes with function_name filter (GET path)
async fn test_list_episodes_with_function_name(client: Client) {
    // Create some test inferences
    for _ in 0..3 {
        let _ = create_test_inference(&client).await;
    }

    // Wait for async writes to be visible
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Filter by the function name used in create_test_inference
    let response = client
        .list_episodes(ListEpisodesRequest {
            limit: 10,
            function_name: Some("basic_test".to_string()),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(
        !response.episodes.is_empty(),
        "Expected at least some episodes for function `basic_test`"
    );

    // Filter by a non-existent function name
    let response = client
        .list_episodes(ListEpisodesRequest {
            limit: 10,
            function_name: Some("nonexistent_function".to_string()),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(
        response.episodes.is_empty(),
        "Expected no episodes for non-existent function"
    );
}

tensorzero::make_gateway_test_functions!(test_list_episodes_with_function_name);

/// Test listing episodes with a boolean metric filter (POST path)
async fn test_list_episodes_with_boolean_filter(client: Client) {
    // Create an inference and submit boolean feedback for it
    let inference_id = create_test_inference(&client).await;

    client
        .feedback(FeedbackParams {
            inference_id: Some(inference_id),
            metric_name: "task_success".to_string(),
            value: json!(true),
            ..Default::default()
        })
        .await
        .unwrap();

    // Wait for async writes to be visible
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Get unfiltered count for comparison
    let unfiltered = client
        .list_episodes(ListEpisodesRequest {
            limit: 100,
            ..Default::default()
        })
        .await
        .unwrap();

    // Filter by the boolean metric we just submitted
    let filtered = client
        .list_episodes(ListEpisodesRequest {
            limit: 100,
            filters: Some(InferenceFilter::BooleanMetric(BooleanMetricFilter {
                metric_name: "task_success".to_string(),
                value: true,
            })),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(
        !filtered.episodes.is_empty(),
        "Expected at least one episode matching the boolean metric filter"
    );
    assert!(
        filtered.episodes.len() <= unfiltered.episodes.len(),
        "Filtered episodes should be a subset of unfiltered episodes"
    );
}

tensorzero::make_gateway_test_functions!(test_list_episodes_with_boolean_filter);

/// Test listing episodes with a float metric filter (POST path)
async fn test_list_episodes_with_float_filter(client: Client) {
    // Create an inference and submit float feedback for it
    let inference_id = create_test_inference(&client).await;

    client
        .feedback(FeedbackParams {
            inference_id: Some(inference_id),
            metric_name: "user_rating".to_string(),
            value: json!(4.5),
            ..Default::default()
        })
        .await
        .unwrap();

    // Wait for async writes to be visible
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Get unfiltered count for comparison
    let unfiltered = client
        .list_episodes(ListEpisodesRequest {
            limit: 100,
            ..Default::default()
        })
        .await
        .unwrap();

    // Filter by the float metric we just submitted (>= 4.0 should match our 4.5)
    let filtered = client
        .list_episodes(ListEpisodesRequest {
            limit: 100,
            filters: Some(InferenceFilter::FloatMetric(FloatMetricFilter {
                metric_name: "user_rating".to_string(),
                value: 4.0,
                comparison_operator: FloatComparisonOperator::GreaterThanOrEqual,
            })),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(
        !filtered.episodes.is_empty(),
        "Expected at least one episode matching the float metric filter"
    );
    assert!(
        filtered.episodes.len() <= unfiltered.episodes.len(),
        "Filtered episodes should be a subset of unfiltered episodes"
    );
}

tensorzero::make_gateway_test_functions!(test_list_episodes_with_float_filter);

/// Test listing episodes with combined filters (AND, POST path)
async fn test_list_episodes_combined_filters(client: Client) {
    // Create an inference and submit both boolean and float feedback for it
    let inference_id = create_test_inference(&client).await;

    client
        .feedback(FeedbackParams {
            inference_id: Some(inference_id),
            metric_name: "task_success".to_string(),
            value: json!(true),
            ..Default::default()
        })
        .await
        .unwrap();

    client
        .feedback(FeedbackParams {
            inference_id: Some(inference_id),
            metric_name: "user_rating".to_string(),
            value: json!(5.0),
            ..Default::default()
        })
        .await
        .unwrap();

    // Wait for async writes to be visible
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Get unfiltered count for comparison
    let unfiltered = client
        .list_episodes(ListEpisodesRequest {
            limit: 100,
            ..Default::default()
        })
        .await
        .unwrap();

    // Combine a boolean filter and a float filter with AND
    let filtered = client
        .list_episodes(ListEpisodesRequest {
            limit: 100,
            function_name: Some("basic_test".to_string()),
            filters: Some(InferenceFilter::And {
                children: vec![
                    InferenceFilter::BooleanMetric(BooleanMetricFilter {
                        metric_name: "task_success".to_string(),
                        value: true,
                    }),
                    InferenceFilter::FloatMetric(FloatMetricFilter {
                        metric_name: "user_rating".to_string(),
                        value: 4.0,
                        comparison_operator: FloatComparisonOperator::GreaterThanOrEqual,
                    }),
                ],
            }),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(
        !filtered.episodes.is_empty(),
        "Expected at least one episode matching the combined filters"
    );
    assert!(
        filtered.episodes.len() <= unfiltered.episodes.len(),
        "Filtered episodes should be a subset of unfiltered episodes"
    );
}

tensorzero::make_gateway_test_functions!(test_list_episodes_combined_filters);

/// Unit test for ListEpisodesRequest JSON serialization
#[test]
fn test_list_episodes_request_serialization() {
    let request = ListEpisodesRequest {
        limit: 10,
        before: None,
        after: None,
        function_name: Some("my_function".to_string()),
        filters: Some(InferenceFilter::BooleanMetric(BooleanMetricFilter {
            metric_name: "task_success".to_string(),
            value: true,
        })),
    };

    let json = serde_json::to_value(&request).unwrap();
    assert_eq!(json["limit"], 10, "limit should serialize correctly");
    assert_eq!(
        json["function_name"], "my_function",
        "function_name should serialize correctly"
    );
    assert!(
        json.get("before").is_none(),
        "None fields should be skipped"
    );
    assert!(json.get("after").is_none(), "None fields should be skipped");
    assert!(
        json.get("filters").is_some(),
        "filters should be present when set"
    );

    // Roundtrip
    let deserialized: ListEpisodesRequest = serde_json::from_value(json).unwrap();
    assert_eq!(deserialized.limit, 10, "limit should roundtrip correctly");
    assert_eq!(
        deserialized.function_name.as_deref(),
        Some("my_function"),
        "function_name should roundtrip correctly"
    );
    assert!(
        deserialized.filters.is_some(),
        "filters should roundtrip correctly"
    );
}
