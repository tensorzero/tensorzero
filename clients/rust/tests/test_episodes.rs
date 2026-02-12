#![cfg(feature = "e2e_tests")]
#![expect(clippy::unwrap_used)]
#![expect(clippy::unreachable)]
use serde_json::json;
use tensorzero::{
    Client, ClientExt, ClientInferenceParams, InferenceOutput, Input, InputMessage,
    InputMessageContent, ListEpisodesParams,
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
        .list_episodes(ListEpisodesParams {
            limit: 10,
            before: None,
            after: None,
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
        .list_episodes(ListEpisodesParams {
            limit: 1,
            before: None,
            after: None,
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
        .list_episodes(ListEpisodesParams {
            limit: 1,
            before: Some(first_episode_id),
            after: None,
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
