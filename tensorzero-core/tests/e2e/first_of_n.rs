#![allow(clippy::print_stdout)]
use serde_json::Value;
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    InferenceOutput, InferenceResponse,
};
use tensorzero_core::inference::types::{ContentBlockChatOutput, Role, Text, TextKind};
use tensorzero_core::providers::dummy::DUMMY_INFER_RESPONSE_CONTENT;
use uuid::Uuid;

use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_json_inference_clickhouse,
    select_model_inferences_clickhouse,
};

/// Test that first_of_n returns the fastest candidate (non-streaming)
#[tokio::test]
async fn e2e_test_first_of_n_fast_wins_non_stream() {
    let gateway = tensorzero::test_helpers::make_embedded_gateway_with_config(
        r#"
        [functions.test_first_of_n]
        type = "chat"

        [functions.test_first_of_n.variants.fast]
        type = "chat_completion"
        model = "test"

        [functions.test_first_of_n.variants.slow]
        type = "chat_completion"
        model = "slow"

        [functions.test_first_of_n.variants.first_of_n_variant]
        type = "experimental_first_of_n"
        timeout_s = 10.0
        candidates = ["fast", "slow"]

        [models.test]
        routing = ["dummy"]
        [models.test.providers.dummy]
        type = "dummy"
        model_name = "test"

        [models.slow]
        routing = ["dummy"]
        [models.slow.providers.dummy]
        type = "dummy"
        model_name = "slow"
        "#,
    )
    .await;

    let episode_id = Uuid::now_v7();
    let params = ClientInferenceParams {
        function_name: Some("test_first_of_n".to_string()),
        variant_name: Some("first_of_n_variant".to_string()),
        episode_id: Some(episode_id),
        input: ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Hello!".to_string(),
                })],
            }],
        },
        ..Default::default()
    };

    let response = gateway.inference(params).await.unwrap();

    let InferenceOutput::NonStreaming(InferenceResponse::Chat(chat_response)) = response else {
        panic!("Expected non-streaming chat response");
    };

    // Check that the response contains the fast model's output
    assert_eq!(chat_response.content.len(), 1);
    let content_block = chat_response.content.first().unwrap();
    match content_block {
        ContentBlockChatOutput::Text(Text { text }) => {
            assert_eq!(text, DUMMY_INFER_RESPONSE_CONTENT);
        }
        _ => panic!("Expected text content block"),
    }

    let inference_id = chat_response.inference_id;

    // Sleep to allow trailing writes
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "first_of_n_variant");

    // Check the ModelInference Table - should have 1 entry (only the fast winner)
    // Note: The slow candidate is cancelled when the fast one wins, so it's not logged
    let results: Vec<Value> = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    assert_eq!(results.len(), 1, "Should have 1 model inference");

    // Verify it's the fast model
    let model_name = results[0].get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, "test", "Should be the fast model");
}

/// Test that first_of_n handles error candidates gracefully
#[tokio::test]
async fn e2e_test_first_of_n_with_error_candidates() {
    let gateway = tensorzero::test_helpers::make_embedded_gateway_with_config(
        r#"
        [functions.test_first_of_n]
        type = "chat"

        [functions.test_first_of_n.variants.error_candidate]
        type = "chat_completion"
        model = "error"

        [functions.test_first_of_n.variants.fast]
        type = "chat_completion"
        model = "test"

        [functions.test_first_of_n.variants.first_of_n_variant]
        type = "experimental_first_of_n"
        timeout_s = 10.0
        candidates = ["error_candidate", "fast"]

        [models.error]
        routing = ["dummy"]
        [models.error.providers.dummy]
        type = "dummy"
        model_name = "error"

        [models.test]
        routing = ["dummy"]
        [models.test.providers.dummy]
        type = "dummy"
        model_name = "test"
        "#,
    )
    .await;

    let episode_id = Uuid::now_v7();
    let params = ClientInferenceParams {
        function_name: Some("test_first_of_n".to_string()),
        variant_name: Some("first_of_n_variant".to_string()),
        episode_id: Some(episode_id),
        input: ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Hello!".to_string(),
                })],
            }],
        },
        ..Default::default()
    };

    let response = gateway.inference(params).await.unwrap();

    let InferenceOutput::NonStreaming(InferenceResponse::Chat(chat_response)) = response else {
        panic!("Expected non-streaming chat response");
    };

    // Should succeed with the fast candidate's result
    assert_eq!(chat_response.content.len(), 1);
    match chat_response.content.first().unwrap() {
        ContentBlockChatOutput::Text(Text { text }) => {
            assert_eq!(text, DUMMY_INFER_RESPONSE_CONTENT);
        }
        _ => panic!("Expected text content block"),
    }

    let inference_id = chat_response.inference_id;

    // Sleep to allow trailing writes
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ModelInference table - should show only the successful attempt
    // The error candidate fails but once a success happens, other candidates are cancelled
    let clickhouse = get_clickhouse().await;
    let results: Vec<Value> = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    assert_eq!(results.len(), 1, "Should have 1 successful model inference");
}

/// Test that first_of_n returns an error when all candidates fail
#[tokio::test]
async fn e2e_test_first_of_n_all_fail() {
    let gateway = tensorzero::test_helpers::make_embedded_gateway_with_config(
        r#"
        [functions.test_first_of_n]
        type = "chat"

        [functions.test_first_of_n.variants.error1]
        type = "chat_completion"
        model = "error"

        [functions.test_first_of_n.variants.error2]
        type = "chat_completion"
        model = "error"

        [functions.test_first_of_n.variants.first_of_n_variant]
        type = "experimental_first_of_n"
        timeout_s = 10.0
        candidates = ["error1", "error2"]

        [models.error]
        routing = ["dummy"]
        [models.error.providers.dummy]
        type = "dummy"
        model_name = "error"
        "#,
    )
    .await;

    let episode_id = Uuid::now_v7();
    let params = ClientInferenceParams {
        function_name: Some("test_first_of_n".to_string()),
        variant_name: Some("first_of_n_variant".to_string()),
        episode_id: Some(episode_id),
        input: ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Hello!".to_string(),
                })],
            }],
        },
        ..Default::default()
    };

    let result = gateway.inference(params).await;

    // Should return an error
    assert!(result.is_err(), "Should fail when all candidates fail");
}

/// Test that first_of_n works with a single candidate
#[tokio::test]
async fn e2e_test_first_of_n_single_candidate() {
    let gateway = tensorzero::test_helpers::make_embedded_gateway_with_config(
        r#"
        [functions.test_first_of_n]
        type = "chat"

        [functions.test_first_of_n.variants.only_candidate]
        type = "chat_completion"
        model = "test"

        [functions.test_first_of_n.variants.first_of_n_variant]
        type = "experimental_first_of_n"
        timeout_s = 10.0
        candidates = ["only_candidate"]

        [models.test]
        routing = ["dummy"]
        [models.test.providers.dummy]
        type = "dummy"
        model_name = "test"
        "#,
    )
    .await;

    let episode_id = Uuid::now_v7();
    let params = ClientInferenceParams {
        function_name: Some("test_first_of_n".to_string()),
        variant_name: Some("first_of_n_variant".to_string()),
        episode_id: Some(episode_id),
        input: ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Hello!".to_string(),
                })],
            }],
        },
        ..Default::default()
    };

    let response = gateway.inference(params).await.unwrap();

    let InferenceOutput::NonStreaming(InferenceResponse::Chat(chat_response)) = response else {
        panic!("Expected non-streaming chat response");
    };

    assert_eq!(chat_response.content.len(), 1);
    match chat_response.content.first().unwrap() {
        ContentBlockChatOutput::Text(Text { text }) => {
            assert_eq!(text, DUMMY_INFER_RESPONSE_CONTENT);
        }
        _ => panic!("Expected text content block"),
    }

    let inference_id = chat_response.inference_id;

    // Sleep to allow trailing writes
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ModelInference table
    let clickhouse = get_clickhouse().await;
    let results: Vec<Value> = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    assert_eq!(results.len(), 1, "Should have 1 model inference");
}

/// Test that first_of_n works with JSON mode
#[tokio::test]
async fn e2e_test_first_of_n_json_mode() {
    // Create a temporary output schema file
    let temp_dir = tempfile::tempdir().unwrap();
    let schema_path = temp_dir.path().join("output_schema.json");
    std::fs::write(
        &schema_path,
        r#"{"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}"#,
    )
    .unwrap();

    let config = format!(
        r#"
        [functions.test_first_of_n_json]
        type = "json"
        output_schema = "{schema_path}"

        [functions.test_first_of_n_json.variants.json_candidate]
        type = "chat_completion"
        model = "json"
        json_mode = "on"

        [functions.test_first_of_n_json.variants.slow_candidate]
        type = "chat_completion"
        model = "slow"
        json_mode = "on"

        [functions.test_first_of_n_json.variants.first_of_n_variant]
        type = "experimental_first_of_n"
        timeout_s = 10.0
        candidates = ["json_candidate", "slow_candidate"]

        [models.json]
        routing = ["dummy"]
        [models.json.providers.dummy]
        type = "dummy"
        model_name = "json"

        [models.slow]
        routing = ["dummy"]
        [models.slow.providers.dummy]
        type = "dummy"
        model_name = "slow"
        "#,
        schema_path = schema_path.display()
    );

    let gateway = tensorzero::test_helpers::make_embedded_gateway_with_config(&config).await;

    let episode_id = Uuid::now_v7();
    let params = ClientInferenceParams {
        function_name: Some("test_first_of_n_json".to_string()),
        variant_name: Some("first_of_n_variant".to_string()),
        episode_id: Some(episode_id),
        input: ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "What is the answer?".to_string(),
                })],
            }],
        },
        ..Default::default()
    };

    let response = gateway.inference(params).await.unwrap();

    let InferenceOutput::NonStreaming(InferenceResponse::Json(json_response)) = response else {
        panic!("Expected non-streaming JSON response");
    };

    let answer = json_response
        .output
        .parsed
        .as_ref()
        .unwrap()
        .get("answer")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(answer, "Hello");

    let inference_id = json_response.inference_id;

    // Sleep to allow trailing writes
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    // Check ModelInference table - should have 1 entry (only the winner)
    let results: Vec<Value> = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    assert_eq!(results.len(), 1, "Should have 1 model inference");

    // Don't need to keep temp_dir longer
    drop(temp_dir);
}
