//! E2E tests for ModelInferenceQueries implementations (ClickHouse and Postgres).
//!
//! Tests verify read and write operations for model inferences work correctly with both backends.

use std::path::Path;

use tensorzero_core::{
    config::{Config, ConfigFileGlob},
    db::{
        inferences::{InferenceOutputSource, InferenceQueries, ListInferencesParams},
        model_inferences::ModelInferenceQueries,
    },
    inference::types::{FinishReason, StoredModelInference},
};
use uuid::Uuid;

async fn get_e2e_config() -> Config {
    Config::load_from_path_optional_verify_credentials(
        &ConfigFileGlob::new_from_path(Path::new("tests/e2e/config/tensorzero.*.toml")).unwrap(),
        false,
    )
    .await
    .unwrap()
    .into_config_without_writing_for_tests()
}

// ===== READ TESTS =====

async fn test_get_model_inferences_for_existing_inference(
    conn: impl ModelInferenceQueries + InferenceQueries,
) {
    let config = get_e2e_config().await;

    // First, get an inference ID from the database.
    // Filter by variant_name to avoid metadata-only inferences in Postgres.
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("write_haiku"),
                variant_name: Some("better_prompt_haiku_4_5"),
                output_source: InferenceOutputSource::Inference,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should have at least one inference to test with"
    );

    let inference_id = inferences[0].id();

    // Now get the model inferences for this inference
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();

    // Each inference should have at least one model inference (the actual API call)
    assert!(
        !model_inferences.is_empty(),
        "Should have at least one model inference for inference {inference_id}"
    );

    // Verify the model inferences have the correct inference_id
    for mi in &model_inferences {
        assert_eq!(
            mi.inference_id, inference_id,
            "Model inference should reference the correct inference"
        );
    }
}
make_db_test!(test_get_model_inferences_for_existing_inference);

async fn test_get_model_inferences_for_json_inference(conn: impl ModelInferenceQueries) {
    // Use a hardcoded extract_entities inference ID known to have model inferences.
    // We can't query dynamically because metadata-only inferences (which have no
    // model inference rows) may be returned first.
    let inference_id = Uuid::parse_str("0196374c-2c6d-7ce0-b508-e3b24ee4579c").expect("Valid UUID");

    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();

    assert!(
        !model_inferences.is_empty(),
        "Should have at least one model inference for JSON inference {inference_id}"
    );

    // Verify basic structure
    for mi in &model_inferences {
        assert_eq!(mi.inference_id, inference_id);
        assert!(!mi.model_name.is_empty(), "Model name should not be empty");
        assert!(
            !mi.model_provider_name.is_empty(),
            "Model provider name should not be empty"
        );
    }
}
make_db_test!(test_get_model_inferences_for_json_inference);

async fn test_get_model_inferences_for_nonexistent_inference(conn: impl ModelInferenceQueries) {
    let nonexistent_id = Uuid::now_v7();

    let model_inferences = conn
        .get_model_inferences_by_inference_id(nonexistent_id)
        .await
        .unwrap();

    assert!(
        model_inferences.is_empty(),
        "Should return empty list for non-existent inference"
    );
}
make_db_test!(test_get_model_inferences_for_nonexistent_inference);

async fn test_model_inference_fields_populated(
    conn: impl ModelInferenceQueries + InferenceQueries,
) {
    let config = get_e2e_config().await;

    // Get an inference with model inferences.
    // Filter by variant_name to avoid metadata-only inferences in Postgres.
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("write_haiku"),
                variant_name: Some("better_prompt_haiku_4_5"),
                output_source: InferenceOutputSource::Inference,
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Find one with model inferences that have populated fields
    for inference in &inferences {
        let model_inferences = conn
            .get_model_inferences_by_inference_id(inference.id())
            .await
            .unwrap();

        for mi in &model_inferences {
            // These fields should always be populated
            assert!(
                mi.raw_request
                    .as_ref()
                    .is_some_and(|raw_request| !raw_request.is_empty()),
                "raw_request should not be empty"
            );
            assert!(
                mi.raw_response
                    .as_ref()
                    .is_some_and(|raw_response| !raw_response.is_empty()),
                "raw_response should not be empty"
            );
            assert!(!mi.model_name.is_empty(), "model_name should not be empty");
            assert!(
                !mi.model_provider_name.is_empty(),
                "model_provider_name should not be empty"
            );

            // input_messages and output should be populated for chat/json functions
            // (they may be empty arrays but should exist)

            // timestamp should be populated when reading from database
            assert!(
                mi.timestamp.is_some(),
                "timestamp should be populated when reading from database"
            );

            // Check that input_tokens and output_tokens are reasonable when present
            if let Some(input_tokens) = mi.input_tokens {
                assert!(
                    input_tokens > 0,
                    "input_tokens should be positive when present"
                );
            }
            if let Some(output_tokens) = mi.output_tokens {
                assert!(
                    output_tokens > 0,
                    "output_tokens should be positive when present"
                );
            }
            // response_time_ms can be 0 for cached responses or very fast responses
        }
    }
}
make_db_test!(test_model_inference_fields_populated);

// ===== WRITE AND ROUND-TRIP TESTS =====

async fn test_insert_and_read_model_inference(conn: impl ModelInferenceQueries) {
    let inference_id = Uuid::now_v7();
    let model_inference_id = Uuid::now_v7();

    let model_inference = StoredModelInference {
        id: model_inference_id,
        inference_id,
        raw_request: Some(r#"{"model": "test-model", "messages": []}"#.to_string()),
        raw_response: Some(
            r#"{"choices": [{"message": {"content": "test response"}}]}"#.to_string(),
        ),
        system: Some("You are a helpful assistant.".to_string()),
        input_messages: Some(vec![]),
        output: Some(vec![]),
        input_tokens: Some(100),
        output_tokens: Some(50),
        response_time_ms: Some(1234),
        model_name: "test-model".to_string(),
        model_provider_name: "test-provider".to_string(),
        ttft_ms: Some(200),
        cached: false,
        finish_reason: Some(FinishReason::Stop),
        snapshot_hash: None,
        timestamp: None, // Computed from UUID on insert
    };

    // Insert the model inference
    conn.insert_model_inferences(std::slice::from_ref(&model_inference))
        .await
        .unwrap();

    // Read it back
    let read_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();

    assert_eq!(
        read_inferences.len(),
        1,
        "Should have exactly one model inference"
    );

    let read = &read_inferences[0];
    assert_eq!(read.id, model_inference_id, "ID should match");
    assert_eq!(read.inference_id, inference_id, "inference_id should match");
    assert_eq!(read.raw_request, model_inference.raw_request);
    assert_eq!(read.raw_response, model_inference.raw_response);
    assert_eq!(read.system, model_inference.system);
    assert_eq!(read.input_tokens, model_inference.input_tokens);
    assert_eq!(read.output_tokens, model_inference.output_tokens);
    assert_eq!(read.response_time_ms, model_inference.response_time_ms);
    assert_eq!(read.model_name, model_inference.model_name);
    assert_eq!(
        read.model_provider_name,
        model_inference.model_provider_name
    );
    assert_eq!(read.ttft_ms, model_inference.ttft_ms);
    assert_eq!(read.cached, model_inference.cached);
    assert_eq!(read.finish_reason, model_inference.finish_reason);
    assert!(
        read.timestamp.is_some(),
        "timestamp should be populated on read"
    );
}
make_db_test!(test_insert_and_read_model_inference);

async fn test_insert_multiple_model_inferences_for_same_inference(
    conn: impl ModelInferenceQueries,
) {
    let inference_id = Uuid::now_v7();

    // Simulate a fallback scenario: two model inferences for the same inference
    let model_inferences = vec![
        StoredModelInference {
            id: Uuid::now_v7(),
            inference_id,
            raw_request: Some(r#"{"model": "primary-model"}"#.to_string()),
            raw_response: Some(r#"{"error": "rate limited"}"#.to_string()),
            system: None,
            input_messages: Some(vec![]),
            output: Some(vec![]),
            input_tokens: Some(100),
            output_tokens: None,
            response_time_ms: Some(500),
            model_name: "primary-model".to_string(),
            model_provider_name: "primary-provider".to_string(),
            ttft_ms: None,
            cached: false,
            finish_reason: None, // Failed, no finish reason
            snapshot_hash: None,
            timestamp: None,
        },
        StoredModelInference {
            id: Uuid::now_v7(),
            inference_id,
            raw_request: Some(r#"{"model": "fallback-model"}"#.to_string()),
            raw_response: Some(r#"{"choices": [{"message": {"content": "success"}}]}"#.to_string()),
            system: None,
            input_messages: Some(vec![]),
            output: Some(vec![]),
            input_tokens: Some(100),
            output_tokens: Some(25),
            response_time_ms: Some(800),
            model_name: "fallback-model".to_string(),
            model_provider_name: "fallback-provider".to_string(),
            ttft_ms: Some(150),
            cached: false,
            finish_reason: Some(FinishReason::Stop),
            snapshot_hash: None,
            timestamp: None,
        },
    ];

    // Insert both
    conn.insert_model_inferences(&model_inferences)
        .await
        .unwrap();

    // Read them back
    let read_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();

    assert_eq!(
        read_inferences.len(),
        2,
        "Should have two model inferences for fallback scenario"
    );

    // Verify both are present
    let model_names: Vec<_> = read_inferences.iter().map(|mi| &mi.model_name).collect();
    assert!(
        model_names.contains(&&"primary-model".to_string()),
        "Should contain primary model"
    );
    assert!(
        model_names.contains(&&"fallback-model".to_string()),
        "Should contain fallback model"
    );
}
make_db_test!(test_insert_multiple_model_inferences_for_same_inference);

async fn test_insert_model_inference_with_all_finish_reasons(conn: impl ModelInferenceQueries) {
    let finish_reasons = [
        FinishReason::Stop,
        FinishReason::StopSequence,
        FinishReason::Length,
        FinishReason::ToolCall,
        FinishReason::ContentFilter,
        FinishReason::Unknown,
    ];

    for finish_reason in finish_reasons {
        let inference_id = Uuid::now_v7();
        let model_inference = StoredModelInference {
            id: Uuid::now_v7(),
            inference_id,
            raw_request: Some("{}".to_string()),
            raw_response: Some("{}".to_string()),
            system: None,
            input_messages: Some(vec![]),
            output: Some(vec![]),
            input_tokens: None,
            output_tokens: None,
            response_time_ms: None,
            model_name: "test-model".to_string(),
            model_provider_name: "test-provider".to_string(),
            ttft_ms: None,
            cached: false,
            finish_reason: Some(finish_reason),
            snapshot_hash: None,
            timestamp: None,
        };

        conn.insert_model_inferences(&[model_inference])
            .await
            .unwrap();

        let read = conn
            .get_model_inferences_by_inference_id(inference_id)
            .await
            .unwrap();

        assert_eq!(read.len(), 1);
        assert_eq!(
            read[0].finish_reason,
            Some(finish_reason),
            "finish_reason should round-trip correctly for {finish_reason:?}"
        );
    }
}
make_db_test!(test_insert_model_inference_with_all_finish_reasons);

async fn test_insert_model_inference_with_null_finish_reason(conn: impl ModelInferenceQueries) {
    let inference_id = Uuid::now_v7();
    let model_inference = StoredModelInference {
        id: Uuid::now_v7(),
        inference_id,
        raw_request: Some("{}".to_string()),
        raw_response: Some("{}".to_string()),
        system: None,
        input_messages: Some(vec![]),
        output: Some(vec![]),
        input_tokens: None,
        output_tokens: None,
        response_time_ms: None,
        model_name: "test-model".to_string(),
        model_provider_name: "test-provider".to_string(),
        ttft_ms: None,
        cached: false,
        finish_reason: None,
        snapshot_hash: None,
        timestamp: None,
    };

    conn.insert_model_inferences(&[model_inference])
        .await
        .unwrap();

    let read = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();

    assert_eq!(read.len(), 1);
    assert_eq!(
        read[0].finish_reason, None,
        "finish_reason should be None when inserted as None"
    );
}
make_db_test!(test_insert_model_inference_with_null_finish_reason);

async fn test_insert_model_inference_cached_flag(conn: impl ModelInferenceQueries) {
    // Test with cached = true
    let inference_id_cached = Uuid::now_v7();
    let model_inference_cached = StoredModelInference {
        id: Uuid::now_v7(),
        inference_id: inference_id_cached,
        raw_request: Some("{}".to_string()),
        raw_response: Some("{}".to_string()),
        system: None,
        input_messages: Some(vec![]),
        output: Some(vec![]),
        input_tokens: None,
        output_tokens: None,
        response_time_ms: None,
        model_name: "test-model".to_string(),
        model_provider_name: "test-provider".to_string(),
        ttft_ms: None,
        cached: true,
        finish_reason: None,
        snapshot_hash: None,
        timestamp: None,
    };

    conn.insert_model_inferences(&[model_inference_cached])
        .await
        .unwrap();

    let read_cached = conn
        .get_model_inferences_by_inference_id(inference_id_cached)
        .await
        .unwrap();

    assert_eq!(read_cached.len(), 1);
    assert!(read_cached[0].cached, "cached should be true");

    // Test with cached = false
    let inference_id_not_cached = Uuid::now_v7();
    let model_inference_not_cached = StoredModelInference {
        id: Uuid::now_v7(),
        inference_id: inference_id_not_cached,
        raw_request: Some("{}".to_string()),
        raw_response: Some("{}".to_string()),
        system: None,
        input_messages: Some(vec![]),
        output: Some(vec![]),
        input_tokens: None,
        output_tokens: None,
        response_time_ms: None,
        model_name: "test-model".to_string(),
        model_provider_name: "test-provider".to_string(),
        ttft_ms: None,
        cached: false,
        finish_reason: None,
        snapshot_hash: None,
        timestamp: None,
    };

    conn.insert_model_inferences(&[model_inference_not_cached])
        .await
        .unwrap();

    let read_not_cached = conn
        .get_model_inferences_by_inference_id(inference_id_not_cached)
        .await
        .unwrap();

    assert_eq!(read_not_cached.len(), 1);
    assert!(!read_not_cached[0].cached, "cached should be false");
}
make_db_test!(test_insert_model_inference_cached_flag);

async fn test_insert_empty_list_is_noop(conn: impl ModelInferenceQueries) {
    // Inserting an empty list should succeed without error
    conn.insert_model_inferences(&[]).await.unwrap();
}
make_db_test!(test_insert_empty_list_is_noop);
