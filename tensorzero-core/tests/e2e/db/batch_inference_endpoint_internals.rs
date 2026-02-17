//! E2E tests for batch inference endpoint internal helper functions.
//!
//! These tests verify the batch inference endpoint orchestration functions
//! that coordinate database operations.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;
use tensorzero_core::config::Config;
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::db::batch_inference::BatchInferenceQueries;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::inferences::InferenceQueries;
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::endpoints::batch_inference::{
    PollInferenceResponse, PollPathParams, get_batch_inferences, get_batch_request,
    get_completed_batch_inference_response, write_batch_request_row,
    write_completed_batch_inference, write_poll_batch_inference,
};
use tensorzero_core::endpoints::inference::{InferenceParams, InferenceResponse};
use tensorzero_core::function::{FunctionConfig, FunctionConfigChat, FunctionConfigJson};
use tensorzero_core::inference::types::batch::{
    BatchModelInferenceRow, BatchRequestRow, BatchStatus, PollBatchInferenceResponse,
    ProviderBatchInferenceOutput, ProviderBatchInferenceResponse, UnparsedBatchRequestRow,
};
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, FinishReason, JsonInferenceOutput, StoredInput, Usage,
};
use tensorzero_core::jsonschema_util::JSONSchema;
use uuid::Uuid;

// ===== HELPER FUNCTIONS =====

fn create_delegating_connection(
    clickhouse: ClickHouseConnectionInfo,
) -> DelegatingDatabaseConnection {
    DelegatingDatabaseConnection::new(clickhouse, PostgresConnectionInfo::new_disabled())
}

/// Helper function to write 2 rows to the BatchModelInference table
async fn write_2_batch_model_inference_rows(
    database: &impl BatchInferenceQueries,
    batch_id: Uuid,
) -> Vec<BatchModelInferenceRow<'static>> {
    let inference_id = Uuid::now_v7();
    let function_name = "test_function";
    let variant_name = "test_variant";
    let episode_id = Uuid::now_v7();
    let input = StoredInput {
        system: None,
        messages: vec![],
    };
    let model_name = "test_model";
    let model_provider_name = "test_model_provider";
    let row1 = BatchModelInferenceRow {
        batch_id,
        inference_id,
        function_name: function_name.into(),
        variant_name: variant_name.into(),
        episode_id,
        input: Some(input.clone()),
        input_messages: Some(vec![]),
        system: None,
        tool_params: None,
        inference_params: Some(Cow::Owned(InferenceParams::default())),
        output_schema: None,
        raw_request: Some(Cow::Borrowed("")),
        model_name: Cow::Borrowed(model_name),
        model_provider_name: Cow::Borrowed(model_provider_name),
        tags: HashMap::new(),
        snapshot_hash: Some(SnapshotHash::new_test()),
    };
    let inference_id2 = Uuid::now_v7();
    let row2 = BatchModelInferenceRow {
        batch_id,
        inference_id: inference_id2,
        function_name: function_name.into(),
        variant_name: variant_name.into(),
        episode_id,
        input: Some(input),
        input_messages: Some(vec![]),
        system: None,
        tool_params: None,
        inference_params: Some(Cow::Owned(InferenceParams::default())),
        output_schema: None,
        raw_request: Some(Cow::Borrowed("")),
        model_name: Cow::Borrowed(model_name),
        model_provider_name: Cow::Borrowed(model_provider_name),
        tags: HashMap::new(),
        snapshot_hash: Some(SnapshotHash::new_test()),
    };
    let rows = vec![row1, row2];
    database.write_batch_model_inferences(&rows).await.unwrap();
    rows
}

// ===== TESTS =====

async fn test_get_batch_request_endpoint(
    database: impl BatchInferenceQueries + TestDatabaseHelpers,
) {
    let batch_id = Uuid::now_v7();
    let batch_params = json!({"foo": "bar"});
    let function_name = "test_function";
    let variant_name = "test_variant";
    let model_name = "test_model";
    let model_provider_name = "test_model_provider";
    let raw_request = "raw request";
    let raw_response = "raw response";
    let batch_request = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id,
        batch_params: &batch_params,
        function_name,
        variant_name,
        model_name,
        raw_request,
        raw_response,
        model_provider_name,
        status: BatchStatus::Pending,
        errors: vec![],
        snapshot_hash: Some(SnapshotHash::new_test()),
    });
    write_batch_request_row(&database, &batch_request)
        .await
        .unwrap();
    database.sleep_for_writes_to_be_visible().await;

    // First, let's query by batch ID
    let query = PollPathParams {
        batch_id,
        inference_id: None,
    };
    let batch_request = get_batch_request(&database, &query).await.unwrap();
    assert_eq!(batch_request.batch_id, batch_id, "batch_id should match");
    assert_eq!(
        batch_request.batch_params.into_owned(),
        batch_params,
        "batch_params should match"
    );
    assert_eq!(
        batch_request.function_name, function_name,
        "function_name should match"
    );
    assert_eq!(
        batch_request.variant_name, variant_name,
        "variant_name should match"
    );
    assert_eq!(
        &*batch_request.model_name, model_name,
        "model_name should match"
    );
    assert_eq!(
        &*batch_request.model_provider_name, model_provider_name,
        "model_provider_name should match"
    );
    assert_eq!(
        batch_request.status,
        BatchStatus::Pending,
        "status should be Pending"
    );
    assert_eq!(
        batch_request.raw_request.as_deref(),
        Some(raw_request),
        "raw_request should match"
    );
    assert_eq!(
        batch_request.raw_response.as_deref(),
        Some(raw_response),
        "raw_response should match"
    );

    // Next, we'll insert a BatchModelInferenceRow
    let inference_id = Uuid::now_v7();
    let episode_id = Uuid::now_v7();
    let input = StoredInput {
        system: None,
        messages: vec![],
    };

    let row = BatchModelInferenceRow {
        batch_id,
        inference_id,
        function_name: function_name.into(),
        variant_name: variant_name.into(),
        episode_id,
        input: Some(input),
        input_messages: Some(vec![]),
        system: None,
        tool_params: None,
        inference_params: Some(Cow::Owned(InferenceParams::default())),
        output_schema: None,
        raw_request: Some(Cow::Borrowed("")),
        model_name: Cow::Borrowed(model_name),
        model_provider_name: Cow::Borrowed(model_provider_name),
        tags: HashMap::new(),
        snapshot_hash: Some(SnapshotHash::new_test()),
    };
    database.write_batch_model_inferences(&[row]).await.unwrap();
    database.sleep_for_writes_to_be_visible().await;

    // Now, let's query by inference ID
    let query = PollPathParams {
        batch_id,
        inference_id: Some(inference_id),
    };
    let batch_request = get_batch_request(&database, &query).await.unwrap();
    assert_eq!(batch_request.batch_id, batch_id, "batch_id should match");
    assert_eq!(
        batch_request.function_name, function_name,
        "function_name should match"
    );
    assert_eq!(
        batch_request.variant_name, variant_name,
        "variant_name should match"
    );
    assert_eq!(
        &*batch_request.model_name, model_name,
        "model_name should match"
    );
    assert_eq!(
        &*batch_request.model_provider_name, model_provider_name,
        "model_provider_name should match"
    );
}
make_db_test!(test_get_batch_request_endpoint);

async fn test_write_poll_batch_inference_endpoint(
    database: impl BatchInferenceQueries
    + InferenceQueries
    + ModelInferenceQueries
    + TestDatabaseHelpers,
) {
    let batch_id = Uuid::now_v7();
    let batch_params = json!({"baz": "bat"});
    let function_name = "test_function2";
    let variant_name = "test_variant2";
    let model_name = "test_model2";
    let model_provider_name = "test_model_provider2";
    let raw_request = "raw request".to_string();
    let raw_response = "raw response".to_string();
    let status = BatchStatus::Pending;
    let errors = vec![];
    let batch_request = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id,
        batch_params: &batch_params,
        function_name,
        variant_name,
        model_name,
        model_provider_name,
        raw_request: &raw_request,
        raw_response: &raw_response,
        status,
        errors,
        snapshot_hash: Some(SnapshotHash::new_test()),
    });
    let config = Config::new_empty()
        .await
        .unwrap()
        .into_config_without_writing_for_tests();

    // Write a pending batch
    let poll_inference_response = write_poll_batch_inference(
        &database,
        &batch_request,
        PollBatchInferenceResponse::Pending {
            raw_request: raw_request.clone(),
            raw_response: raw_response.clone(),
        },
        Arc::from("dummy"),
        &config,
    )
    .await
    .unwrap();
    assert_eq!(
        poll_inference_response,
        PollInferenceResponse::Pending,
        "Response should be Pending"
    );
    database.sleep_for_writes_to_be_visible().await;

    let query = PollPathParams {
        batch_id,
        inference_id: None,
    };
    let batch_request_result = get_batch_request(&database, &query).await.unwrap();
    assert_eq!(
        batch_request_result.batch_id, batch_id,
        "batch_id should match"
    );
    assert_eq!(
        batch_request_result.status,
        BatchStatus::Pending,
        "status should be Pending"
    );

    // Write a failed batch
    let status = BatchStatus::Failed;
    let batch_request = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id,
        batch_params: &batch_params,
        function_name,
        variant_name,
        model_name,
        model_provider_name,
        raw_request: &raw_request,
        raw_response: &raw_response,
        status,
        errors: vec![],
        snapshot_hash: Some(SnapshotHash::new_test()),
    });
    let poll_inference_response = write_poll_batch_inference(
        &database,
        &batch_request,
        PollBatchInferenceResponse::Failed {
            raw_request: raw_request.clone(),
            raw_response: raw_response.clone(),
        },
        Arc::from("dummy"),
        &config,
    )
    .await
    .unwrap();
    assert_eq!(
        poll_inference_response,
        PollInferenceResponse::Failed,
        "Response should be Failed"
    );

    database.sleep_for_writes_to_be_visible().await;
    let query = PollPathParams {
        batch_id,
        inference_id: None,
    };
    // This should return the failed batch as it is more recent
    let batch_request = get_batch_request(&database, &query).await.unwrap();
    assert_eq!(batch_request.batch_id, batch_id, "batch_id should match");
    assert_eq!(
        batch_request.status,
        BatchStatus::Failed,
        "status should be Failed"
    );
}
make_db_test!(test_write_poll_batch_inference_endpoint);

/// ClickHouse-specific test to verify that BatchRequest has snapshot_hash
async fn test_batch_request_has_snapshot_hash(clickhouse: ClickHouseConnectionInfo) {
    let database = create_delegating_connection(clickhouse.clone());
    let batch_id = Uuid::now_v7();
    let batch_params = json!({"baz": "bat"});
    let function_name = "test_function";
    let variant_name = "test_variant";
    let model_name = "test_model";
    let model_provider_name = "test_model_provider";
    let raw_request = "raw request".to_string();
    let raw_response = "raw response".to_string();
    let batch_request = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id,
        batch_params: &batch_params,
        function_name,
        variant_name,
        model_name,
        model_provider_name,
        raw_request: &raw_request,
        raw_response: &raw_response,
        status: BatchStatus::Pending,
        errors: vec![],
        snapshot_hash: Some(SnapshotHash::new_test()),
    });
    let config = Config::new_empty()
        .await
        .unwrap()
        .into_config_without_writing_for_tests();

    write_poll_batch_inference(
        &database,
        &batch_request,
        PollBatchInferenceResponse::Pending {
            raw_request: raw_request.clone(),
            raw_response: raw_response.clone(),
        },
        Arc::from("dummy"),
        &config,
    )
    .await
    .unwrap();
    database.clickhouse.sleep_for_writes_to_be_visible().await;

    let batch_request_query = format!(
        "SELECT snapshot_hash FROM BatchRequest WHERE batch_id = '{batch_id}' ORDER BY timestamp DESC LIMIT 1 FORMAT JSONEachRow"
    );
    let response = clickhouse
        .run_query_synchronous_no_params(batch_request_query)
        .await
        .unwrap();
    let batch_request_row: serde_json::Value = serde_json::from_str(&response.response).unwrap();
    assert!(
        !batch_request_row["snapshot_hash"].is_null(),
        "BatchRequest should have snapshot_hash"
    );
}
make_clickhouse_only_test!(test_batch_request_has_snapshot_hash);

async fn test_get_batch_inferences_endpoint(
    database: impl BatchInferenceQueries + TestDatabaseHelpers,
) {
    let batch_id = Uuid::now_v7();
    let mut expected_batch_rows = write_2_batch_model_inference_rows(&database, batch_id).await;
    let other_batch_id = Uuid::now_v7();
    let other_batch_rows = write_2_batch_model_inference_rows(&database, other_batch_id).await;
    database.sleep_for_writes_to_be_visible().await;

    let mut batch_inferences = get_batch_inferences(
        &database,
        batch_id,
        &[
            expected_batch_rows[0].inference_id,
            expected_batch_rows[1].inference_id,
            other_batch_rows[0].inference_id,
            other_batch_rows[1].inference_id,
        ],
    )
    .await
    .unwrap();

    assert_eq!(
        batch_inferences.len(),
        2,
        "Should return 2 batch inferences for the matching batch_id"
    );

    // Sort both arrays by inference_id to ensure matching order
    batch_inferences.sort_by_key(|row| row.inference_id);
    expected_batch_rows.sort_by_key(|row| row.inference_id);

    assert_eq!(
        batch_inferences[0], expected_batch_rows[0],
        "First batch inference should match"
    );
    assert_eq!(
        batch_inferences[1], expected_batch_rows[1],
        "Second batch inference should match"
    );
}
make_db_test!(test_get_batch_inferences_endpoint);

async fn test_write_read_completed_batch_inference_chat(
    database: impl BatchInferenceQueries
    + InferenceQueries
    + ModelInferenceQueries
    + TestDatabaseHelpers,
) {
    let batch_id = Uuid::now_v7();
    let batch_params = json!({"baz": "bat"});
    let function_name = "test_function";
    let variant_name = "test_variant";
    let model_name = "test_model";
    let model_provider_name = "test_model_provider";
    let raw_request = "raw request".to_string();
    let raw_response = "raw response".to_string();
    let status = BatchStatus::Pending;
    let errors = vec![];
    let batch_request = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id,
        batch_params: &batch_params,
        function_name,
        variant_name,
        model_name,
        model_provider_name,
        raw_request: &raw_request,
        raw_response: &raw_response,
        status,
        errors,
        snapshot_hash: Some(SnapshotHash::new_test()),
    });
    let function_config = Arc::new(FunctionConfig::Chat(FunctionConfigChat {
        variants: HashMap::new(),
        ..Default::default()
    }));
    let mut config = Config::new_empty()
        .await
        .unwrap()
        .into_config_without_writing_for_tests();
    config.functions = HashMap::from([(function_name.to_string(), function_config)]);

    let batch_model_inference_rows = write_2_batch_model_inference_rows(&database, batch_id).await;
    database.sleep_for_writes_to_be_visible().await;

    let inference_id1 = batch_model_inference_rows[0].inference_id;
    let output_1 = ProviderBatchInferenceOutput {
        id: inference_id1,
        output: vec!["hello world".to_string().into()],
        raw_response: String::new(),
        usage: Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            cost: None,
        },
        finish_reason: Some(FinishReason::Stop),
    };
    let inference_id2 = batch_model_inference_rows[1].inference_id;
    let output_2 = ProviderBatchInferenceOutput {
        id: inference_id2,
        output: vec!["goodbye world".to_string().into()],
        raw_response: String::new(),
        usage: Usage {
            input_tokens: Some(20),
            output_tokens: Some(30),
            cost: None,
        },
        finish_reason: Some(FinishReason::ToolCall),
    };
    let response = ProviderBatchInferenceResponse {
        elements: HashMap::from([(inference_id1, output_1), (inference_id2, output_2)]),
        raw_request: raw_request.clone(),
        raw_response: raw_response.clone(),
    };
    let mut inference_responses = write_completed_batch_inference(
        &database,
        &batch_request,
        response,
        Arc::from("dummy"),
        &config,
    )
    .await
    .unwrap();

    // Sort inferences by inference_id to ensure consistent ordering
    inference_responses.sort_by_key(tensorzero::InferenceResponse::inference_id);

    assert_eq!(
        inference_responses.len(),
        2,
        "Should return 2 inference responses"
    );
    let inference_response_1 = &inference_responses[0];
    let inference_response_2 = &inference_responses[1];

    match inference_response_1 {
        InferenceResponse::Chat(chat_inference_response) => {
            assert_eq!(
                chat_inference_response.inference_id, inference_id1,
                "inference_id should match"
            );
            assert_eq!(
                chat_inference_response.finish_reason,
                Some(FinishReason::Stop),
                "finish_reason should be Stop"
            );
            match &chat_inference_response.content[0] {
                ContentBlockChatOutput::Text(text_block) => {
                    assert_eq!(text_block.text, "hello world", "text should match");
                }
                _ => panic!("Unexpected content block type"),
            }
            assert_eq!(
                chat_inference_response.usage.input_tokens,
                Some(10),
                "input_tokens should match"
            );
            assert_eq!(
                chat_inference_response.usage.output_tokens,
                Some(20),
                "output_tokens should match"
            );
        }
        InferenceResponse::Json(_) => panic!("Unexpected inference response type"),
    }

    match inference_response_2 {
        InferenceResponse::Chat(chat_inference_response) => {
            assert_eq!(
                chat_inference_response.inference_id, inference_id2,
                "inference_id should match"
            );
            match &chat_inference_response.content[0] {
                ContentBlockChatOutput::Text(text_block) => {
                    assert_eq!(text_block.text, "goodbye world", "text should match");
                }
                _ => panic!("Unexpected content block type"),
            }
            assert_eq!(
                chat_inference_response.usage.input_tokens,
                Some(20),
                "input_tokens should match"
            );
            assert_eq!(
                chat_inference_response.usage.output_tokens,
                Some(30),
                "output_tokens should match"
            );
        }
        InferenceResponse::Json(_) => panic!("Unexpected inference response type"),
    }

    database.sleep_for_writes_to_be_visible().await;

    // Verify ChatInference rows were written via trait method
    let completed_rows = database
        .get_completed_chat_batch_inferences(batch_id, function_name, variant_name, None)
        .await
        .unwrap();
    assert_eq!(
        completed_rows.len(),
        2,
        "Should return 2 completed chat inferences"
    );
    let rows_by_id: HashMap<_, _> = completed_rows.iter().map(|r| (r.inference_id, r)).collect();
    let row1 = rows_by_id
        .get(&inference_id1)
        .expect("Should find inference_id1");
    assert_eq!(row1.variant_name, variant_name, "variant_name should match");
    let output1: Vec<ContentBlockChatOutput> =
        serde_json::from_str(row1.output.as_deref().expect("output should be present")).unwrap();
    assert_eq!(output1.len(), 1, "Should have 1 content block");
    match &output1[0] {
        ContentBlockChatOutput::Text(text) => {
            assert_eq!(text.text, "hello world", "text should match");
        }
        _ => panic!("Expected Text content block"),
    }
    assert_eq!(row1.input_tokens, Some(10), "input_tokens should match");
    assert_eq!(row1.output_tokens, Some(20), "output_tokens should match");
    assert_eq!(
        row1.finish_reason,
        Some(FinishReason::Stop),
        "finish_reason should match"
    );
    let row2 = rows_by_id
        .get(&inference_id2)
        .expect("Should find inference_id2");
    assert_eq!(row2.variant_name, variant_name, "variant_name should match");
    let output2: Vec<ContentBlockChatOutput> =
        serde_json::from_str(row2.output.as_deref().expect("output should be present")).unwrap();
    match &output2[0] {
        ContentBlockChatOutput::Text(text) => {
            assert_eq!(text.text, "goodbye world", "text should match");
        }
        _ => panic!("Expected Text content block"),
    }

    // Verify ModelInference rows were written via trait method
    let model_inferences = database
        .get_model_inferences_by_inference_id(inference_id1)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1, "Should have 1 model inference");
    assert_eq!(
        model_inferences[0].inference_id, inference_id1,
        "inference_id should match"
    );
    assert_eq!(
        model_inferences[0].model_name, model_name,
        "model_name should match"
    );
    assert_eq!(
        model_inferences[0].model_provider_name, model_provider_name,
        "model_provider_name should match"
    );
    assert!(
        model_inferences[0].snapshot_hash.is_some(),
        "ModelInference should have snapshot_hash"
    );
    let model_inferences = database
        .get_model_inferences_by_inference_id(inference_id2)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1, "Should have 1 model inference");
    assert_eq!(
        model_inferences[0].inference_id, inference_id2,
        "inference_id should match"
    );
    assert_eq!(
        model_inferences[0].model_name, model_name,
        "model_name should match"
    );
    assert_eq!(
        model_inferences[0].model_provider_name, model_provider_name,
        "model_provider_name should match"
    );
    assert!(
        model_inferences[0].snapshot_hash.is_some(),
        "ModelInference should have snapshot_hash"
    );

    // Read back using `get_completed_batch_inference_response`
    let query = PollPathParams {
        batch_id,
        inference_id: None,
    };
    let completed_inference_response = get_completed_batch_inference_response(
        &database,
        &batch_request,
        &query,
        &config.functions[function_name],
    )
    .await
    .unwrap();
    assert_eq!(
        completed_inference_response.batch_id, batch_id,
        "batch_id should match"
    );
    assert_eq!(
        completed_inference_response.inferences.len(),
        2,
        "Should return 2 inferences"
    );

    // Create HashMaps keyed by inference_id for both sets of inferences
    let completed_inferences: HashMap<_, _> = completed_inference_response
        .inferences
        .iter()
        .map(|r| (r.inference_id(), r))
        .collect();
    let original_inferences: HashMap<_, _> = [inference_response_1, inference_response_2]
        .into_iter()
        .map(|r| (r.inference_id(), r))
        .collect();

    // Compare the HashMaps
    assert_eq!(
        completed_inferences, original_inferences,
        "Completed inferences should match original inferences"
    );

    // Read back with a specific inference_id
    let query = PollPathParams {
        batch_id,
        inference_id: Some(inference_id1),
    };
    let completed_inference_response = get_completed_batch_inference_response(
        &database,
        &batch_request,
        &query,
        &config.functions[function_name],
    )
    .await
    .unwrap();
    assert_eq!(
        &completed_inference_response.inferences[0], original_inferences[&inference_id1],
        "Single inference query should return matching inference"
    );
}
make_db_test!(test_write_read_completed_batch_inference_chat);

async fn test_write_read_completed_batch_inference_json(
    database: impl BatchInferenceQueries
    + InferenceQueries
    + ModelInferenceQueries
    + TestDatabaseHelpers,
) {
    let batch_id = Uuid::now_v7();
    let batch_params = json!({"baz": "bat"});
    let function_name = "test_function";
    let variant_name = "test_variant";
    let model_name = "test_model";
    let model_provider_name = "test_model_provider";
    let raw_request = "raw request".to_string();
    let raw_response = "raw response".to_string();
    let status = BatchStatus::Pending;
    let batch_request = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id,
        batch_params: &batch_params,
        function_name,
        variant_name,
        raw_request: &raw_request,
        raw_response: &raw_response,
        model_name,
        model_provider_name,
        status,
        errors: vec![],
        snapshot_hash: Some(SnapshotHash::new_test()),
    });
    let output_schema = JSONSchema::from_value(json!({
        "type": "object",
        "properties": {
            "answer": {
                "type": "string"
            }
        },
        "required": ["answer"]
    }))
    .unwrap();
    let function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
        variants: HashMap::new(),
        output_schema,
        ..Default::default()
    }));
    let mut config = Config::new_empty()
        .await
        .unwrap()
        .into_config_without_writing_for_tests();
    config.functions = HashMap::from([(function_name.to_string(), function_config)]);

    let batch_model_inference_rows = write_2_batch_model_inference_rows(&database, batch_id).await;
    database.sleep_for_writes_to_be_visible().await;

    let inference_id1 = batch_model_inference_rows[0].inference_id;
    let output_1 = ProviderBatchInferenceOutput {
        id: inference_id1,
        output: vec!["{\"answer\": \"hello world\"}".to_string().into()],
        raw_response: String::new(),
        usage: Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            cost: None,
        },
        finish_reason: Some(FinishReason::Stop),
    };
    let inference_id2 = batch_model_inference_rows[1].inference_id;
    let output_2 = ProviderBatchInferenceOutput {
        id: inference_id2,
        output: vec!["{\"response\": \"goodbye world\"}".to_string().into()],
        raw_response: String::new(),
        usage: Usage {
            input_tokens: Some(20),
            output_tokens: Some(30),
            cost: None,
        },
        finish_reason: Some(FinishReason::ToolCall),
    };
    let response = ProviderBatchInferenceResponse {
        elements: HashMap::from([(inference_id1, output_1), (inference_id2, output_2)]),
        raw_request: raw_request.clone(),
        raw_response: raw_response.clone(),
    };
    let inference_responses = write_completed_batch_inference(
        &database,
        &batch_request,
        response,
        Arc::from("dummy"),
        &config,
    )
    .await
    .unwrap();

    assert_eq!(
        inference_responses.len(),
        2,
        "Should return 2 inference responses"
    );

    // Create a map of inferences by inference_id for easier lookup
    let inferences_by_id: HashMap<_, _> = inference_responses
        .iter()
        .map(|r| (r.inference_id(), r))
        .collect();

    // Now we can safely access each response by its ID
    let response_1 = inferences_by_id
        .get(&inference_id1)
        .expect("Missing response for inference_id1");
    let response_2 = inferences_by_id
        .get(&inference_id2)
        .expect("Missing response for inference_id2");

    match response_1 {
        InferenceResponse::Json(json_inference_response) => {
            assert_eq!(
                json_inference_response.inference_id, inference_id1,
                "inference_id should match"
            );
            assert_eq!(
                json_inference_response.output.raw,
                Some("{\"answer\": \"hello world\"}".to_string()),
                "raw output should match"
            );
            assert_eq!(
                json_inference_response.output.parsed.as_ref().unwrap()["answer"],
                "hello world",
                "parsed output should match"
            );
            assert_eq!(
                json_inference_response.usage.input_tokens,
                Some(10),
                "input_tokens should match"
            );
            assert_eq!(
                json_inference_response.finish_reason,
                Some(FinishReason::Stop),
                "finish_reason should be Stop"
            );
        }
        InferenceResponse::Chat(_) => panic!("Unexpected inference response type"),
    }

    match response_2 {
        InferenceResponse::Json(json_inference_response) => {
            assert_eq!(
                json_inference_response.inference_id, inference_id2,
                "inference_id should match"
            );
            assert_eq!(
                json_inference_response.output.raw,
                Some("{\"response\": \"goodbye world\"}".to_string()),
                "raw output should match"
            );
            assert!(
                json_inference_response.output.parsed.is_none(),
                "parsed output should be None for invalid schema"
            );
            assert_eq!(
                json_inference_response.finish_reason,
                Some(FinishReason::ToolCall),
                "finish_reason should be ToolCall"
            );
        }
        InferenceResponse::Chat(_) => panic!("Unexpected inference response type"),
    }

    database.sleep_for_writes_to_be_visible().await;

    // Verify JsonInference rows were written via trait method
    let completed_rows = database
        .get_completed_json_batch_inferences(batch_id, function_name, variant_name, None)
        .await
        .unwrap();
    assert_eq!(
        completed_rows.len(),
        2,
        "Should return 2 completed JSON inferences"
    );
    let rows_by_id: HashMap<_, _> = completed_rows.iter().map(|r| (r.inference_id, r)).collect();
    let row1 = rows_by_id
        .get(&inference_id1)
        .expect("Should find inference_id1");
    assert_eq!(row1.variant_name, variant_name, "variant_name should match");
    let output1: JsonInferenceOutput =
        serde_json::from_str(row1.output.as_deref().expect("output should be present")).unwrap();
    assert_eq!(
        output1.parsed.unwrap()["answer"],
        "hello world",
        "parsed output should match"
    );
    assert_eq!(
        output1.raw,
        Some("{\"answer\": \"hello world\"}".to_string()),
        "raw output should match"
    );
    assert_eq!(row1.input_tokens, Some(10), "input_tokens should match");
    assert_eq!(row1.output_tokens, Some(20), "output_tokens should match");
    assert_eq!(
        row1.finish_reason,
        Some(FinishReason::Stop),
        "finish_reason should match"
    );
    let row2 = rows_by_id
        .get(&inference_id2)
        .expect("Should find inference_id2");
    assert_eq!(row2.variant_name, variant_name, "variant_name should match");
    let output2: JsonInferenceOutput =
        serde_json::from_str(row2.output.as_deref().expect("output should be present")).unwrap();
    assert!(
        output2.parsed.is_none(),
        "parsed output should be None for invalid schema"
    );
    assert_eq!(
        output2.raw,
        Some("{\"response\": \"goodbye world\"}".to_string()),
        "raw output should match"
    );

    // Verify ModelInference rows were written via trait method
    let model_inferences = database
        .get_model_inferences_by_inference_id(inference_id1)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1, "Should have 1 model inference");
    assert_eq!(
        model_inferences[0].inference_id, inference_id1,
        "inference_id should match"
    );
    assert_eq!(
        model_inferences[0].model_name, model_name,
        "model_name should match"
    );
    assert_eq!(
        model_inferences[0].model_provider_name, model_provider_name,
        "model_provider_name should match"
    );
    assert!(
        model_inferences[0].snapshot_hash.is_some(),
        "ModelInference should have snapshot_hash"
    );
    let model_inferences = database
        .get_model_inferences_by_inference_id(inference_id2)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1, "Should have 1 model inference");
    assert_eq!(
        model_inferences[0].inference_id, inference_id2,
        "inference_id should match"
    );
    assert_eq!(
        model_inferences[0].model_name, model_name,
        "model_name should match"
    );
    assert_eq!(
        model_inferences[0].model_provider_name, model_provider_name,
        "model_provider_name should match"
    );
    assert!(
        model_inferences[0].snapshot_hash.is_some(),
        "ModelInference should have snapshot_hash"
    );

    // Read back using `get_completed_batch_inference_response`
    let query = PollPathParams {
        batch_id,
        inference_id: None,
    };
    let completed_inference_response = get_completed_batch_inference_response(
        &database,
        &batch_request,
        &query,
        &config.functions[function_name],
    )
    .await
    .unwrap();
    assert_eq!(
        completed_inference_response.batch_id, batch_id,
        "batch_id should match"
    );
    assert_eq!(
        completed_inference_response.inferences.len(),
        2,
        "Should return 2 inferences"
    );

    // Create HashMaps keyed by inference_id for both sets of inferences
    let completed_inferences: HashMap<_, _> = completed_inference_response
        .inferences
        .iter()
        .map(|r| (r.inference_id(), r))
        .collect();
    let original_inferences: HashMap<_, _> = [response_1, response_2]
        .into_iter()
        .map(|r| (r.inference_id(), *r))
        .collect();

    // Compare the HashMaps
    assert_eq!(
        completed_inferences, original_inferences,
        "Completed inferences should match original inferences"
    );

    // Read back with a specific inference_id
    let query = PollPathParams {
        batch_id,
        inference_id: Some(inference_id1),
    };
    let completed_inference_response = get_completed_batch_inference_response(
        &database,
        &batch_request,
        &query,
        &config.functions[function_name],
    )
    .await
    .unwrap();
    assert_eq!(
        &completed_inference_response.inferences[0], original_inferences[&inference_id1],
        "Single inference query should return matching inference"
    );
}
make_db_test!(test_write_read_completed_batch_inference_json);
