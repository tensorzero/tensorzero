//! E2E tests for batch inference queries (ClickHouse and Postgres).

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::db::batch_inference::BatchInferenceQueries;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::endpoints::inference::InferenceParams;
use tensorzero_core::inference::types::StoredInput;
use tensorzero_core::inference::types::batch::{
    BatchModelInferenceRow, BatchRequestRow, BatchStatus,
};
use uuid::Uuid;

/// Helper function to create a batch request row
fn create_batch_request_row(
    batch_id: Uuid,
    function_name: &str,
    variant_name: &str,
) -> BatchRequestRow<'static> {
    let id = Uuid::now_v7();
    BatchRequestRow {
        batch_id,
        id,
        batch_params: Cow::Owned(json!({"test": "params"})),
        model_name: Arc::from("test_model"),
        model_provider_name: Cow::Borrowed("test_provider"),
        status: BatchStatus::Completed,
        function_name: Cow::Owned(function_name.to_string()),
        variant_name: Cow::Owned(variant_name.to_string()),
        raw_request: Some(Cow::Borrowed("{}")),
        raw_response: Some(Cow::Borrowed("{}")),
        errors: vec![],
        snapshot_hash: None,
    }
}

/// Helper function to create a batch model inference row
fn create_batch_model_inference_row(
    batch_id: Uuid,
    inference_id: Uuid,
    episode_id: Uuid,
    function_name: &str,
    variant_name: &str,
) -> BatchModelInferenceRow<'static> {
    let input = StoredInput {
        system: None,
        messages: vec![],
    };
    BatchModelInferenceRow {
        batch_id,
        inference_id,
        function_name: Cow::Owned(function_name.to_string()),
        variant_name: Cow::Owned(variant_name.to_string()),
        episode_id,
        input: Some(input),
        input_messages: Some(vec![]),
        system: None,
        tool_params: None,
        inference_params: Some(Cow::Owned(InferenceParams::default())),
        output_schema: None,
        raw_request: Some(Cow::Borrowed("{}")),
        model_name: Cow::Borrowed("test_model"),
        model_provider_name: Cow::Borrowed("test_provider"),
        tags: HashMap::new(),
        snapshot_hash: Some(SnapshotHash::new_test()),
    }
}

async fn test_get_batch_request_by_batch_id(
    conn: impl BatchInferenceQueries + TestDatabaseHelpers,
) {
    let batch_id = Uuid::now_v7();
    let function_name = "test_batch_fn";
    let variant_name = "test_batch_var";

    let row = create_batch_request_row(batch_id, function_name, variant_name);
    conn.write_batch_request(&row).await.unwrap();
    conn.sleep_for_writes_to_be_visible().await;

    let result = conn.get_batch_request(batch_id, None).await.unwrap();

    assert!(
        result.is_some(),
        "Should find the batch request by batch_id"
    );
    let row = result.unwrap();
    assert_eq!(row.batch_id, batch_id, "batch_id should match");
    assert_eq!(
        row.function_name.as_ref(),
        function_name,
        "function_name should match"
    );
    assert_eq!(
        row.variant_name.as_ref(),
        variant_name,
        "variant_name should match"
    );
    assert_eq!(
        row.status,
        BatchStatus::Completed,
        "status should be Completed"
    );
}
make_db_test!(test_get_batch_request_by_batch_id);

async fn test_get_batch_request_by_batch_id_and_inference_id(
    conn: impl BatchInferenceQueries + TestDatabaseHelpers,
) {
    let batch_id = Uuid::now_v7();
    let inference_id = Uuid::now_v7();
    let episode_id = Uuid::now_v7();
    let function_name = "test_batch_fn_inf";
    let variant_name = "test_batch_var_inf";

    let batch_row = create_batch_request_row(batch_id, function_name, variant_name);
    conn.write_batch_request(&batch_row).await.unwrap();

    let inference_row = create_batch_model_inference_row(
        batch_id,
        inference_id,
        episode_id,
        function_name,
        variant_name,
    );
    conn.write_batch_model_inferences(&[inference_row])
        .await
        .unwrap();
    conn.sleep_for_writes_to_be_visible().await;

    let result = conn
        .get_batch_request(batch_id, Some(inference_id))
        .await
        .unwrap();

    assert!(
        result.is_some(),
        "Should find the batch request by batch_id and inference_id"
    );
    let row = result.unwrap();
    assert_eq!(row.batch_id, batch_id, "batch_id should match");
    assert_eq!(
        row.function_name.as_ref(),
        function_name,
        "function_name should match"
    );
}
make_db_test!(test_get_batch_request_by_batch_id_and_inference_id);

async fn test_get_batch_request_not_found(conn: impl BatchInferenceQueries + TestDatabaseHelpers) {
    let nonexistent_batch_id = Uuid::now_v7();

    let result = conn
        .get_batch_request(nonexistent_batch_id, None)
        .await
        .unwrap();

    assert!(
        result.is_none(),
        "Should return None for nonexistent batch_id"
    );
}
make_db_test!(test_get_batch_request_not_found);

async fn test_get_batch_request_wrong_inference_id(
    conn: impl BatchInferenceQueries + TestDatabaseHelpers,
) {
    let batch_id = Uuid::now_v7();
    let wrong_inference_id = Uuid::now_v7();
    let function_name = "test_batch_fn_wrong";
    let variant_name = "test_batch_var_wrong";

    let row = create_batch_request_row(batch_id, function_name, variant_name);
    conn.write_batch_request(&row).await.unwrap();
    conn.sleep_for_writes_to_be_visible().await;

    // Query with a non-existent inference_id should return None
    let result = conn
        .get_batch_request(batch_id, Some(wrong_inference_id))
        .await
        .unwrap();

    assert!(
        result.is_none(),
        "Should return None when inference_id doesn't belong to the batch"
    );
}
make_db_test!(test_get_batch_request_wrong_inference_id);

async fn test_get_batch_model_inferences(conn: impl BatchInferenceQueries + TestDatabaseHelpers) {
    let batch_id = Uuid::now_v7();
    let inference_id_1 = Uuid::now_v7();
    let inference_id_2 = Uuid::now_v7();
    let episode_id_1 = Uuid::now_v7();
    let episode_id_2 = Uuid::now_v7();
    let function_name = "test_batch_model_fn";
    let variant_name = "test_batch_model_var";

    let batch_row = create_batch_request_row(batch_id, function_name, variant_name);
    conn.write_batch_request(&batch_row).await.unwrap();

    let inference_row_1 = create_batch_model_inference_row(
        batch_id,
        inference_id_1,
        episode_id_1,
        function_name,
        variant_name,
    );
    let inference_row_2 = create_batch_model_inference_row(
        batch_id,
        inference_id_2,
        episode_id_2,
        function_name,
        variant_name,
    );
    conn.write_batch_model_inferences(&[inference_row_1, inference_row_2])
        .await
        .unwrap();
    conn.sleep_for_writes_to_be_visible().await;

    let result = conn
        .get_batch_model_inferences(batch_id, &[inference_id_1, inference_id_2])
        .await
        .unwrap();

    assert_eq!(result.len(), 2, "Should return 2 batch model inferences");

    let ids: Vec<Uuid> = result.iter().map(|r| r.inference_id).collect();
    assert!(
        ids.contains(&inference_id_1),
        "Should contain inference_id_1"
    );
    assert!(
        ids.contains(&inference_id_2),
        "Should contain inference_id_2"
    );
}
make_db_test!(test_get_batch_model_inferences);

async fn test_get_batch_model_inferences_partial(
    conn: impl BatchInferenceQueries + TestDatabaseHelpers,
) {
    let batch_id = Uuid::now_v7();
    let inference_id_1 = Uuid::now_v7();
    let inference_id_2 = Uuid::now_v7();
    let nonexistent_inference_id = Uuid::now_v7();
    let episode_id_1 = Uuid::now_v7();
    let episode_id_2 = Uuid::now_v7();
    let function_name = "test_batch_partial_fn";
    let variant_name = "test_batch_partial_var";

    let batch_row = create_batch_request_row(batch_id, function_name, variant_name);
    conn.write_batch_request(&batch_row).await.unwrap();

    let inference_row_1 = create_batch_model_inference_row(
        batch_id,
        inference_id_1,
        episode_id_1,
        function_name,
        variant_name,
    );
    let inference_row_2 = create_batch_model_inference_row(
        batch_id,
        inference_id_2,
        episode_id_2,
        function_name,
        variant_name,
    );
    conn.write_batch_model_inferences(&[inference_row_1, inference_row_2])
        .await
        .unwrap();
    conn.sleep_for_writes_to_be_visible().await;

    // Query with one valid and one invalid inference_id
    let result = conn
        .get_batch_model_inferences(batch_id, &[inference_id_1, nonexistent_inference_id])
        .await
        .unwrap();

    assert_eq!(result.len(), 1, "Should return only the existing inference");
    assert_eq!(
        result[0].inference_id, inference_id_1,
        "Should return inference_id_1"
    );
}
make_db_test!(test_get_batch_model_inferences_partial);

async fn test_get_batch_model_inferences_empty_ids(
    conn: impl BatchInferenceQueries + TestDatabaseHelpers,
) {
    let batch_id = Uuid::now_v7();

    let result = conn
        .get_batch_model_inferences(batch_id, &[])
        .await
        .unwrap();

    assert!(
        result.is_empty(),
        "Should return empty vec for empty inference_ids"
    );
}
make_db_test!(test_get_batch_model_inferences_empty_ids);

async fn test_get_batch_model_inferences_wrong_batch(
    conn: impl BatchInferenceQueries + TestDatabaseHelpers,
) {
    let batch_id = Uuid::now_v7();
    let wrong_batch_id = Uuid::now_v7();
    let inference_id = Uuid::now_v7();
    let episode_id = Uuid::now_v7();
    let function_name = "test_batch_wrong_fn";
    let variant_name = "test_batch_wrong_var";

    let batch_row = create_batch_request_row(batch_id, function_name, variant_name);
    conn.write_batch_request(&batch_row).await.unwrap();

    let inference_row = create_batch_model_inference_row(
        batch_id,
        inference_id,
        episode_id,
        function_name,
        variant_name,
    );
    conn.write_batch_model_inferences(&[inference_row])
        .await
        .unwrap();
    conn.sleep_for_writes_to_be_visible().await;

    // Query with wrong batch_id should return empty
    let result = conn
        .get_batch_model_inferences(wrong_batch_id, &[inference_id])
        .await
        .unwrap();

    assert!(
        result.is_empty(),
        "Should return empty when batch_id doesn't match"
    );
}
make_db_test!(test_get_batch_model_inferences_wrong_batch);
