//! E2E tests for batch inference ClickHouse queries.

use std::borrow::Cow;
use std::collections::HashMap;

use serde_json::json;
use tensorzero_core::db::batch_inference::BatchInferenceQueries;
use tensorzero_core::db::clickhouse::TableName;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::endpoints::batch_inference::write_batch_request_row;
use tensorzero_core::endpoints::inference::InferenceParams;
use tensorzero_core::inference::types::StoredInput;
use tensorzero_core::inference::types::batch::{
    BatchModelInferenceRow, BatchRequestRow, BatchStatus, UnparsedBatchRequestRow,
};
use tokio::time::{Duration, sleep};
use uuid::Uuid;

/// Helper function to create and write a batch request row
async fn create_batch_request(
    clickhouse: &tensorzero_core::db::clickhouse::ClickHouseConnectionInfo,
    batch_id: Uuid,
    function_name: &str,
    variant_name: &str,
) {
    let batch_params = json!({"test": "params"});
    let batch_request = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id,
        batch_params: &batch_params,
        function_name,
        variant_name,
        model_name: "test_model",
        model_provider_name: "test_provider",
        raw_request: "{}",
        raw_response: "{}",
        status: BatchStatus::Completed,
        errors: vec![],
    });
    write_batch_request_row(clickhouse, &batch_request)
        .await
        .unwrap();
}

/// Helper function to create and write a batch model inference row
async fn create_batch_model_inference(
    clickhouse: &tensorzero_core::db::clickhouse::ClickHouseConnectionInfo,
    batch_id: Uuid,
    inference_id: Uuid,
    episode_id: Uuid,
    function_name: &str,
    variant_name: &str,
) {
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
        input,
        input_messages: vec![],
        system: None,
        tool_params: None,
        inference_params: Cow::Owned(InferenceParams::default()),
        output_schema: None,
        raw_request: Cow::Borrowed("{}"),
        model_name: Cow::Borrowed("test_model"),
        model_provider_name: Cow::Borrowed("test_provider"),
        tags: HashMap::new(),
    };
    clickhouse
        .write_batched(&[row], TableName::BatchModelInference)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_get_batch_request_by_batch_id() {
    let clickhouse = get_clickhouse().await;
    let batch_id = Uuid::now_v7();
    let function_name = "test_batch_fn";
    let variant_name = "test_batch_var";

    create_batch_request(&clickhouse, batch_id, function_name, variant_name).await;
    sleep(Duration::from_millis(200)).await;

    let result = clickhouse.get_batch_request(batch_id, None).await.unwrap();

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

#[tokio::test]
async fn test_get_batch_request_by_batch_id_and_inference_id() {
    let clickhouse = get_clickhouse().await;
    let batch_id = Uuid::now_v7();
    let inference_id = Uuid::now_v7();
    let episode_id = Uuid::now_v7();
    let function_name = "test_batch_fn_inf";
    let variant_name = "test_batch_var_inf";

    create_batch_request(&clickhouse, batch_id, function_name, variant_name).await;
    create_batch_model_inference(
        &clickhouse,
        batch_id,
        inference_id,
        episode_id,
        function_name,
        variant_name,
    )
    .await;
    sleep(Duration::from_millis(200)).await;

    let result = clickhouse
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

#[tokio::test]
async fn test_get_batch_request_not_found() {
    let clickhouse = get_clickhouse().await;
    let nonexistent_batch_id = Uuid::now_v7();

    let result = clickhouse
        .get_batch_request(nonexistent_batch_id, None)
        .await
        .unwrap();

    assert!(
        result.is_none(),
        "Should return None for nonexistent batch_id"
    );
}

#[tokio::test]
async fn test_get_batch_request_wrong_inference_id() {
    let clickhouse = get_clickhouse().await;
    let batch_id = Uuid::now_v7();
    let wrong_inference_id = Uuid::now_v7();
    let function_name = "test_batch_fn_wrong";
    let variant_name = "test_batch_var_wrong";

    create_batch_request(&clickhouse, batch_id, function_name, variant_name).await;
    sleep(Duration::from_millis(200)).await;

    // Query with a non-existent inference_id should return None
    let result = clickhouse
        .get_batch_request(batch_id, Some(wrong_inference_id))
        .await
        .unwrap();

    assert!(
        result.is_none(),
        "Should return None when inference_id doesn't belong to the batch"
    );
}

#[tokio::test]
async fn test_get_batch_model_inferences() {
    let clickhouse = get_clickhouse().await;
    let batch_id = Uuid::now_v7();
    let inference_id_1 = Uuid::now_v7();
    let inference_id_2 = Uuid::now_v7();
    let episode_id_1 = Uuid::now_v7();
    let episode_id_2 = Uuid::now_v7();
    let function_name = "test_batch_model_fn";
    let variant_name = "test_batch_model_var";

    create_batch_request(&clickhouse, batch_id, function_name, variant_name).await;
    create_batch_model_inference(
        &clickhouse,
        batch_id,
        inference_id_1,
        episode_id_1,
        function_name,
        variant_name,
    )
    .await;
    create_batch_model_inference(
        &clickhouse,
        batch_id,
        inference_id_2,
        episode_id_2,
        function_name,
        variant_name,
    )
    .await;
    sleep(Duration::from_millis(200)).await;

    let result = clickhouse
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

#[tokio::test]
async fn test_get_batch_model_inferences_partial() {
    let clickhouse = get_clickhouse().await;
    let batch_id = Uuid::now_v7();
    let inference_id_1 = Uuid::now_v7();
    let inference_id_2 = Uuid::now_v7();
    let nonexistent_inference_id = Uuid::now_v7();
    let episode_id_1 = Uuid::now_v7();
    let episode_id_2 = Uuid::now_v7();
    let function_name = "test_batch_partial_fn";
    let variant_name = "test_batch_partial_var";

    create_batch_request(&clickhouse, batch_id, function_name, variant_name).await;
    create_batch_model_inference(
        &clickhouse,
        batch_id,
        inference_id_1,
        episode_id_1,
        function_name,
        variant_name,
    )
    .await;
    create_batch_model_inference(
        &clickhouse,
        batch_id,
        inference_id_2,
        episode_id_2,
        function_name,
        variant_name,
    )
    .await;
    sleep(Duration::from_millis(200)).await;

    // Query with one valid and one invalid inference_id
    let result = clickhouse
        .get_batch_model_inferences(batch_id, &[inference_id_1, nonexistent_inference_id])
        .await
        .unwrap();

    assert_eq!(result.len(), 1, "Should return only the existing inference");
    assert_eq!(
        result[0].inference_id, inference_id_1,
        "Should return inference_id_1"
    );
}

#[tokio::test]
async fn test_get_batch_model_inferences_empty_ids() {
    let clickhouse = get_clickhouse().await;
    let batch_id = Uuid::now_v7();

    let result = clickhouse
        .get_batch_model_inferences(batch_id, &[])
        .await
        .unwrap();

    assert!(
        result.is_empty(),
        "Should return empty vec for empty inference_ids"
    );
}

#[tokio::test]
async fn test_get_batch_model_inferences_wrong_batch() {
    let clickhouse = get_clickhouse().await;
    let batch_id = Uuid::now_v7();
    let wrong_batch_id = Uuid::now_v7();
    let inference_id = Uuid::now_v7();
    let episode_id = Uuid::now_v7();
    let function_name = "test_batch_wrong_fn";
    let variant_name = "test_batch_wrong_var";

    create_batch_request(&clickhouse, batch_id, function_name, variant_name).await;
    create_batch_model_inference(
        &clickhouse,
        batch_id,
        inference_id,
        episode_id,
        function_name,
        variant_name,
    )
    .await;
    sleep(Duration::from_millis(200)).await;

    // Query with wrong batch_id should return empty
    let result = clickhouse
        .get_batch_model_inferences(wrong_batch_id, &[inference_id])
        .await
        .unwrap();

    assert!(
        result.is_empty(),
        "Should return empty when batch_id doesn't match"
    );
}
