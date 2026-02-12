//! E2E tests for Postgres batch write behavior.
//!
//! These tests verify that the `PostgresBatchSender` correctly buffers rows and flushes
//! them based on `max_rows` and `flush_interval_ms` configuration.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use uuid::Uuid;

use tensorzero_core::config::BatchWritesConfig;
use tensorzero_core::db::inferences::InferenceQueries;
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::postgres::batching::PostgresBatchSender;
use tensorzero_core::endpoints::inference::InferenceParams;
use tensorzero_core::inference::types::extra_body::UnfilteredInferenceExtraBody;
use tensorzero_core::inference::types::stored_input::StoredInput;
use tensorzero_core::inference::types::{
    ChatInferenceDatabaseInsert, FinishReason, JsonInferenceDatabaseInsert, JsonInferenceOutput,
    StoredModelInference,
};
use tensorzero_core::tool::ToolCallConfigDatabaseInsert;

use crate::db::get_test_postgres;

/// Creates a test pool from the environment, identical to `get_test_postgres` but returns the raw pool.
async fn get_test_pool() -> sqlx::PgPool {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("Environment variable TENSORZERO_POSTGRES_URL must be set");
    sqlx::postgres::PgPoolOptions::new()
        .connect(&postgres_url)
        .await
        .expect("Failed to connect to Postgres")
}

fn make_chat_inference(function_name: &str) -> ChatInferenceDatabaseInsert {
    ChatInferenceDatabaseInsert {
        id: Uuid::now_v7(),
        function_name: function_name.to_string(),
        variant_name: "test_variant".to_string(),
        episode_id: Uuid::now_v7(),
        input: StoredInput::default(),
        output: vec![],
        tool_params: Some(ToolCallConfigDatabaseInsert::default()),
        inference_params: InferenceParams::default(),
        processing_time_ms: Some(42),
        ttft_ms: None,
        tags: HashMap::new(),
        extra_body: UnfilteredInferenceExtraBody::default(),
        snapshot_hash: None,
    }
}

fn make_json_inference(function_name: &str) -> JsonInferenceDatabaseInsert {
    JsonInferenceDatabaseInsert {
        id: Uuid::now_v7(),
        function_name: function_name.to_string(),
        variant_name: "test_variant".to_string(),
        episode_id: Uuid::now_v7(),
        input: StoredInput::default(),
        output: JsonInferenceOutput {
            raw: Some("{}".to_string()),
            parsed: Some(serde_json::json!({})),
        },
        auxiliary_content: vec![],
        inference_params: InferenceParams::default(),
        processing_time_ms: Some(42),
        output_schema: serde_json::json!({}),
        ttft_ms: None,
        tags: HashMap::new(),
        extra_body: UnfilteredInferenceExtraBody::default(),
        snapshot_hash: None,
    }
}

fn make_model_inference(inference_id: Uuid) -> StoredModelInference {
    StoredModelInference {
        id: Uuid::now_v7(),
        inference_id,
        raw_request: "{}".to_string(),
        raw_response: "{}".to_string(),
        system: None,
        input_messages: vec![],
        output: vec![],
        input_tokens: Some(10),
        output_tokens: Some(20),
        response_time_ms: Some(100),
        model_name: "test-model".to_string(),
        model_provider_name: "test-provider".to_string(),
        ttft_ms: None,
        cached: false,
        finish_reason: Some(FinishReason::Stop),
        snapshot_hash: None,
        timestamp: None,
        cost: None,
    }
}

async fn count_by_function_name(pool: &sqlx::PgPool, table: &str, function_name: &str) -> i64 {
    let mut qb =
        sqlx::QueryBuilder::<sqlx::Postgres>::new("SELECT COUNT(*)::BIGINT FROM tensorzero.");
    qb.push(table);
    qb.push(" WHERE function_name = ");
    qb.push_bind(function_name);
    let count: i64 = qb
        .build_query_scalar()
        .fetch_one(pool)
        .await
        .expect("count query should succeed");
    count
}

async fn count_model_inferences_by_inference_id(pool: &sqlx::PgPool, inference_id: Uuid) -> i64 {
    sqlx::query_scalar!(
        r#"SELECT COUNT(*)::BIGINT as "count!" FROM tensorzero.model_inferences WHERE inference_id = $1"#,
        inference_id
    )
    .fetch_one(pool)
    .await
    .expect("count query should succeed")
}

/// Rows buffered below `max_rows` should not be written until the flush interval elapses.
/// Sending exactly `max_rows` should trigger a flush.
#[tokio::test(flavor = "multi_thread")]
async fn test_chat_inferences_flush_on_max_rows() {
    let pool = get_test_pool().await;
    let max_rows = 5;
    let function_name = format!("test_batch_chat_max_rows_{}", Uuid::now_v7());

    let sender = Arc::new(
        PostgresBatchSender::new(
            pool.clone(),
            BatchWritesConfig {
                enabled: true,
                flush_interval_ms: 60_000, // Very long — should not trigger
                max_rows,
                ..Default::default()
            },
        )
        .expect("should create batch sender"),
    );
    let conn = PostgresConnectionInfo::new_with_pool_and_batcher(pool.clone(), sender.clone());

    // Send max_rows - 1 rows: should NOT be flushed yet
    let rows: Vec<_> = (0..max_rows - 1)
        .map(|_| make_chat_inference(&function_name))
        .collect();
    conn.insert_chat_inferences(&rows)
        .await
        .expect("insert should succeed");

    // Give the writer a moment to process (it shouldn't flush)
    tokio::time::sleep(Duration::from_millis(200)).await;

    let count = count_by_function_name(&pool, "chat_inferences", &function_name).await;
    assert_eq!(
        count, 0,
        "Rows below max_rows should not be flushed before the flush interval"
    );

    // Send 1 more row to hit max_rows exactly
    let extra = make_chat_inference(&function_name);
    conn.insert_chat_inferences(std::slice::from_ref(&extra))
        .await
        .expect("insert should succeed");

    // Give the writer time to flush
    tokio::time::sleep(Duration::from_millis(500)).await;

    let count = count_by_function_name(&pool, "chat_inferences", &function_name).await;
    assert_eq!(
        count, max_rows as i64,
        "All rows should be flushed once max_rows is reached"
    );
}

/// Same test for JSON inferences.
#[tokio::test(flavor = "multi_thread")]
async fn test_json_inferences_flush_on_max_rows() {
    let pool = get_test_pool().await;
    let max_rows = 5;
    let function_name = format!("test_batch_json_max_rows_{}", Uuid::now_v7());

    let sender = Arc::new(
        PostgresBatchSender::new(
            pool.clone(),
            BatchWritesConfig {
                enabled: true,
                flush_interval_ms: 60_000,
                max_rows,
                ..Default::default()
            },
        )
        .expect("should create batch sender"),
    );
    let conn = PostgresConnectionInfo::new_with_pool_and_batcher(pool.clone(), sender.clone());

    // Send max_rows - 1 rows
    let rows: Vec<_> = (0..max_rows - 1)
        .map(|_| make_json_inference(&function_name))
        .collect();
    conn.insert_json_inferences(&rows)
        .await
        .expect("insert should succeed");

    tokio::time::sleep(Duration::from_millis(200)).await;

    let count = count_by_function_name(&pool, "json_inferences", &function_name).await;
    assert_eq!(
        count, 0,
        "Rows below max_rows should not be flushed before the flush interval"
    );

    // Send 1 more row to hit max_rows
    let extra = make_json_inference(&function_name);
    conn.insert_json_inferences(std::slice::from_ref(&extra))
        .await
        .expect("insert should succeed");

    tokio::time::sleep(Duration::from_millis(500)).await;

    let count = count_by_function_name(&pool, "json_inferences", &function_name).await;
    assert_eq!(
        count, max_rows as i64,
        "All rows should be flushed once max_rows is reached"
    );
}

/// Same test for model inferences.
#[tokio::test(flavor = "multi_thread")]
async fn test_model_inferences_flush_on_max_rows() {
    let pool = get_test_pool().await;
    let max_rows = 5;
    let inference_id = Uuid::now_v7();

    let sender = Arc::new(
        PostgresBatchSender::new(
            pool.clone(),
            BatchWritesConfig {
                enabled: true,
                flush_interval_ms: 60_000,
                max_rows,
                ..Default::default()
            },
        )
        .expect("should create batch sender"),
    );
    let conn = PostgresConnectionInfo::new_with_pool_and_batcher(pool.clone(), sender.clone());

    // Send max_rows - 1 rows
    let rows: Vec<_> = (0..max_rows - 1)
        .map(|_| make_model_inference(inference_id))
        .collect();
    conn.insert_model_inferences(&rows)
        .await
        .expect("insert should succeed");

    tokio::time::sleep(Duration::from_millis(200)).await;

    let count = count_model_inferences_by_inference_id(&pool, inference_id).await;
    assert_eq!(
        count, 0,
        "Rows below max_rows should not be flushed before the flush interval"
    );

    // Send 1 more row to hit max_rows
    let extra = make_model_inference(inference_id);
    conn.insert_model_inferences(std::slice::from_ref(&extra))
        .await
        .expect("insert should succeed");

    tokio::time::sleep(Duration::from_millis(500)).await;

    let count = count_model_inferences_by_inference_id(&pool, inference_id).await;
    assert_eq!(
        count, max_rows as i64,
        "All rows should be flushed once max_rows is reached"
    );
}

/// Rows should be flushed when the flush interval elapses, even if max_rows is not reached.
#[tokio::test(flavor = "multi_thread")]
async fn test_flush_on_timeout() {
    let pool = get_test_pool().await;
    let function_name = format!("test_batch_timeout_{}", Uuid::now_v7());

    let sender = Arc::new(
        PostgresBatchSender::new(
            pool.clone(),
            BatchWritesConfig {
                enabled: true,
                flush_interval_ms: 1000,
                max_rows: 1000, // Very large — should not trigger via row count
                ..Default::default()
            },
        )
        .expect("should create batch sender"),
    );
    let conn = PostgresConnectionInfo::new_with_pool_and_batcher(pool.clone(), sender.clone());

    let rows: Vec<_> = (0..3)
        .map(|_| make_chat_inference(&function_name))
        .collect();
    conn.insert_chat_inferences(&rows)
        .await
        .expect("insert should succeed");

    // Should not be flushed immediately
    let count = count_by_function_name(&pool, "chat_inferences", &function_name).await;
    assert_eq!(count, 0, "Rows should not be flushed immediately");

    // Wait for the flush interval plus some margin
    tokio::time::sleep(Duration::from_millis(1500)).await;

    let count = count_by_function_name(&pool, "chat_inferences", &function_name).await;
    assert_eq!(
        count, 3,
        "Rows should be flushed after the flush interval elapses"
    );
}

/// Closing channels (dropping the sender) should drain all remaining buffered rows.
#[tokio::test(flavor = "multi_thread")]
async fn test_drain_on_close() {
    let pool = get_test_pool().await;
    let function_name = format!("test_batch_drain_{}", Uuid::now_v7());
    let inference_id = Uuid::now_v7();

    let sender = Arc::new(
        PostgresBatchSender::new(
            pool.clone(),
            BatchWritesConfig {
                enabled: true,
                flush_interval_ms: 60_000, // Very long
                max_rows: 1000,            // Very large
                ..Default::default()
            },
        )
        .expect("should create batch sender"),
    );
    let writer_handle = sender.writer_handle.clone();
    let conn = PostgresConnectionInfo::new_with_pool_and_batcher(pool.clone(), sender);

    // Send some rows (below both thresholds)
    let chat_rows: Vec<_> = (0..3)
        .map(|_| make_chat_inference(&function_name))
        .collect();
    conn.insert_chat_inferences(&chat_rows)
        .await
        .expect("insert should succeed");

    let model_rows: Vec<_> = (0..2).map(|_| make_model_inference(inference_id)).collect();
    conn.insert_model_inferences(&model_rows)
        .await
        .expect("insert should succeed");

    // Not yet flushed
    let count = count_by_function_name(&pool, "chat_inferences", &function_name).await;
    assert_eq!(count, 0, "Should not be flushed yet");

    // Drop the connection info, which drops the Arc<PostgresBatchSender>,
    // closing the channels and allowing the writer to drain.
    drop(conn);

    // Wait for the writer to finish
    writer_handle.await.expect("writer should finish cleanly");

    let chat_count = count_by_function_name(&pool, "chat_inferences", &function_name).await;
    assert_eq!(
        chat_count, 3,
        "All chat inference rows should be drained on close"
    );

    let model_count = count_model_inferences_by_inference_id(&pool, inference_id).await;
    assert_eq!(
        model_count, 2,
        "All model inference rows should be drained on close"
    );
}

/// Without batching enabled, `PostgresConnectionInfo::new_with_pool` should write directly.
#[tokio::test(flavor = "multi_thread")]
async fn test_direct_writes_without_batching() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();
    let function_name = format!("test_direct_write_{}", Uuid::now_v7());

    // Verify batch_sender is None for the plain connection
    assert!(
        conn.batch_sender().is_none(),
        "Plain connection should have no batch sender"
    );

    let rows: Vec<_> = (0..3)
        .map(|_| make_chat_inference(&function_name))
        .collect();
    conn.insert_chat_inferences(&rows)
        .await
        .expect("direct insert should succeed");

    // Data should be visible immediately
    let count = count_by_function_name(&pool, "chat_inferences", &function_name).await;
    assert_eq!(
        count, 3,
        "Direct writes should be visible immediately without batching"
    );
}

/// Rows exceeding max_rows should flush the first batch and buffer the remainder.
#[tokio::test(flavor = "multi_thread")]
async fn test_overflow_beyond_max_rows() {
    let pool = get_test_pool().await;
    let max_rows = 3;
    let function_name = format!("test_batch_overflow_{}", Uuid::now_v7());

    let sender = Arc::new(
        PostgresBatchSender::new(
            pool.clone(),
            BatchWritesConfig {
                enabled: true,
                flush_interval_ms: 60_000,
                max_rows,
                ..Default::default()
            },
        )
        .expect("should create batch sender"),
    );
    let writer_handle = sender.writer_handle.clone();
    let conn = PostgresConnectionInfo::new_with_pool_and_batcher(pool.clone(), sender);

    // Send max_rows + 2 rows in one call
    let total = max_rows + 2;
    let rows: Vec<_> = (0..total)
        .map(|_| make_chat_inference(&function_name))
        .collect();
    conn.insert_chat_inferences(&rows)
        .await
        .expect("insert should succeed");

    // Wait for the first batch to flush
    tokio::time::sleep(Duration::from_millis(500)).await;

    let count = count_by_function_name(&pool, "chat_inferences", &function_name).await;
    assert_eq!(
        count, max_rows as i64,
        "First batch of max_rows should be flushed; remainder stays buffered"
    );

    // Drop to drain the remaining rows
    drop(conn);
    writer_handle.await.expect("writer should finish cleanly");

    let count = count_by_function_name(&pool, "chat_inferences", &function_name).await;
    assert_eq!(
        count, total as i64,
        "All rows (including overflow) should be flushed after drain"
    );
}
