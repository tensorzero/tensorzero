//! E2E tests for inference count PostgreSQL queries.

use sqlx::ConnectOptions;
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use tensorzero_core::db::TimeWindow;
use tensorzero_core::db::inference_count::{
    CountInferencesParams, CountInferencesWithDemonstrationFeedbacksParams,
    CountInferencesWithFeedbackParams, InferenceCountQueries,
};
use tensorzero_core::db::postgres::{
    PostgresConnectionInfo, manual_run_postgres_migrations_with_url,
};
use tensorzero_core::{
    config::{MetricConfig, MetricConfigLevel, MetricConfigOptimize, MetricConfigType},
    db::inference_count::GetFunctionThroughputByVariantParams,
    function::FunctionConfigType,
};

// ===== HELPER FUNCTIONS =====

/// Inserts a test chat inference into the database.
async fn insert_chat_inference(
    pool: &sqlx::PgPool,
    id: uuid::Uuid,
    function_name: &str,
    variant_name: &str,
    episode_id: uuid::Uuid,
) {
    sqlx::query(
        r"INSERT INTO tensorzero.chat_inference (id, function_name, variant_name, episode_id, input, output, tags)
           VALUES ($1, $2, $3, $4, $5, $6, $7)",
    )
    .bind(id)
    .bind(function_name)
    .bind(variant_name)
    .bind(episode_id)
    .bind(serde_json::json!({"messages": []}))
    .bind(serde_json::json!({"content": "test"}))
    .bind(serde_json::json!({}))
    .execute(pool)
    .await
    .expect("Failed to insert chat inference");
}

/// Inserts a test JSON inference into the database.
async fn insert_json_inference(
    pool: &sqlx::PgPool,
    id: uuid::Uuid,
    function_name: &str,
    variant_name: &str,
    episode_id: uuid::Uuid,
) {
    sqlx::query(
        r"INSERT INTO tensorzero.json_inference (id, function_name, variant_name, episode_id, input, output, tags)
           VALUES ($1, $2, $3, $4, $5, $6, $7)",
    )
    .bind(id)
    .bind(function_name)
    .bind(variant_name)
    .bind(episode_id)
    .bind(serde_json::json!({"messages": []}))
    .bind(serde_json::json!({"result": "test"}))
    .bind(serde_json::json!({}))
    .execute(pool)
    .await
    .expect("Failed to insert json inference");
}

// ===== COUNT INFERENCES FOR FUNCTION TESTS =====

#[sqlx::test]
async fn test_count_inferences_for_function_empty(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let params = CountInferencesParams {
        function_name: "nonexistent_function",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let count = conn.count_inferences_for_function(params).await.unwrap();
    assert_eq!(count, 0, "Empty table should return 0 count");
}

#[sqlx::test]
async fn test_count_inferences_for_function_chat(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts.clone()).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool.clone());

    // Insert test data
    let episode_id = uuid::Uuid::now_v7();
    for _ in 0..5 {
        let id = uuid::Uuid::now_v7();
        insert_chat_inference(&pool, id, "test_chat_func", "variant_a", episode_id).await;
    }

    let params = CountInferencesParams {
        function_name: "test_chat_func",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let count = conn.count_inferences_for_function(params).await.unwrap();
    assert_eq!(count, 5, "Should count 5 chat inferences");
}

#[sqlx::test]
async fn test_count_inferences_for_function_json(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts.clone()).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool.clone());

    // Insert test data
    let episode_id = uuid::Uuid::now_v7();
    for _ in 0..3 {
        let id = uuid::Uuid::now_v7();
        insert_json_inference(&pool, id, "test_json_func", "variant_b", episode_id).await;
    }

    let params = CountInferencesParams {
        function_name: "test_json_func",
        function_type: FunctionConfigType::Json,
        variant_name: None,
    };

    let count = conn.count_inferences_for_function(params).await.unwrap();
    assert_eq!(count, 3, "Should count 3 json inferences");
}

#[sqlx::test]
async fn test_count_inferences_for_function_with_variant(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts.clone()).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool.clone());

    // Insert test data with different variants
    let episode_id = uuid::Uuid::now_v7();
    for _ in 0..4 {
        let id = uuid::Uuid::now_v7();
        insert_chat_inference(&pool, id, "multi_variant_func", "variant_a", episode_id).await;
    }
    for _ in 0..2 {
        let id = uuid::Uuid::now_v7();
        insert_chat_inference(&pool, id, "multi_variant_func", "variant_b", episode_id).await;
    }

    // Count only variant_a
    let params = CountInferencesParams {
        function_name: "multi_variant_func",
        function_type: FunctionConfigType::Chat,
        variant_name: Some("variant_a"),
    };

    let count = conn.count_inferences_for_function(params).await.unwrap();
    assert_eq!(count, 4, "Should count 4 inferences for variant_a");

    // Count only variant_b
    let params = CountInferencesParams {
        function_name: "multi_variant_func",
        function_type: FunctionConfigType::Chat,
        variant_name: Some("variant_b"),
    };

    let count = conn.count_inferences_for_function(params).await.unwrap();
    assert_eq!(count, 2, "Should count 2 inferences for variant_b");
}

// ===== COUNT INFERENCES BY VARIANT TESTS =====

#[sqlx::test]
async fn test_count_inferences_by_variant(pool_opts: PgPoolOptions, conn_opts: PgConnectOptions) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts.clone()).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool.clone());

    // Insert test data with different variants
    let episode_id = uuid::Uuid::now_v7();
    for _ in 0..5 {
        let id = uuid::Uuid::now_v7();
        insert_chat_inference(&pool, id, "by_variant_func", "variant_a", episode_id).await;
    }
    for _ in 0..3 {
        let id = uuid::Uuid::now_v7();
        insert_chat_inference(&pool, id, "by_variant_func", "variant_b", episode_id).await;
    }

    let params = CountInferencesParams {
        function_name: "by_variant_func",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let rows = conn.count_inferences_by_variant(params).await.unwrap();

    assert_eq!(rows.len(), 2, "Should return 2 variants");

    // Results should be ordered by inference_count DESC
    assert_eq!(
        rows[0].variant_name, "variant_a",
        "variant_a should be first (higher count)"
    );
    assert_eq!(rows[0].inference_count, 5);
    assert_eq!(
        rows[1].variant_name, "variant_b",
        "variant_b should be second"
    );
    assert_eq!(rows[1].inference_count, 3);

    // Each row should have a valid last_used timestamp
    let rfc3339_millis_regex =
        regex::Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$").unwrap();
    for row in &rows {
        assert!(
            rfc3339_millis_regex.is_match(&row.last_used_at),
            "last_used should be in RFC 3339 format with milliseconds, got: {} for variant {}",
            row.last_used_at,
            row.variant_name
        );
    }
}

#[sqlx::test]
async fn test_count_inferences_by_variant_empty(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let params = CountInferencesParams {
        function_name: "nonexistent_function",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let rows = conn.count_inferences_by_variant(params).await.unwrap();
    assert!(
        rows.is_empty(),
        "Should return empty for nonexistent function"
    );
}

// ===== COUNT INFERENCES FOR EPISODE TESTS =====

#[sqlx::test]
async fn test_count_inferences_for_episode(pool_opts: PgPoolOptions, conn_opts: PgConnectOptions) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts.clone()).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool.clone());

    // Create episode with both chat and json inferences
    let episode_id = uuid::Uuid::now_v7();

    // Insert chat inferences
    for _ in 0..3 {
        let id = uuid::Uuid::now_v7();
        insert_chat_inference(&pool, id, "episode_func", "variant_a", episode_id).await;
    }

    // Insert json inferences
    for _ in 0..2 {
        let id = uuid::Uuid::now_v7();
        insert_json_inference(&pool, id, "episode_json_func", "variant_b", episode_id).await;
    }

    let count = conn.count_inferences_for_episode(episode_id).await.unwrap();
    assert_eq!(
        count, 5,
        "Should count 5 total inferences for episode (3 chat + 2 json)"
    );
}

#[sqlx::test]
async fn test_count_inferences_for_episode_empty(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let episode_id = uuid::Uuid::now_v7();
    let count = conn.count_inferences_for_episode(episode_id).await.unwrap();
    assert_eq!(count, 0, "Should return 0 for episode with no inferences");
}

// ===== LIST FUNCTIONS WITH INFERENCE COUNT TESTS =====

#[sqlx::test]
async fn test_list_functions_with_inference_count(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts.clone()).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool.clone());

    let episode_id = uuid::Uuid::now_v7();

    // Insert chat inferences for func_a
    for _ in 0..4 {
        let id = uuid::Uuid::now_v7();
        insert_chat_inference(&pool, id, "func_a", "variant_a", episode_id).await;
    }

    // Insert json inferences for func_b
    for _ in 0..2 {
        let id = uuid::Uuid::now_v7();
        insert_json_inference(&pool, id, "func_b", "variant_b", episode_id).await;
    }

    // Wait a bit to ensure different timestamps
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    // Insert more recent inference for func_a to ensure it's first
    let id = uuid::Uuid::now_v7();
    insert_chat_inference(&pool, id, "func_a", "variant_a", episode_id).await;

    let rows = conn.list_functions_with_inference_count().await.unwrap();

    assert_eq!(rows.len(), 2, "Should return 2 functions");

    // Results should be ordered by last_inference_timestamp DESC
    // func_a has the most recent inference
    assert_eq!(
        rows[0].function_name, "func_a",
        "func_a should be first (most recent)"
    );
    assert_eq!(
        rows[0].inference_count, 5,
        "func_a should have 5 inferences"
    );

    assert_eq!(rows[1].function_name, "func_b", "func_b should be second");
    assert_eq!(
        rows[1].inference_count, 2,
        "func_b should have 2 inferences"
    );

    // Verify timestamps are ordered DESC
    assert!(
        rows[0].last_inference_timestamp >= rows[1].last_inference_timestamp,
        "Results should be ordered by last_inference_timestamp DESC"
    );
}

#[sqlx::test]
async fn test_list_functions_with_inference_count_empty(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let rows = conn.list_functions_with_inference_count().await.unwrap();
    assert!(
        rows.is_empty(),
        "Should return empty when no inferences exist"
    );
}

// ===== GET FUNCTION THROUGHPUT BY VARIANT TESTS =====

#[sqlx::test]
async fn test_get_function_throughput_by_variant_cumulative(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts.clone()).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool.clone());

    let episode_id = uuid::Uuid::now_v7();

    // Insert chat inferences
    for _ in 0..3 {
        let id = uuid::Uuid::now_v7();
        insert_chat_inference(&pool, id, "throughput_func", "variant_a", episode_id).await;
    }
    for _ in 0..2 {
        let id = uuid::Uuid::now_v7();
        insert_chat_inference(&pool, id, "throughput_func", "variant_b", episode_id).await;
    }

    // Insert json inferences (should also be counted)
    for _ in 0..1 {
        let id = uuid::Uuid::now_v7();
        insert_json_inference(&pool, id, "throughput_func", "variant_c", episode_id).await;
    }

    let params = GetFunctionThroughputByVariantParams {
        function_name: "throughput_func",
        time_window: TimeWindow::Cumulative,
        max_periods: 10,
    };

    let rows = conn
        .get_function_throughput_by_variant(params)
        .await
        .unwrap();

    assert_eq!(rows.len(), 3, "Should return 3 variants");

    // All rows should have epoch period_start for cumulative
    for row in &rows {
        assert_eq!(
            row.period_start.timestamp(),
            0,
            "Cumulative should have epoch (1970-01-01) as period_start"
        );
    }

    // Check totals per variant
    let total: u32 = rows.iter().map(|r| r.count).sum();
    assert_eq!(total, 6, "Total count should be 6 (3 + 2 + 1)");
}

#[sqlx::test]
async fn test_get_function_throughput_by_variant_windowed(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts.clone()).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool.clone());

    let episode_id = uuid::Uuid::now_v7();

    // Insert some inferences
    for _ in 0..5 {
        let id = uuid::Uuid::now_v7();
        insert_chat_inference(&pool, id, "windowed_func", "variant_a", episode_id).await;
    }

    let params = GetFunctionThroughputByVariantParams {
        function_name: "windowed_func",
        time_window: TimeWindow::Hour,
        max_periods: 10,
    };

    let rows = conn
        .get_function_throughput_by_variant(params)
        .await
        .unwrap();

    // Should have at least 1 row (all inferences are recent)
    assert!(
        !rows.is_empty(),
        "Should return at least 1 row for windowed query"
    );

    // Total count should be 5
    let total: u32 = rows.iter().map(|r| r.count).sum();
    assert_eq!(total, 5, "Total count should be 5");
}

#[sqlx::test]
async fn test_get_function_throughput_by_variant_empty(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let params = GetFunctionThroughputByVariantParams {
        function_name: "nonexistent_func",
        time_window: TimeWindow::Cumulative,
        max_periods: 10,
    };

    let rows = conn
        .get_function_throughput_by_variant(params)
        .await
        .unwrap();
    assert!(
        rows.is_empty(),
        "Should return empty for nonexistent function"
    );
}

// ===== NOT IMPLEMENTED METHODS TESTS =====

#[sqlx::test]
async fn test_count_inferences_with_feedback_not_implemented(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let metric_config = MetricConfig {
        r#type: MetricConfigType::Boolean,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
        description: None,
    };

    let params = CountInferencesWithFeedbackParams {
        function_name: "test_func",
        function_type: FunctionConfigType::Chat,
        metric_name: "test_metric",
        metric_config: &metric_config,
        metric_threshold: None,
    };

    let result = conn.count_inferences_with_feedback(params).await;
    assert!(
        result.is_err(),
        "Should return error for not implemented method"
    );

    let error = result.unwrap_err();
    assert!(
        error.to_string().contains("not implemented"),
        "Error message should indicate not implemented"
    );
}

#[sqlx::test]
async fn test_count_inferences_with_demonstration_feedback_not_implemented(
    pool_opts: PgPoolOptions,
    conn_opts: PgConnectOptions,
) {
    manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
        .await
        .unwrap();
    let pool = pool_opts.connect_with(conn_opts).await.unwrap();
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let params = CountInferencesWithDemonstrationFeedbacksParams {
        function_name: "test_func",
        function_type: FunctionConfigType::Chat,
    };

    let result = conn
        .count_inferences_with_demonstration_feedback(params)
        .await;
    assert!(
        result.is_err(),
        "Should return error for not implemented method"
    );

    let error = result.unwrap_err();
    assert!(
        error.to_string().contains("not implemented"),
        "Error message should indicate not implemented"
    );
}
