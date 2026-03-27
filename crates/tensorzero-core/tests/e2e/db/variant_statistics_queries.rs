//! E2E tests for variant statistics aggregation (ClickHouse and Postgres).
//!
//! Verifies that the `VariantStatistics` rollup (ClickHouse) and
//! `variant_statistics` table (Postgres) correctly aggregate metrics
//! from chat/json inferences and model inferences.

use std::collections::HashMap;

use rust_decimal::Decimal;
use serde::Deserialize;
use sqlx::Row as _;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::inferences::InferenceQueries;
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::inference::types::{
    ChatInferenceDatabaseInsert, FinishReason, StoredModelInference,
};
use tonic::async_trait;

// ===== TEST QUERY INFRASTRUCTURE =====

#[derive(Debug)]
struct VariantStatsTestRow {
    function_name: String,
    variant_name: String,
    count: u64,
    total_input_tokens: Option<u64>,
    total_output_tokens: Option<u64>,
    total_cost: Option<Decimal>,
    count_with_cost: u64,
}

/// Trait for querying variant statistics in tests. Implemented separately
/// for each backend since the query mechanism differs.
#[async_trait]
trait VariantStatisticsTestQueries: Send + Sync {
    async fn query_variant_stats(
        &self,
        function_name: &str,
    ) -> Result<Vec<VariantStatsTestRow>, String>;
}

#[derive(Debug, Deserialize)]
struct ClickHouseVariantStatsRow {
    function_name: String,
    variant_name: String,
    count: u64,
    total_input_tokens: Option<i64>,
    total_output_tokens: Option<i64>,
    #[serde(default, with = "rust_decimal::serde::float_option")]
    total_cost: Option<Decimal>,
    count_with_cost: u64,
}

#[async_trait]
impl VariantStatisticsTestQueries for ClickHouseConnectionInfo {
    async fn query_variant_stats(
        &self,
        function_name: &str,
    ) -> Result<Vec<VariantStatsTestRow>, String> {
        let query = format!(
            r"SELECT
                function_name,
                variant_name,
                countMerge(count) AS count,
                sumMerge(total_input_tokens) AS total_input_tokens,
                sumMerge(total_output_tokens) AS total_output_tokens,
                sumMerge(total_cost) AS total_cost,
                countMerge(count_with_cost) AS count_with_cost
            FROM VariantStatistics
            WHERE function_name = '{function_name}'
            GROUP BY function_name, variant_name
            FORMAT JSONEachRow"
        );
        let response = self
            .run_query_synchronous_no_params(query)
            .await
            .map_err(|e| format!("ClickHouse query failed: {e}"))?;
        let mut rows = Vec::new();
        for line in response.response.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let ch_row: ClickHouseVariantStatsRow = serde_json::from_str(line)
                .map_err(|e| format!("Failed to parse ClickHouse row `{line}`: {e}"))?;
            rows.push(VariantStatsTestRow {
                function_name: ch_row.function_name,
                variant_name: ch_row.variant_name,
                count: ch_row.count,
                total_input_tokens: ch_row.total_input_tokens.map(|v| v as u64),
                total_output_tokens: ch_row.total_output_tokens.map(|v| v as u64),
                total_cost: ch_row.total_cost,
                count_with_cost: ch_row.count_with_cost,
            });
        }
        Ok(rows)
    }
}

#[async_trait]
impl VariantStatisticsTestQueries for PostgresConnectionInfo {
    async fn query_variant_stats(
        &self,
        function_name: &str,
    ) -> Result<Vec<VariantStatsTestRow>, String> {
        let pool = self
            .get_pool()
            .ok_or_else(|| "Postgres pool not available".to_string())?;
        let rows = sqlx::query(
            "SELECT function_name, variant_name, inference_count, \
             total_input_tokens, total_output_tokens, total_cost, count_with_cost \
             FROM tensorzero.variant_statistics WHERE function_name = $1",
        )
        .bind(function_name)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("Postgres query failed: {e}"))?;

        Ok(rows
            .iter()
            .map(|r| VariantStatsTestRow {
                function_name: r.get("function_name"),
                variant_name: r.get("variant_name"),
                count: r.get::<i64, _>("inference_count") as u64,
                total_input_tokens: r
                    .get::<Option<i64>, _>("total_input_tokens")
                    .map(|v| v as u64),
                total_output_tokens: r
                    .get::<Option<i64>, _>("total_output_tokens")
                    .map(|v| v as u64),
                total_cost: r.get::<Option<Decimal>, _>("total_cost"),
                count_with_cost: r.get::<Option<i64>, _>("count_with_cost").unwrap_or(0) as u64,
            })
            .collect())
    }
}

// ===== TEST DATA HELPERS =====

/// Create chat inferences and model inferences for variant statistics testing.
///
/// Creates 2 chat inferences for the given function/variant, with 3 model inferences:
/// - Model inference 1: linked to chat 1, input_tokens=100, output_tokens=50, cost=0.0005
/// - Model inference 2: linked to chat 1 (retry), input_tokens=200, output_tokens=100, cost=0.0015
/// - Model inference 3: linked to chat 2, input_tokens=300, output_tokens=150, cost=None
///
/// Expected aggregation:
/// - count: 2 (chat inferences)
/// - total_input_tokens: 600 (100+200+300)
/// - total_output_tokens: 300 (50+100+150)
/// - total_cost: 0.002 (0.0005+0.0015, NULL excluded)
/// - count_with_cost: 2 (model inferences with non-null cost)
fn make_variant_stats_test_data(
    function_name: &str,
    variant_name: &str,
) -> (Vec<ChatInferenceDatabaseInsert>, Vec<StoredModelInference>) {
    let chat1_id = uuid::Uuid::now_v7();
    let chat2_id = uuid::Uuid::now_v7();

    let chat_inferences = vec![
        ChatInferenceDatabaseInsert {
            id: chat1_id,
            function_name: function_name.to_string(),
            variant_name: variant_name.to_string(),
            episode_id: uuid::Uuid::now_v7(),
            input: None,
            output: None,
            tool_params: None,
            inference_params: None,
            processing_time_ms: Some(100),
            ttft_ms: Some(50),
            tags: HashMap::new(),
            extra_body: None,
            snapshot_hash: None,
        },
        ChatInferenceDatabaseInsert {
            id: chat2_id,
            function_name: function_name.to_string(),
            variant_name: variant_name.to_string(),
            episode_id: uuid::Uuid::now_v7(),
            input: None,
            output: None,
            tool_params: None,
            inference_params: None,
            processing_time_ms: Some(200),
            ttft_ms: Some(75),
            tags: HashMap::new(),
            extra_body: None,
            snapshot_hash: None,
        },
    ];

    let model_inferences = vec![
        StoredModelInference {
            id: uuid::Uuid::now_v7(),
            inference_id: chat1_id,
            raw_request: Some("{}".to_string()),
            raw_response: Some("{}".to_string()),
            system: None,
            input_messages: Some(vec![]),
            output: Some(vec![]),
            input_tokens: Some(100),
            output_tokens: Some(50),
            response_time_ms: Some(100),
            model_name: "test-model".to_string(),
            model_provider_name: "test-provider".to_string(),
            ttft_ms: None,
            cached: false,
            cost: Some(Decimal::new(500, 6)), // 0.000500
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            finish_reason: Some(FinishReason::Stop),
            snapshot_hash: None,
            timestamp: None,
        },
        // Second model inference for same chat inference (simulates retry/fallback)
        StoredModelInference {
            id: uuid::Uuid::now_v7(),
            inference_id: chat1_id,
            raw_request: Some("{}".to_string()),
            raw_response: Some("{}".to_string()),
            system: None,
            input_messages: Some(vec![]),
            output: Some(vec![]),
            input_tokens: Some(200),
            output_tokens: Some(100),
            response_time_ms: Some(150),
            model_name: "test-model".to_string(),
            model_provider_name: "test-provider".to_string(),
            ttft_ms: None,
            cached: false,
            cost: Some(Decimal::new(1500, 6)), // 0.001500
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            finish_reason: Some(FinishReason::Stop),
            snapshot_hash: None,
            timestamp: None,
        },
        StoredModelInference {
            id: uuid::Uuid::now_v7(),
            inference_id: chat2_id,
            raw_request: Some("{}".to_string()),
            raw_response: Some("{}".to_string()),
            system: None,
            input_messages: Some(vec![]),
            output: Some(vec![]),
            input_tokens: Some(300),
            output_tokens: Some(150),
            response_time_ms: Some(50),
            model_name: "test-model".to_string(),
            model_provider_name: "test-provider".to_string(),
            ttft_ms: None,
            cached: false,
            cost: None, // NULL cost — excluded from SUM and count_with_cost
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            finish_reason: Some(FinishReason::Stop),
            snapshot_hash: None,
            timestamp: None,
        },
    ];

    (chat_inferences, model_inferences)
}

// ===== TESTS =====

/// Inserts chat inferences and model inferences with known values and verifies
/// the variant statistics aggregation is correct on both backends.
///
/// Tests:
/// - Inference count from chat inferences
/// - Token sums across multiple model inferences (including retries)
/// - Cost sum with NULL exclusion
/// - count_with_cost tracks only model inferences with non-null cost
async fn test_variant_statistics_aggregation(
    conn: impl InferenceQueries
    + ModelInferenceQueries
    + TestDatabaseHelpers
    + VariantStatisticsTestQueries,
) {
    let function_name = format!("variant-stats-test-{}", uuid::Uuid::now_v7());
    let variant_name = "test-variant";
    let (chat_inferences, model_inferences) =
        make_variant_stats_test_data(&function_name, variant_name);

    // Insert chat inferences first
    conn.insert_chat_inferences(&chat_inferences)
        .await
        .expect("Failed to insert chat inferences");

    // Flush to ensure InferenceById is populated (needed for ClickHouse MV JOIN)
    conn.flush_pending_writes().await;

    // Insert model inferences (linked via inference_id to the chat inferences)
    conn.insert_model_inferences(&model_inferences)
        .await
        .expect("Failed to insert model inferences");

    // Prepare variant statistics (flush for ClickHouse, refresh for Postgres)
    conn.prepare_variant_statistics().await;

    // Poll for results — ClickHouse MVs may take a moment to process
    let stats = crate::utils::poll_for_result::poll_for_result(
        || conn.query_variant_stats(&function_name),
        |stats| !stats.is_empty(),
        "Timed out waiting for variant statistics to appear",
    )
    .await;

    assert_eq!(stats.len(), 1, "Should have exactly 1 row for our variant");
    let row = &stats[0];

    assert_eq!(row.function_name, function_name);
    assert_eq!(row.variant_name, variant_name);
    assert_eq!(row.count, 2, "Should count 2 chat inferences");
    assert_eq!(
        row.total_input_tokens,
        Some(600),
        "Input tokens should sum to 100 + 200 + 300 = 600"
    );
    assert_eq!(
        row.total_output_tokens,
        Some(300),
        "Output tokens should sum to 50 + 100 + 150 = 300"
    );
    // SUM of costs: 0.000500 + 0.001500 = 0.002000 (NULL excluded)
    assert_eq!(
        row.total_cost,
        Some(Decimal::new(2000, 6)),
        "Cost should sum to 0.002000 (NULL cost excluded)"
    );
    // count_with_cost: 2 model inferences have non-null cost
    assert_eq!(
        row.count_with_cost, 2,
        "count_with_cost should be 2 (model inferences with non-null cost)"
    );
}
make_db_test!(test_variant_statistics_aggregation);
