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
use tensorzero_core::inference::types::stored_input::StoredInput;
use tensorzero_core::inference::types::{
    ChatInferenceDatabaseInsert, FinishReason, JsonInferenceDatabaseInsert, StoredModelInference,
};
use tonic::async_trait;

// ===== TEST QUERY INFRASTRUCTURE =====

#[derive(Debug)]
struct VariantStatsTestRow {
    #[expect(dead_code)]
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
    /// Query variant statistics cumulative (grouped by function_name + variant_name).
    async fn query_variant_stats(
        &self,
        function_name: &str,
    ) -> Result<Vec<VariantStatsTestRow>, String>;

    /// Query variant statistics per-minute (grouped by function_name + variant_name + minute).
    async fn query_variant_stats_per_minute(
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

impl From<ClickHouseVariantStatsRow> for VariantStatsTestRow {
    fn from(ch: ClickHouseVariantStatsRow) -> Self {
        Self {
            function_name: ch.function_name,
            variant_name: ch.variant_name,
            count: ch.count,
            total_input_tokens: ch.total_input_tokens.map(|v| v as u64),
            total_output_tokens: ch.total_output_tokens.map(|v| v as u64),
            total_cost: ch.total_cost,
            count_with_cost: ch.count_with_cost,
        }
    }
}

fn parse_clickhouse_rows(response: &str) -> Result<Vec<VariantStatsTestRow>, String> {
    let mut rows = Vec::new();
    for line in response.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ch_row: ClickHouseVariantStatsRow = serde_json::from_str(line)
            .map_err(|e| format!("Failed to parse ClickHouse row `{line}`: {e}"))?;
        rows.push(ch_row.into());
    }
    Ok(rows)
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
            ORDER BY variant_name
            FORMAT JSONEachRow"
        );
        let response = self
            .run_query_synchronous_no_params(query)
            .await
            .map_err(|e| format!("ClickHouse query failed: {e}"))?;
        parse_clickhouse_rows(&response.response)
    }

    async fn query_variant_stats_per_minute(
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
            GROUP BY function_name, variant_name, minute
            ORDER BY variant_name, minute
            FORMAT JSONEachRow"
        );
        let response = self
            .run_query_synchronous_no_params(query)
            .await
            .map_err(|e| format!("ClickHouse query failed: {e}"))?;
        parse_clickhouse_rows(&response.response)
    }
}

fn pg_row_to_variant_stats(r: &sqlx::postgres::PgRow) -> VariantStatsTestRow {
    VariantStatsTestRow {
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
            "SELECT function_name, variant_name, \
             SUM(inference_count)::BIGINT AS inference_count, \
             SUM(total_input_tokens)::BIGINT AS total_input_tokens, \
             SUM(total_output_tokens)::BIGINT AS total_output_tokens, \
             SUM(total_cost) AS total_cost, \
             SUM(count_with_cost)::BIGINT AS count_with_cost \
             FROM tensorzero.variant_statistics WHERE function_name = $1 \
             GROUP BY function_name, variant_name \
             ORDER BY variant_name",
        )
        .bind(function_name)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("Postgres query failed: {e}"))?;

        Ok(rows.iter().map(pg_row_to_variant_stats).collect())
    }

    async fn query_variant_stats_per_minute(
        &self,
        function_name: &str,
    ) -> Result<Vec<VariantStatsTestRow>, String> {
        let pool = self
            .get_pool()
            .ok_or_else(|| "Postgres pool not available".to_string())?;
        let rows = sqlx::query(
            "SELECT function_name, variant_name, inference_count, \
             total_input_tokens, total_output_tokens, total_cost, count_with_cost \
             FROM tensorzero.variant_statistics WHERE function_name = $1 \
             ORDER BY variant_name, minute",
        )
        .bind(function_name)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("Postgres query failed: {e}"))?;

        Ok(rows.iter().map(pg_row_to_variant_stats).collect())
    }
}

// ===== SHARED HELPERS =====

fn make_chat_inference(
    id: uuid::Uuid,
    function_name: &str,
    variant_name: &str,
    processing_time_ms: Option<u32>,
    ttft_ms: Option<u32>,
) -> ChatInferenceDatabaseInsert {
    ChatInferenceDatabaseInsert {
        id,
        function_name: function_name.to_string(),
        variant_name: variant_name.to_string(),
        episode_id: uuid::Uuid::now_v7(),
        input: Some(StoredInput::default()),
        output: Some(vec![]),
        tool_params: None,
        inference_params: None,
        processing_time_ms,
        ttft_ms,
        tags: HashMap::new(),
        extra_body: None,
        snapshot_hash: None,
    }
}

fn make_json_inference(
    id: uuid::Uuid,
    function_name: &str,
    variant_name: &str,
    processing_time_ms: Option<u32>,
    ttft_ms: Option<u32>,
) -> JsonInferenceDatabaseInsert {
    JsonInferenceDatabaseInsert {
        id,
        function_name: function_name.to_string(),
        variant_name: variant_name.to_string(),
        episode_id: uuid::Uuid::now_v7(),
        input: Some(StoredInput::default()),
        output: None,
        auxiliary_content: None,
        inference_params: None,
        processing_time_ms,
        output_schema: Some(serde_json::json!({"type": "object"})),
        ttft_ms,
        tags: HashMap::new(),
        extra_body: None,
        snapshot_hash: None,
    }
}

fn make_model_inference(
    inference_id: uuid::Uuid,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    cost: Option<Decimal>,
) -> StoredModelInference {
    StoredModelInference {
        id: uuid::Uuid::now_v7(),
        inference_id,
        raw_request: Some("{}".to_string()),
        raw_response: Some("{}".to_string()),
        system: None,
        input_messages: Some(vec![]),
        output: Some(vec![]),
        input_tokens,
        output_tokens,
        response_time_ms: Some(100),
        model_name: "test-model".to_string(),
        model_provider_name: "test-provider".to_string(),
        ttft_ms: None,
        cached: false,
        cost,
        provider_cache_read_input_tokens: None,
        provider_cache_write_input_tokens: None,
        finish_reason: Some(FinishReason::Stop),
        snapshot_hash: None,
        timestamp: None,
    }
}

fn make_model_inference_at(
    ts: uuid::Timestamp,
    inference_id: uuid::Uuid,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    cost: Option<Decimal>,
) -> StoredModelInference {
    let mut mi = make_model_inference(inference_id, input_tokens, output_tokens, cost);
    mi.id = uuid::Uuid::new_v7(ts);
    mi
}

/// Insert chat + model inferences and prepare variant statistics.
/// Returns after polling confirms the variant is visible.
async fn insert_and_prepare(
    conn: &(impl InferenceQueries + ModelInferenceQueries + TestDatabaseHelpers),
    chat_inferences: &[ChatInferenceDatabaseInsert],
    json_inferences: &[JsonInferenceDatabaseInsert],
    model_inferences: &[StoredModelInference],
) {
    // Insert chat and json inferences first (populates InferenceById in ClickHouse)
    if !chat_inferences.is_empty() {
        conn.insert_chat_inferences(chat_inferences)
            .await
            .expect("Failed to insert chat inferences");
    }
    if !json_inferences.is_empty() {
        conn.insert_json_inferences(json_inferences)
            .await
            .expect("Failed to insert json inferences");
    }

    // Flush to ensure InferenceById is populated (needed for ClickHouse MV JOIN)
    conn.flush_pending_writes().await;

    // Insert model inferences (linked via inference_id)
    if !model_inferences.is_empty() {
        conn.insert_model_inferences(model_inferences)
            .await
            .expect("Failed to insert model inferences");
    }

    // Prepare variant statistics (flush for ClickHouse, refresh for Postgres)
    conn.prepare_variant_statistics().await;
}

// ===== TESTS =====

/// Basic aggregation: 2 chat inferences, 3 model inferences (one retry).
/// Verifies token sums, cost SUM with NULL exclusion, and count_with_cost.
async fn test_variant_statistics_basic_aggregation(
    conn: impl InferenceQueries
    + ModelInferenceQueries
    + TestDatabaseHelpers
    + VariantStatisticsTestQueries,
) {
    let function_name = format!("vs-basic-{}", uuid::Uuid::now_v7());
    let variant_name = "v1";

    let chat1_id = uuid::Uuid::now_v7();
    let chat2_id = uuid::Uuid::now_v7();

    let chats = vec![
        make_chat_inference(chat1_id, &function_name, variant_name, Some(100), Some(50)),
        make_chat_inference(chat2_id, &function_name, variant_name, Some(200), Some(75)),
    ];
    let models = vec![
        make_model_inference(chat1_id, Some(100), Some(50), Some(Decimal::new(500, 6))),
        // Retry for same chat inference
        make_model_inference(chat1_id, Some(200), Some(100), Some(Decimal::new(1500, 6))),
        // Second chat inference — no cost
        make_model_inference(chat2_id, Some(300), Some(150), None),
    ];

    insert_and_prepare(&conn, &chats, &[], &models).await;

    let stats = crate::utils::poll_for_result::poll_for_result(
        || conn.query_variant_stats(&function_name),
        |s| !s.is_empty(),
        "Timed out waiting for variant statistics",
    )
    .await;

    assert_eq!(stats.len(), 1, "Expected 1 variant row");
    let row = &stats[0];
    assert_eq!(row.count, 2, "2 chat inferences");
    assert_eq!(row.total_input_tokens, Some(600), "100+200+300");
    assert_eq!(row.total_output_tokens, Some(300), "50+100+150");
    assert_eq!(
        row.total_cost,
        Some(Decimal::new(2000, 6)),
        "0.0005 + 0.0015 = 0.002"
    );
    assert_eq!(row.count_with_cost, 2, "2 model inferences with cost");
}
make_db_test!(test_variant_statistics_basic_aggregation);

// ===== MULTIPLE VARIANTS TEST =====

/// Three variants under the same function, each with different counts and costs.
/// Verifies variant-level isolation: each variant's metrics are independent.
async fn test_variant_statistics_multiple_variants(
    conn: impl InferenceQueries
    + ModelInferenceQueries
    + TestDatabaseHelpers
    + VariantStatisticsTestQueries,
) {
    let function_name = format!("vs-multi-{}", uuid::Uuid::now_v7());

    // Variant A: 3 chat inferences, 3 model inferences (all with cost)
    let a1 = uuid::Uuid::now_v7();
    let a2 = uuid::Uuid::now_v7();
    let a3 = uuid::Uuid::now_v7();

    // Variant B: 2 chat inferences, 2 model inferences (one with cost, one without)
    let b1 = uuid::Uuid::now_v7();
    let b2 = uuid::Uuid::now_v7();

    // Variant C: 1 chat inference, 1 model inference (no cost)
    let c1 = uuid::Uuid::now_v7();

    let chats = vec![
        make_chat_inference(a1, &function_name, "variant-a", Some(50), Some(10)),
        make_chat_inference(a2, &function_name, "variant-a", Some(60), Some(15)),
        make_chat_inference(a3, &function_name, "variant-a", Some(70), Some(20)),
        make_chat_inference(b1, &function_name, "variant-b", Some(100), Some(30)),
        make_chat_inference(b2, &function_name, "variant-b", Some(120), Some(35)),
        make_chat_inference(c1, &function_name, "variant-c", Some(200), None),
    ];

    let models = vec![
        // Variant A
        make_model_inference(a1, Some(10), Some(5), Some(Decimal::new(100, 6))),
        make_model_inference(a2, Some(20), Some(10), Some(Decimal::new(200, 6))),
        make_model_inference(a3, Some(30), Some(15), Some(Decimal::new(300, 6))),
        // Variant B
        make_model_inference(b1, Some(50), Some(25), Some(Decimal::new(500, 6))),
        make_model_inference(b2, Some(60), Some(30), None),
        // Variant C
        make_model_inference(c1, Some(100), Some(50), None),
    ];

    insert_and_prepare(&conn, &chats, &[], &models).await;

    let stats = crate::utils::poll_for_result::poll_for_result(
        || conn.query_variant_stats(&function_name),
        |s| s.len() == 3,
        "Timed out waiting for 3 variant rows",
    )
    .await;

    assert_eq!(stats.len(), 3);

    // Variant A: 3 inferences, tokens=60/30, cost=0.0006, count_with_cost=3
    let a = stats
        .iter()
        .find(|r| r.variant_name == "variant-a")
        .unwrap();
    assert_eq!(a.count, 3);
    assert_eq!(a.total_input_tokens, Some(60), "10+20+30");
    assert_eq!(a.total_output_tokens, Some(30), "5+10+15");
    assert_eq!(a.total_cost, Some(Decimal::new(600, 6)));
    assert_eq!(a.count_with_cost, 3);

    // Variant B: 2 inferences, tokens=110/55, cost=0.0005, count_with_cost=1
    let b = stats
        .iter()
        .find(|r| r.variant_name == "variant-b")
        .unwrap();
    assert_eq!(b.count, 2);
    assert_eq!(b.total_input_tokens, Some(110), "50+60");
    assert_eq!(b.total_output_tokens, Some(55), "25+30");
    assert_eq!(b.total_cost, Some(Decimal::new(500, 6)));
    assert_eq!(b.count_with_cost, 1);

    // Variant C: 1 inference, tokens=100/50, cost=None, count_with_cost=0
    let c = stats
        .iter()
        .find(|r| r.variant_name == "variant-c")
        .unwrap();
    assert_eq!(c.count, 1);
    assert_eq!(c.total_input_tokens, Some(100));
    assert_eq!(c.total_output_tokens, Some(50));
    assert_eq!(c.count_with_cost, 0);
}
make_db_test!(test_variant_statistics_multiple_variants);

// ===== CROSS-MINUTE BUCKETING TEST =====

/// Inferences spread across 3 different minute buckets.
/// Verifies per-minute bucketing produces the correct number of rows
/// and each bucket has the right counts and totals.
async fn test_variant_statistics_cross_minute(
    conn: impl InferenceQueries
    + ModelInferenceQueries
    + TestDatabaseHelpers
    + VariantStatisticsTestQueries,
) {
    let function_name = format!("vs-xmin-{}", uuid::Uuid::now_v7());
    let variant_name = "v1";

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Minute A (7 min ago): 2 chat inferences with cost
    let min_a = now_secs - 7 * 60;
    let ts_a1 = uuid::Timestamp::from_unix_time(min_a, 0, 0, 0);
    let ts_a2 = uuid::Timestamp::from_unix_time(min_a, 500_000_000, 0, 0);
    let a1 = uuid::Uuid::new_v7(ts_a1);
    let a2 = uuid::Uuid::new_v7(ts_a2);

    // Minute B (5 min ago): 3 chat inferences, mixed cost
    let min_b = now_secs - 5 * 60;
    let ts_b1 = uuid::Timestamp::from_unix_time(min_b, 0, 0, 0);
    let ts_b2 = uuid::Timestamp::from_unix_time(min_b, 300_000_000, 0, 0);
    let ts_b3 = uuid::Timestamp::from_unix_time(min_b, 600_000_000, 0, 0);
    let b1 = uuid::Uuid::new_v7(ts_b1);
    let b2 = uuid::Uuid::new_v7(ts_b2);
    let b3 = uuid::Uuid::new_v7(ts_b3);

    // Minute C (3 min ago): 1 chat inference, no cost
    let min_c = now_secs - 3 * 60;
    let ts_c1 = uuid::Timestamp::from_unix_time(min_c, 0, 0, 0);
    let c1 = uuid::Uuid::new_v7(ts_c1);

    let chats = vec![
        make_chat_inference(a1, &function_name, variant_name, Some(80), Some(20)),
        make_chat_inference(a2, &function_name, variant_name, Some(90), Some(25)),
        make_chat_inference(b1, &function_name, variant_name, Some(100), Some(30)),
        make_chat_inference(b2, &function_name, variant_name, Some(110), Some(35)),
        make_chat_inference(b3, &function_name, variant_name, Some(120), Some(40)),
        make_chat_inference(c1, &function_name, variant_name, Some(200), None),
    ];

    let models = vec![
        // Minute A
        make_model_inference_at(ts_a1, a1, Some(10), Some(5), Some(Decimal::new(100, 6))),
        make_model_inference_at(ts_a2, a2, Some(20), Some(10), Some(Decimal::new(200, 6))),
        // Minute B
        make_model_inference_at(ts_b1, b1, Some(30), Some(15), Some(Decimal::new(300, 6))),
        make_model_inference_at(ts_b2, b2, Some(40), Some(20), None),
        make_model_inference_at(ts_b3, b3, Some(50), Some(25), Some(Decimal::new(500, 6))),
        // Minute C
        make_model_inference_at(ts_c1, c1, Some(60), Some(30), None),
    ];

    insert_and_prepare(&conn, &chats, &[], &models).await;

    // Poll for per-minute rows
    let stats = crate::utils::poll_for_result::poll_for_result(
        || conn.query_variant_stats_per_minute(&function_name),
        |s| s.len() == 3,
        "Timed out waiting for 3 per-minute rows",
    )
    .await;

    assert_eq!(stats.len(), 3, "Should have 3 minute buckets");

    // Verify totals across all minutes
    let total_count: u64 = stats.iter().map(|r| r.count).sum();
    let total_input: u64 = stats.iter().filter_map(|r| r.total_input_tokens).sum();
    let total_output: u64 = stats.iter().filter_map(|r| r.total_output_tokens).sum();
    let total_cwc: u64 = stats.iter().map(|r| r.count_with_cost).sum();

    assert_eq!(total_count, 6, "6 total chat inferences");
    assert_eq!(total_input, 210, "10+20+30+40+50+60");
    assert_eq!(total_output, 105, "5+10+15+20+25+30");
    assert_eq!(total_cwc, 4, "4 model inferences with cost (a1,a2,b1,b3)");

    // Minute A: count=2, tokens=30/15, cwc=2
    let min_a_row = stats
        .iter()
        .find(|r| r.count == 2 && r.count_with_cost == 2);
    assert!(
        min_a_row.is_some(),
        "Should find minute A row (count=2, cwc=2)"
    );
    let min_a_row = min_a_row.unwrap();
    assert_eq!(min_a_row.total_input_tokens, Some(30));
    assert_eq!(min_a_row.total_output_tokens, Some(15));
    assert_eq!(min_a_row.total_cost, Some(Decimal::new(300, 6)));

    // Minute B: count=3, cwc=2 (one has no cost)
    let min_b_row = stats.iter().find(|r| r.count == 3);
    assert!(min_b_row.is_some(), "Should find minute B row (count=3)");
    let min_b_row = min_b_row.unwrap();
    assert_eq!(min_b_row.total_input_tokens, Some(120), "30+40+50");
    assert_eq!(min_b_row.total_output_tokens, Some(60), "15+20+25");
    assert_eq!(min_b_row.count_with_cost, 2);
    assert_eq!(min_b_row.total_cost, Some(Decimal::new(800, 6)));

    // Minute C: count=1, cwc=0
    let min_c_row = stats.iter().find(|r| r.count == 1);
    assert!(min_c_row.is_some(), "Should find minute C row (count=1)");
    let min_c_row = min_c_row.unwrap();
    assert_eq!(min_c_row.total_input_tokens, Some(60));
    assert_eq!(min_c_row.total_output_tokens, Some(30));
    assert_eq!(min_c_row.count_with_cost, 0);
}
make_db_test!(test_variant_statistics_cross_minute);

// ===== MIXED CHAT + JSON INFERENCES TEST =====

/// Both chat and json inferences contribute to the same variant.
/// Verifies that VariantStatisticsChatView and VariantStatisticsJsonView
/// both write to the same target table and merge correctly.
async fn test_variant_statistics_mixed_chat_json(
    conn: impl InferenceQueries
    + ModelInferenceQueries
    + TestDatabaseHelpers
    + VariantStatisticsTestQueries,
) {
    let function_name = format!("vs-mixed-{}", uuid::Uuid::now_v7());
    let variant_name = "v1";

    // 2 chat inferences + 3 json inferences = 5 total inferences
    let chat1 = uuid::Uuid::now_v7();
    let chat2 = uuid::Uuid::now_v7();
    let json1 = uuid::Uuid::now_v7();
    let json2 = uuid::Uuid::now_v7();
    let json3 = uuid::Uuid::now_v7();

    let chats = vec![
        make_chat_inference(chat1, &function_name, variant_name, Some(100), Some(50)),
        make_chat_inference(chat2, &function_name, variant_name, Some(150), Some(60)),
    ];

    let jsons = vec![
        make_json_inference(json1, &function_name, variant_name, Some(200), Some(70)),
        make_json_inference(json2, &function_name, variant_name, Some(250), Some(80)),
        make_json_inference(json3, &function_name, variant_name, Some(300), None),
    ];

    // 5 model inferences — one per inference
    let models = vec![
        make_model_inference(chat1, Some(10), Some(5), Some(Decimal::new(100, 6))),
        make_model_inference(chat2, Some(20), Some(10), Some(Decimal::new(200, 6))),
        make_model_inference(json1, Some(30), Some(15), Some(Decimal::new(300, 6))),
        make_model_inference(json2, Some(40), Some(20), Some(Decimal::new(400, 6))),
        make_model_inference(json3, Some(50), Some(25), None),
    ];

    insert_and_prepare(&conn, &chats, &jsons, &models).await;

    let stats = crate::utils::poll_for_result::poll_for_result(
        || conn.query_variant_stats(&function_name),
        |s| s.iter().any(|r| r.count == 5),
        "Timed out waiting for 5 inferences in variant statistics",
    )
    .await;

    assert_eq!(stats.len(), 1);
    let row = &stats[0];
    assert_eq!(row.count, 5, "2 chat + 3 json = 5 total inferences");
    assert_eq!(row.total_input_tokens, Some(150), "10+20+30+40+50");
    assert_eq!(row.total_output_tokens, Some(75), "5+10+15+20+25");
    assert_eq!(
        row.total_cost,
        Some(Decimal::new(1000, 6)),
        "0.0001+0.0002+0.0003+0.0004 = 0.001"
    );
    assert_eq!(row.count_with_cost, 4, "4 model inferences with cost");
}
make_db_test!(test_variant_statistics_mixed_chat_json);

// ===== HIGH-VOLUME AGGREGATION TEST =====

/// 20 chat inferences across 2 variants with varying token counts.
/// Exercises aggregation at modest scale and verifies exact sums.
async fn test_variant_statistics_high_volume(
    conn: impl InferenceQueries
    + ModelInferenceQueries
    + TestDatabaseHelpers
    + VariantStatisticsTestQueries,
) {
    let function_name = format!("vs-volume-{}", uuid::Uuid::now_v7());

    let mut chats = Vec::new();
    let mut models = Vec::new();

    // Variant "fast": 12 inferences, each with 1 model inference
    // input_tokens = i*10, output_tokens = i*5, cost = i * 0.0001
    let mut fast_expected_input: u64 = 0;
    let mut fast_expected_output: u64 = 0;
    let mut fast_expected_cost = Decimal::ZERO;
    for i in 1..=12u32 {
        let id = uuid::Uuid::now_v7();
        chats.push(make_chat_inference(
            id,
            &function_name,
            "fast",
            Some(i * 10),
            Some(i * 5),
        ));
        let cost = Decimal::new(i64::from(i) * 100, 6);
        models.push(make_model_inference(
            id,
            Some(i * 10),
            Some(i * 5),
            Some(cost),
        ));
        fast_expected_input += u64::from(i * 10);
        fast_expected_output += u64::from(i * 5);
        fast_expected_cost += cost;
    }

    // Variant "slow": 8 inferences, each with 2 model inferences (retry pattern)
    // First attempt: input_tokens = i*15, output_tokens = i*7, cost = i * 0.0002
    // Retry: input_tokens = i*15, output_tokens = i*7, no cost (retries often don't have cost)
    let mut slow_expected_input: u64 = 0;
    let mut slow_expected_output: u64 = 0;
    let mut slow_expected_cost = Decimal::ZERO;
    let mut slow_expected_cwc: u64 = 0;
    for i in 1..=8u32 {
        let id = uuid::Uuid::now_v7();
        chats.push(make_chat_inference(
            id,
            &function_name,
            "slow",
            Some(i * 20),
            Some(i * 8),
        ));
        let cost = Decimal::new(i64::from(i) * 200, 6);
        // First attempt with cost
        models.push(make_model_inference(
            id,
            Some(i * 15),
            Some(i * 7),
            Some(cost),
        ));
        // Retry without cost
        models.push(make_model_inference(id, Some(i * 15), Some(i * 7), None));
        slow_expected_input += u64::from(i * 15) * 2; // both attempts
        slow_expected_output += u64::from(i * 7) * 2;
        slow_expected_cost += cost; // only first attempt
        slow_expected_cwc += 1; // only first attempt has cost
    }

    insert_and_prepare(&conn, &chats, &[], &models).await;

    let stats = crate::utils::poll_for_result::poll_for_result(
        || conn.query_variant_stats(&function_name),
        |s| s.len() == 2,
        "Timed out waiting for 2 variant rows in high-volume test",
    )
    .await;

    assert_eq!(stats.len(), 2);

    let fast = stats.iter().find(|r| r.variant_name == "fast").unwrap();
    assert_eq!(fast.count, 12, "12 fast inferences");
    assert_eq!(fast.total_input_tokens, Some(fast_expected_input));
    assert_eq!(fast.total_output_tokens, Some(fast_expected_output));
    assert_eq!(fast.total_cost, Some(fast_expected_cost));
    assert_eq!(fast.count_with_cost, 12, "all fast inferences have cost");

    let slow = stats.iter().find(|r| r.variant_name == "slow").unwrap();
    assert_eq!(slow.count, 8, "8 slow inferences");
    assert_eq!(slow.total_input_tokens, Some(slow_expected_input));
    assert_eq!(slow.total_output_tokens, Some(slow_expected_output));
    assert_eq!(slow.total_cost, Some(slow_expected_cost));
    assert_eq!(
        slow.count_with_cost, slow_expected_cwc,
        "only first attempts have cost"
    );
}
make_db_test!(test_variant_statistics_high_volume);

// ===== INFERENCE WITH NO MODEL INFERENCES TEST =====

/// A chat inference with no associated model inferences.
/// Verifies count is tracked but token/cost columns are NULL/0.
async fn test_variant_statistics_no_model_inferences(
    conn: impl InferenceQueries
    + ModelInferenceQueries
    + TestDatabaseHelpers
    + VariantStatisticsTestQueries,
) {
    let function_name = format!("vs-nomodel-{}", uuid::Uuid::now_v7());
    let variant_name = "orphan";

    let chat1 = uuid::Uuid::now_v7();
    let chat2 = uuid::Uuid::now_v7();

    let chats = vec![
        make_chat_inference(chat1, &function_name, variant_name, Some(100), Some(50)),
        make_chat_inference(chat2, &function_name, variant_name, Some(200), Some(75)),
    ];

    // No model inferences at all
    insert_and_prepare(&conn, &chats, &[], &[]).await;

    let stats = crate::utils::poll_for_result::poll_for_result(
        || conn.query_variant_stats(&function_name),
        |s| s.iter().any(|r| r.count == 2),
        "Timed out waiting for variant statistics without model inferences",
    )
    .await;

    assert_eq!(stats.len(), 1);
    let row = &stats[0];
    assert_eq!(row.count, 2, "2 chat inferences");
    // No model inferences means no token/cost data.
    // ClickHouse: sumMerge of empty state = NULL. Postgres: SUM of no rows = NULL.
    assert_eq!(row.count_with_cost, 0);
}
make_db_test!(test_variant_statistics_no_model_inferences);
