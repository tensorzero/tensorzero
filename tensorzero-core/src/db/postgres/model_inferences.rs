//! Model inference queries for Postgres.
//!
//! This module implements read and write operations for the model_inferences table in Postgres.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::types::Json;
use sqlx::{QueryBuilder, Row};
use uuid::Uuid;

use crate::config::snapshot::SnapshotHash;
use crate::db::model_inferences::ModelInferenceQueries;
use crate::db::query_helpers::uuid_to_datetime;
use crate::error::Error;
use crate::inference::types::{
    ContentBlockOutput, FinishReason, StoredModelInference, StoredRequestMessage,
};

use super::PostgresConnectionInfo;

// =====================================================================
// ModelInferenceQueries trait implementation
// =====================================================================

#[async_trait]
impl ModelInferenceQueries for PostgresConnectionInfo {
    async fn get_model_inferences_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<Vec<StoredModelInference>, Error> {
        let pool = self.get_pool_result()?;

        let mut qb = build_get_model_inferences_query(inference_id);
        let rows: Vec<StoredModelInference> = qb.build_query_as().fetch_all(pool).await?;

        Ok(rows)
    }

    async fn insert_model_inferences(&self, rows: &[StoredModelInference]) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        let pool = self.get_pool_result()?;

        let mut qb = build_insert_model_inferences_query(rows)?;

        qb.build().execute(pool).await?;

        Ok(())
    }
}

// =====================================================================
// Query builder functions (for unit testing)
// =====================================================================

/// Builds a query to get model inferences by inference_id.
fn build_get_model_inferences_query(inference_id: Uuid) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT
            id,
            inference_id,
            raw_request,
            raw_response,
            system,
            input_messages,
            output,
            input_tokens,
            output_tokens,
            response_time_ms,
            model_name,
            model_provider_name,
            ttft_ms,
            cached,
            finish_reason,
            snapshot_hash,
            created_at
        FROM tensorzero.model_inferences
        WHERE inference_id = ",
    );
    qb.push_bind(inference_id);

    qb
}

/// Builds a query to insert model inferences.
fn build_insert_model_inferences_query(
    rows: &[StoredModelInference],
) -> Result<QueryBuilder<sqlx::Postgres>, Error> {
    // Pre-compute timestamps from UUIDs to propagate errors before entering push_values
    let timestamps: Vec<DateTime<Utc>> = rows
        .iter()
        .map(|row| uuid_to_datetime(row.id))
        .collect::<Result<_, _>>()?;

    let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.model_inferences (
            id, inference_id, raw_request, raw_response, system,
            input_messages, output, input_tokens, output_tokens,
            response_time_ms, model_name, model_provider_name,
            ttft_ms, cached, finish_reason, snapshot_hash, created_at
        ) ",
    );

    qb.push_values(rows.iter().zip(&timestamps), |mut b, (row, created_at)| {
        let snapshot_hash_bytes: Option<Vec<u8>> =
            row.snapshot_hash.as_ref().map(|h| h.as_bytes().to_vec());

        b.push_bind(row.id)
            .push_bind(row.inference_id)
            .push_bind(&row.raw_request)
            .push_bind(&row.raw_response)
            .push_bind(&row.system)
            .push_bind(Json(&row.input_messages))
            .push_bind(Json(&row.output))
            .push_bind(row.input_tokens.map(|v| v as i32))
            .push_bind(row.output_tokens.map(|v| v as i32))
            .push_bind(row.response_time_ms.map(|v| v as i32))
            .push_bind(&row.model_name)
            .push_bind(&row.model_provider_name)
            .push_bind(row.ttft_ms.map(|v| v as i32))
            .push_bind(row.cached)
            .push_bind(row.finish_reason)
            .push_bind(snapshot_hash_bytes)
            .push_bind(created_at);
    });

    Ok(qb)
}

// =====================================================================
// FromRow implementation for StoredModelInference
// =====================================================================

/// Manual implementation of FromRow for StoredModelInference.
/// This allows direct deserialization from Postgres rows.
impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for StoredModelInference {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let id: Uuid = row.try_get("id")?;
        let inference_id: Uuid = row.try_get("inference_id")?;
        let raw_request: String = row.try_get("raw_request")?;
        let raw_response: String = row.try_get("raw_response")?;
        let system: Option<String> = row.try_get("system")?;
        let input_messages: Json<Vec<StoredRequestMessage>> = row.try_get("input_messages")?;
        let output: Json<Vec<ContentBlockOutput>> = row.try_get("output")?;
        let input_tokens: Option<i32> = row.try_get("input_tokens")?;
        let output_tokens: Option<i32> = row.try_get("output_tokens")?;
        let response_time_ms: Option<i32> = row.try_get("response_time_ms")?;
        let model_name: String = row.try_get("model_name")?;
        let model_provider_name: String = row.try_get("model_provider_name")?;
        let ttft_ms: Option<i32> = row.try_get("ttft_ms")?;
        let cached: bool = row.try_get("cached")?;
        let finish_reason: Option<FinishReason> = row.try_get("finish_reason")?;
        let snapshot_hash_bytes: Option<Vec<u8>> = row.try_get("snapshot_hash")?;
        let created_at: DateTime<Utc> = row.try_get("created_at")?;

        // Convert snapshot_hash from bytes
        let snapshot_hash = snapshot_hash_bytes.map(|bytes| SnapshotHash::from_bytes(&bytes));

        Ok(StoredModelInference {
            id,
            inference_id,
            raw_request,
            raw_response,
            system,
            input_messages: input_messages.0,
            output: output.0,
            input_tokens: input_tokens.map(|v| v as u32),
            output_tokens: output_tokens.map(|v| v as u32),
            response_time_ms: response_time_ms.map(|v| v as u32),
            model_name,
            model_provider_name,
            ttft_ms: ttft_ms.map(|v| v as u32),
            cached,
            finish_reason,
            snapshot_hash,
            timestamp: Some(created_at.to_rfc3339()),
        })
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_helpers::assert_query_equals;

    #[test]
    fn test_build_get_model_inferences_query() {
        let inference_id = Uuid::nil();
        let qb = build_get_model_inferences_query(inference_id);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                id,
                inference_id,
                raw_request,
                raw_response,
                system,
                input_messages,
                output,
                input_tokens,
                output_tokens,
                response_time_ms,
                model_name,
                model_provider_name,
                ttft_ms,
                cached,
                finish_reason,
                snapshot_hash,
                created_at
            FROM tensorzero.model_inferences
            WHERE inference_id = $1
            ",
        );
    }

    #[test]
    fn test_build_insert_model_inferences_query_single_row() {
        let rows = vec![StoredModelInference {
            id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            raw_request: "request".to_string(),
            raw_response: "response".to_string(),
            system: Some("system".to_string()),
            input_messages: vec![],
            output: vec![],
            input_tokens: Some(10),
            output_tokens: Some(20),
            response_time_ms: Some(100),
            model_name: "test_model".to_string(),
            model_provider_name: "test_provider".to_string(),
            ttft_ms: Some(50),
            cached: false,
            finish_reason: Some(FinishReason::Stop),
            snapshot_hash: None,
            timestamp: None,
        }];

        let qb = build_insert_model_inferences_query(&rows).expect("Should build query");
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            INSERT INTO tensorzero.model_inferences (
                id, inference_id, raw_request, raw_response, system,
                input_messages, output, input_tokens, output_tokens,
                response_time_ms, model_name, model_provider_name,
                ttft_ms, cached, finish_reason, snapshot_hash, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            ",
        );
    }

    #[test]
    fn test_build_insert_model_inferences_query_multiple_rows() {
        let rows = vec![
            StoredModelInference {
                id: Uuid::now_v7(),
                inference_id: Uuid::now_v7(),
                raw_request: "request1".to_string(),
                raw_response: "response1".to_string(),
                system: None,
                input_messages: vec![],
                output: vec![],
                input_tokens: None,
                output_tokens: None,
                response_time_ms: None,
                model_name: "model1".to_string(),
                model_provider_name: "provider1".to_string(),
                ttft_ms: None,
                cached: false,
                finish_reason: None,
                snapshot_hash: None,
                timestamp: None,
            },
            StoredModelInference {
                id: Uuid::now_v7(),
                inference_id: Uuid::now_v7(),
                raw_request: "request2".to_string(),
                raw_response: "response2".to_string(),
                system: Some("system2".to_string()),
                input_messages: vec![],
                output: vec![],
                input_tokens: Some(100),
                output_tokens: Some(200),
                response_time_ms: Some(500),
                model_name: "model2".to_string(),
                model_provider_name: "provider2".to_string(),
                ttft_ms: Some(25),
                cached: true,
                finish_reason: Some(FinishReason::ToolCall),
                snapshot_hash: None,
                timestamp: None,
            },
        ];

        let qb = build_insert_model_inferences_query(&rows).expect("Should build query");
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            INSERT INTO tensorzero.model_inferences (
                id, inference_id, raw_request, raw_response, system,
                input_messages, output, input_tokens, output_tokens,
                response_time_ms, model_name, model_provider_name,
                ttft_ms, cached, finish_reason, snapshot_hash, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17),
            ($18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34)
            ",
        );
    }
}
