//! Batch inference queries for Postgres.
//!
//! This module implements read operations for batch inference tables in Postgres.

use async_trait::async_trait;
use serde_json::Value;
use sqlx::QueryBuilder;
use sqlx::Row;
use sqlx::types::Json;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::config::snapshot::SnapshotHash;
use crate::db::batch_inference::{BatchInferenceQueries, CompletedBatchInferenceRow};
use crate::db::query_helpers::uuid_to_datetime;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::batch::{BatchModelInferenceRow, BatchRequestRow, BatchStatus};
use crate::inference::types::{StoredInput, StoredRequestMessage};
use crate::tool::config::AllowedTools;
use crate::tool::storage::ToolCallConfigDatabaseInsert;
use crate::tool::types::{ProviderTool, Tool};
use crate::tool::wire::ToolChoice;

use super::PostgresConnectionInfo;
use crate::endpoints::inference::InferenceParams;

#[async_trait]
impl BatchInferenceQueries for PostgresConnectionInfo {
    async fn get_batch_request(
        &self,
        batch_id: Uuid,
        inference_id: Option<Uuid>,
    ) -> Result<Option<BatchRequestRow<'static>>, Error> {
        let pool = self.get_pool_result()?;

        let mut qb = build_get_batch_request_query(batch_id, inference_id);

        let row: Option<BatchRequestRow<'static>> = qb
            .build_query_as()
            .fetch_optional(pool)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!("Failed to get batch request: {e}"),
                })
            })?;

        Ok(row)
    }

    async fn get_batch_model_inferences(
        &self,
        batch_id: Uuid,
        inference_ids: &[Uuid],
    ) -> Result<Vec<BatchModelInferenceRow<'static>>, Error> {
        if inference_ids.is_empty() {
            return Ok(vec![]);
        }

        let pool = self.get_pool_result()?;

        sqlx::query_as::<_, BatchModelInferenceRow<'static>>(
            r"
            SELECT
                inference_id,
                batch_id,
                function_name,
                variant_name,
                episode_id,
                input,
                input_messages,
                system,
                dynamic_tools,
                dynamic_provider_tools,
                allowed_tools,
                tool_choice,
                parallel_tool_calls,
                inference_params,
                output_schema,
                raw_request,
                model_name,
                model_provider_name,
                tags,
                snapshot_hash
            FROM tensorzero.batch_model_inferences
            WHERE batch_id = $1 AND inference_id = ANY($2)
            ",
        )
        .bind(batch_id)
        .bind(inference_ids)
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to get batch model inferences: {e}"),
            })
        })
    }

    async fn get_completed_chat_batch_inferences(
        &self,
        batch_id: Uuid,
        function_name: &str,
        variant_name: &str,
        inference_id: Option<Uuid>,
    ) -> Result<Vec<CompletedBatchInferenceRow>, Error> {
        let pool = self.get_pool_result()?;

        let mut qb = build_get_completed_chat_batch_inferences_query(
            batch_id,
            function_name,
            variant_name,
            inference_id,
        );

        let rows: Vec<CompletedBatchInferenceRow> =
            qb.build_query_as().fetch_all(pool).await.map_err(|e| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!("Failed to get completed chat batch inferences: {e}"),
                })
            })?;

        Ok(rows)
    }

    async fn get_completed_json_batch_inferences(
        &self,
        batch_id: Uuid,
        function_name: &str,
        variant_name: &str,
        inference_id: Option<Uuid>,
    ) -> Result<Vec<CompletedBatchInferenceRow>, Error> {
        let pool = self.get_pool_result()?;

        let mut qb = build_get_completed_json_batch_inferences_query(
            batch_id,
            function_name,
            variant_name,
            inference_id,
        );

        let rows: Vec<CompletedBatchInferenceRow> =
            qb.build_query_as().fetch_all(pool).await.map_err(|e| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!("Failed to get completed json batch inferences: {e}"),
                })
            })?;

        Ok(rows)
    }

    // ===== Write methods =====

    async fn write_batch_request(&self, row: &BatchRequestRow<'_>) -> Result<(), Error> {
        let pool = self.get_pool_result()?;

        let status = match row.status {
            BatchStatus::Pending => "pending",
            BatchStatus::Completed => "completed",
            BatchStatus::Failed => "failed",
        };

        let created_at = uuid_to_datetime(row.id)?;

        let snapshot_hash_bytes = row.snapshot_hash.as_ref().map(|h| h.as_bytes().to_vec());

        sqlx::query(
            r"
            INSERT INTO tensorzero.batch_requests (
                batch_id, id, batch_params, model_name, model_provider_name,
                status, function_name, variant_name, raw_request, raw_response, errors, snapshot_hash, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ",
        )
        .bind(row.batch_id)
        .bind(row.id)
        .bind(Json::from(&row.batch_params))
        .bind(row.model_name.as_ref())
        .bind(row.model_provider_name.as_ref())
        .bind(status)
        .bind(row.function_name.as_ref())
        .bind(row.variant_name.as_ref())
        .bind(row.raw_request.as_ref())
        .bind(row.raw_response.as_ref())
        .bind(Json::from(&row.errors))
        .bind(snapshot_hash_bytes)
        .bind(created_at)
        .execute(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to insert batch request: {e}"),
            })
        })?;

        Ok(())
    }

    async fn write_batch_model_inferences(
        &self,
        rows: &[BatchModelInferenceRow<'_>],
    ) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        let pool = self.get_pool_result()?;

        // Pre-compute output_schemas to propagate errors before entering push_values
        let output_schemas: Vec<Option<Value>> = rows
            .iter()
            .map(|row| {
                row.output_schema
                    .as_ref()
                    .map(|s| serde_json::from_str(s))
                    .transpose()
                    .map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!("Failed to parse output_schema: {e}"),
                        })
                    })
            })
            .collect::<Result<_, _>>()?;

        // Pre-compute timestamps from inference_id UUIDs
        let timestamps: Vec<_> = rows
            .iter()
            .map(|row| uuid_to_datetime(row.inference_id))
            .collect::<Result<_, _>>()?;

        let empty_tools: Vec<Tool> = vec![];
        let empty_provider_tools: Vec<ProviderTool> = vec![];

        let mut qb = QueryBuilder::new(
            r"
            INSERT INTO tensorzero.batch_model_inferences (
                inference_id, batch_id, function_name, variant_name, episode_id,
                input, input_messages, system, dynamic_tools, dynamic_provider_tools,
                allowed_tools, tool_choice, parallel_tool_calls, inference_params,
                output_schema, raw_request, model_name, model_provider_name, tags, snapshot_hash, created_at
            )
            ",
        );

        qb.push_values(
            rows.iter().zip(&output_schemas).zip(&timestamps),
            |mut b, ((row, output_schema), created_at)| {
                let tool_params_ref = row.tool_params.as_ref();
                let snapshot_hash_bytes = row.snapshot_hash.as_ref().map(|h| h.as_bytes().to_vec());

                b.push_bind(row.inference_id)
                    .push_bind(row.batch_id)
                    .push_bind(row.function_name.as_ref())
                    .push_bind(row.variant_name.as_ref())
                    .push_bind(row.episode_id)
                    .push_bind(Json::from(&row.input))
                    .push_bind(Json::from(&row.input_messages))
                    .push_bind(row.system.as_deref())
                    .push_bind(Json::from(
                        tool_params_ref
                            .map(|tp| &tp.dynamic_tools)
                            // We always store an empty array when not provided
                            .unwrap_or(&empty_tools),
                    ))
                    .push_bind(Json::from(
                        tool_params_ref
                            .map(|tp| &tp.dynamic_provider_tools)
                            // We always store an empty array when not provided
                            .unwrap_or(&empty_provider_tools),
                    ))
                    .push_bind(tool_params_ref.map(|tp| Json::from(&tp.allowed_tools)))
                    .push_bind(tool_params_ref.map(|tp| Json::from(&tp.tool_choice)))
                    .push_bind(tool_params_ref.and_then(|tp| tp.parallel_tool_calls))
                    .push_bind(Json::from(row.inference_params.as_ref()))
                    .push_bind(output_schema)
                    .push_bind(row.raw_request.as_ref())
                    .push_bind(row.model_name.as_ref())
                    .push_bind(row.model_provider_name.as_ref())
                    .push_bind(Json::from(&row.tags))
                    .push_bind(snapshot_hash_bytes)
                    .push_bind(created_at);
            },
        );

        qb.build().execute(pool).await.map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to insert batch model inferences: {e}"),
            })
        })?;

        Ok(())
    }
}

// =====================================================================
// Query builder functions (for unit testing)
// =====================================================================

/// Builds a query to get a batch request by batch_id, optionally verifying that
/// an inference_id belongs to the batch.
fn build_get_batch_request_query(
    batch_id: Uuid,
    inference_id: Option<Uuid>,
) -> QueryBuilder<sqlx::Postgres> {
    match inference_id {
        None => {
            // Query by batch_id only
            let mut qb = QueryBuilder::new(
                r"
                SELECT
                    batch_id,
                    id,
                    batch_params,
                    model_name,
                    model_provider_name,
                    status,
                    function_name,
                    variant_name,
                    raw_request,
                    raw_response,
                    errors,
                    snapshot_hash
                FROM tensorzero.batch_requests
                WHERE batch_id = ",
            );
            qb.push_bind(batch_id);
            qb.push(
                r"
                ORDER BY created_at DESC
                LIMIT 1
                ",
            );
            qb
        }
        Some(inference_id) => {
            // Query by batch_id and verify inference_id belongs to the batch
            let mut qb = QueryBuilder::new(
                r"
                SELECT
                    br.batch_id,
                    br.id,
                    br.batch_params,
                    br.model_name,
                    br.model_provider_name,
                    br.status,
                    br.function_name,
                    br.variant_name,
                    br.raw_request,
                    br.raw_response,
                    br.errors,
                    br.snapshot_hash
                FROM tensorzero.batch_model_inferences bi
                JOIN tensorzero.batch_requests br ON bi.batch_id = br.batch_id
                WHERE bi.inference_id = ",
            );
            qb.push_bind(inference_id);
            qb.push(" AND bi.batch_id = ");
            qb.push_bind(batch_id);
            qb.push(
                r"
                ORDER BY br.created_at DESC
                LIMIT 1
                ",
            );
            qb
        }
    }
}

/// Builds a query to get completed chat batch inferences.
fn build_get_completed_chat_batch_inferences_query(
    batch_id: Uuid,
    function_name: &str,
    variant_name: &str,
    inference_id: Option<Uuid>,
) -> QueryBuilder<sqlx::Postgres> {
    match inference_id {
        None => {
            // Get all inferences for the batch
            let mut qb = QueryBuilder::new(
                r"
                WITH batch_inferences AS (
                    SELECT inference_id
                    FROM tensorzero.batch_model_inferences
                    WHERE batch_id = ",
            );
            qb.push_bind(batch_id);
            // The ARRAY_AGG function here takes all finish_reasons from model inferences, orders them by model inference ID, and
            // returns the first one (Postgres uses 1-based indexing).
            // This returns the most recent finish reason for each inference, which matches ClickHouse's behavior.
            qb.push(
                r"
                )
                SELECT
                    ci.id as inference_id,
                    ci.episode_id as episode_id,
                    ci.variant_name as variant_name,
                    ci.output::text as output,
                    SUM(mi.input_tokens)::INTEGER as input_tokens,
                    SUM(mi.output_tokens)::INTEGER as output_tokens,
                    (ARRAY_AGG(mi.finish_reason ORDER BY mi.id DESC))[1] as finish_reason
                FROM tensorzero.chat_inferences ci
                LEFT JOIN tensorzero.model_inferences mi ON ci.id = mi.inference_id
                WHERE ci.id IN (SELECT inference_id FROM batch_inferences)
                AND ci.function_name = ",
            );
            qb.push_bind(function_name.to_string());
            qb.push(" AND ci.variant_name = ");
            qb.push_bind(variant_name.to_string());
            qb.push(" GROUP BY ci.id, ci.episode_id, ci.variant_name, ci.output");
            qb
        }
        Some(inference_id) => {
            // Get a specific inference
            // The ARRAY_AGG function here takes all finish_reasons from model inferences, orders them by model inference ID, and
            // returns the first one (Postgres uses 1-based indexing).
            // This returns the most recent finish reason for each inference, which matches ClickHouse's behavior.
            let mut qb = QueryBuilder::new(
                r"
                SELECT
                    ci.id as inference_id,
                    ci.episode_id as episode_id,
                    ci.variant_name as variant_name,
                    ci.output::text as output,
                    SUM(mi.input_tokens)::INTEGER as input_tokens,
                    SUM(mi.output_tokens)::INTEGER as output_tokens,
                    (ARRAY_AGG(mi.finish_reason ORDER BY mi.id DESC))[1] as finish_reason
                FROM tensorzero.chat_inferences ci
                LEFT JOIN tensorzero.model_inferences mi ON ci.id = mi.inference_id
                WHERE ci.id = ",
            );
            qb.push_bind(inference_id);
            qb.push(" AND ci.function_name = ");
            qb.push_bind(function_name.to_string());
            qb.push(" AND ci.variant_name = ");
            qb.push_bind(variant_name.to_string());
            qb.push(" GROUP BY ci.id, ci.episode_id, ci.variant_name, ci.output");
            qb
        }
    }
}

/// Builds a query to get completed json batch inferences.
fn build_get_completed_json_batch_inferences_query(
    batch_id: Uuid,
    function_name: &str,
    variant_name: &str,
    inference_id: Option<Uuid>,
) -> QueryBuilder<sqlx::Postgres> {
    match inference_id {
        None => {
            // Get all inferences for the batch
            let mut qb = QueryBuilder::new(
                r"
                WITH batch_inferences AS (
                    SELECT inference_id
                    FROM tensorzero.batch_model_inferences
                    WHERE batch_id = ",
            );
            qb.push_bind(batch_id);

            // The ARRAY_AGG function here takes all finish_reasons from model inferences, orders them by model inference ID, and
            // returns the first one (Postgres uses 1-based indexing).
            // This returns the most recent finish reason for each inference, which matches ClickHouse's behavior.
            qb.push(
                r"
                )
                SELECT
                    ji.id as inference_id,
                    ji.episode_id as episode_id,
                    ji.variant_name as variant_name,
                    ji.output::text as output,
                    SUM(mi.input_tokens)::INTEGER as input_tokens,
                    SUM(mi.output_tokens)::INTEGER as output_tokens,
                    (ARRAY_AGG(mi.finish_reason ORDER BY mi.id DESC))[1] as finish_reason
                FROM tensorzero.json_inferences ji
                LEFT JOIN tensorzero.model_inferences mi ON ji.id = mi.inference_id
                WHERE ji.id IN (SELECT inference_id FROM batch_inferences)
                AND ji.function_name = ",
            );
            qb.push_bind(function_name.to_string());
            qb.push(" AND ji.variant_name = ");
            qb.push_bind(variant_name.to_string());
            qb.push(" GROUP BY ji.id, ji.episode_id, ji.variant_name, ji.output");
            qb
        }
        Some(inference_id) => {
            // Get a specific inference
            // The ARRAY_AGG function here takes all finish_reasons from model inferences, orders them by model inference ID, and
            // returns the first one (Postgres uses 1-based indexing).
            // This returns the most recent finish reason for each inference, which matches ClickHouse's behavior.
            let mut qb = QueryBuilder::new(
                r"
                SELECT
                    ji.id as inference_id,
                    ji.episode_id as episode_id,
                    ji.variant_name as variant_name,
                    ji.output::text as output,
                    SUM(mi.input_tokens)::INTEGER as input_tokens,
                    SUM(mi.output_tokens)::INTEGER as output_tokens,
                    (ARRAY_AGG(mi.finish_reason ORDER BY mi.id DESC))[1] as finish_reason
                FROM tensorzero.json_inferences ji
                LEFT JOIN tensorzero.model_inferences mi ON ji.id = mi.inference_id
                WHERE ji.id = ",
            );
            qb.push_bind(inference_id);
            qb.push(" AND ji.function_name = ");
            qb.push_bind(function_name.to_string());
            qb.push(" AND ji.variant_name = ");
            qb.push_bind(variant_name.to_string());
            qb.push(" GROUP BY ji.id, ji.episode_id, ji.variant_name, ji.output");
            qb
        }
    }
}

// =====================================================================
// FromRow implementations for batch data
// =====================================================================

/// Manual implementation of FromRow for BatchRequestRow.
/// This allows direct deserialization from Postgres rows.
impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for BatchRequestRow<'static> {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let batch_params: Json<Value> = row.try_get("batch_params")?;
        let status: BatchStatus = row.try_get("status")?;
        let errors: Option<Json<Vec<Value>>> = row.try_get("errors")?;
        let model_name: String = row.try_get("model_name")?;

        let snapshot_hash: Option<Vec<u8>> = row.try_get("snapshot_hash")?;
        let snapshot_hash = snapshot_hash.map(|bytes| SnapshotHash::from_bytes(&bytes));

        Ok(BatchRequestRow {
            batch_id: row.try_get("batch_id")?,
            id: row.try_get("id")?,
            batch_params: Cow::Owned(batch_params.0),
            model_name: Arc::from(model_name.as_str()),
            model_provider_name: Cow::Owned(row.try_get("model_provider_name")?),
            status,
            function_name: Cow::Owned(row.try_get("function_name")?),
            variant_name: Cow::Owned(row.try_get("variant_name")?),
            raw_request: Cow::Owned(row.try_get("raw_request")?),
            raw_response: Cow::Owned(row.try_get("raw_response")?),
            errors: errors.map(|e| e.0).unwrap_or_default(),
            snapshot_hash,
        })
    }
}

/// Manual implementation of FromRow for BatchModelInferenceRow.
/// This allows direct deserialization from Postgres rows.
impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for BatchModelInferenceRow<'static> {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let input: Json<StoredInput> = row.try_get("input")?;
        let input_messages: Json<Vec<StoredRequestMessage>> = row.try_get("input_messages")?;
        let dynamic_tools: Json<Vec<Tool>> = row.try_get("dynamic_tools")?;
        let dynamic_provider_tools: Json<Vec<ProviderTool>> =
            row.try_get("dynamic_provider_tools")?;
        let allowed_tools: Option<Json<AllowedTools>> = row.try_get("allowed_tools")?;
        let tool_choice: Option<Json<ToolChoice>> = row.try_get("tool_choice")?;
        let parallel_tool_calls: Option<bool> = row.try_get("parallel_tool_calls")?;
        let inference_params: Json<InferenceParams> = row.try_get("inference_params")?;
        let output_schema: Option<Value> = row.try_get("output_schema")?;
        let tags: Json<HashMap<String, String>> = row.try_get("tags")?;

        let tool_params = ToolCallConfigDatabaseInsert::from_stored_values(
            dynamic_tools.0,
            dynamic_provider_tools.0,
            allowed_tools.map(|v| v.0),
            tool_choice.map(|v| v.0),
            parallel_tool_calls,
        );

        let snapshot_hash: Option<Vec<u8>> = row.try_get("snapshot_hash")?;
        let snapshot_hash = snapshot_hash.map(|bytes| SnapshotHash::from_bytes(&bytes));

        Ok(BatchModelInferenceRow {
            inference_id: row.try_get("inference_id")?,
            batch_id: row.try_get("batch_id")?,
            function_name: Cow::Owned(row.try_get("function_name")?),
            variant_name: Cow::Owned(row.try_get("variant_name")?),
            episode_id: row.try_get("episode_id")?,
            input: input.0,
            input_messages: input_messages.0,
            system: row.try_get::<Option<String>, _>("system")?.map(Cow::Owned),
            tool_params,
            inference_params: Cow::Owned(inference_params.0),
            output_schema: output_schema.map(|v| serde_json::to_string(&v).unwrap_or_default()),
            raw_request: Cow::Owned(row.try_get("raw_request")?),
            model_name: Cow::Owned(row.try_get("model_name")?),
            model_provider_name: Cow::Owned(row.try_get("model_provider_name")?),
            tags: tags.0,
            snapshot_hash,
        })
    }
}

/// Manual implementation of FromRow for CompletedBatchInferenceRow.
/// This allows direct deserialization from Postgres rows.
impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for CompletedBatchInferenceRow {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let finish_reason_str: Option<String> = row.try_get("finish_reason")?;
        let finish_reason = finish_reason_str
            .map(|fr| serde_json::from_value(Value::String(fr)))
            .transpose()
            .map_err(|e| sqlx::Error::ColumnDecode {
                index: "finish_reason".to_string(),
                source: Box::new(e),
            })?;

        let input_tokens: Option<i32> = row.try_get("input_tokens")?;
        let output_tokens: Option<i32> = row.try_get("output_tokens")?;

        Ok(CompletedBatchInferenceRow {
            inference_id: row.try_get("inference_id")?,
            episode_id: row.try_get("episode_id")?,
            variant_name: row.try_get("variant_name")?,
            output: row.try_get("output")?,
            input_tokens: input_tokens.map(|t| t as u32),
            output_tokens: output_tokens.map(|t| t as u32),
            finish_reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_helpers::assert_query_equals;

    #[test]
    fn test_build_get_batch_request_query_by_batch_id_only() {
        let batch_id = Uuid::now_v7();
        let qb = build_get_batch_request_query(batch_id, None);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                batch_id,
                id,
                batch_params,
                model_name,
                model_provider_name,
                status,
                function_name,
                variant_name,
                raw_request,
                raw_response,
                errors,
                snapshot_hash
            FROM tensorzero.batch_requests
            WHERE batch_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            ",
        );
    }

    #[test]
    fn test_build_get_batch_request_query_with_inference_id() {
        let batch_id = Uuid::now_v7();
        let inference_id = Uuid::now_v7();
        let qb = build_get_batch_request_query(batch_id, Some(inference_id));
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                br.batch_id,
                br.id,
                br.batch_params,
                br.model_name,
                br.model_provider_name,
                br.status,
                br.function_name,
                br.variant_name,
                br.raw_request,
                br.raw_response,
                br.errors,
                br.snapshot_hash
            FROM tensorzero.batch_model_inferences bi
            JOIN tensorzero.batch_requests br ON bi.batch_id = br.batch_id
            WHERE bi.inference_id = $1 AND bi.batch_id = $2
            ORDER BY br.created_at DESC
            LIMIT 1
            ",
        );
    }

    #[test]
    fn test_build_get_completed_chat_batch_inferences_query_all() {
        let batch_id = Uuid::now_v7();
        let qb = build_get_completed_chat_batch_inferences_query(
            batch_id,
            "test_function",
            "test_variant",
            None,
        );
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH batch_inferences AS (
                SELECT inference_id
                FROM tensorzero.batch_model_inferences
                WHERE batch_id = $1
            )
            SELECT
                ci.id as inference_id,
                ci.episode_id as episode_id,
                ci.variant_name as variant_name,
                ci.output::text as output,
                SUM(mi.input_tokens)::INTEGER as input_tokens,
                SUM(mi.output_tokens)::INTEGER as output_tokens,
                (ARRAY_AGG(mi.finish_reason ORDER BY mi.id DESC))[1] as finish_reason
            FROM tensorzero.chat_inferences ci
            LEFT JOIN tensorzero.model_inferences mi ON ci.id = mi.inference_id
            WHERE ci.id IN (SELECT inference_id FROM batch_inferences)
            AND ci.function_name = $2 AND ci.variant_name = $3
            GROUP BY ci.id, ci.episode_id, ci.variant_name, ci.output
            ",
        );
    }

    #[test]
    fn test_build_get_completed_chat_batch_inferences_query_single() {
        let batch_id = Uuid::now_v7();
        let inference_id = Uuid::now_v7();
        let qb = build_get_completed_chat_batch_inferences_query(
            batch_id,
            "test_function",
            "test_variant",
            Some(inference_id),
        );
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                ci.id as inference_id,
                ci.episode_id as episode_id,
                ci.variant_name as variant_name,
                ci.output::text as output,
                SUM(mi.input_tokens)::INTEGER as input_tokens,
                SUM(mi.output_tokens)::INTEGER as output_tokens,
                (ARRAY_AGG(mi.finish_reason ORDER BY mi.id DESC))[1] as finish_reason
            FROM tensorzero.chat_inferences ci
            LEFT JOIN tensorzero.model_inferences mi ON ci.id = mi.inference_id
            WHERE ci.id = $1 AND ci.function_name = $2 AND ci.variant_name = $3
            GROUP BY ci.id, ci.episode_id, ci.variant_name, ci.output
            ",
        );
    }

    #[test]
    fn test_build_get_completed_json_batch_inferences_query_all() {
        let batch_id = Uuid::now_v7();
        let qb = build_get_completed_json_batch_inferences_query(
            batch_id,
            "test_function",
            "test_variant",
            None,
        );
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH batch_inferences AS (
                SELECT inference_id
                FROM tensorzero.batch_model_inferences
                WHERE batch_id = $1
            )
            SELECT
                ji.id as inference_id,
                ji.episode_id as episode_id,
                ji.variant_name as variant_name,
                ji.output::text as output,
                SUM(mi.input_tokens)::INTEGER as input_tokens,
                SUM(mi.output_tokens)::INTEGER as output_tokens,
                (ARRAY_AGG(mi.finish_reason ORDER BY mi.id DESC))[1] as finish_reason
            FROM tensorzero.json_inferences ji
            LEFT JOIN tensorzero.model_inferences mi ON ji.id = mi.inference_id
            WHERE ji.id IN (SELECT inference_id FROM batch_inferences)
            AND ji.function_name = $2 AND ji.variant_name = $3
            GROUP BY ji.id, ji.episode_id, ji.variant_name, ji.output
            ",
        );
    }

    #[test]
    fn test_build_get_completed_json_batch_inferences_query_single() {
        let batch_id = Uuid::now_v7();
        let inference_id = Uuid::now_v7();
        let qb = build_get_completed_json_batch_inferences_query(
            batch_id,
            "test_function",
            "test_variant",
            Some(inference_id),
        );
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                ji.id as inference_id,
                ji.episode_id as episode_id,
                ji.variant_name as variant_name,
                ji.output::text as output,
                SUM(mi.input_tokens)::INTEGER as input_tokens,
                SUM(mi.output_tokens)::INTEGER as output_tokens,
                (ARRAY_AGG(mi.finish_reason ORDER BY mi.id DESC))[1] as finish_reason
            FROM tensorzero.json_inferences ji
            LEFT JOIN tensorzero.model_inferences mi ON ji.id = mi.inference_id
            WHERE ji.id = $1 AND ji.function_name = $2 AND ji.variant_name = $3
            GROUP BY ji.id, ji.episode_id, ji.variant_name, ji.output
            ",
        );
    }
}
