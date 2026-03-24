//! Postgres queries for evaluation statistics.

use async_trait::async_trait;
use sqlx::postgres::PgRow;
use sqlx::{PgPool, QueryBuilder, Row};
use uuid::Uuid;

use crate::db::evaluation_queries::{
    EvaluationQueries, EvaluationResultRow, EvaluationRunInfoByIdRow, EvaluationRunInfoRow,
    EvaluationRunSearchResult, EvaluationStatisticsRow, InferenceEvaluationHumanFeedbackRow,
    InferenceEvaluationRunInsert, InferenceEvaluationRunMetadata,
    InferenceEvaluationRunMetricMetadata, RawEvaluationResultRow,
};
use crate::endpoints::inference::InferenceResponse;
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfigType;
use crate::serde_util::serialize_with_sorted_keys;
use crate::statistics_util::{wald_confint, wilson_confint};

use super::PostgresConnectionInfo;

/// Raw statistics row from Postgres before CI computation.
#[derive(sqlx::FromRow)]
struct RawEvaluationStatisticsRow {
    evaluation_run_id: Uuid,
    metric_name: String,
    is_boolean: bool,
    datapoint_count: i32,
    mean_metric: f64,
    stdev: Option<f64>,
}

impl RawEvaluationStatisticsRow {
    fn into_evaluation_statistics_row(self) -> EvaluationStatisticsRow {
        let (ci_lower, ci_upper) = if self.is_boolean {
            wilson_confint(self.mean_metric, self.datapoint_count as u32)
                .map(|(l, u)| (Some(l), Some(u)))
                .unwrap_or((None, None))
        } else if let Some(stdev) = self.stdev {
            wald_confint(self.mean_metric, stdev, self.datapoint_count as u32)
                .map(|(l, u)| (Some(l), Some(u)))
                .unwrap_or((None, None))
        } else {
            (None, None)
        };

        EvaluationStatisticsRow {
            evaluation_run_id: self.evaluation_run_id,
            metric_name: self.metric_name,
            datapoint_count: self.datapoint_count as u32,
            mean_metric: self.mean_metric,
            ci_lower,
            ci_upper,
        }
    }
}

// =====================================================================
// EvaluationQueries trait implementation
// =====================================================================

#[async_trait]
impl EvaluationQueries for PostgresConnectionInfo {
    async fn get_inference_evaluation_run_metadata(
        &self,
        evaluation_run_ids: &[Uuid],
    ) -> Result<Vec<(Uuid, InferenceEvaluationRunMetadata)>, Error> {
        if evaluation_run_ids.is_empty() {
            return Ok(Vec::new());
        }

        let pool = self.get_pool_result()?;

        #[derive(sqlx::FromRow)]
        struct InferenceEvaluationRunMetadataRow {
            run_id: Uuid,
            evaluation_name: String,
            function_name: String,
            function_type: FunctionConfigType,
            metrics: sqlx::types::Json<Vec<InferenceEvaluationRunMetricMetadata>>,
        }
        let rows: Vec<InferenceEvaluationRunMetadataRow> = sqlx::query_as(
            r"
            SELECT run_id, evaluation_name, function_name, function_type, metrics
            FROM tensorzero.inference_evaluation_runs
            WHERE run_id = ANY($1)
            ",
        )
        .bind(evaluation_run_ids)
        .fetch_all(pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|row| {
                (
                    row.run_id,
                    InferenceEvaluationRunMetadata {
                        evaluation_name: row.evaluation_name,
                        function_name: row.function_name,
                        function_type: row.function_type,
                        metrics: row.metrics.0,
                    },
                )
            })
            .collect())
    }

    async fn insert_inference_evaluation_run(
        &self,
        run: &InferenceEvaluationRunInsert,
    ) -> Result<(), Error> {
        let pool = self.get_pool_result()?;

        let variant_names_json = serde_json::to_value(&run.variant_names).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize `variant_names`: {e}"),
            })
        })?;
        let metrics_json = serde_json::to_value(&run.metrics).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize `metrics`: {e}"),
            })
        })?;

        sqlx::query(
            r"
            INSERT INTO tensorzero.inference_evaluation_runs (
                run_id,
                evaluation_name,
                function_name,
                function_type,
                dataset_name,
                variant_names,
                metrics,
                source,
                snapshot_hash
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (run_id) DO UPDATE SET
                evaluation_name = EXCLUDED.evaluation_name,
                function_name = EXCLUDED.function_name,
                function_type = EXCLUDED.function_type,
                dataset_name = EXCLUDED.dataset_name,
                variant_names = EXCLUDED.variant_names,
                metrics = EXCLUDED.metrics,
                source = EXCLUDED.source,
                snapshot_hash = EXCLUDED.snapshot_hash,
                updated_at = NOW()
            ",
        )
        .bind(run.run_id)
        .bind(run.evaluation_name.as_str())
        .bind(run.function_name.as_str())
        .bind(run.function_type.as_str())
        .bind(run.dataset_name.as_str())
        .bind(variant_names_json)
        .bind(metrics_json)
        .bind(run.source.to_string())
        .bind(run.snapshot_hash.clone())
        .execute(pool)
        .await?;

        Ok(())
    }

    async fn count_total_evaluation_runs(&self) -> Result<u64, Error> {
        // This is most likely performant enough because we expect evaluations to have small scale;
        // If not, switch to approximated counts via pg_class.reltuples.
        let pool = self.get_pool_result()?;
        let count: i64 =
            sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM tensorzero.inference_evaluation_runs")
                .fetch_one(pool)
                .await?;
        Ok(count as u64)
    }

    async fn list_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EvaluationRunInfoRow>, Error> {
        let pool = self.get_pool_result()?;
        let mut qb = build_list_evaluation_runs_query(limit, offset);
        let rows: Vec<EvaluationRunInfoRow> = qb.build_query_as().fetch_all(pool).await?;
        Ok(rows)
    }

    async fn count_datapoints_for_evaluation(
        &self,
        function_name: &str,
        evaluation_run_ids: &[Uuid],
    ) -> Result<u64, Error> {
        if evaluation_run_ids.is_empty() {
            return Ok(0);
        }
        let pool = self.get_pool_result()?;
        let mut qb = build_count_datapoints_for_evaluation_query(function_name, evaluation_run_ids);
        let row: PgRow = qb.build().fetch_one(pool).await?;
        let count: i64 = row.get("count");
        Ok(count as u64)
    }

    async fn search_evaluation_runs(
        &self,
        evaluation_name: Option<&str>,
        function_name: Option<&str>,
        query: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EvaluationRunSearchResult>, Error> {
        let pool = self.get_pool_result()?;
        let mut qb = build_search_evaluation_runs_query(
            evaluation_name,
            function_name,
            query,
            limit,
            offset,
        );
        let rows: Vec<EvaluationRunSearchResult> = qb.build_query_as().fetch_all(pool).await?;
        Ok(rows)
    }

    async fn get_evaluation_run_infos(
        &self,
        evaluation_run_ids: &[Uuid],
        function_name: &str,
    ) -> Result<Vec<EvaluationRunInfoByIdRow>, Error> {
        if evaluation_run_ids.is_empty() {
            return Ok(vec![]);
        }
        let pool = self.get_pool_result()?;
        let mut qb = build_get_evaluation_run_infos_query(evaluation_run_ids, function_name);
        let rows: Vec<EvaluationRunInfoByIdRow> = qb.build_query_as().fetch_all(pool).await?;
        Ok(rows)
    }

    async fn get_evaluation_run_infos_for_datapoint(
        &self,
        datapoint_id: &Uuid,
        function_name: &str,
        function_type: FunctionConfigType,
    ) -> Result<Vec<EvaluationRunInfoByIdRow>, Error> {
        let pool = self.get_pool_result()?;
        let mut qb = build_get_evaluation_run_infos_for_datapoint_query(
            datapoint_id,
            function_name,
            function_type,
        );
        let rows: Vec<EvaluationRunInfoByIdRow> = qb.build_query_as().fetch_all(pool).await?;
        Ok(rows)
    }

    async fn get_evaluation_statistics(
        &self,
        function_name: &str,
        function_type: FunctionConfigType,
        metric_names: &[String],
        evaluation_run_ids: &[Uuid],
    ) -> Result<Vec<EvaluationStatisticsRow>, Error> {
        if evaluation_run_ids.is_empty() || metric_names.is_empty() {
            return Ok(vec![]);
        }
        let pool = self.get_pool_result()?;
        let raw_rows = get_evaluation_statistics_raw(
            pool,
            function_name,
            function_type,
            metric_names,
            evaluation_run_ids,
        )
        .await?;

        Ok(raw_rows
            .into_iter()
            .map(|row| row.into_evaluation_statistics_row())
            .collect())
    }

    async fn get_evaluation_results(
        &self,
        function_name: &str,
        evaluation_run_ids: &[Uuid],
        function_type: FunctionConfigType,
        metric_names: &[String],
        datapoint_id: Option<&Uuid>,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EvaluationResultRow>, Error> {
        if evaluation_run_ids.is_empty() {
            return Ok(vec![]);
        }
        let pool = self.get_pool_result()?;
        let mut qb = build_get_evaluation_results_query(
            function_name,
            evaluation_run_ids,
            function_type,
            metric_names,
            datapoint_id,
            limit,
            offset,
        );
        let rows: Vec<RawEvaluationResultRow> = qb.build_query_as().fetch_all(pool).await?;

        rows.into_iter()
            .map(|raw| match function_type {
                FunctionConfigType::Chat => raw.into_chat().map(EvaluationResultRow::Chat),
                FunctionConfigType::Json => raw.into_json().map(EvaluationResultRow::Json),
            })
            .collect()
    }

    fn serialize_output_for_feedback(
        &self,
        inference_response: &InferenceResponse,
    ) -> Result<String, Error> {
        // Serialize through Value so we can sort keys before producing the final string.
        // Postgres JSONB does not preserve key order, so the stored output may have
        // different ordering than direct struct serialization.
        let output_value = match inference_response {
            InferenceResponse::Chat(c) => serde_json::to_value(&c.content),
            InferenceResponse::Json(j) => serde_json::to_value(&j.output),
        }
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize inference output for feedback lookup: {e}"),
            })
        })?;
        serialize_with_sorted_keys(&output_value)
    }

    async fn get_inference_evaluation_human_feedback(
        &self,
        metric_name: &str,
        datapoint_id: &Uuid,
        output: &str,
    ) -> Result<Option<InferenceEvaluationHumanFeedbackRow>, Error> {
        let pool = self.get_pool_result()?;
        let mut qb =
            build_get_inference_evaluation_human_feedback_query(metric_name, datapoint_id, output);
        let row: Option<PgRow> = qb.build().fetch_optional(pool).await?;

        match row {
            Some(row) => {
                let value_str: String = row.get("value");
                let value: serde_json::Value = serde_json::from_str(&value_str).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Failed to deserialize human feedback value: {e}"),
                    })
                })?;
                // TODO(shuyangli): Change `InferenceEvaluationHumanFeedbackRow.evaluator_inference_id` to `Option<Uuid>`
                // Postgres stores NULL for missing evaluator_inference_id;
                // ClickHouse defaults to nil UUID. Normalize to nil here for consistency.
                let evaluator_inference_id: Option<Uuid> = row.get("evaluator_inference_id");
                let evaluator_inference_id = evaluator_inference_id.unwrap_or(Uuid::nil());
                Ok(Some(InferenceEvaluationHumanFeedbackRow {
                    value,
                    evaluator_inference_id,
                }))
            }
            None => Ok(None),
        }
    }
}

// =====================================================================
// Query builder functions
// =====================================================================

fn build_list_evaluation_runs_query(limit: u32, offset: u32) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT
            run_id as evaluation_run_id,
            evaluation_name,
            function_name,
            COALESCE(variant_names->>0, '') as variant_name,
            dataset_name,
            created_at,
            encode(snapshot_hash, 'hex') as snapshot_hash
        FROM tensorzero.inference_evaluation_runs
        ORDER BY run_id DESC
        LIMIT ",
    );
    qb.push_bind(limit as i64);
    qb.push(" OFFSET ");
    qb.push_bind(offset as i64);

    qb
}

fn build_count_datapoints_for_evaluation_query(
    function_name: &str,
    evaluation_run_ids: &[Uuid],
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        WITH all_inference_ids AS (
            SELECT DISTINCT id as inference_id
            FROM tensorzero.chat_inferences
            WHERE tags->>'tensorzero::evaluation_run_id' = ANY(",
    );
    let run_id_strings: Vec<String> = evaluation_run_ids.iter().map(|id| id.to_string()).collect();
    qb.push_bind(run_id_strings.clone());
    qb.push(") AND function_name = ");
    qb.push_bind(function_name.to_string());
    qb.push(
        r"
            UNION ALL
            SELECT DISTINCT id as inference_id
            FROM tensorzero.json_inferences
            WHERE tags->>'tensorzero::evaluation_run_id' = ANY(",
    );
    qb.push_bind(run_id_strings);
    qb.push(") AND function_name = ");
    qb.push_bind(function_name.to_string());
    qb.push(
        r"
        )
        SELECT COUNT(DISTINCT tags->>'tensorzero::datapoint_id')::BIGINT as count
        FROM (
            SELECT tags FROM tensorzero.chat_inferences WHERE id IN (SELECT inference_id FROM all_inference_ids)
            UNION ALL
            SELECT tags FROM tensorzero.json_inferences WHERE id IN (SELECT inference_id FROM all_inference_ids)
        ) sub
        WHERE tags ? 'tensorzero::datapoint_id'
        ",
    );

    qb
}

fn build_search_evaluation_runs_query(
    evaluation_name: Option<&str>,
    function_name: Option<&str>,
    query: &str,
    limit: u32,
    offset: u32,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT
            run_id AS evaluation_run_id,
            evaluation_name,
            dataset_name,
            COALESCE(variant_names->>0, '') AS variant_name
        FROM tensorzero.inference_evaluation_runs
        WHERE TRUE",
    );
    if let Some(evaluation_name) = evaluation_name {
        qb.push(" AND evaluation_name = ");
        qb.push_bind(evaluation_name.to_string());
    }
    if let Some(fn_name) = function_name {
        qb.push(" AND function_name = ");
        qb.push_bind(fn_name.to_string());
    }
    if !query.is_empty() {
        qb.push(
            r"
        AND (run_id::TEXT ILIKE ",
        );
        qb.push_bind(format!("%{query}%"));
        qb.push(" OR evaluation_name ILIKE ");
        qb.push_bind(format!("%{query}%"));
        qb.push(" OR dataset_name ILIKE ");
        qb.push_bind(format!("%{query}%"));
        qb.push(" OR COALESCE(variant_names->>0, '') ILIKE ");
        qb.push_bind(format!("%{query}%"));
        qb.push(")");
    }
    qb.push(
        r"
        ORDER BY run_id DESC
        LIMIT ",
    );
    qb.push_bind(limit as i64);
    qb.push(" OFFSET ");
    qb.push_bind(offset as i64);

    qb
}

fn build_get_evaluation_run_infos_query(
    evaluation_run_ids: &[Uuid],
    function_name: &str,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT
            run_id as evaluation_run_id,
            COALESCE(variant_names->>0, '') as variant_name,
            created_at
        FROM tensorzero.inference_evaluation_runs
        WHERE run_id = ANY(",
    );
    qb.push_bind(evaluation_run_ids.to_vec());
    qb.push(") AND function_name = ");
    qb.push_bind(function_name.to_string());
    qb.push(" ORDER BY run_id DESC");

    qb
}

fn build_get_evaluation_run_infos_for_datapoint_query(
    datapoint_id: &Uuid,
    function_name: &str,
    function_type: FunctionConfigType,
) -> QueryBuilder<sqlx::Postgres> {
    let inference_table = function_type.postgres_table_name();

    let mut qb = QueryBuilder::new(
        "SELECT (tags->>'tensorzero::evaluation_run_id')::UUID as evaluation_run_id, (ARRAY_AGG(variant_name))[1] as variant_name, MAX(created_at) as created_at FROM ",
    );
    qb.push(inference_table);
    qb.push(" WHERE tags->>'tensorzero::datapoint_id' = ");
    qb.push_bind(datapoint_id.to_string());
    qb.push(" AND function_name = ");
    qb.push_bind(function_name.to_string());
    qb.push(
        r" AND tags ? 'tensorzero::evaluation_run_id'
        GROUP BY tags->>'tensorzero::evaluation_run_id'
        ",
    );

    qb
}

/// Builds the statistics query using Postgres QueryBuilder.
fn build_get_evaluation_statistics_query(
    function_name: &str,
    function_type: FunctionConfigType,
    metric_names: &[String],
    evaluation_run_ids: &[Uuid],
) -> QueryBuilder<sqlx::Postgres> {
    let inference_table = function_type.postgres_table_name();
    let run_id_strings: Vec<String> = evaluation_run_ids.iter().map(|id| id.to_string()).collect();
    let metric_names_owned: Vec<String> = metric_names.to_vec();

    let mut qb = QueryBuilder::new(
        "WITH filtered_inference AS (SELECT id, tags->>'tensorzero::evaluation_run_id' as evaluation_run_id FROM ",
    );
    qb.push(inference_table);
    qb.push(" WHERE tags->>'tensorzero::evaluation_run_id' = ANY(");
    qb.push_bind(run_id_strings);
    qb.push(") AND function_name = ");
    qb.push_bind(function_name.to_string());
    qb.push(
        r"),
        float_feedback AS (
            SELECT DISTINCT ON (target_id, metric_name)
                metric_name, value, target_id
            FROM tensorzero.float_metric_feedback
            WHERE metric_name = ANY(",
    );
    qb.push_bind(metric_names_owned.clone());
    qb.push(
        r") AND target_id IN (SELECT id FROM filtered_inference)
            ORDER BY target_id, metric_name, created_at DESC
        ),
        boolean_feedback AS (
            SELECT DISTINCT ON (target_id, metric_name)
                metric_name, value, target_id
            FROM tensorzero.boolean_metric_feedback
            WHERE metric_name = ANY(",
    );
    qb.push_bind(metric_names_owned);
    qb.push(
        r") AND target_id IN (SELECT id FROM filtered_inference)
            ORDER BY target_id, metric_name, created_at DESC
        ),
        float_stats AS (
            SELECT
                fi.evaluation_run_id::UUID as evaluation_run_id,
                ff.metric_name,
                false as is_boolean,
                COUNT(*)::INT as datapoint_count,
                AVG(ff.value)::DOUBLE PRECISION as mean_metric,
                STDDEV_SAMP(ff.value)::DOUBLE PRECISION as stdev
            FROM filtered_inference fi
            INNER JOIN float_feedback ff ON ff.target_id = fi.id
            WHERE ff.value IS NOT NULL
            GROUP BY fi.evaluation_run_id, ff.metric_name
        ),
        boolean_stats AS (
            SELECT
                fi.evaluation_run_id::UUID as evaluation_run_id,
                bf.metric_name,
                true as is_boolean,
                COUNT(*)::INT as datapoint_count,
                AVG(bf.value::INT)::DOUBLE PRECISION as mean_metric,
                NULL::DOUBLE PRECISION as stdev
            FROM filtered_inference fi
            INNER JOIN boolean_feedback bf ON bf.target_id = fi.id
            WHERE bf.value IS NOT NULL
            GROUP BY fi.evaluation_run_id, bf.metric_name
        )
        SELECT * FROM float_stats
        UNION ALL
        SELECT * FROM boolean_stats
        ORDER BY evaluation_run_id DESC, metric_name ASC
        ",
    );

    qb
}

fn build_get_evaluation_results_query(
    function_name: &str,
    evaluation_run_ids: &[Uuid],
    function_type: FunctionConfigType,
    metric_names: &[String],
    datapoint_id: Option<&Uuid>,
    limit: u32,
    offset: u32,
) -> QueryBuilder<sqlx::Postgres> {
    let inference_table = function_type.postgres_table_name();
    let inference_data_table = function_type.postgres_inference_data_table_name();
    let datapoint_table = function_type.postgres_datapoint_table_name();
    let run_id_strings: Vec<String> = evaluation_run_ids.iter().map(|id| id.to_string()).collect();
    let metric_names_owned: Vec<String> = metric_names.to_vec();

    // CTE 1: all_inference_ids - find inferences matching evaluation run IDs
    let mut qb =
        QueryBuilder::new("WITH all_inference_ids AS (SELECT DISTINCT id as inference_id FROM ");
    qb.push(inference_table);
    qb.push(" WHERE tags->>'tensorzero::evaluation_run_id' = ANY(");
    qb.push_bind(run_id_strings);
    qb.push(") AND function_name = ");
    qb.push_bind(function_name.to_string());

    // CTE 2: all_datapoint_ids - find distinct datapoint IDs with optional filter and pagination
    qb.push(
        r"),
        all_datapoint_ids AS (
            SELECT DISTINCT (tags->>'tensorzero::datapoint_id')::UUID as datapoint_id
            FROM ",
    );
    qb.push(inference_table);
    qb.push(
        r" WHERE id IN (SELECT inference_id FROM all_inference_ids)
            AND tags ? 'tensorzero::datapoint_id'",
    );

    if let Some(dp_id) = datapoint_id {
        qb.push(" AND (tags->>'tensorzero::datapoint_id')::UUID = ");
        qb.push_bind(*dp_id);
    }

    qb.push(" ORDER BY datapoint_id DESC");

    // Only apply pagination if no specific datapoint_id is requested
    if datapoint_id.is_none() {
        qb.push(" LIMIT ");
        qb.push_bind(limit as i64);
        qb.push(" OFFSET ");
        qb.push_bind(offset as i64);
    }

    // CTE 3: filtered_dp - datapoints in the result set
    qb.push("), filtered_dp AS (SELECT * FROM ");
    qb.push(datapoint_table);
    qb.push(" WHERE function_name = ");
    qb.push_bind(function_name.to_string());
    qb.push(" AND id IN (SELECT datapoint_id FROM all_datapoint_ids)");

    // CTE 4: filtered_inference - inferences matching the evaluation runs
    qb.push("), filtered_inference AS (SELECT * FROM ");
    qb.push(inference_table);
    qb.push(" WHERE id IN (SELECT inference_id FROM all_inference_ids) AND function_name = ");
    qb.push_bind(function_name.to_string());

    // CTE 5: filtered_inference_data - inference data (input/output) from the split data table
    qb.push("), filtered_inference_data AS (SELECT * FROM ");
    qb.push(inference_data_table);
    qb.push(" WHERE id IN (SELECT inference_id FROM all_inference_ids)");

    // CTE 6: filtered_feedback - latest feedback per (target_id, metric_name) from both boolean and float
    qb.push(
        r"),
        filtered_feedback AS (
            SELECT * FROM (
                SELECT DISTINCT ON (target_id, metric_name)
                    metric_name,
                    value::TEXT as value,
                    NULLIF(tags->>'tensorzero::evaluator_inference_id', '')::UUID as evaluator_inference_id,
                    id as feedback_id,
                    (COALESCE(tags->>'tensorzero::human_feedback', '') = 'true') as is_human_feedback,
                    target_id
                FROM tensorzero.boolean_metric_feedback
                WHERE metric_name = ANY(",
    );
    qb.push_bind(metric_names_owned.clone());
    qb.push(
        r") AND target_id IN (SELECT inference_id FROM all_inference_ids)
                ORDER BY target_id, metric_name, created_at DESC
            ) bool_fb
            UNION ALL
            SELECT * FROM (
                SELECT DISTINCT ON (target_id, metric_name)
                    metric_name,
                    value::TEXT as value,
                    NULLIF(tags->>'tensorzero::evaluator_inference_id', '')::UUID as evaluator_inference_id,
                    id as feedback_id,
                    (COALESCE(tags->>'tensorzero::human_feedback', '') = 'true') as is_human_feedback,
                    target_id
                FROM tensorzero.float_metric_feedback
                WHERE metric_name = ANY(",
    );
    qb.push_bind(metric_names_owned);
    qb.push(
        r") AND target_id IN (SELECT inference_id FROM all_inference_ids)
                ORDER BY target_id, metric_name, created_at DESC
            ) float_fb
        )
        SELECT
            filtered_dp.input as input,
            filtered_dp.id as datapoint_id,
            filtered_dp.name as name,
            filtered_dp.output::TEXT as reference_output,
            filtered_inference_data.output::TEXT as generated_output,
            (filtered_inference.tags->>'tensorzero::evaluation_run_id')::UUID as evaluation_run_id,
            filtered_inference.tags->>'tensorzero::dataset_name' as dataset_name,
            filtered_feedback.evaluator_inference_id as evaluator_inference_id,
            filtered_inference.id as inference_id,
            filtered_inference.episode_id as episode_id,
            filtered_feedback.metric_name as metric_name,
            filtered_feedback.value as metric_value,
            filtered_feedback.feedback_id as feedback_id,
            COALESCE(filtered_feedback.is_human_feedback, false) as is_human_feedback,
            filtered_dp.staled_at::TEXT as staled_at,
            filtered_inference.variant_name as variant_name
        FROM filtered_dp
        INNER JOIN filtered_inference
            ON (filtered_inference.tags->>'tensorzero::datapoint_id')::UUID = filtered_dp.id
        LEFT JOIN filtered_inference_data
            ON filtered_inference_data.id = filtered_inference.id
            AND filtered_inference_data.created_at = filtered_inference.created_at
        LEFT JOIN filtered_feedback
            ON filtered_feedback.target_id = filtered_inference.id
        ORDER BY datapoint_id DESC, metric_name DESC
        ",
    );

    qb
}

fn build_get_inference_evaluation_human_feedback_query(
    metric_name: &str,
    datapoint_id: &Uuid,
    output: &str,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT value, evaluator_inference_id
        FROM tensorzero.inference_evaluation_human_feedback
        WHERE metric_name = ",
    );
    qb.push_bind(metric_name.to_string());
    qb.push(" AND datapoint_id = ");
    qb.push_bind(*datapoint_id);
    qb.push(" AND output = ");
    qb.push_bind(output.to_string());
    qb.push(" LIMIT 1");

    qb
}

// =====================================================================
// Helper functions
// =====================================================================

async fn get_evaluation_statistics_raw(
    pool: &PgPool,
    function_name: &str,
    function_type: FunctionConfigType,
    metric_names: &[String],
    evaluation_run_ids: &[Uuid],
) -> Result<Vec<RawEvaluationStatisticsRow>, Error> {
    let mut qb = build_get_evaluation_statistics_query(
        function_name,
        function_type,
        metric_names,
        evaluation_run_ids,
    );
    let rows: Vec<RawEvaluationStatisticsRow> = qb.build_query_as().fetch_all(pool).await?;
    Ok(rows)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_helpers::assert_query_equals;

    #[test]
    fn test_build_list_evaluation_runs_query() {
        let qb = build_list_evaluation_runs_query(100, 0);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                run_id as evaluation_run_id,
                evaluation_name,
                function_name,
                COALESCE(variant_names->>0, '') as variant_name,
                dataset_name,
                created_at,
                encode(snapshot_hash, 'hex') as snapshot_hash
            FROM tensorzero.inference_evaluation_runs
            ORDER BY run_id DESC
            LIMIT $1 OFFSET $2
            ",
        );
    }

    #[test]
    fn test_build_search_evaluation_runs_query() {
        let qb = build_search_evaluation_runs_query(
            Some("test_eval"),
            Some("test_func"),
            "search_term",
            50,
            10,
        );
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                run_id AS evaluation_run_id,
                evaluation_name,
                dataset_name,
                COALESCE(variant_names->>0, '') AS variant_name
            FROM tensorzero.inference_evaluation_runs
            WHERE TRUE AND evaluation_name = $1 AND function_name = $2
            AND (run_id::TEXT ILIKE $3 OR evaluation_name ILIKE $4 OR dataset_name ILIKE $5 OR COALESCE(variant_names->>0, '') ILIKE $6)
            ORDER BY run_id DESC
            LIMIT $7 OFFSET $8
            ",
        );
    }

    #[test]
    fn test_build_search_evaluation_runs_query_no_function_name() {
        let qb = build_search_evaluation_runs_query(Some("test_eval"), None, "search_term", 50, 10);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                run_id AS evaluation_run_id,
                evaluation_name,
                dataset_name,
                COALESCE(variant_names->>0, '') AS variant_name
            FROM tensorzero.inference_evaluation_runs
            WHERE TRUE AND evaluation_name = $1
            AND (run_id::TEXT ILIKE $2 OR evaluation_name ILIKE $3 OR dataset_name ILIKE $4 OR COALESCE(variant_names->>0, '') ILIKE $5)
            ORDER BY run_id DESC
            LIMIT $6 OFFSET $7
            ",
        );
    }

    #[test]
    fn test_build_search_evaluation_runs_query_function_only_empty_search() {
        let qb = build_search_evaluation_runs_query(None, Some("test_func"), "", 50, 10);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                run_id AS evaluation_run_id,
                evaluation_name,
                dataset_name,
                COALESCE(variant_names->>0, '') AS variant_name
            FROM tensorzero.inference_evaluation_runs
            WHERE TRUE AND function_name = $1
            ORDER BY run_id DESC
            LIMIT $2 OFFSET $3
            ",
        );
    }

    #[test]
    fn test_build_get_evaluation_run_infos_query() {
        let run_ids = vec![Uuid::now_v7()];
        let qb = build_get_evaluation_run_infos_query(&run_ids, "test_func");
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                run_id as evaluation_run_id,
                COALESCE(variant_names->>0, '') as variant_name,
                created_at
            FROM tensorzero.inference_evaluation_runs
            WHERE run_id = ANY($1) AND function_name = $2 ORDER BY run_id DESC
            ",
        );
    }

    #[test]
    fn test_build_get_evaluation_run_infos_for_datapoint_query_chat() {
        let datapoint_id = Uuid::now_v7();
        let qb = build_get_evaluation_run_infos_for_datapoint_query(
            &datapoint_id,
            "test_func",
            FunctionConfigType::Chat,
        );
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT (tags->>'tensorzero::evaluation_run_id')::UUID as evaluation_run_id,
                (ARRAY_AGG(variant_name))[1] as variant_name,
                MAX(created_at) as created_at
            FROM tensorzero.chat_inferences
            WHERE tags->>'tensorzero::datapoint_id' = $1
            AND function_name = $2
            AND tags ? 'tensorzero::evaluation_run_id'
            GROUP BY tags->>'tensorzero::evaluation_run_id'
            ",
        );
    }

    #[test]
    fn test_build_get_inference_evaluation_human_feedback_query() {
        let datapoint_id = Uuid::now_v7();
        let qb = build_get_inference_evaluation_human_feedback_query(
            "test_metric",
            &datapoint_id,
            "test output",
        );
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT value, evaluator_inference_id
            FROM tensorzero.inference_evaluation_human_feedback
            WHERE metric_name = $1 AND datapoint_id = $2 AND output = $3 LIMIT 1
            ",
        );
    }
}
