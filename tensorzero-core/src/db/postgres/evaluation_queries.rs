//! Postgres queries for evaluation statistics.

use async_trait::async_trait;
use sqlx::postgres::PgRow;
use sqlx::{PgPool, QueryBuilder, Row};
use uuid::Uuid;

use crate::db::evaluation_queries::{
    EvaluationQueries, EvaluationResultRow, EvaluationRunInfoByIdRow, EvaluationRunInfoRow,
    EvaluationRunSearchResult, EvaluationStatisticsRow, InferenceEvaluationHumanFeedbackRow,
    RawEvaluationResultRow,
};
use crate::error::Error;
use crate::function::FunctionConfigType;
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
    async fn count_total_evaluation_runs(&self) -> Result<u64, Error> {
        let pool = self.get_pool_result()?;
        // Count distinct evaluation_run_ids per table and sum.
        // Evaluation runs target a single function type, so the sets are disjoint.
        let count: i64 = sqlx::query_scalar(
            "SELECT (
                COALESCE((
                    SELECT COUNT(DISTINCT tags->>'tensorzero::evaluation_run_id')
                    FROM tensorzero.chat_inferences
                    WHERE tags ? 'tensorzero::evaluation_run_id'
                    AND NOT function_name LIKE 'tensorzero::%'
                ), 0) +
                COALESCE((
                    SELECT COUNT(DISTINCT tags->>'tensorzero::evaluation_run_id')
                    FROM tensorzero.json_inferences
                    WHERE tags ? 'tensorzero::evaluation_run_id'
                    AND NOT function_name LIKE 'tensorzero::%'
                ), 0)
            )::BIGINT",
        )
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
        evaluation_name: &str,
        function_name: &str,
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
                    Error::new(crate::error::ErrorDetails::Serialization {
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
        WITH all_eval_inferences AS (
            SELECT
                tags->>'tensorzero::evaluation_run_id' as evaluation_run_id,
                tags->>'tensorzero::evaluation_name' as evaluation_name,
                tags->>'tensorzero::dataset_name' as dataset_name,
                function_name,
                variant_name,
                id,
                created_at
            FROM tensorzero.chat_inferences
            WHERE tags ? 'tensorzero::evaluation_run_id'
            AND NOT function_name LIKE 'tensorzero::%'
            UNION ALL
            SELECT
                tags->>'tensorzero::evaluation_run_id' as evaluation_run_id,
                tags->>'tensorzero::evaluation_name' as evaluation_name,
                tags->>'tensorzero::dataset_name' as dataset_name,
                function_name,
                variant_name,
                id,
                created_at
            FROM tensorzero.json_inferences
            WHERE tags ? 'tensorzero::evaluation_run_id'
            AND NOT function_name LIKE 'tensorzero::%'
        )
        SELECT
            evaluation_run_id::UUID as evaluation_run_id,
            (ARRAY_AGG(evaluation_name))[1] as evaluation_name,
            (ARRAY_AGG(function_name))[1] as function_name,
            (ARRAY_AGG(variant_name))[1] as variant_name,
            (ARRAY_AGG(dataset_name))[1] as dataset_name,
            MAX(created_at) as last_inference_timestamp
        FROM all_eval_inferences
        GROUP BY evaluation_run_id
        ORDER BY evaluation_run_id::UUID DESC
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
    evaluation_name: &str,
    function_name: &str,
    query: &str,
    limit: u32,
    offset: u32,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        WITH evaluation_inferences AS (
            SELECT DISTINCT
                tags->>'tensorzero::evaluation_run_id' as evaluation_run_id,
                variant_name
            FROM tensorzero.chat_inferences
            WHERE tags->>'tensorzero::evaluation_name' = ",
    );
    qb.push_bind(evaluation_name.to_string());
    qb.push(" AND function_name = ");
    qb.push_bind(function_name.to_string());
    qb.push(
        r"
            AND tags ? 'tensorzero::evaluation_run_id'
            UNION ALL
            SELECT DISTINCT
                tags->>'tensorzero::evaluation_run_id' as evaluation_run_id,
                variant_name
            FROM tensorzero.json_inferences
            WHERE tags->>'tensorzero::evaluation_name' = ",
    );
    qb.push_bind(evaluation_name.to_string());
    qb.push(" AND function_name = ");
    qb.push_bind(function_name.to_string());
    qb.push(
        r"
            AND tags ? 'tensorzero::evaluation_run_id'
        )
        SELECT DISTINCT evaluation_run_id::UUID as evaluation_run_id, variant_name
        FROM evaluation_inferences
        WHERE (evaluation_run_id ILIKE ",
    );
    qb.push_bind(format!("%{query}%"));
    qb.push(" OR variant_name ILIKE ");
    qb.push_bind(format!("%{query}%"));
    qb.push(
        r")
        ORDER BY evaluation_run_id::UUID DESC
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
    let run_id_strings: Vec<String> = evaluation_run_ids.iter().map(|id| id.to_string()).collect();

    let mut qb = QueryBuilder::new(
        r"
        WITH all_eval_inferences AS (
            SELECT
                tags->>'tensorzero::evaluation_run_id' as evaluation_run_id,
                variant_name,
                created_at
            FROM tensorzero.chat_inferences
            WHERE tags->>'tensorzero::evaluation_run_id' = ANY(",
    );
    qb.push_bind(run_id_strings.clone());
    qb.push(") AND function_name = ");
    qb.push_bind(function_name.to_string());
    qb.push(
        r"
            UNION ALL
            SELECT
                tags->>'tensorzero::evaluation_run_id' as evaluation_run_id,
                variant_name,
                created_at
            FROM tensorzero.json_inferences
            WHERE tags->>'tensorzero::evaluation_run_id' = ANY(",
    );
    qb.push_bind(run_id_strings);
    qb.push(") AND function_name = ");
    qb.push_bind(function_name.to_string());
    qb.push(
        r"
        )
        SELECT
            evaluation_run_id::UUID as evaluation_run_id,
            (ARRAY_AGG(variant_name))[1] as variant_name,
            MAX(created_at) as most_recent_inference_date
        FROM all_eval_inferences
        GROUP BY evaluation_run_id
        ORDER BY evaluation_run_id::UUID DESC
        ",
    );

    qb
}

fn build_get_evaluation_run_infos_for_datapoint_query(
    datapoint_id: &Uuid,
    function_name: &str,
    function_type: FunctionConfigType,
) -> QueryBuilder<sqlx::Postgres> {
    let inference_table = function_type.postgres_table_name();

    let mut qb = QueryBuilder::new(
        "SELECT (tags->>'tensorzero::evaluation_run_id')::UUID as evaluation_run_id, (ARRAY_AGG(variant_name))[1] as variant_name, MAX(created_at) as most_recent_inference_date FROM ",
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
            WITH all_eval_inferences AS (
                SELECT
                    tags->>'tensorzero::evaluation_run_id' as evaluation_run_id,
                    tags->>'tensorzero::evaluation_name' as evaluation_name,
                    tags->>'tensorzero::dataset_name' as dataset_name,
                    function_name,
                    variant_name,
                    id,
                    created_at
                FROM tensorzero.chat_inferences
                WHERE tags ? 'tensorzero::evaluation_run_id'
                AND NOT function_name LIKE 'tensorzero::%'
                UNION ALL
                SELECT
                    tags->>'tensorzero::evaluation_run_id' as evaluation_run_id,
                    tags->>'tensorzero::evaluation_name' as evaluation_name,
                    tags->>'tensorzero::dataset_name' as dataset_name,
                    function_name,
                    variant_name,
                    id,
                    created_at
                FROM tensorzero.json_inferences
                WHERE tags ? 'tensorzero::evaluation_run_id'
                AND NOT function_name LIKE 'tensorzero::%'
            )
            SELECT
                evaluation_run_id::UUID as evaluation_run_id,
                (ARRAY_AGG(evaluation_name))[1] as evaluation_name,
                (ARRAY_AGG(function_name))[1] as function_name,
                (ARRAY_AGG(variant_name))[1] as variant_name,
                (ARRAY_AGG(dataset_name))[1] as dataset_name,
                MAX(created_at) as last_inference_timestamp
            FROM all_eval_inferences
            GROUP BY evaluation_run_id
            ORDER BY evaluation_run_id::UUID DESC
            LIMIT $1 OFFSET $2
            ",
        );
    }

    #[test]
    fn test_build_search_evaluation_runs_query() {
        let qb =
            build_search_evaluation_runs_query("test_eval", "test_func", "search_term", 50, 10);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            WITH evaluation_inferences AS (
                SELECT DISTINCT
                    tags->>'tensorzero::evaluation_run_id' as evaluation_run_id,
                    variant_name
                FROM tensorzero.chat_inferences
                WHERE tags->>'tensorzero::evaluation_name' = $1 AND function_name = $2
                AND tags ? 'tensorzero::evaluation_run_id'
                UNION ALL
                SELECT DISTINCT
                    tags->>'tensorzero::evaluation_run_id' as evaluation_run_id,
                    variant_name
                FROM tensorzero.json_inferences
                WHERE tags->>'tensorzero::evaluation_name' = $3 AND function_name = $4
                AND tags ? 'tensorzero::evaluation_run_id'
            )
            SELECT DISTINCT evaluation_run_id::UUID as evaluation_run_id, variant_name
            FROM evaluation_inferences
            WHERE (evaluation_run_id ILIKE $5 OR variant_name ILIKE $6)
            ORDER BY evaluation_run_id::UUID DESC
            LIMIT $7 OFFSET $8
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
            WITH all_eval_inferences AS (
                SELECT
                    tags->>'tensorzero::evaluation_run_id' as evaluation_run_id,
                    variant_name,
                    created_at
                FROM tensorzero.chat_inferences
                WHERE tags->>'tensorzero::evaluation_run_id' = ANY($1) AND function_name = $2
                UNION ALL
                SELECT
                    tags->>'tensorzero::evaluation_run_id' as evaluation_run_id,
                    variant_name,
                    created_at
                FROM tensorzero.json_inferences
                WHERE tags->>'tensorzero::evaluation_run_id' = ANY($3) AND function_name = $4
            )
            SELECT
                evaluation_run_id::UUID as evaluation_run_id,
                (ARRAY_AGG(variant_name))[1] as variant_name,
                MAX(created_at) as most_recent_inference_date
            FROM all_eval_inferences
            GROUP BY evaluation_run_id
            ORDER BY evaluation_run_id::UUID DESC
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
                MAX(created_at) as most_recent_inference_date
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
