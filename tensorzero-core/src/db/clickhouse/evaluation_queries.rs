//! ClickHouse queries for evaluation statistics.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::Deserialize;

use super::ClickHouseConnectionInfo;
use super::escape_string_for_clickhouse_literal;
use super::select_queries::{parse_count, parse_json_rows};
use crate::db::evaluation_queries::EvaluationQueries;
use crate::db::evaluation_queries::EvaluationResultRow;
use crate::db::evaluation_queries::EvaluationRunInfoByIdRow;
use crate::db::evaluation_queries::EvaluationRunInfoRow;
use crate::db::evaluation_queries::EvaluationRunSearchResult;
use crate::db::evaluation_queries::EvaluationStatisticsRow;
use crate::db::evaluation_queries::InferenceEvaluationHumanFeedbackRow;
use crate::db::evaluation_queries::RawEvaluationResultRow;
use crate::error::Error;
use crate::function::FunctionConfigType;
use crate::statistics_util::{wald_confint, wilson_confint};

/// Raw statistics row from ClickHouse before CI computation.
/// This is an intermediate representation used to compute confidence intervals in Rust.
#[derive(Debug, Deserialize)]
struct RawEvaluationStatisticsRow {
    evaluation_run_id: uuid::Uuid,
    metric_name: String,
    metric_type: String, // "float" or "boolean"
    datapoint_count: u32,
    mean_metric: f64,
    stdev: Option<f64>, // Only present for float metrics
}

impl RawEvaluationStatisticsRow {
    /// Converts the raw row to the final `EvaluationStatisticsRow` by computing
    /// confidence intervals in Rust.
    fn into_evaluation_statistics_row(self) -> EvaluationStatisticsRow {
        let (ci_lower, ci_upper) = if self.metric_type == "float" {
            // Use Wald confidence interval for continuous (float) metrics
            if let Some(stdev) = self.stdev {
                wald_confint(self.mean_metric, stdev, self.datapoint_count)
                    .map(|(l, u)| (Some(l), Some(u)))
                    .unwrap_or((None, None))
            } else {
                (None, None)
            }
        } else {
            // Use Wilson confidence interval for Bernoulli (boolean) metrics
            wilson_confint(self.mean_metric, self.datapoint_count)
                .map(|(l, u)| (Some(l), Some(u)))
                .unwrap_or((None, None))
        };

        EvaluationStatisticsRow {
            evaluation_run_id: self.evaluation_run_id,
            metric_name: self.metric_name,
            datapoint_count: self.datapoint_count,
            mean_metric: self.mean_metric,
            ci_lower,
            ci_upper,
        }
    }
}

// Private helper for constructing the subquery for datapoint IDs
fn get_evaluation_result_datapoint_id_subquery(
    function_name: &str,
    evaluation_run_ids: &[uuid::Uuid],
    datapoint_id: Option<&uuid::Uuid>,
    limit: Option<u32>,
    offset: Option<u32>,
) -> (String, HashMap<String, String>) {
    let limit_clause = if let Some(limit) = limit {
        format!("LIMIT {limit}")
    } else {
        String::new()
    };
    let offset_clause = if let Some(offset) = offset {
        format!("OFFSET {offset}")
    } else {
        String::new()
    };

    let mut params = HashMap::new();
    params.insert("function_name".to_string(), function_name.to_string());

    let eval_run_ids_str: Vec<String> = evaluation_run_ids
        .iter()
        .map(|id| format!("'{id}'"))
        .collect();
    let eval_run_ids_joined = format!("[{}]", eval_run_ids_str.join(","));
    params.insert("evaluation_run_ids".to_string(), eval_run_ids_joined);

    // Add optional datapoint_id filter
    let datapoint_filter = if let Some(dp_id) = datapoint_id {
        params.insert("datapoint_id".to_string(), dp_id.to_string());
        "AND value = {datapoint_id:String}"
    } else {
        ""
    };

    let query = format!(
        "all_inference_ids AS (
            SELECT DISTINCT inference_id
            FROM TagInference WHERE key = 'tensorzero::evaluation_run_id'
            AND function_name = {{function_name:String}}
            AND value IN ({{evaluation_run_ids:Array(String)}})
        ),
        all_datapoint_ids AS (
            SELECT DISTINCT value as datapoint_id
            FROM TagInference
            WHERE key = 'tensorzero::datapoint_id'
            AND function_name = {{function_name:String}}
            AND inference_id IN (SELECT inference_id FROM all_inference_ids)
            {datapoint_filter}
            ORDER BY toUInt128(toUUID(datapoint_id)) DESC
            {limit_clause}
            {offset_clause}
        )"
    );
    (query, params)
}

#[async_trait]
impl EvaluationQueries for ClickHouseConnectionInfo {
    async fn count_total_evaluation_runs(&self) -> Result<u64, Error> {
        let query = "SELECT toUInt32(uniqExact(value)) as count
                     FROM TagInference
                     WHERE key = 'tensorzero::evaluation_run_id'
                     FORMAT JSONEachRow"
            .to_string();

        let response = self.run_query_synchronous(query, &HashMap::new()).await?;
        parse_count(&response.response)
    }

    async fn list_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EvaluationRunInfoRow>, Error> {
        let query = r"
            SELECT
                evaluation_run_id,
                any(evaluation_name) AS evaluation_name,
                any(inference_function_name) AS function_name,
                any(variant_name) AS variant_name,
                any(dataset_name) AS dataset_name,
                formatDateTime(UUIDv7ToDateTime(uint_to_uuid(max(max_inference_id))), '%Y-%m-%dT%H:%i:%SZ') AS last_inference_timestamp
            FROM (
                SELECT
                    maxIf(value, key = 'tensorzero::evaluation_run_id') AS evaluation_run_id,
                    maxIf(value, key = 'tensorzero::evaluation_name') AS evaluation_name,
                    maxIf(value, key = 'tensorzero::dataset_name') AS dataset_name,
                    any(function_name) AS inference_function_name,
                    any(variant_name) AS variant_name,
                    max(toUInt128(inference_id)) AS max_inference_id
                FROM TagInference
                WHERE key IN ('tensorzero::evaluation_run_id', 'tensorzero::evaluation_name', 'tensorzero::dataset_name')
                GROUP BY inference_id
            )
            WHERE NOT startsWith(inference_function_name, 'tensorzero::')
            GROUP BY evaluation_run_id
            ORDER BY toUInt128(toUUID(evaluation_run_id)) DESC
            LIMIT {limit:UInt32}
            OFFSET {offset:UInt32}
            FORMAT JSONEachRow
        "
        .to_string();

        let limit_str = limit.to_string();
        let offset_str = offset.to_string();
        let mut params = HashMap::new();
        params.insert("limit", limit_str.as_str());
        params.insert("offset", offset_str.as_str());

        let response = self.run_query_synchronous(query, &params).await?;

        parse_json_rows(response.response.as_str())
    }

    async fn count_datapoints_for_evaluation(
        &self,
        function_name: &str,
        evaluation_run_ids: &[uuid::Uuid],
    ) -> Result<u64, Error> {
        let (datapoint_id_subquery, params_owned) = get_evaluation_result_datapoint_id_subquery(
            function_name,
            evaluation_run_ids,
            None, // datapoint_id filter
            /* limit= */ None,
            /* offset= */ None,
        );

        let query = format!(
            r"WITH {datapoint_id_subquery}
            SELECT toUInt32(count()) as count
            FROM all_datapoint_ids
            FORMAT JSONEachRow
        "
        );
        let params: HashMap<_, _> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_count(&response.response)
    }

    async fn search_evaluation_runs(
        &self,
        evaluation_name: &str,
        function_name: &str,
        query: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EvaluationRunSearchResult>, Error> {
        let sql_query = r"
            WITH
                evaluation_inference_ids AS (
                    SELECT inference_id
                    FROM TagInference
                    WHERE key = 'tensorzero::evaluation_name'
                    AND value = {evaluation_name:String}
                )
            SELECT DISTINCT value as evaluation_run_id, variant_name
            FROM TagInference
            WHERE key = 'tensorzero::evaluation_run_id'
                AND function_name = {function_name:String}
                AND inference_id IN (SELECT inference_id FROM evaluation_inference_ids)
                AND (positionCaseInsensitive(value, {query:String}) > 0 OR positionCaseInsensitive(variant_name, {query:String}) > 0)
            ORDER BY toUInt128(toUUID(evaluation_run_id)) DESC
            LIMIT {limit:UInt32}
            OFFSET {offset:UInt32}
            FORMAT JSONEachRow
        "
        .to_string();

        let evaluation_name_str = evaluation_name.to_string();
        let function_name_str = function_name.to_string();
        let query_str = query.to_string();
        let limit_str = limit.to_string();
        let offset_str = offset.to_string();

        let mut params = HashMap::new();
        params.insert("evaluation_name", evaluation_name_str.as_str());
        params.insert("function_name", function_name_str.as_str());
        params.insert("query", query_str.as_str());
        params.insert("limit", limit_str.as_str());
        params.insert("offset", offset_str.as_str());

        let response = self.run_query_synchronous(sql_query, &params).await?;
        parse_json_rows(response.response.as_str())
    }

    async fn get_evaluation_run_infos(
        &self,
        evaluation_run_ids: &[uuid::Uuid],
        function_name: &str,
    ) -> Result<Vec<EvaluationRunInfoByIdRow>, Error> {
        // Format evaluation_run_ids as array for ClickHouse
        let eval_run_ids_str: Vec<String> = evaluation_run_ids
            .iter()
            .map(|id| format!("'{id}'"))
            .collect();
        let eval_run_ids_joined = format!("[{}]", eval_run_ids_str.join(","));

        let sql_query = r"
            SELECT
                any(run_tag.value) as evaluation_run_id,
                any(run_tag.variant_name) as variant_name,
                formatDateTime(
                    max(UUIDv7ToDateTime(inference_id)),
                    '%Y-%m-%dT%H:%i:%SZ'
                ) as most_recent_inference_date
            FROM
                TagInference AS run_tag
            WHERE
                run_tag.key = 'tensorzero::evaluation_run_id'
                AND run_tag.value IN ({evaluation_run_ids:Array(String)})
                AND run_tag.function_name = {function_name:String}
            GROUP BY
                run_tag.value
            ORDER BY
                toUInt128(toUUID(evaluation_run_id)) DESC
            FORMAT JSONEachRow
        "
        .to_string();

        let function_name_str = function_name.to_string();

        let mut params = HashMap::new();
        params.insert("evaluation_run_ids", eval_run_ids_joined.as_str());
        params.insert("function_name", function_name_str.as_str());

        let response = self.run_query_synchronous(sql_query, &params).await?;
        parse_json_rows(response.response.as_str())
    }

    async fn get_evaluation_run_infos_for_datapoint(
        &self,
        datapoint_id: &uuid::Uuid,
        function_name: &str,
        function_type: FunctionConfigType,
    ) -> Result<Vec<EvaluationRunInfoByIdRow>, Error> {
        let inference_table_name = function_type.table_name();

        let sql_query = format!(
            r"
            WITH datapoint_inference_ids AS (
                SELECT inference_id
                FROM TagInference
                WHERE key = 'tensorzero::datapoint_id'
                AND value = {{datapoint_id:String}}
            )
            SELECT
                any(tags['tensorzero::evaluation_run_id']) as evaluation_run_id,
                any(variant_name) as variant_name,
                formatDateTime(
                    max(UUIDv7ToDateTime(id)),
                    '%Y-%m-%dT%H:%i:%SZ'
                ) as most_recent_inference_date
            FROM {inference_table_name}
            WHERE id IN (SELECT inference_id FROM datapoint_inference_ids)
            AND function_name = {{function_name:String}}
            GROUP BY
                tags['tensorzero::evaluation_run_id']
            FORMAT JSONEachRow
        "
        );

        let datapoint_id_str = datapoint_id.to_string();
        let function_name_str = function_name.to_string();

        let mut params = HashMap::new();
        params.insert("datapoint_id", datapoint_id_str.as_str());
        params.insert("function_name", function_name_str.as_str());

        let response = self.run_query_synchronous(sql_query, &params).await?;
        parse_json_rows(response.response.as_str())
    }

    async fn get_evaluation_statistics(
        &self,
        function_name: &str,
        function_type: FunctionConfigType,
        metric_names: &[String],
        evaluation_run_ids: &[uuid::Uuid],
    ) -> Result<Vec<EvaluationStatisticsRow>, Error> {
        let inference_table_name = function_type.table_name();

        // Build the datapoint ID subquery
        let (datapoint_id_subquery, params_owned) = get_evaluation_result_datapoint_id_subquery(
            function_name,
            evaluation_run_ids,
            None, // datapoint_id filter
            /* limit= */ None,
            /* offset= */ None,
        );

        // Build metric names array for ClickHouse
        let metric_names_str: Vec<String> = metric_names.iter().map(|s| format!("'{s}'")).collect();
        let metric_names_joined = format!("[{}]", metric_names_str.join(","));

        // Query returns raw statistics (mean, count, stdev) without CI computation.
        // CI is computed in Rust using wald_confint (float) or wilson_confint (boolean).
        let sql_query = format!(
            r"WITH {datapoint_id_subquery},
            filtered_inference AS (
                SELECT
                    id,
                    tags['tensorzero::evaluation_run_id'] AS evaluation_run_id
                FROM {inference_table_name}
                WHERE id IN (SELECT inference_id FROM all_inference_ids)
                AND function_name = {{function_name:String}}
            ),
            float_feedback AS (
                SELECT metric_name,
                       argMax(value, timestamp) as value,
                       target_id
                FROM FloatMetricFeedback
                WHERE metric_name IN ({{metric_names:Array(String)}})
                AND target_id IN (SELECT inference_id FROM all_inference_ids)
                GROUP BY target_id, metric_name
            ),
            boolean_feedback AS (
                SELECT metric_name,
                       argMax(value, timestamp) as value,
                       target_id
                FROM BooleanMetricFeedback
                WHERE metric_name IN ({{metric_names:Array(String)}})
                AND target_id IN (SELECT inference_id FROM all_inference_ids)
                GROUP BY target_id, metric_name
            ),
            float_stats AS (
                SELECT
                    filtered_inference.evaluation_run_id,
                    float_feedback.metric_name AS metric_name,
                    'float' AS metric_type,
                    toUInt32(count()) AS datapoint_count,
                    avg(toFloat64(float_feedback.value)) AS mean_metric,
                    stddevSamp(toFloat64(float_feedback.value)) AS stdev
                FROM filtered_inference
                INNER JOIN float_feedback
                    ON float_feedback.target_id = filtered_inference.id
                    AND float_feedback.value IS NOT NULL
                GROUP BY
                    filtered_inference.evaluation_run_id,
                    float_feedback.metric_name
            ),
            boolean_stats AS (
                SELECT
                    filtered_inference.evaluation_run_id,
                    boolean_feedback.metric_name AS metric_name,
                    'boolean' AS metric_type,
                    toUInt32(count()) AS datapoint_count,
                    avg(toFloat64(boolean_feedback.value)) AS mean_metric,
                    NULL AS stdev
                FROM filtered_inference
                INNER JOIN boolean_feedback
                    ON boolean_feedback.target_id = filtered_inference.id
                    AND boolean_feedback.value IS NOT NULL
                GROUP BY
                    filtered_inference.evaluation_run_id,
                    boolean_feedback.metric_name
            )
            SELECT * FROM float_stats
            UNION ALL
            SELECT * FROM boolean_stats
            ORDER BY
                toUInt128(toUUID(evaluation_run_id)) DESC,
                metric_name ASC
            FORMAT JSONEachRow
            "
        );

        let function_name_str = function_name.to_string();
        let mut params: HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        params.insert("function_name", function_name_str.as_str());
        params.insert("metric_names", metric_names_joined.as_str());

        let response = self.run_query_synchronous(sql_query, &params).await?;
        let raw_rows: Vec<RawEvaluationStatisticsRow> =
            parse_json_rows(response.response.as_str())?;

        // Compute confidence intervals in Rust
        Ok(raw_rows
            .into_iter()
            .map(|row| row.into_evaluation_statistics_row())
            .collect())
    }

    async fn get_evaluation_results(
        &self,
        function_name: &str,
        evaluation_run_ids: &[uuid::Uuid],
        function_type: FunctionConfigType,
        metric_names: &[String],
        datapoint_id: Option<&uuid::Uuid>,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EvaluationResultRow>, Error> {
        let inference_table_name = function_type.table_name();
        let datapoint_table_name = function_type.datapoint_table_name();

        // Build the datapoint ID subquery with pagination
        let (datapoint_id_subquery, params_owned) = get_evaluation_result_datapoint_id_subquery(
            function_name,
            evaluation_run_ids,
            datapoint_id,
            Some(limit),
            Some(offset),
        );

        // Format metric_names as array for ClickHouse
        let metric_names_str: Vec<String> = metric_names.iter().map(|m| format!("'{m}'")).collect();
        let metric_names_joined = format!("[{}]", metric_names_str.join(","));

        // The query uses CTEs to:
        // 1. Find all datapoints in the evaluation runs (with pagination)
        // 2. Filter inferences to those datapoints
        // 3. Get feedback (both boolean and float metrics) for those inferences
        // 4. Get the datapoint reference outputs
        // 5. Join everything together
        let sql_query = format!(
            r"
            WITH {datapoint_id_subquery},
            filtered_dp AS (
                SELECT * FROM {datapoint_table_name}
                WHERE function_name = {{function_name:String}}
                AND id IN (SELECT datapoint_id FROM all_datapoint_ids)
            ),
            filtered_inference AS (
                SELECT * FROM {inference_table_name}
                WHERE id IN (SELECT inference_id FROM all_inference_ids)
                AND function_name = {{function_name:String}}
            ),
            filtered_feedback AS (
                SELECT metric_name,
                    argMax(toString(value), timestamp) as value,
                    argMax(tags['tensorzero::evaluator_inference_id'], timestamp) as evaluator_inference_id,
                    argMax(id, timestamp) as feedback_id,
                    argMax(tags['tensorzero::human_feedback'], timestamp) == 'true' as is_human_feedback,
                    target_id
                FROM BooleanMetricFeedback
                WHERE metric_name IN ({{metric_names:Array(String)}})
                AND target_id IN (SELECT inference_id FROM all_inference_ids)
                GROUP BY target_id, metric_name
                UNION ALL
                SELECT metric_name,
                    argMax(toString(value), timestamp) as value,
                    argMax(tags['tensorzero::evaluator_inference_id'], timestamp) as evaluator_inference_id,
                    argMax(id, timestamp) as feedback_id,
                    argMax(tags['tensorzero::human_feedback'], timestamp) == 'true' as is_human_feedback,
                    target_id
                FROM FloatMetricFeedback
                WHERE metric_name IN ({{metric_names:Array(String)}})
                AND target_id IN (SELECT inference_id FROM all_inference_ids)
                GROUP BY target_id, metric_name
            )
            SELECT
                filtered_dp.input as input,
                filtered_dp.id as datapoint_id,
                filtered_dp.name as name,
                filtered_dp.output as reference_output,
                filtered_inference.output as generated_output,
                toUUID(filtered_inference.tags['tensorzero::evaluation_run_id']) as evaluation_run_id,
                filtered_inference.tags['tensorzero::dataset_name'] as dataset_name,
                if(length(filtered_feedback.evaluator_inference_id) > 0, filtered_feedback.evaluator_inference_id, null) as evaluator_inference_id,
                filtered_inference.id as inference_id,
                filtered_inference.episode_id as episode_id,
                filtered_feedback.metric_name as metric_name,
                filtered_feedback.value as metric_value,
                filtered_feedback.feedback_id as feedback_id,
                toBool(filtered_feedback.is_human_feedback) as is_human_feedback,
                filtered_dp.staled_at as staled_at,
                filtered_inference.variant_name as variant_name
            FROM filtered_dp
            INNER JOIN filtered_inference
                ON toUUIDOrNull(filtered_inference.tags['tensorzero::datapoint_id']) = filtered_dp.id
            LEFT JOIN filtered_feedback
                ON filtered_feedback.target_id = filtered_inference.id
            ORDER BY toUInt128(datapoint_id) DESC, metric_name DESC
            FORMAT JSONEachRow
            "
        );

        let mut params: HashMap<_, _> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        params.insert("metric_names", metric_names_joined.as_str());

        let response = self.run_query_synchronous(sql_query, &params).await?;
        let raw_rows: Vec<RawEvaluationResultRow> = parse_json_rows(response.response.as_str())?;

        // Convert raw rows to typed enum variants based on function type
        raw_rows
            .into_iter()
            .map(|row| match function_type {
                FunctionConfigType::Chat => row.into_chat().map(EvaluationResultRow::Chat),
                FunctionConfigType::Json => row.into_json().map(EvaluationResultRow::Json),
            })
            .collect()
    }

    async fn get_inference_evaluation_human_feedback(
        &self,
        metric_name: &str,
        datapoint_id: &uuid::Uuid,
        output: &str,
    ) -> Result<Option<InferenceEvaluationHumanFeedbackRow>, Error> {
        let sql_query = r"
            SELECT value, evaluator_inference_id
            FROM StaticEvaluationHumanFeedback
            WHERE metric_name = {metric_name:String}
            AND datapoint_id = {datapoint_id:UUID}
            AND output = {output:String}
            LIMIT 1
            FORMAT JSONEachRow
        "
        .to_string();

        let metric_name_str = metric_name.to_string();
        let datapoint_id_str = datapoint_id.to_string();
        let escaped_output = escape_string_for_clickhouse_literal(output);

        let mut params = HashMap::new();
        params.insert("metric_name", metric_name_str.as_str());
        params.insert("datapoint_id", datapoint_id_str.as_str());
        params.insert("output", escaped_output.as_str());

        let response = self.run_query_synchronous(sql_query, &params).await?;
        let rows: Vec<InferenceEvaluationHumanFeedbackRow> =
            parse_json_rows(response.response.as_str())?;
        Ok(rows.into_iter().next())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::db::{
        clickhouse::{
            ClickHouseConnectionInfo, ClickHouseResponse, ClickHouseResponseMetadata,
            clickhouse_client::MockClickHouseClient,
            query_builder::test_util::assert_query_contains,
        },
        evaluation_queries::{EvaluationQueries, EvaluationResultRow},
    };
    use crate::function::FunctionConfigType;

    use uuid::Uuid;

    #[tokio::test]
    async fn test_count_total_evaluation_runs() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(
                    query,
                    "SELECT toUInt32(uniqExact(value)) as count
                     FROM TagInference
                     WHERE key = 'tensorzero::evaluation_run_id'
                     FORMAT JSONEachRow",
                );
                assert_eq!(params.len(), 0, "Should have no parameters");
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":42}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn.count_total_evaluation_runs().await.unwrap();

        assert_eq!(result, 42, "Should return count of 42");
    }

    #[tokio::test]
    async fn test_list_evaluation_runs_with_defaults() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                // Verify the query contains the expected structure
                assert_query_contains(query, "SELECT");
                assert_query_contains(query, "evaluation_run_id");
                assert_query_contains(query, "FROM TagInference");
                assert_query_contains(query, "LIMIT {limit:UInt32}");
                assert_query_contains(query, "OFFSET {offset:UInt32}");

                // Verify parameters
                assert_eq!(params.get("limit"), Some(&"100"));
                assert_eq!(params.get("offset"), Some(&"0"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","evaluation_name":"test_eval","function_name":"test_func","variant_name":"test_variant","dataset_name":"test_dataset","last_inference_timestamp":"2025-05-20T16:52:58Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn.list_evaluation_runs(100, 0).await.unwrap();

        assert_eq!(result.len(), 1, "Should return one evaluation run");
        assert_eq!(result[0].evaluation_name, "test_eval");
        assert_eq!(result[0].function_name, "test_func");
        assert_eq!(result[0].variant_name, "test_variant");
        assert_eq!(result[0].dataset_name, "test_dataset");
    }

    #[tokio::test]
    async fn test_list_evaluation_runs_with_custom_pagination() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|_query, params| {
                // Verify custom pagination parameters
                assert_eq!(params.get("limit"), Some(&"50"));
                assert_eq!(params.get("offset"), Some(&"100"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn.list_evaluation_runs(50, 100).await.unwrap();

        assert_eq!(result.len(), 0, "Should return empty results");
    }

    #[tokio::test]
    async fn test_list_evaluation_runs_multiple_results() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","evaluation_name":"eval1","function_name":"func1","variant_name":"variant1","dataset_name":"dataset1","last_inference_timestamp":"2025-05-20T16:52:58Z"}
{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","evaluation_name":"eval2","function_name":"func2","variant_name":"variant2","dataset_name":"dataset2","last_inference_timestamp":"2025-05-20T17:52:58Z"}
{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95f","evaluation_name":"eval3","function_name":"func3","variant_name":"variant3","dataset_name":"dataset3","last_inference_timestamp":"2025-05-20T18:52:58Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 3,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn.list_evaluation_runs(100, 0).await.unwrap();

        assert_eq!(result.len(), 3, "Should return three evaluation runs");
        assert_eq!(result[0].evaluation_name, "eval1");
        assert_eq!(result[1].evaluation_name, "eval2");
        assert_eq!(result[2].evaluation_name, "eval3");
    }

    #[tokio::test]
    async fn test_list_evaluation_runs_filters_out_tensorzero_functions() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _params| {
                // Verify the query filters out tensorzero:: functions
                assert_query_contains(
                    query,
                    "NOT startsWith(inference_function_name, 'tensorzero::')",
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let _result = conn.list_evaluation_runs(100, 0).await.unwrap();
    }

    #[tokio::test]
    async fn test_count_datapoints_for_evaluation() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(
                    query,
                    "
                WITH all_inference_ids AS (
                    SELECT DISTINCT inference_id
                    FROM TagInference WHERE key = 'tensorzero::evaluation_run_id'
                    AND function_name = {function_name:String}
                    AND value IN ({evaluation_run_ids:Array(String)})
                ),
                all_datapoint_ids AS (
                    SELECT DISTINCT value as datapoint_id
                    FROM TagInference
                    WHERE key = 'tensorzero::datapoint_id'
                    AND function_name = {function_name:String}
                    AND inference_id IN (SELECT inference_id FROM all_inference_ids)
                    ORDER BY toUInt128(toUUID(datapoint_id)) DESC
                )
                SELECT toUInt32(count()) as count
                FROM all_datapoint_ids
                FORMAT JSONEachRow",
                );
                assert_eq!(params.get("function_name"), Some(&"test_function"));
                assert_eq!(
                    params.get("evaluation_run_ids"),
                    Some(&"['01234567-89ab-cdef-0123-456789abcdef']")
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":42}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .count_datapoints_for_evaluation(
                "test_function",
                &[Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap()],
            )
            .await
            .unwrap();

        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_count_datapoints_multiple_run_ids() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|_query, params| {
                // Query doesn't change from single run ID, but run ID param changes.
                assert_eq!(
                    params.get("evaluation_run_ids"),
                    Some(&"['01234567-89ab-cdef-0123-456789abcdef','11234567-89ab-cdef-0123-456789abcdef']")
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":100}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let run_ids = vec![
            Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap(),
            Uuid::parse_str("11234567-89ab-cdef-0123-456789abcdef").unwrap(),
        ];

        let result = conn
            .count_datapoints_for_evaluation("test_function", &run_ids)
            .await
            .unwrap();

        assert_eq!(result, 100);
    }

    #[tokio::test]
    async fn test_count_datapoints_zero_results() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":0}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .count_datapoints_for_evaluation(
                "nonexistent_function",
                &[Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap()],
            )
            .await
            .unwrap();

        assert_eq!(result, 0);
    }

    #[tokio::test]
    async fn test_search_evaluation_runs() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "WITH
                    evaluation_inference_ids AS (
                        SELECT inference_id
                        FROM TagInference
                        WHERE key = 'tensorzero::evaluation_name'
                        AND value = {evaluation_name:String}
                    )
                SELECT DISTINCT value as evaluation_run_id, variant_name
                FROM TagInference
                WHERE key = 'tensorzero::evaluation_run_id'
                    AND function_name = {function_name:String}
                    AND inference_id IN (SELECT inference_id FROM evaluation_inference_ids)
                    AND (positionCaseInsensitive(value, {query:String}) > 0 OR positionCaseInsensitive(variant_name, {query:String}) > 0)
                ORDER BY toUInt128(toUUID(evaluation_run_id)) DESC
                LIMIT {limit:UInt32}
                OFFSET {offset:UInt32}
                FORMAT JSONEachRow");
                assert_eq!(params.get("evaluation_name"), Some(&"test_eval"));
                assert_eq!(params.get("function_name"), Some(&"test_func"));
                assert_eq!(params.get("query"), Some(&"variant"));
                assert_eq!(params.get("limit"), Some(&"100"));
                assert_eq!(params.get("offset"), Some(&"0"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","variant_name":"variant1"}
{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","variant_name":"variant2"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .search_evaluation_runs("test_eval", "test_func", "variant", 100, 0)
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].variant_name, "variant1");
        assert_eq!(result[1].variant_name, "variant2");
    }

    #[tokio::test]
    async fn test_search_evaluation_runs_empty() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .search_evaluation_runs("test_eval", "test_func", "nonexistent", 100, 0)
            .await
            .unwrap();

        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_run_infos() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "SELECT
                    any(run_tag.value) as evaluation_run_id,
                    any(run_tag.variant_name) as variant_name,
                    formatDateTime(
                        max(UUIDv7ToDateTime(inference_id)),
                        '%Y-%m-%dT%H:%i:%SZ'
                    ) as most_recent_inference_date
                FROM
                    TagInference AS run_tag
                WHERE
                    run_tag.key = 'tensorzero::evaluation_run_id'
                    AND run_tag.value IN ({evaluation_run_ids:Array(String)})
                    AND run_tag.function_name = {function_name:String}
                GROUP BY
                    run_tag.value
                ORDER BY
                    toUInt128(toUUID(evaluation_run_id)) DESC
                FORMAT JSONEachRow");
                assert_eq!(params.get("function_name"), Some(&"test_func"));
                assert_eq!(
                    params.get("evaluation_run_ids"),
                    Some(&"['0196ee9c-d808-74f3-8000-02ec7409b95d']")
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","variant_name":"test_variant","most_recent_inference_date":"2025-05-20T16:52:58Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_run_infos(
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap()],
                "test_func",
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].variant_name, "test_variant");
        assert_eq!(
            result[0].evaluation_run_id,
            Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap()
        );
    }

    #[tokio::test]
    async fn test_get_evaluation_run_infos_multiple() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|_query, params| {
                assert_eq!(
                    params.get("evaluation_run_ids"),
                    Some(&"['0196ee9c-d808-74f3-8000-02ec7409b95d','0196ee9c-d808-74f3-8000-02ec7409b95e']")
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","variant_name":"variant1","most_recent_inference_date":"2025-05-20T16:52:58Z"}
{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","variant_name":"variant2","most_recent_inference_date":"2025-05-20T17:52:58Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_run_infos(
                &[
                    Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap(),
                    Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap(),
                ],
                "test_func",
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].variant_name, "variant1");
        assert_eq!(result[1].variant_name, "variant2");
    }

    #[tokio::test]
    async fn test_get_evaluation_run_infos_empty() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_run_infos(
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap()],
                "nonexistent_func",
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_run_infos_for_datapoint_chat() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(
                    query,
                    "WITH datapoint_inference_ids AS (
                        SELECT inference_id
                        FROM TagInference
                        WHERE key = 'tensorzero::datapoint_id'
                        AND value = {datapoint_id:String}
                    )
                    SELECT
                        any(tags['tensorzero::evaluation_run_id']) as evaluation_run_id,
                        any(variant_name) as variant_name,
                        formatDateTime(
                            max(UUIDv7ToDateTime(id)),
                            '%Y-%m-%dT%H:%i:%SZ'
                        ) as most_recent_inference_date
                    FROM ChatInference
                    WHERE id IN (SELECT inference_id FROM datapoint_inference_ids)
                    AND function_name = {function_name:String}
                    GROUP BY
                        tags['tensorzero::evaluation_run_id']
                    FORMAT JSONEachRow",
                );
                assert_eq!(
                    params.get("datapoint_id"),
                    Some(&"0196ee9c-d808-74f3-8000-02ec7409b95d")
                );
                assert_eq!(params.get("function_name"), Some(&"test_func"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","variant_name":"test_variant","most_recent_inference_date":"2025-05-20T16:52:58Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_run_infos_for_datapoint(
                &Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap(),
                "test_func",
                FunctionConfigType::Chat,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].variant_name, "test_variant");
        assert_eq!(
            result[0].evaluation_run_id,
            Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap()
        );
    }

    #[tokio::test]
    async fn test_get_evaluation_run_infos_for_datapoint_json() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(
                    query,
                    "WITH datapoint_inference_ids AS (
                        SELECT inference_id
                        FROM TagInference
                        WHERE key = 'tensorzero::datapoint_id'
                        AND value = {datapoint_id:String}
                    )
                    SELECT
                        any(tags['tensorzero::evaluation_run_id']) as evaluation_run_id,
                        any(variant_name) as variant_name,
                        formatDateTime(
                            max(UUIDv7ToDateTime(id)),
                            '%Y-%m-%dT%H:%i:%SZ'
                        ) as most_recent_inference_date
                    FROM JsonInference
                    WHERE id IN (SELECT inference_id FROM datapoint_inference_ids)
                    AND function_name = {function_name:String}
                    GROUP BY
                        tags['tensorzero::evaluation_run_id']
                    FORMAT JSONEachRow",
                );
                assert_eq!(
                    params.get("datapoint_id"),
                    Some(&"0196ee9c-d808-74f3-8000-02ec7409b95d")
                );
                assert_eq!(params.get("function_name"), Some(&"test_func"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","variant_name":"test_variant","most_recent_inference_date":"2025-05-20T16:52:58Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_run_infos_for_datapoint(
                &Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap(),
                "test_func",
                FunctionConfigType::Json,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].variant_name, "test_variant");
        assert_eq!(
            result[0].evaluation_run_id,
            Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap()
        );
    }

    #[tokio::test]
    async fn test_get_evaluation_run_infos_for_datapoint_multiple() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","variant_name":"variant1","most_recent_inference_date":"2025-05-20T16:52:58Z"}
{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","variant_name":"variant2","most_recent_inference_date":"2025-05-20T17:52:58Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_run_infos_for_datapoint(
                &Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap(),
                "test_func",
                FunctionConfigType::Chat,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].variant_name, "variant1");
        assert_eq!(result[1].variant_name, "variant2");
    }

    #[tokio::test]
    async fn test_get_evaluation_run_infos_for_datapoint_empty() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_run_infos_for_datapoint(
                &Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap(),
                "nonexistent_func",
                FunctionConfigType::Json,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 0);
    }

    // ============================================================================
    // get_evaluation_statistics tests
    // ============================================================================

    #[tokio::test]
    async fn test_get_evaluation_statistics_chat_function() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "FROM ChatInference");
                assert_query_contains(query, "float_stats");
                assert_query_contains(query, "boolean_stats");
                assert_query_contains(query, "FloatMetricFeedback");
                assert_query_contains(query, "BooleanMetricFeedback");
                assert_query_contains(query, "'float' AS metric_type");
                assert_query_contains(query, "'boolean' AS metric_type");
                assert_query_contains(query, "stddevSamp");
                assert_eq!(params.get("function_name"), Some(&"test_func"));
                assert_eq!(
                    params.get("metric_names"),
                    Some(&"['metric1','metric2']")
                );
                true
            })
            .returning(|_, _| {
                // Return raw stats with metric_type and stdev; CI is computed in Rust
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","metric_name":"metric1","metric_type":"float","datapoint_count":100,"mean_metric":0.75,"stdev":0.1}
{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","metric_name":"metric2","metric_type":"boolean","datapoint_count":100,"mean_metric":0.8,"stdev":null}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_statistics(
                "test_func",
                FunctionConfigType::Chat,
                &["metric1".to_string(), "metric2".to_string()],
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap()],
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        // Float metric uses Wald CI
        assert_eq!(result[0].metric_name, "metric1");
        assert_eq!(result[0].datapoint_count, 100);
        assert!((result[0].mean_metric - 0.75).abs() < 0.001);
        // Wald CI: 0.75 ± 1.96 * (0.1 / sqrt(100)) = 0.75 ± 0.0196
        assert!((result[0].ci_lower.unwrap() - 0.7304).abs() < 0.001);
        assert!((result[0].ci_upper.unwrap() - 0.7696).abs() < 0.001);

        // Boolean metric uses Wilson CI
        assert_eq!(result[1].metric_name, "metric2");
        assert_eq!(result[1].datapoint_count, 100);
        assert!((result[1].mean_metric - 0.8).abs() < 0.001);
        // Wilson CI for p=0.8, n=100
        assert!(result[1].ci_lower.is_some());
        assert!(result[1].ci_upper.is_some());
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_json_function() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _params| {
                assert_query_contains(query, "FROM JsonInference");
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","metric_name":"accuracy","metric_type":"boolean","datapoint_count":5,"mean_metric":0.9,"stdev":null}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_statistics(
                "test_func",
                FunctionConfigType::Json,
                &["accuracy".to_string()],
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap()],
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].metric_name, "accuracy");
        assert_eq!(result[0].datapoint_count, 5);
        assert!((result[0].mean_metric - 0.9).abs() < 0.001);
        // Wilson CI is computed in Rust
        assert!(result[0].ci_lower.is_some());
        assert!(result[0].ci_upper.is_some());
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_multiple_runs() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|_query, params| {
                assert_eq!(
                    params.get("evaluation_run_ids"),
                    Some(&"['0196ee9c-d808-74f3-8000-02ec7409b95d','0196ee9c-d808-74f3-8000-02ec7409b95e']")
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","metric_name":"metric1","metric_type":"float","datapoint_count":10,"mean_metric":0.75,"stdev":0.1}
{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","metric_name":"metric1","metric_type":"float","datapoint_count":15,"mean_metric":0.80,"stdev":0.12}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_statistics(
                "test_func",
                FunctionConfigType::Chat,
                &["metric1".to_string()],
                &[
                    Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap(),
                    Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap(),
                ],
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        // Both should have CI computed
        assert!(result[0].ci_lower.is_some());
        assert!(result[0].ci_upper.is_some());
        assert!(result[1].ci_lower.is_some());
        assert!(result[1].ci_upper.is_some());
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_empty_results() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_statistics(
                "nonexistent_func",
                FunctionConfigType::Chat,
                &["metric1".to_string()],
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap()],
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_single_datapoint_float() {
        // For float metrics with single datapoint, stdev is null so CI cannot be computed
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","metric_name":"metric1","metric_type":"float","datapoint_count":1,"mean_metric":1.0,"stdev":null}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_statistics(
                "test_func",
                FunctionConfigType::Chat,
                &["metric1".to_string()],
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap()],
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        // Float metric with null stdev cannot compute Wald CI
        assert!(result[0].ci_lower.is_none());
        assert!(result[0].ci_upper.is_none());
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_single_datapoint_boolean() {
        // For boolean metrics with single datapoint, Wilson CI can still be computed
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","metric_name":"metric1","metric_type":"boolean","datapoint_count":1,"mean_metric":1.0,"stdev":null}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_statistics(
                "test_func",
                FunctionConfigType::Chat,
                &["metric1".to_string()],
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap()],
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        // Boolean metric can still compute Wilson CI even with single datapoint
        assert!(result[0].ci_lower.is_some());
        assert!(result[0].ci_upper.is_some());
        // For p=1.0, n=1, Wilson lower is approximately 0.206
        assert!((result[0].ci_lower.unwrap() - 0.206543).abs() < 0.001);
        assert!((result[0].ci_upper.unwrap() - 1.0).abs() < 0.001);
    }

    // ============================================================================
    // get_evaluation_results tests
    // ============================================================================

    #[tokio::test]
    async fn test_get_evaluation_results() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                // Verify the query structure
                assert_query_contains(query, "WITH all_inference_ids AS");
                assert_query_contains(query, "all_datapoint_ids AS");
                assert_query_contains(query, "filtered_dp AS");
                assert_query_contains(query, "filtered_inference AS");
                assert_query_contains(query, "filtered_feedback AS");
                assert_query_contains(query, "FROM ChatInference");
                assert_query_contains(query, "FROM ChatInferenceDatapoint");
                assert_query_contains(query, "LEFT JOIN filtered_feedback");
                assert_query_contains(query, "LIMIT 10");
                assert_query_contains(query, "OFFSET 0");
                assert_query_contains(query, "ORDER BY toUInt128(datapoint_id) DESC");

                // Verify parameters
                assert_eq!(params.get("function_name"), Some(&"test_func"));
                assert_eq!(
                    params.get("evaluation_run_ids"),
                    Some(&"['0196ee9c-d808-74f3-8000-02ec7409b95e']")
                );
                assert_eq!(
                    params.get("metric_names"),
                    Some(&"['tensorzero::evaluation_name::test::evaluator_name::exact_match']")
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"inference_id":"0196ee9c-d808-74f3-8000-02ec7409b95f","episode_id":"0196ee9c-d808-74f3-8000-02ec7409b960","datapoint_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","evaluator_inference_id":null,"input":"{\"messages\":[]}","generated_output":"[{\"type\":\"text\",\"text\":\"hello\"}]","reference_output":"[{\"type\":\"text\",\"text\":\"hello\"}]","dataset_name":"test_dataset","metric_name":"tensorzero::evaluation_name::test::evaluator_name::exact_match","metric_value":"true","feedback_id":"0196ee9c-d808-74f3-8000-02ec7409b961","is_human_feedback":false,"variant_name":"test_variant","name":"test_datapoint","staled_at":null}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_results(
                "test_func",
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap()],
                FunctionConfigType::Chat,
                &["tensorzero::evaluation_name::test::evaluator_name::exact_match".to_string()],
                None,
                10,
                0,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        match &result[0] {
            EvaluationResultRow::Chat(row) => {
                assert_eq!(row.variant_name, "test_variant");
                assert_eq!(row.dataset_name, "test_dataset");
                assert_eq!(
                    row.metric_name,
                    Some(
                        "tensorzero::evaluation_name::test::evaluator_name::exact_match"
                            .to_string()
                    )
                );
                assert_eq!(row.metric_value, Some("true".to_string()));
                assert!(!row.is_human_feedback);
            }
            EvaluationResultRow::Json(_) => {
                panic!("Expected Chat result")
            }
        }
    }

    #[tokio::test]
    async fn test_get_evaluation_results_multiple_results() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"inference_id":"0196ee9c-d808-74f3-8000-02ec7409b95f","episode_id":"0196ee9c-d808-74f3-8000-02ec7409b960","datapoint_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","evaluator_inference_id":null,"input":"{\"messages\":[]}","generated_output":"{\"raw\":\"{}\",\"parsed\":{}}","reference_output":"{\"raw\":\"{}\",\"parsed\":{}}","dataset_name":"ds1","metric_name":"metric1","metric_value":"true","feedback_id":"0196ee9c-d808-74f3-8000-02ec7409b961","is_human_feedback":false,"variant_name":"variant1","name":null,"staled_at":null}
{"inference_id":"0196ee9c-d808-74f3-8000-02ec7409b962","episode_id":"0196ee9c-d808-74f3-8000-02ec7409b963","datapoint_id":"0196ee9c-d808-74f3-8000-02ec7409b964","evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b965","evaluator_inference_id":"0196ee9c-d808-74f3-8000-02ec7409b966","input":"{\"messages\":[]}","generated_output":"{\"raw\":\"{}\",\"parsed\":{}}","reference_output":"{\"raw\":\"{}\",\"parsed\":{}}","dataset_name":"ds2","metric_name":"metric2","metric_value":"0.95","feedback_id":"0196ee9c-d808-74f3-8000-02ec7409b967","is_human_feedback":true,"variant_name":"variant2","name":"named_dp","staled_at":"2025-05-20T16:52:58Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_results(
                "test_func",
                &[
                    Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap(),
                    Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b965").unwrap(),
                ],
                FunctionConfigType::Json,
                &["metric1".to_string(), "metric2".to_string()],
                None,
                100,
                0,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        match &result[0] {
            EvaluationResultRow::Json(row) => {
                assert_eq!(row.variant_name, "variant1");
                assert!(!row.is_human_feedback);
                assert!(row.evaluator_inference_id.is_none());
            }
            EvaluationResultRow::Chat(_) => {
                panic!("Expected Json result")
            }
        }

        match &result[1] {
            EvaluationResultRow::Json(row) => {
                assert_eq!(row.variant_name, "variant2");
                assert!(row.is_human_feedback);
                assert!(row.evaluator_inference_id.is_some());
                assert_eq!(row.name, Some("named_dp".to_string()));
                assert!(row.staled_at.is_some());
            }
            EvaluationResultRow::Chat(_) => {
                panic!("Expected Json result")
            }
        }
    }

    #[tokio::test]
    async fn test_get_evaluation_results_empty() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_results(
                "nonexistent_func",
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap()],
                FunctionConfigType::Chat,
                &["metric".to_string()],
                None,
                100,
                0,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_results_with_pagination() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _params| {
                // Verify pagination is passed through
                assert_query_contains(query, "LIMIT 50");
                assert_query_contains(query, "OFFSET 100");
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_results(
                "test_func",
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap()],
                FunctionConfigType::Json,
                &["metric".to_string()],
                None,
                50,
                100,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 0);
    }

    // ============================================================================
    // get_evaluation_results with datapoint_id filter tests
    // ============================================================================

    #[tokio::test]
    async fn test_get_evaluation_results_with_datapoint_id_filter() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                // Verify the query structure includes datapoint filter
                assert_query_contains(query, "WITH all_inference_ids AS");
                assert_query_contains(query, "all_datapoint_ids AS");
                assert_query_contains(query, "filtered_dp AS");
                assert_query_contains(query, "filtered_inference AS");
                assert_query_contains(query, "filtered_feedback AS");
                assert_query_contains(query, "FROM ChatInference");
                assert_query_contains(query, "FROM ChatInferenceDatapoint");
                // Verify the datapoint_id filter is present
                assert_query_contains(query, "AND value = {datapoint_id:String}");
                assert_query_contains(query, "LEFT JOIN filtered_feedback");

                // Verify parameters include datapoint_id
                assert_eq!(
                    params.get("datapoint_id"),
                    Some(&"0196ee9c-d808-74f3-8000-02ec7409b95d")
                );
                assert_eq!(params.get("function_name"), Some(&"test_func"));
                assert_eq!(
                    params.get("evaluation_run_ids"),
                    Some(&"['0196ee9c-d808-74f3-8000-02ec7409b95e']")
                );
                assert_eq!(
                    params.get("metric_names"),
                    Some(&"['tensorzero::evaluation_name::test::evaluator_name::exact_match']")
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"inference_id":"0196ee9c-d808-74f3-8000-02ec7409b95f","episode_id":"0196ee9c-d808-74f3-8000-02ec7409b960","datapoint_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","evaluator_inference_id":null,"input":"{\"messages\":[]}","generated_output":"[{\"type\":\"text\",\"text\":\"hello\"}]","reference_output":"[{\"type\":\"text\",\"text\":\"hello\"}]","dataset_name":"test_dataset","metric_name":"tensorzero::evaluation_name::test::evaluator_name::exact_match","metric_value":"true","feedback_id":"0196ee9c-d808-74f3-8000-02ec7409b961","is_human_feedback":false,"variant_name":"test_variant","name":"test_datapoint","staled_at":null}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let datapoint_id = Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap();
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_results(
                "test_func",
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap()],
                FunctionConfigType::Chat,
                &["tensorzero::evaluation_name::test::evaluator_name::exact_match".to_string()],
                Some(&datapoint_id),
                u32::MAX,
                0,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        match &result[0] {
            EvaluationResultRow::Chat(row) => {
                assert_eq!(row.variant_name, "test_variant");
                assert_eq!(row.dataset_name, "test_dataset");
                assert_eq!(
                    row.metric_name,
                    Some(
                        "tensorzero::evaluation_name::test::evaluator_name::exact_match"
                            .to_string()
                    )
                );
                assert_eq!(row.metric_value, Some("true".to_string()));
                assert!(!row.is_human_feedback);
                assert_eq!(row.name, Some("test_datapoint".to_string()));
            }
            EvaluationResultRow::Json(_) => {
                panic!("Expected Chat result")
            }
        }
    }

    #[tokio::test]
    async fn test_get_evaluation_results_with_datapoint_id_json() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _params| {
                assert_query_contains(query, "FROM JsonInference");
                assert_query_contains(query, "FROM JsonInferenceDatapoint");
                assert_query_contains(query, "AND value = {datapoint_id:String}");
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"inference_id":"0196ee9c-d808-74f3-8000-02ec7409b95f","episode_id":"0196ee9c-d808-74f3-8000-02ec7409b960","datapoint_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","evaluator_inference_id":null,"input":"{}","generated_output":"{}","reference_output":"{}","dataset_name":"test_dataset","metric_name":"metric1","metric_value":"0.95","feedback_id":"0196ee9c-d808-74f3-8000-02ec7409b961","is_human_feedback":false,"variant_name":"test_variant","name":null,"staled_at":null}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let datapoint_id = Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap();
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_results(
                "test_func",
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap()],
                FunctionConfigType::Json,
                &["metric1".to_string()],
                Some(&datapoint_id),
                u32::MAX,
                0,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        match &result[0] {
            EvaluationResultRow::Json(row) => {
                assert_eq!(row.variant_name, "test_variant");
                assert_eq!(row.metric_value, Some("0.95".to_string()));
            }
            EvaluationResultRow::Chat(_) => {
                panic!("Expected Json result")
            }
        }
    }

    #[tokio::test]
    async fn test_get_evaluation_results_with_datapoint_id_multiple_run_ids() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                // Should have datapoint_id and multiple run IDs
                assert_query_contains(query, "AND value = {datapoint_id:String}");
                assert_eq!(
                    params.get("evaluation_run_ids"),
                    Some(&"['0196ee9c-d808-74f3-8000-02ec7409b95e','0196ee9c-d808-74f3-8000-02ec7409b95f']")
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"inference_id":"0196ee9c-d808-74f3-8000-02ec7409b95f","episode_id":"0196ee9c-d808-74f3-8000-02ec7409b960","datapoint_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","evaluator_inference_id":null,"input":"{\"messages\":[]}","generated_output":"[]","reference_output":"[]","dataset_name":"ds1","metric_name":"metric1","metric_value":"true","feedback_id":"0196ee9c-d808-74f3-8000-02ec7409b961","is_human_feedback":false,"variant_name":"variant1","name":null,"staled_at":null}
{"inference_id":"0196ee9c-d808-74f3-8000-02ec7409b962","episode_id":"0196ee9c-d808-74f3-8000-02ec7409b963","datapoint_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95f","evaluator_inference_id":"0196ee9c-d808-74f3-8000-02ec7409b966","input":"{\"messages\":[]}","generated_output":"[]","reference_output":"[]","dataset_name":"ds1","metric_name":"metric1","metric_value":"0.75","feedback_id":"0196ee9c-d808-74f3-8000-02ec7409b967","is_human_feedback":true,"variant_name":"variant2","name":null,"staled_at":"2025-05-20T16:52:58Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });

        let datapoint_id = Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap();
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_results(
                "test_func",
                &[
                    Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap(),
                    Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95f").unwrap(),
                ],
                FunctionConfigType::Chat,
                &["metric1".to_string()],
                Some(&datapoint_id),
                u32::MAX,
                0,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        match &result[0] {
            EvaluationResultRow::Chat(row) => {
                assert_eq!(row.variant_name, "variant1");
                assert!(!row.is_human_feedback);
                assert!(row.evaluator_inference_id.is_none());
            }
            EvaluationResultRow::Json(_) => {
                panic!("Expected Chat result")
            }
        }
        match &result[1] {
            EvaluationResultRow::Chat(row) => {
                assert_eq!(row.variant_name, "variant2");
                assert!(row.is_human_feedback);
                assert!(row.evaluator_inference_id.is_some());
                assert!(row.staled_at.is_some());
            }
            EvaluationResultRow::Json(_) => {
                panic!("Expected Chat result")
            }
        }
    }

    #[tokio::test]
    async fn test_get_evaluation_results_with_datapoint_id_empty() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let datapoint_id = Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap();
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_results(
                "nonexistent_func",
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap()],
                FunctionConfigType::Chat,
                &["metric".to_string()],
                Some(&datapoint_id),
                u32::MAX,
                0,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_results_with_datapoint_id_multiple_metrics() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                // Should have datapoint_id and multiple metric names
                assert_query_contains(query, "AND value = {datapoint_id:String}");
                assert_eq!(
                    params.get("metric_names"),
                    Some(&"['metric1','metric2']")
                );
                true
            })
            .returning(|_, _| {
                // Return rows for different metrics for the same inference
                Ok(ClickHouseResponse {
                    response: r#"{"inference_id":"0196ee9c-d808-74f3-8000-02ec7409b95f","episode_id":"0196ee9c-d808-74f3-8000-02ec7409b960","datapoint_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","evaluator_inference_id":null,"input":"{\"messages\":[]}","generated_output":"[]","reference_output":"[]","dataset_name":"test_dataset","metric_name":"metric1","metric_value":"true","feedback_id":"0196ee9c-d808-74f3-8000-02ec7409b961","is_human_feedback":false,"variant_name":"test_variant","name":null,"staled_at":null}
{"inference_id":"0196ee9c-d808-74f3-8000-02ec7409b95f","episode_id":"0196ee9c-d808-74f3-8000-02ec7409b960","datapoint_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","evaluator_inference_id":"0196ee9c-d808-74f3-8000-02ec7409b968","input":"{\"messages\":[]}","generated_output":"[]","reference_output":"[]","dataset_name":"test_dataset","metric_name":"metric2","metric_value":"0.95","feedback_id":"0196ee9c-d808-74f3-8000-02ec7409b969","is_human_feedback":false,"variant_name":"test_variant","name":null,"staled_at":null}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });

        let datapoint_id = Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap();
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_results(
                "test_func",
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap()],
                FunctionConfigType::Chat,
                &["metric1".to_string(), "metric2".to_string()],
                Some(&datapoint_id),
                u32::MAX,
                0,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        match &result[0] {
            EvaluationResultRow::Chat(row) => {
                assert_eq!(row.metric_name, Some("metric1".to_string()));
                assert_eq!(row.metric_value, Some("true".to_string()));
            }
            EvaluationResultRow::Json(_) => {
                panic!("Expected Chat result")
            }
        }
        match &result[1] {
            EvaluationResultRow::Chat(row) => {
                assert_eq!(row.metric_name, Some("metric2".to_string()));
                assert_eq!(row.metric_value, Some("0.95".to_string()));
            }
            EvaluationResultRow::Json(_) => {
                panic!("Expected Chat result")
            }
        }
    }

    #[tokio::test]
    async fn test_get_evaluation_results_no_feedback() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        // Test case where inference exists but no feedback (LEFT JOIN returns nulls)
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"inference_id":"0196ee9c-d808-74f3-8000-02ec7409b95f","episode_id":"0196ee9c-d808-74f3-8000-02ec7409b960","datapoint_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","evaluation_run_id":"0196ee9c-d808-74f3-8000-02ec7409b95e","evaluator_inference_id":null,"input":"{\"messages\":[]}","generated_output":"[]","reference_output":"[]","dataset_name":"ds","metric_name":null,"metric_value":null,"feedback_id":null,"is_human_feedback":false,"variant_name":"variant","name":null,"staled_at":null}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_evaluation_results(
                "test_func",
                &[Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap()],
                FunctionConfigType::Chat,
                &["metric".to_string()],
                None,
                100,
                0,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        match &result[0] {
            EvaluationResultRow::Chat(row) => {
                assert!(row.metric_name.is_none());
                assert!(row.metric_value.is_none());
                assert!(row.feedback_id.is_none());
            }
            EvaluationResultRow::Json(_) => {
                panic!("Expected Chat result")
            }
        }
    }

    // ============================================================================
    // get_inference_evaluation_human_feedback tests
    // ============================================================================

    #[tokio::test]
    async fn test_get_inference_evaluation_human_feedback_found() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(
                    query,
                    "SELECT value, evaluator_inference_id
                    FROM StaticEvaluationHumanFeedback
                    WHERE metric_name = {metric_name:String}
                    AND datapoint_id = {datapoint_id:UUID}
                    AND output = {output:String}
                    LIMIT 1
                    FORMAT JSONEachRow",
                );
                assert_eq!(params.get("metric_name"), Some(&"test_metric"));
                assert_eq!(
                    params.get("datapoint_id"),
                    Some(&"0196ee9c-d808-74f3-8000-02ec7409b95d")
                );
                assert_eq!(params.get("output"), Some(&r#"{"raw":"test output"}"#));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"value":"0.95","evaluator_inference_id":"0196ee9c-d808-74f3-8000-02ec7409b95e"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let datapoint_id = Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap();
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_inference_evaluation_human_feedback(
                "test_metric",
                &datapoint_id,
                r#"{"raw":"test output"}"#,
            )
            .await
            .unwrap();

        assert!(result.is_some());
        let feedback = result.unwrap();
        assert_eq!(feedback.value, serde_json::json!(0.95));
        assert_eq!(
            feedback.evaluator_inference_id,
            Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95e").unwrap()
        );
    }

    #[tokio::test]
    async fn test_get_inference_evaluation_human_feedback_not_found() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let datapoint_id = Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap();
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_inference_evaluation_human_feedback(
                "nonexistent_metric",
                &datapoint_id,
                r#"{"raw":"test output"}"#,
            )
            .await
            .unwrap();

        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_get_inference_evaluation_human_feedback_boolean_value() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"value":"true","evaluator_inference_id":"0196ee9c-d808-74f3-8000-02ec7409b95e"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let datapoint_id = Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap();
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_inference_evaluation_human_feedback("test_metric", &datapoint_id, "test output")
            .await
            .unwrap();

        assert!(result.is_some());
        let feedback = result.unwrap();
        assert_eq!(feedback.value, serde_json::json!(true));
    }

    #[tokio::test]
    async fn test_get_inference_evaluation_human_feedback_object_value() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"value":"{\"score\":0.8,\"reason\":\"good\"}","evaluator_inference_id":"0196ee9c-d808-74f3-8000-02ec7409b95e"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let datapoint_id = Uuid::parse_str("0196ee9c-d808-74f3-8000-02ec7409b95d").unwrap();
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_inference_evaluation_human_feedback("test_metric", &datapoint_id, "test output")
            .await
            .unwrap();

        assert!(result.is_some());
        let feedback = result.unwrap();
        assert_eq!(
            feedback.value,
            serde_json::json!({"score": 0.8, "reason": "good"})
        );
    }
}
