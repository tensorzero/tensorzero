//! ClickHouse queries for evaluation statistics.

use std::collections::HashMap;

use async_trait::async_trait;

use super::ClickHouseConnectionInfo;
use super::select_queries::{parse_count, parse_json_rows};
use crate::db::evaluation_queries::EvaluationQueries;
use crate::db::evaluation_queries::EvaluationRunInfoByIdRow;
use crate::db::evaluation_queries::EvaluationRunInfoRow;
use crate::db::evaluation_queries::EvaluationRunSearchResult;
use crate::error::Error;

// Private helper for constructing the subquery for datapoint IDs
fn get_evaluation_result_datapoint_id_subquery(
    function_name: &str,
    evaluation_run_ids: &[uuid::Uuid],
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

    let query = format!(
        "all_inference_ids AS (
            SELECT DISTINCT inference_id
            FROM TagInference FINAL WHERE key = 'tensorzero::evaluation_run_id'
            AND function_name = {{function_name:String}}
            AND value IN ({{evaluation_run_ids:Array(String)}})
        ),
        all_datapoint_ids AS (
            SELECT DISTINCT value as datapoint_id
            FROM TagInference FINAL
            WHERE key = 'tensorzero::datapoint_id'
            AND function_name = {{function_name:String}}
            AND inference_id IN (SELECT inference_id FROM all_inference_ids)
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
                FROM TagInference FINAL
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
            FROM TagInference FINAL
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
                TagInference AS run_tag FINAL
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
    ) -> Result<Vec<EvaluationRunInfoByIdRow>, Error> {
        let sql_query = r"
            WITH datapoint_inference_ids AS (
                SELECT inference_id
                FROM TagInference FINAL
                WHERE key = 'tensorzero::datapoint_id'
                AND value = {datapoint_id:String}
                AND function_name = {function_name:String}
            )
            SELECT
                any(run_tag.value) as evaluation_run_id,
                any(run_tag.variant_name) as variant_name,
                formatDateTime(
                    max(UUIDv7ToDateTime(run_tag.inference_id)),
                    '%Y-%m-%dT%H:%i:%SZ'
                ) as most_recent_inference_date
            FROM TagInference AS run_tag FINAL
            WHERE
                run_tag.key = 'tensorzero::evaluation_run_id'
                AND run_tag.inference_id IN (SELECT inference_id FROM datapoint_inference_ids)
                AND run_tag.function_name = {function_name:String}
            GROUP BY
                run_tag.value
            ORDER BY
                toUInt128(toUUID(evaluation_run_id)) DESC
            FORMAT JSONEachRow
        "
        .to_string();

        let datapoint_id_str = datapoint_id.to_string();
        let function_name_str = function_name.to_string();

        let mut params = HashMap::new();
        params.insert("datapoint_id", datapoint_id_str.as_str());
        params.insert("function_name", function_name_str.as_str());

        let response = self.run_query_synchronous(sql_query, &params).await?;
        parse_json_rows(response.response.as_str())
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
        evaluation_queries::EvaluationQueries,
    };

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
                assert_query_contains(query, "FROM TagInference FINAL");
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
                    FROM TagInference FINAL WHERE key = 'tensorzero::evaluation_run_id'
                    AND function_name = {function_name:String}
                    AND value IN ({evaluation_run_ids:Array(String)})
                ),
                all_datapoint_ids AS (
                    SELECT DISTINCT value as datapoint_id
                    FROM TagInference FINAL
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
                FROM TagInference FINAL
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
                    TagInference AS run_tag FINAL
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
    async fn test_get_evaluation_run_infos_for_datapoint() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(
                    query,
                    "WITH datapoint_inference_ids AS (
                        SELECT inference_id
                        FROM TagInference FINAL
                        WHERE key = 'tensorzero::datapoint_id'
                        AND value = {datapoint_id:String}
                        AND function_name = {function_name:String}
                    )
                    SELECT
                        any(run_tag.value) as evaluation_run_id,
                        any(run_tag.variant_name) as variant_name,
                        formatDateTime(
                            max(UUIDv7ToDateTime(run_tag.inference_id)),
                            '%Y-%m-%dT%H:%i:%SZ'
                        ) as most_recent_inference_date
                    FROM TagInference AS run_tag FINAL
                    WHERE
                        run_tag.key = 'tensorzero::evaluation_run_id'
                        AND run_tag.inference_id IN (SELECT inference_id FROM datapoint_inference_ids)
                        AND run_tag.function_name = {function_name:String}
                    GROUP BY
                        run_tag.value
                    ORDER BY
                        toUInt128(toUUID(evaluation_run_id)) DESC
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
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 0);
    }
}
