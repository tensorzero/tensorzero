//! ClickHouse queries for evaluation statistics.

use std::collections::HashMap;

use async_trait::async_trait;

use super::ClickHouseConnectionInfo;
use super::select_queries::{parse_count, parse_json_rows};
use crate::db::evaluation_queries::EvaluationQueries;
use crate::db::evaluation_queries::EvaluationRunInfoRow;
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
}
