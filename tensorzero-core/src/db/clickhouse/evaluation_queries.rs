//! ClickHouse queries for evaluation statistics.

use std::collections::HashMap;

use async_trait::async_trait;

use super::ClickHouseConnectionInfo;
use super::select_queries::{parse_count, parse_json_rows};
use crate::db::evaluation_queries::EvaluationQueries;
use crate::db::evaluation_queries::EvaluationRunInfoRow;
use crate::error::Error;

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
}
