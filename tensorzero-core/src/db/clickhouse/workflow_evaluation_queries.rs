//! ClickHouse queries for workflow evaluation statistics.

use std::collections::HashMap;

use async_trait::async_trait;

use super::ClickHouseConnectionInfo;
use super::select_queries::parse_json_rows;
use crate::db::workflow_evaluation_queries::WorkflowEvaluationProjectRow;
use crate::db::workflow_evaluation_queries::WorkflowEvaluationQueries;
use crate::error::Error;

#[async_trait]
impl WorkflowEvaluationQueries for ClickHouseConnectionInfo {
    async fn list_workflow_evaluation_projects(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<WorkflowEvaluationProjectRow>, Error> {
        let query = r"
            SELECT
                project_name as name,
                toUInt32(count()) as count,
                formatDateTime(max(updated_at), '%Y-%m-%dT%H:%i:%SZ') as last_updated
            FROM DynamicEvaluationRunByProjectName
            GROUP BY project_name
            ORDER BY last_updated DESC
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
        workflow_evaluation_queries::WorkflowEvaluationQueries,
    };

    #[tokio::test]
    async fn test_list_workflow_evaluation_projects() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "SELECT");
                assert_query_contains(query, "project_name as name");
                assert_query_contains(query, "toUInt32(count()) as count");
                assert_query_contains(query, "FROM DynamicEvaluationRunByProjectName");
                assert_query_contains(query, "GROUP BY project_name");
                assert_query_contains(query, "ORDER BY last_updated DESC");
                assert_query_contains(query, "LIMIT {limit:UInt32}");
                assert_query_contains(query, "OFFSET {offset:UInt32}");

                assert_eq!(params.get("limit"), Some(&"10"));
                assert_eq!(params.get("offset"), Some(&"0"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response:
                        r#"{"name":"project1","count":5,"last_updated":"2025-05-20T16:52:58Z"}"#
                            .to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn.list_workflow_evaluation_projects(10, 0).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "project1");
        assert_eq!(result[0].count, 5);
    }

    #[tokio::test]
    async fn test_list_workflow_evaluation_projects_with_custom_pagination() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|_query, params| {
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

        let result = conn
            .list_workflow_evaluation_projects(50, 100)
            .await
            .unwrap();

        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_list_workflow_evaluation_projects_multiple_results() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response:
                        r#"{"name":"project1","count":5,"last_updated":"2025-05-20T16:52:58Z"}
{"name":"project2","count":10,"last_updated":"2025-05-20T17:52:58Z"}
{"name":"project3","count":3,"last_updated":"2025-05-20T18:52:58Z"}"#
                            .to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 3,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .list_workflow_evaluation_projects(100, 0)
            .await
            .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].name, "project1");
        assert_eq!(result[1].name, "project2");
        assert_eq!(result[2].name, "project3");
    }
}
