//! ClickHouse queries for workflow evaluation statistics.

use std::collections::HashMap;

use async_trait::async_trait;
use uuid::Uuid;

use super::ClickHouseConnectionInfo;
use super::select_queries::{parse_count, parse_json_rows};
use crate::db::workflow_evaluation_queries::{
    WorkflowEvaluationProjectRow, WorkflowEvaluationQueries, WorkflowEvaluationRunRow,
    WorkflowEvaluationRunWithEpisodeCountRow,
};
use crate::error::{Error, ErrorDetails};

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

    async fn count_workflow_evaluation_projects(&self) -> Result<u32, Error> {
        let query = r"
            SELECT
                toUInt32(countDistinct(project_name)) as count
            FROM DynamicEvaluationRunByProjectName
            WHERE project_name IS NOT NULL
            FORMAT JSONEachRow
        "
        .to_string();

        let response = self.run_query_synchronous_no_params(query).await?;
        let count = parse_count(response.response.as_str())?;

        u32::try_from(count).map_err(|error| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: format!("Failed to convert workflow evaluation project count: {error}"),
            })
        })
    }

    async fn search_workflow_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
        project_name: Option<&str>,
        search_query: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunRow>, Error> {
        // Build WHERE clause predicates
        let mut predicates: Vec<String> = Vec::new();

        if project_name.is_some() {
            predicates.push("project_name = {project_name:String}".to_string());
        }

        if search_query.is_some() {
            predicates.push(
                "(positionCaseInsensitive(run_display_name, {search_query:String}) > 0 \
                 OR positionCaseInsensitive(toString(uint_to_uuid(run_id_uint)), {search_query:String}) > 0)"
                    .to_string(),
            );
        }

        let where_clause = if predicates.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", predicates.join(" AND "))
        };

        let query = format!(
            r"
            SELECT
                run_display_name as name,
                uint_to_uuid(run_id_uint) as id,
                variant_pins,
                tags,
                project_name,
                formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') as timestamp
            FROM DynamicEvaluationRun
            {where_clause}
            ORDER BY updated_at DESC
            LIMIT {{limit:UInt32}}
            OFFSET {{offset:UInt32}}
            FORMAT JSONEachRow
            "
        );

        let limit_str = limit.to_string();
        let offset_str = offset.to_string();
        let project_name_str = project_name.unwrap_or_default().to_string();
        let search_query_str = search_query.unwrap_or_default().to_string();

        let mut params = HashMap::new();
        params.insert("limit", limit_str.as_str());
        params.insert("offset", offset_str.as_str());
        if project_name.is_some() {
            params.insert("project_name", project_name_str.as_str());
        }
        if search_query.is_some() {
            params.insert("search_query", search_query_str.as_str());
        }

        let response = self.run_query_synchronous(query, &params).await?;

        parse_json_rows(response.response.as_str())
    }

    async fn list_workflow_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
        run_id: Option<Uuid>,
        project_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunWithEpisodeCountRow>, Error> {
        // Build WHERE clause - only one of run_id or project_name should be provided
        let where_clause = if run_id.is_some() {
            "WHERE toUInt128(toUUID({run_id:String})) = run_id_uint"
        } else if project_name.is_some() {
            "WHERE project_name = {project_name:String}"
        } else {
            ""
        };

        let query = format!(
            r"
            WITH FilteredDynamicEvaluationRuns AS (
                SELECT
                    run_display_name as name,
                    uint_to_uuid(run_id_uint) as id,
                    run_id_uint,
                    variant_pins,
                    tags,
                    project_name,
                    formatDateTime(UUIDv7ToDateTime(uint_to_uuid(run_id_uint)), '%Y-%m-%dT%H:%i:%SZ') as timestamp
                FROM DynamicEvaluationRun
                {where_clause}
                ORDER BY run_id_uint DESC
                LIMIT {{limit:UInt32}}
                OFFSET {{offset:UInt32}}
            ),
            DynamicEvaluationRunsEpisodeCounts AS (
                SELECT
                    run_id_uint,
                    toUInt32(count()) as num_episodes
                FROM DynamicEvaluationRunEpisodeByRunId
                WHERE run_id_uint IN (SELECT run_id_uint FROM FilteredDynamicEvaluationRuns)
                GROUP BY run_id_uint
            )
            SELECT
                name,
                id,
                variant_pins,
                tags,
                project_name,
                COALESCE(num_episodes, 0) AS num_episodes,
                timestamp
            FROM FilteredDynamicEvaluationRuns
            LEFT JOIN DynamicEvaluationRunsEpisodeCounts USING run_id_uint
            ORDER BY run_id_uint DESC
            FORMAT JSONEachRow
            "
        );

        let limit_str = limit.to_string();
        let offset_str = offset.to_string();
        let run_id_str = run_id.map(|id| id.to_string()).unwrap_or_default();
        let project_name_str = project_name.unwrap_or_default().to_string();

        let mut params = HashMap::new();
        params.insert("limit", limit_str.as_str());
        params.insert("offset", offset_str.as_str());
        if run_id.is_some() {
            params.insert("run_id", run_id_str.as_str());
        }
        if project_name.is_some() {
            params.insert("project_name", project_name_str.as_str());
        }

        let response = self.run_query_synchronous(query, &params).await?;

        parse_json_rows(response.response.as_str())
    }

    async fn count_workflow_evaluation_runs(&self) -> Result<u32, Error> {
        let query = r"
            SELECT toUInt32(count()) as count FROM DynamicEvaluationRun
            FORMAT JSONEachRow
        "
        .to_string();

        let response = self.run_query_synchronous_no_params(query).await?;
        let count = parse_count(response.response.as_str())?;

        u32::try_from(count).map_err(|error| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: format!("Failed to convert workflow evaluation run count: {error}"),
            })
        })
    }

    async fn get_workflow_evaluation_runs(
        &self,
        run_ids: &[Uuid],
        project_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunRow>, Error> {
        if run_ids.is_empty() {
            return Ok(vec![]);
        }

        let mut params = HashMap::new();

        let project_name_filter = if let Some(project_name_str) = project_name {
            params.insert("project_name", project_name_str);
            "AND project_name = {project_name:String}"
        } else {
            ""
        };

        let query = format!(
            r"
            SELECT
                run_display_name AS name,
                uint_to_uuid(run_id_uint) AS id,
                variant_pins,
                tags,
                project_name,
                formatDateTime(
                    UUIDv7ToDateTime(uint_to_uuid(run_id_uint)),
                    '%Y-%m-%dT%H:%i:%SZ'
                ) AS timestamp
            FROM DynamicEvaluationRun
            WHERE run_id_uint IN (
                SELECT arrayJoin(
                    arrayMap(x -> toUInt128(toUUID(x)), {{run_ids:Array(String)}})
                )
            )
            {project_name_filter}
            ORDER BY run_id_uint DESC
            FORMAT JSONEachRow
            "
        );

        // Format run_ids as ClickHouse array with single quotes
        let run_ids_str: Vec<String> = run_ids.iter().map(|id| format!("'{id}'")).collect();
        let run_ids_array = format!("[{}]", run_ids_str.join(","));
        params.insert("run_ids", run_ids_array.as_str());

        let response = self.run_query_synchronous(query, &params).await?;

        parse_json_rows(response.response.as_str())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use uuid::Uuid;

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

    #[tokio::test]
    async fn test_count_workflow_evaluation_projects() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(
                    query,
                    "
                SELECT toUInt32(countDistinct(project_name)) as count
                FROM DynamicEvaluationRunByProjectName
                WHERE project_name IS NOT NULL
                FORMAT JSONEachRow",
                );
                assert!(params.is_empty());
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":2}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let count = conn.count_workflow_evaluation_projects().await.unwrap();

        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_runs_empty_ids() {
        let mock_clickhouse_client = MockClickHouseClient::new();
        // No expectations set - should not call the database

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn.get_workflow_evaluation_runs(&[], None).await.unwrap();

        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_runs_with_ids() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "SELECT");
                assert_query_contains(query, "run_display_name AS name");
                assert_query_contains(query, "uint_to_uuid(run_id_uint) AS id");
                assert_query_contains(query, "variant_pins");
                assert_query_contains(query, "tags");
                assert_query_contains(query, "project_name");
                assert_query_contains(query, "FROM DynamicEvaluationRun");
                assert_query_contains(query, "WHERE run_id_uint IN");
                assert_query_contains(query, "ORDER BY run_id_uint DESC");

                assert!(params.contains_key("run_ids"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"name":"test_run","id":"01968d04-142c-7e53-8ea7-3a3255b518dc","variant_pins":{},"tags":{},"project_name":"test_project","timestamp":"2025-05-01T18:02:56Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();
        let result = conn
            .get_workflow_evaluation_runs(&[run_id], None)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, Some("test_run".to_string()));
        assert_eq!(result[0].id, run_id);
        assert_eq!(result[0].project_name, Some("test_project".to_string()));
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_runs_with_project_name() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "AND project_name = {project_name:String}");
                assert!(params.contains_key("run_ids"));
                assert_eq!(params.get("project_name"), Some(&"my_project"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"name":"filtered_run","id":"01968d04-142c-7e53-8ea7-3a3255b518dc","variant_pins":{},"tags":{},"project_name":"my_project","timestamp":"2025-05-01T18:02:56Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();
        let result = conn
            .get_workflow_evaluation_runs(&[run_id], Some("my_project"))
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].project_name, Some("my_project".to_string()));
    }
}
