//! DICL (Dynamic In-Context Learning) queries for ClickHouse.
//!
//! This module implements similarity search for DICL examples using ClickHouse's
//! cosineDistance function and ExternalDataInfo for batch inserts.

use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

use async_trait::async_trait;

use super::{ClickHouseConnectionInfo, ExternalDataInfo};
use crate::db::{DICLExampleWithDistance, DICLQueries, StoredDICLExample};
use crate::error::{Error, ErrorDetails};

#[async_trait]
impl DICLQueries for ClickHouseConnectionInfo {
    async fn insert_dicl_example(&self, example: &StoredDICLExample) -> Result<(), Error> {
        self.insert_dicl_examples(std::slice::from_ref(example))
            .await?;
        Ok(())
    }

    async fn insert_dicl_examples(&self, examples: &[StoredDICLExample]) -> Result<u64, Error> {
        if examples.is_empty() {
            return Ok(0);
        }

        // Process in batches to avoid query size limits
        const BATCH_SIZE: usize = 100;

        let mut total_inserted = 0u64;

        for batch in examples.chunks(BATCH_SIZE) {
            let rows_affected = insert_dicl_batch(self, batch).await?;
            total_inserted += rows_affected;
        }

        Ok(total_inserted)
    }

    async fn get_similar_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
        embedding: &[f32],
        limit: u32,
    ) -> Result<Vec<DICLExampleWithDistance>, Error> {
        // Format the embedding as a string for ClickHouse: [0.1,0.2,0.3]
        // Note: embedding and limit are interpolated because ClickHouse parameterized queries
        // don't support array literals in cosineDistance() or LIMIT.
        let formatted_embedding = format_embedding_for_clickhouse(embedding);

        let query = format!(
            r"SELECT input, output, cosineDistance(embedding, {formatted_embedding}) as cosine_distance
                   FROM DynamicInContextLearningExample
                   WHERE function_name = {{function_name:String}} AND variant_name = {{variant_name:String}}
                   ORDER BY cosine_distance ASC
                   LIMIT {limit}
                   FORMAT JSONEachRow"
        );

        let params = HashMap::from([
            ("function_name", function_name),
            ("variant_name", variant_name),
        ]);

        let result = self.run_query_synchronous(query, &params).await?;

        // Parse each line into RawDiclExample
        let examples: Vec<DICLExampleWithDistance> = result
            .response
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                let raw: RawDiclExample = serde_json::from_str(line)?;
                Ok(DICLExampleWithDistance {
                    input: raw.input,
                    output: raw.output,
                    cosine_distance: raw.cosine_distance,
                })
            })
            .collect::<Result<Vec<_>, serde_json::Error>>()
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to parse DICL examples: {e}"),
                })
            })?;

        Ok(examples)
    }

    async fn has_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
    ) -> Result<bool, Error> {
        let query = r"
            SELECT 1
            FROM DynamicInContextLearningExample
            WHERE function_name = {function_name:String}
            AND variant_name = {variant_name:String}
            LIMIT 1
        ";

        let params = HashMap::from([
            ("function_name", function_name),
            ("variant_name", variant_name),
        ]);

        let result = self
            .run_query_synchronous(query.to_string(), &params)
            .await?;

        // If the query returns "1", examples exist; if empty, they don't
        Ok(result.response.trim() == "1")
    }

    async fn delete_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
        namespace: Option<&str>,
    ) -> Result<u64, Error> {
        let mut params: HashMap<&str, &str> = HashMap::from([
            ("function_name", function_name),
            ("variant_name", variant_name),
        ]);

        // ClickHouse doesn't return affected row count directly for DELETE,
        // so we first count the rows that will be deleted
        let count_query = if let Some(ns) = namespace {
            params.insert("namespace", ns);
            r"SELECT count() FROM DynamicInContextLearningExample
              WHERE function_name = {function_name:String} AND variant_name = {variant_name:String} AND namespace = {namespace:String}"
                .to_string()
        } else {
            r"SELECT count() FROM DynamicInContextLearningExample
              WHERE function_name = {function_name:String} AND variant_name = {variant_name:String}"
                .to_string()
        };

        let count_result = self.run_query_synchronous(count_query, &params).await?;
        let count: u64 = count_result.response.trim().parse().unwrap_or(0);

        if count == 0 {
            return Ok(0);
        }

        // Now perform the actual delete
        let delete_query = if namespace.is_some() {
            r"ALTER TABLE DynamicInContextLearningExample DELETE
              WHERE function_name = {function_name:String} AND variant_name = {variant_name:String} AND namespace = {namespace:String}"
                .to_string()
        } else {
            r"ALTER TABLE DynamicInContextLearningExample DELETE
              WHERE function_name = {function_name:String} AND variant_name = {variant_name:String}"
                .to_string()
        };

        self.run_query_synchronous(delete_query, &params).await?;

        Ok(count)
    }
}

/// Raw DICL example from ClickHouse query result.
#[derive(Debug, Deserialize)]
struct RawDiclExample {
    input: String,
    output: String,
    cosine_distance: f32,
}

/// Format an embedding as a ClickHouse-compatible string: [0.1,0.2,0.3]
fn format_embedding_for_clickhouse(embedding: &[f32]) -> String {
    let values: Vec<String> = embedding.iter().map(|v| v.to_string()).collect();
    format!("[{}]", values.join(","))
}

/// Insert a batch of DICL examples using ExternalDataInfo.
async fn insert_dicl_batch(
    clickhouse: &ClickHouseConnectionInfo,
    examples: &[StoredDICLExample],
) -> Result<u64, Error> {
    if examples.is_empty() {
        return Ok(0);
    }

    let serialized_rows: Result<Vec<String>, Error> = examples
        .iter()
        .map(|example| {
            // Convert f32 embedding to f64 for JSON serialization consistency
            let embedding_f64: Vec<f64> = example.embedding.iter().map(|&v| v as f64).collect();

            let row = json!({
                "id": example.id,
                "function_name": example.function_name,
                "variant_name": example.variant_name,
                "namespace": example.namespace,
                "input": example.input,
                "output": example.output,
                "embedding": embedding_f64,
            });

            serde_json::to_string(&row).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to serialize DICL example: {e}"),
                })
            })
        })
        .collect();

    let rows = serialized_rows?;

    let query = r"
        INSERT INTO DynamicInContextLearningExample
            (
                id,
                function_name,
                variant_name,
                namespace,
                input,
                output,
                embedding
            )
            SELECT
                new_data.id,
                new_data.function_name,
                new_data.variant_name,
                new_data.namespace,
                new_data.input,
                new_data.output,
                new_data.embedding
            FROM new_data
    ";

    let external_data = ExternalDataInfo {
        external_data_name: "new_data".to_string(),
        structure: "id UUID, function_name LowCardinality(String), variant_name LowCardinality(String), namespace String, input String, output String, embedding Array(Float32)".to_string(),
        format: "JSONEachRow".to_string(),
        data: rows.join("\n"),
    };

    let result = clickhouse
        .run_query_with_external_data(external_data, query.to_string())
        .await?;

    Ok(result.metadata.written_rows)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Utc;
    use uuid::Uuid;

    use super::*;
    use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
    use crate::db::clickhouse::{ClickHouseResponse, ClickHouseResponseMetadata};
    use crate::db::test_helpers::assert_query_contains;

    #[test]
    fn test_format_embedding_for_clickhouse() {
        let embedding = vec![0.1, 0.2, 0.3];
        let result = format_embedding_for_clickhouse(&embedding);
        assert_eq!(result, "[0.1,0.2,0.3]");
    }

    #[test]
    fn test_format_embedding_empty() {
        let embedding: Vec<f32> = vec![];
        let result = format_embedding_for_clickhouse(&embedding);
        assert_eq!(result, "[]");
    }

    #[test]
    fn test_format_embedding_single() {
        let embedding = vec![1.5];
        let result = format_embedding_for_clickhouse(&embedding);
        assert_eq!(result, "[1.5]");
    }

    #[test]
    fn test_format_embedding_negative_values() {
        let embedding = vec![-0.5, 0.0, 0.5];
        let result = format_embedding_for_clickhouse(&embedding);
        assert_eq!(result, "[-0.5,0,0.5]");
    }

    #[tokio::test]
    async fn test_get_similar_dicl_examples() {
        let mut mock = MockClickHouseClient::new();

        mock.expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "SELECT input, output, cosineDistance(embedding,");
                assert_query_contains(query, "FROM DynamicInContextLearningExample");
                assert_query_contains(
                    query,
                    "WHERE function_name = {function_name:String} AND variant_name = {variant_name:String}",
                );
                assert_query_contains(query, "ORDER BY cosine_distance ASC");
                assert_query_contains(query, "LIMIT 3");
                assert_query_contains(query, "FORMAT JSONEachRow");
                assert_eq!(params.get("function_name"), Some(&"test_fn"), "function_name param should match");
                assert_eq!(params.get("variant_name"), Some(&"test_var"), "variant_name param should match");
                true
            })
            .returning(|_, _| {
                let response = [
                    r#"{"input":"{\"messages\":[]}","output":"output1","cosine_distance":0.1}"#,
                    r#"{"input":"{\"messages\":[]}","output":"output2","cosine_distance":0.5}"#,
                ]
                .join("\n");
                Ok(ClickHouseResponse {
                    response,
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
        let results = conn
            .get_similar_dicl_examples("test_fn", "test_var", &[0.1, 0.2, 0.3], 3)
            .await
            .unwrap();

        assert_eq!(results.len(), 2, "Should return two examples");
        assert_eq!(results[0].output, "output1");
        assert!(
            (results[0].cosine_distance - 0.1).abs() < f32::EPSILON,
            "First result should have cosine_distance 0.1"
        );
        assert_eq!(results[1].output, "output2");
        assert!(
            (results[1].cosine_distance - 0.5).abs() < f32::EPSILON,
            "Second result should have cosine_distance 0.5"
        );
    }

    #[tokio::test]
    async fn test_get_similar_dicl_examples_empty_response() {
        let mut mock = MockClickHouseClient::new();

        mock.expect_run_query_synchronous().returning(|_, _| {
            Ok(ClickHouseResponse {
                response: String::new(),
                metadata: ClickHouseResponseMetadata {
                    read_rows: 0,
                    written_rows: 0,
                },
            })
        });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
        let results = conn
            .get_similar_dicl_examples("fn", "var", &[1.0], 5)
            .await
            .unwrap();

        assert!(results.is_empty(), "Should return empty vec for no matches");
    }

    #[tokio::test]
    async fn test_has_dicl_examples_true() {
        let mut mock = MockClickHouseClient::new();

        mock.expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "SELECT 1");
                assert_query_contains(query, "FROM DynamicInContextLearningExample");
                assert_query_contains(query, "WHERE function_name = {function_name:String}");
                assert_query_contains(query, "AND variant_name = {variant_name:String}");
                assert_query_contains(query, "LIMIT 1");
                assert_eq!(
                    params.get("function_name"),
                    Some(&"my_fn"),
                    "function_name param should match"
                );
                assert_eq!(
                    params.get("variant_name"),
                    Some(&"my_var"),
                    "variant_name param should match"
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: "1\n".to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
        let exists = conn.has_dicl_examples("my_fn", "my_var").await.unwrap();
        assert!(exists, "Should return true when examples exist");
    }

    #[tokio::test]
    async fn test_has_dicl_examples_false() {
        let mut mock = MockClickHouseClient::new();

        mock.expect_run_query_synchronous().returning(|_, _| {
            Ok(ClickHouseResponse {
                response: String::new(),
                metadata: ClickHouseResponseMetadata {
                    read_rows: 0,
                    written_rows: 0,
                },
            })
        });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
        let exists = conn.has_dicl_examples("fn", "var").await.unwrap();
        assert!(!exists, "Should return false when no examples exist");
    }

    #[tokio::test]
    async fn test_insert_dicl_examples() {
        let mut mock = MockClickHouseClient::new();

        mock.expect_run_query_with_external_data()
            .withf(|external_data, query| {
                assert_query_contains(query, "INSERT INTO DynamicInContextLearningExample");
                assert_eq!(
                    external_data.external_data_name, "new_data",
                    "External data name should be `new_data`"
                );
                assert_eq!(
                    external_data.format, "JSONEachRow",
                    "Format should be JSONEachRow"
                );
                assert!(
                    external_data.structure.contains("embedding Array(Float32)"),
                    "Structure should include embedding column"
                );
                // Verify the data contains our example
                assert!(
                    external_data.data.contains("test_fn"),
                    "Data should contain function name"
                );
                assert!(
                    external_data.data.contains("test_var"),
                    "Data should contain variant name"
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 2,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));

        let examples = vec![
            StoredDICLExample {
                id: Uuid::now_v7(),
                function_name: "test_fn".to_string(),
                variant_name: "test_var".to_string(),
                namespace: String::new(),
                input: r#"{"messages":[]}"#.to_string(),
                output: "output1".to_string(),
                embedding: vec![0.1, 0.2, 0.3],
                created_at: Utc::now(),
            },
            StoredDICLExample {
                id: Uuid::now_v7(),
                function_name: "test_fn".to_string(),
                variant_name: "test_var".to_string(),
                namespace: String::new(),
                input: r#"{"messages":[]}"#.to_string(),
                output: "output2".to_string(),
                embedding: vec![0.4, 0.5, 0.6],
                created_at: Utc::now(),
            },
        ];

        let rows = conn.insert_dicl_examples(&examples).await.unwrap();
        assert_eq!(rows, 2, "Should report 2 rows inserted");
    }

    #[tokio::test]
    async fn test_insert_dicl_examples_empty() {
        let mock = MockClickHouseClient::new();
        // No expectations set — no queries should be issued for empty input
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));

        let rows = conn.insert_dicl_examples(&[]).await.unwrap();
        assert_eq!(rows, 0, "Should return 0 for empty input");
    }

    #[tokio::test]
    async fn test_delete_dicl_examples_with_namespace() {
        let mut mock = MockClickHouseClient::new();
        let mut seq = mockall::Sequence::new();

        // First call: count query
        mock.expect_run_query_synchronous()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|query, params| {
                assert_query_contains(&query, "SELECT count()");
                assert_query_contains(&query, "namespace = {namespace:String}");
                assert_eq!(params.get("function_name"), Some(&"fn1"));
                assert_eq!(params.get("variant_name"), Some(&"var1"));
                assert_eq!(params.get("namespace"), Some(&"ns1"));
                Ok(ClickHouseResponse {
                    response: "5\n".to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        // Second call: delete query
        mock.expect_run_query_synchronous()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|query, params| {
                assert_query_contains(&query, "ALTER TABLE DynamicInContextLearningExample DELETE");
                assert_query_contains(&query, "function_name = {function_name:String}");
                assert_query_contains(&query, "variant_name = {variant_name:String}");
                assert_query_contains(&query, "namespace = {namespace:String}");
                assert_eq!(params.get("namespace"), Some(&"ns1"));
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
        let deleted = conn
            .delete_dicl_examples("fn1", "var1", Some("ns1"))
            .await
            .unwrap();
        assert_eq!(deleted, 5, "Should report 5 rows deleted");
    }

    #[tokio::test]
    async fn test_delete_dicl_examples_without_namespace() {
        let mut mock = MockClickHouseClient::new();
        let mut seq = mockall::Sequence::new();

        // First call: count query (no namespace filter)
        mock.expect_run_query_synchronous()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|query, _params| {
                assert_query_contains(&query, "SELECT count()");
                assert!(
                    !query.contains("namespace"),
                    "Count query should not filter by namespace"
                );
                Ok(ClickHouseResponse {
                    response: "3\n".to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        // Second call: delete query (no namespace filter)
        mock.expect_run_query_synchronous()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|query, _params| {
                assert_query_contains(&query, "ALTER TABLE DynamicInContextLearningExample DELETE");
                assert!(
                    !query.contains("namespace"),
                    "Delete query should not filter by namespace"
                );
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
        let deleted = conn
            .delete_dicl_examples("fn1", "var1", None)
            .await
            .unwrap();
        assert_eq!(deleted, 3, "Should report 3 rows deleted");
    }

    #[tokio::test]
    async fn test_delete_dicl_examples_none_found() {
        let mut mock = MockClickHouseClient::new();

        // Count returns 0, so no delete should be issued
        mock.expect_run_query_synchronous().returning(|_, _| {
            Ok(ClickHouseResponse {
                response: "0\n".to_string(),
                metadata: ClickHouseResponseMetadata {
                    read_rows: 1,
                    written_rows: 0,
                },
            })
        });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
        let deleted = conn
            .delete_dicl_examples("fn1", "var1", None)
            .await
            .unwrap();
        assert_eq!(deleted, 0, "Should return 0 when no examples to delete");
    }
}
