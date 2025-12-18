//! ClickHouse queries for inference statistics.

use std::collections::HashMap;

use async_trait::async_trait;

use super::ClickHouseConnectionInfo;
use super::select_queries::parse_count;
use crate::config::{MetricConfig, MetricConfigOptimize, MetricConfigType};
use crate::db::inference_stats::{
    CountByVariant, CountInferencesParams, CountInferencesWithDemonstrationFeedbacksParams,
    CountInferencesWithFeedbackParams, InferenceStatsQueries,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfigType;

/// Builds the SQL query for counting inferences.
fn build_count_inferences_query<'a>(
    params: &'a CountInferencesParams<'a>,
) -> (String, HashMap<&'a str, &'a str>) {
    let mut query_params = HashMap::new();
    query_params.insert("function_name", params.function_name);

    let variant_clause = match params.variant_name {
        Some(variant_name) => {
            query_params.insert("variant_name", variant_name);
            "AND variant_name = {variant_name:String}"
        }
        None => "",
    };

    let table_name = params.function_type.table_name();

    let query = format!(
        "SELECT COUNT() AS count
         FROM {table_name}
         WHERE function_name = {{function_name:String}}
           {variant_clause}
         FORMAT JSONEachRow"
    );

    (query, query_params)
}

/// Builds the SQL query for counting inferences grouped by variant.
fn build_count_inferences_by_variant_query<'a>(
    params: &'a CountInferencesParams<'a>,
) -> (String, HashMap<&'a str, &'a str>) {
    let mut query_params = HashMap::new();
    query_params.insert("function_name", params.function_name);

    let variant_clause = match params.variant_name {
        Some(variant_name) => {
            query_params.insert("variant_name", variant_name);
            "AND variant_name = {variant_name:String}"
        }
        None => "",
    };

    let table_name = params.function_type.table_name();

    let query = format!(
        "SELECT
            variant_name,
            COUNT() AS inference_count,
            formatDateTime(max(timestamp), '%Y-%m-%dT%H:%i:%S.000Z') AS last_used_at
        FROM {table_name}
        WHERE function_name = {{function_name:String}}
            {variant_clause}
        GROUP BY variant_name
        ORDER BY inference_count DESC
        FORMAT JSONEachRow"
    );

    (query, query_params)
}

/// Build query for counting feedbacks for a boolean/float metric.
/// If `metric_threshold` is Some, filters to only count feedbacks meeting the threshold criteria based on metric type and optimize direction.
fn build_count_metric_feedbacks_query(
    function_name: &str,
    function_type: FunctionConfigType,
    metric_name: &str,
    metric_config: &MetricConfig,
    metric_threshold: Option<f64>,
) -> (String, HashMap<String, String>) {
    let inference_table = function_type.table_name();
    let feedback_table = metric_config.r#type.to_clickhouse_table_name();
    let join_key = metric_config.level.inference_column_name();

    let mut query_params = HashMap::new();

    let value_condition = match metric_threshold {
        None => String::new(),
        Some(threshold) => match metric_config.r#type {
            MetricConfigType::Boolean => match metric_config.optimize {
                MetricConfigOptimize::Max => "AND value = 1",
                MetricConfigOptimize::Min => "AND value = 0",
            }
            .to_string(),
            MetricConfigType::Float => {
                query_params.insert("threshold".to_string(), threshold.to_string());

                let operator = match metric_config.optimize {
                    MetricConfigOptimize::Max => ">",
                    MetricConfigOptimize::Min => "<",
                };
                format!("AND value {operator} {{threshold:Float64}}")
            }
        },
    };

    let query = format!(
        r"SELECT toUInt32(COUNT(*)) as count
        FROM {inference_table} i
        JOIN (
            SELECT target_id, value,
                ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
            FROM {feedback_table}
            WHERE metric_name = {{metric_name:String}}
            {value_condition}
        ) f ON i.{join_key} = f.target_id AND f.rn = 1
        WHERE i.function_name = {{function_name:String}}
        FORMAT JSONEachRow"
    );

    query_params.insert("function_name".to_string(), function_name.to_string());
    query_params.insert("metric_name".to_string(), metric_name.to_string());

    (query, query_params)
}

/// Build query for counting demonstration feedbacks
fn build_count_demonstration_feedbacks_query(
    params: CountInferencesWithDemonstrationFeedbacksParams<'_>,
) -> (String, HashMap<String, String>) {
    let inference_table = params.function_type.table_name();

    let query = format!(
        r"SELECT toUInt32(COUNT(*)) as count
        FROM {inference_table} i
        JOIN (
            SELECT inference_id,
                ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn
            FROM DemonstrationFeedback
        ) f ON i.id = f.inference_id AND f.rn = 1
        WHERE i.function_name = {{function_name:String}}
        FORMAT JSONEachRow"
    );

    let mut query_params = HashMap::new();
    query_params.insert(
        "function_name".to_string(),
        params.function_name.to_string(),
    );

    (query, query_params)
}

#[async_trait]
impl InferenceStatsQueries for ClickHouseConnectionInfo {
    async fn count_inferences_for_function(
        &self,
        params: CountInferencesParams<'_>,
    ) -> Result<u64, Error> {
        let (query, query_params) = build_count_inferences_query(&params);
        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_count(&response.response)
    }

    async fn count_inferences_by_variant(
        &self,
        params: CountInferencesParams<'_>,
    ) -> Result<Vec<CountByVariant>, Error> {
        let (query, query_params) = build_count_inferences_by_variant_query(&params);
        let response = self.run_query_synchronous(query, &query_params).await?;

        let result: Vec<CountByVariant> = response
            .response
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                let datapoint: Result<CountByVariant, Error> =
                    serde_json::from_str(line).map_err(|e| {
                        Error::new(ErrorDetails::ClickHouseDeserialization {
                            message: format!("Failed to deserialize CountByVariant info: {e}"),
                        })
                    });
                datapoint
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(result)
    }

    async fn count_inferences_with_feedback(
        &self,
        params: CountInferencesWithFeedbackParams<'_>,
    ) -> Result<u64, Error> {
        let (query, params_owned) = build_count_metric_feedbacks_query(
            params.function_name,
            params.function_type,
            params.metric_name,
            params.metric_config,
            params.metric_threshold,
        );
        let query_params: HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_count(&response.response)
    }

    async fn count_inferences_with_demonstration_feedback(
        &self,
        params: CountInferencesWithDemonstrationFeedbacksParams<'_>,
    ) -> Result<u64, Error> {
        let (query, params_owned) = build_count_demonstration_feedbacks_query(params);
        let query_params: HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_count(&response.response)
    }

    async fn count_inferences_for_episode(&self, episode_id: uuid::Uuid) -> Result<u64, Error> {
        let mut query_params_owned = HashMap::new();
        query_params_owned.insert("episode_id".to_string(), episode_id.to_string());

        let query = "SELECT COUNT() AS count
             FROM InferenceByEpisodeId FINAL
             WHERE episode_id_uint = toUInt128(toUUID({episode_id:String}))
             FORMAT JSONEachRow"
            .to_string();

        let query_params: HashMap<&str, &str> = query_params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_count(&response.response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::config::MetricConfigLevel;
    use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
    use crate::db::clickhouse::query_builder::test_util::{
        assert_query_contains, assert_query_does_not_contain,
    };
    use crate::db::clickhouse::{ClickHouseResponse, ClickHouseResponseMetadata};

    #[tokio::test]
    async fn test_count_inferences_for_function_chat_no_variant() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT COUNT() AS count
                     FROM ChatInference
                     WHERE function_name = {function_name:String}",
                );
                assert_query_does_not_contain(query, "variant_name");
                assert_eq!(parameters.get("function_name"), Some(&"write_haiku"));
                assert_eq!(parameters.len(), 1);
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
        let params = CountInferencesParams {
            function_name: "write_haiku",
            function_type: FunctionConfigType::Chat,
            variant_name: None,
        };

        let result = conn.count_inferences_for_function(params).await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_count_inferences_for_function_json_no_variant() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT COUNT() AS count
                     FROM JsonInference
                     WHERE function_name = {function_name:String}",
                );
                assert_query_does_not_contain(query, "variant_name");
                assert_eq!(parameters.get("function_name"), Some(&"extract_entities"));
                assert_eq!(parameters.len(), 1);
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
        let params = CountInferencesParams {
            function_name: "extract_entities",
            function_type: FunctionConfigType::Json,
            variant_name: None,
        };

        let result = conn.count_inferences_for_function(params).await.unwrap();
        assert_eq!(result, 100);
    }

    #[tokio::test]
    async fn test_count_inferences_for_function_chat_with_variant() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT COUNT() AS count
                     FROM ChatInference
                     WHERE function_name = {function_name:String}
                     AND variant_name = {variant_name:String}",
                );
                assert_eq!(parameters.get("function_name"), Some(&"write_haiku"));
                assert_eq!(
                    parameters.get("variant_name"),
                    Some(&"initial_prompt_gpt4o_mini")
                );
                assert_eq!(parameters.len(), 2);
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":15}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let params = CountInferencesParams {
            function_name: "write_haiku",
            function_type: FunctionConfigType::Chat,
            variant_name: Some("initial_prompt_gpt4o_mini"),
        };

        let result = conn.count_inferences_for_function(params).await.unwrap();
        assert_eq!(result, 15);
    }

    #[tokio::test]
    async fn test_count_inferences_for_function_json_with_variant() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT COUNT() AS count
                     FROM JsonInference
                     WHERE function_name = {function_name:String}
                     AND variant_name = {variant_name:String}",
                );
                assert_eq!(parameters.get("function_name"), Some(&"extract_entities"));
                assert_eq!(
                    parameters.get("variant_name"),
                    Some(&"gpt4o_initial_prompt")
                );
                assert_eq!(parameters.len(), 2);
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":25}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let params = CountInferencesParams {
            function_name: "extract_entities",
            function_type: FunctionConfigType::Json,
            variant_name: Some("gpt4o_initial_prompt"),
        };

        let result = conn.count_inferences_for_function(params).await.unwrap();
        assert_eq!(result, 25);
    }

    #[tokio::test]
    async fn test_count_inferences_by_variant_chat() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT
                        variant_name,
                        COUNT() AS inference_count,
                        formatDateTime(max(timestamp), '%Y-%m-%dT%H:%i:%S.000Z') AS last_used_at
                    FROM ChatInference
                    WHERE function_name = {function_name:String}
                    GROUP BY variant_name
                    ORDER BY inference_count DESC",
                );
                assert_eq!(parameters.get("function_name"), Some(&"write_haiku"));
                assert_eq!(parameters.len(), 1);
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"variant_name":"variant_a","inference_count":30,"last_used_at":"2024-01-01T00:00:00.000Z"}
{"variant_name":"variant_b","inference_count":20,"last_used_at":"2024-01-01T00:00:00.000Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata { read_rows: 2, written_rows: 0 },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let params = CountInferencesParams {
            function_name: "write_haiku",
            function_type: FunctionConfigType::Chat,
            variant_name: None,
        };

        let result = conn.count_inferences_by_variant(params).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].variant_name, "variant_a");
        assert_eq!(result[0].inference_count, 30);
        assert_eq!(result[1].variant_name, "variant_b");
        assert_eq!(result[1].inference_count, 20);
    }

    #[tokio::test]
    async fn test_count_inferences_by_variant_json() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT
                        variant_name,
                        COUNT() AS inference_count,
                        formatDateTime(max(timestamp), '%Y-%m-%dT%H:%i:%S.000Z') AS last_used_at
                    FROM JsonInference
                    WHERE function_name = {function_name:String}
                    GROUP BY variant_name
                    ORDER BY inference_count DESC",
                );
                assert_eq!(parameters.get("function_name"), Some(&"extract_entities"));
                assert_eq!(parameters.len(), 1);
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"variant_name":"v1","inference_count":50,"last_used_at":"2024-01-01T00:00:00.000Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata { read_rows: 1, written_rows: 0 },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let params = CountInferencesParams {
            function_name: "extract_entities",
            function_type: FunctionConfigType::Json,
            variant_name: None,
        };

        let result = conn.count_inferences_by_variant(params).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].variant_name, "v1");
        assert_eq!(result[0].inference_count, 50);
    }

    #[tokio::test]
    async fn test_count_inferences_with_feedback_boolean() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT toUInt32(COUNT(*)) as count
                    FROM ChatInference i
                    JOIN (
                        SELECT target_id, value,
                            ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
                        FROM BooleanMetricFeedback
                        WHERE metric_name = {metric_name:String}
                    ) f ON i.id = f.target_id AND f.rn = 1
                    WHERE i.function_name = {function_name:String}",
                );
                assert_eq!(parameters.get("function_name"), Some(&"test_function"));
                assert_eq!(parameters.get("metric_name"), Some(&"test_metric"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":10}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Boolean,
            optimize: MetricConfigOptimize::Max,
            level: MetricConfigLevel::Inference,
        };
        let params = CountInferencesWithFeedbackParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "test_metric",
            metric_config: &metric_config,
            metric_threshold: None,
        };

        let result = conn.count_inferences_with_feedback(params).await.unwrap();
        assert_eq!(result, 10);
    }

    #[tokio::test]
    async fn test_count_inferences_with_feedback_float_episode_level() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _parameters| {
                assert_query_contains(
                    query,
                    "FROM JsonInference i
                    JOIN (
                        SELECT target_id, value,
                            ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
                        FROM FloatMetricFeedback
                        WHERE metric_name = {metric_name:String}
                    ) f ON i.episode_id = f.target_id AND f.rn = 1",
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":5}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Float,
            optimize: MetricConfigOptimize::Min,
            level: MetricConfigLevel::Episode,
        };
        let params = CountInferencesWithFeedbackParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Json,
            metric_name: "test_metric",
            metric_config: &metric_config,
            metric_threshold: None,
        };

        let result = conn.count_inferences_with_feedback(params).await.unwrap();
        assert_eq!(result, 5);
    }

    #[tokio::test]
    async fn test_count_inferences_with_demonstration_feedback() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT toUInt32(COUNT(*)) as count
                    FROM ChatInference i
                    JOIN (
                        SELECT inference_id,
                            ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn
                        FROM DemonstrationFeedback
                    ) f ON i.id = f.inference_id AND f.rn = 1
                    WHERE i.function_name = {function_name:String}",
                );
                assert_eq!(parameters.get("function_name"), Some(&"test_function"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":7}"#.to_string(),
                    metadata: ClickHouseResponseMetadata { read_rows: 1, written_rows: 0 },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let params = CountInferencesWithDemonstrationFeedbacksParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
        };

        let result = conn
            .count_inferences_with_demonstration_feedback(params)
            .await
            .unwrap();
        assert_eq!(result, 7);
    }

    #[tokio::test]
    async fn test_count_inferences_with_feedback_curated_boolean_max() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _parameters| {
                assert_query_contains(
                    query,
                    "WHERE metric_name = {metric_name:String}
                    AND value = 1",
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":3}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Boolean,
            optimize: MetricConfigOptimize::Max,
            level: MetricConfigLevel::Inference,
        };
        let params = CountInferencesWithFeedbackParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "test_metric",
            metric_config: &metric_config,
            metric_threshold: Some(0.0),
        };

        let result = conn.count_inferences_with_feedback(params).await.unwrap();
        assert_eq!(result, 3);
    }

    #[tokio::test]
    async fn test_count_inferences_with_feedback_curated_boolean_min() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _parameters| {
                assert_query_contains(
                    query,
                    "WHERE metric_name = {metric_name:String}
                    AND value = 0",
                );
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
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Boolean,
            optimize: MetricConfigOptimize::Min,
            level: MetricConfigLevel::Inference,
        };
        let params = CountInferencesWithFeedbackParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "test_metric",
            metric_config: &metric_config,
            metric_threshold: Some(0.0),
        };

        let result = conn.count_inferences_with_feedback(params).await.unwrap();
        assert_eq!(result, 2);
    }

    #[tokio::test]
    async fn test_count_inferences_with_feedback_curated_float_max() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "WHERE metric_name = {metric_name:String}
                    AND value > {threshold:Float64}",
                );
                assert_eq!(parameters.get("threshold"), Some(&"0.8"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":8}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Float,
            optimize: MetricConfigOptimize::Max,
            level: MetricConfigLevel::Inference,
        };
        let params = CountInferencesWithFeedbackParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "test_metric",
            metric_config: &metric_config,
            metric_threshold: Some(0.8),
        };

        let result = conn.count_inferences_with_feedback(params).await.unwrap();
        assert_eq!(result, 8);
    }

    #[tokio::test]
    async fn test_count_inferences_with_feedback_curated_float_min() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "WHERE metric_name = {metric_name:String}
                    AND value < {threshold:Float64}",
                );
                assert_eq!(parameters.get("threshold"), Some(&"0.5"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":4}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Float,
            optimize: MetricConfigOptimize::Min,
            level: MetricConfigLevel::Inference,
        };
        let params = CountInferencesWithFeedbackParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "test_metric",
            metric_config: &metric_config,
            metric_threshold: Some(0.5),
        };

        let result = conn.count_inferences_with_feedback(params).await.unwrap();
        assert_eq!(result, 4);
    }

    #[tokio::test]
    async fn test_count_inferences_for_episode_query() {
        let episode_id = uuid::Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT COUNT() AS count
                     FROM InferenceByEpisodeId FINAL
                     WHERE episode_id_uint = toUInt128(toUUID({episode_id:String}))
                     FORMAT JSONEachRow",
                );
                assert_eq!(
                    parameters.get("episode_id"),
                    Some(&episode_id.to_string().as_str())
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":30}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let count = conn.count_inferences_for_episode(episode_id).await.unwrap();
        assert_eq!(count, 30);
    }
}
