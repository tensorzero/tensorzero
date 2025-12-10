//! ClickHouse queries for inference statistics.

use std::collections::HashMap;

use async_trait::async_trait;

use super::ClickHouseConnectionInfo;
use super::select_queries::parse_count;
use crate::config::{MetricConfig, MetricConfigOptimize, MetricConfigType};
use crate::db::inference_stats::{
    CountInferencesParams, CountInferencesWithDemonstrationFeedbacksParams,
    CountInferencesWithFeedbackParams, InferenceStatsQueries,
};
use crate::error::Error;
use crate::function::FunctionConfigType;

/// Builds the SQL query for counting inferences.
fn build_count_inferences_query<'a>(
    params: &'a CountInferencesParams<'a>,
) -> (String, HashMap<&'a str, &'a str>) {
    let mut query_params = HashMap::new();
    query_params.insert("function_name", params.function_name);

    let table_name = params.function_type.table_name();

    let query = match params.variant_name {
        Some(variant_name) => {
            query_params.insert("variant_name", variant_name);
            format!(
                "SELECT COUNT() AS count
         FROM {table_name}
         WHERE function_name = {{function_name:String}}
           AND variant_name = {{variant_name:String}}
         FORMAT JSONEachRow"
            )
        }
        None => {
            format!(
                "SELECT COUNT() AS count
         FROM {table_name}
         WHERE function_name = {{function_name:String}}
         FORMAT JSONEachRow"
            )
        }
    };

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MetricConfigLevel;
    use crate::db::clickhouse::query_builder::test_util::{
        assert_query_contains, assert_query_does_not_contain,
    };

    #[test]
    fn test_build_count_inferences_query_chat_no_variant() {
        let params = CountInferencesParams {
            function_name: "write_haiku",
            function_type: FunctionConfigType::Chat,
            variant_name: None,
        };
        let (query, query_params) = build_count_inferences_query(&params);
        assert_query_contains(&query, "FROM ChatInference");
        assert_query_contains(&query, "function_name = {function_name:String}");
        assert_query_does_not_contain(&query, "variant_name");
        assert_eq!(query_params.len(), 1);
        assert_eq!(query_params.get("function_name"), Some(&"write_haiku"));
    }

    #[test]
    fn test_build_count_inferences_query_json_no_variant() {
        let params = CountInferencesParams {
            function_name: "extract_entities",
            function_type: FunctionConfigType::Json,
            variant_name: None,
        };
        let (query, query_params) = build_count_inferences_query(&params);
        assert_query_contains(&query, "FROM JsonInference");
        assert_query_contains(&query, "function_name = {function_name:String}");
        assert_query_does_not_contain(&query, "variant_name");
        assert_eq!(query_params.len(), 1);
        assert_eq!(query_params.get("function_name"), Some(&"extract_entities"));
    }

    #[test]
    fn test_build_count_inferences_query_chat_with_variant() {
        let params = CountInferencesParams {
            function_name: "write_haiku",
            function_type: FunctionConfigType::Chat,
            variant_name: Some("initial_prompt_gpt4o_mini"),
        };
        let (query, query_params) = build_count_inferences_query(&params);
        assert_query_contains(&query, "FROM ChatInference");
        assert_query_contains(&query, "function_name = {function_name:String}");
        assert_query_contains(&query, "variant_name = {variant_name:String}");
        assert_eq!(query_params.len(), 2);
        assert_eq!(query_params.get("function_name"), Some(&"write_haiku"));
        assert_eq!(
            query_params.get("variant_name"),
            Some(&"initial_prompt_gpt4o_mini")
        );
    }

    #[test]
    fn test_build_count_inferences_query_json_with_variant() {
        let params = CountInferencesParams {
            function_name: "extract_entities",
            function_type: FunctionConfigType::Json,
            variant_name: Some("gpt4o_initial_prompt"),
        };
        let (query, query_params) = build_count_inferences_query(&params);
        assert_query_contains(&query, "FROM JsonInference");
        assert_query_contains(&query, "function_name = {function_name:String}");
        assert_query_contains(&query, "variant_name = {variant_name:String}");
        assert_eq!(query_params.len(), 2);
        assert_eq!(query_params.get("function_name"), Some(&"extract_entities"));
        assert_eq!(
            query_params.get("variant_name"),
            Some(&"gpt4o_initial_prompt")
        );
    }

    /// Normalize whitespace and newlines in a query for comparison
    fn normalize_whitespace(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Assert that the query contains a section (ignoring whitespace and newline differences)
    fn assert_query_contains_normalized(query: &str, expected_section: &str) {
        let normalized_query = normalize_whitespace(query);
        let normalized_section = normalize_whitespace(expected_section);
        assert!(
            normalized_query.contains(&normalized_section),
            "Query does not contain expected section.\nExpected section: {normalized_section}\nFull query: {normalized_query}"
        );
    }

    #[test]
    fn test_build_count_metric_feedbacks_query_boolean() {
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Boolean,
            optimize: MetricConfigOptimize::Max,
            level: MetricConfigLevel::Inference,
        };

        let (query, query_params) = build_count_metric_feedbacks_query(
            "test_function",
            FunctionConfigType::Chat,
            "test_metric",
            &metric_config,
            None,
        );

        assert_query_contains_normalized(&query, "SELECT toUInt32(COUNT(*)) as count");
        assert_query_contains_normalized(&query, "FROM ChatInference i");
        assert_query_contains_normalized(&query, "FROM BooleanMetricFeedback");
        assert_query_contains_normalized(&query, "WHERE metric_name = {metric_name:String}");
        assert_query_contains_normalized(&query, "ON i.id = f.target_id AND f.rn = 1");
        assert_query_contains_normalized(&query, "WHERE i.function_name = {function_name:String}");
        assert_eq!(
            query_params.get("function_name"),
            Some(&"test_function".to_string())
        );
        assert_eq!(
            query_params.get("metric_name"),
            Some(&"test_metric".to_string())
        );
    }

    #[test]
    fn test_build_count_metric_feedbacks_query_float_episode_level() {
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Float,
            optimize: MetricConfigOptimize::Min,
            level: MetricConfigLevel::Episode,
        };

        let (query, _) = build_count_metric_feedbacks_query(
            "test_function",
            FunctionConfigType::Json,
            "test_metric",
            &metric_config,
            None,
        );

        assert_query_contains_normalized(&query, "FROM JsonInference i");
        assert_query_contains_normalized(&query, "FROM FloatMetricFeedback");
        assert_query_contains_normalized(&query, "ON i.episode_id = f.target_id AND f.rn = 1");
    }

    #[test]
    fn test_build_count_demonstration_feedbacks_query() {
        let params = CountInferencesWithDemonstrationFeedbacksParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
        };

        let (query, query_params) = build_count_demonstration_feedbacks_query(params);

        assert_query_contains_normalized(&query, "SELECT toUInt32(COUNT(*)) as count");
        assert_query_contains_normalized(&query, "FROM ChatInference i");
        assert_query_contains_normalized(&query, "FROM DemonstrationFeedback");
        assert_query_contains_normalized(&query, "ON i.id = f.inference_id AND f.rn = 1");
        assert_query_contains_normalized(&query, "WHERE i.function_name = {function_name:String}");
        assert_eq!(
            query_params.get("function_name"),
            Some(&"test_function".to_string())
        );
    }

    #[test]
    fn test_build_count_metric_feedbacks_query_curated_boolean_max() {
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Boolean,
            optimize: MetricConfigOptimize::Max,
            level: MetricConfigLevel::Inference,
        };

        let (query, _) = build_count_metric_feedbacks_query(
            "test_function",
            FunctionConfigType::Chat,
            "test_metric",
            &metric_config,
            Some(0.0),
        );

        assert_query_contains_normalized(&query, "AND value = 1");
    }

    #[test]
    fn test_build_count_metric_feedbacks_query_curated_boolean_min() {
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Boolean,
            optimize: MetricConfigOptimize::Min,
            level: MetricConfigLevel::Inference,
        };

        let (query, _) = build_count_metric_feedbacks_query(
            "test_function",
            FunctionConfigType::Chat,
            "test_metric",
            &metric_config,
            Some(0.0),
        );

        assert_query_contains_normalized(&query, "AND value = 0");
    }

    #[test]
    fn test_build_count_metric_feedbacks_query_curated_float_max() {
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Float,
            optimize: MetricConfigOptimize::Max,
            level: MetricConfigLevel::Inference,
        };

        let (query, query_params) = build_count_metric_feedbacks_query(
            "test_function",
            FunctionConfigType::Chat,
            "test_metric",
            &metric_config,
            Some(0.8),
        );

        assert_query_contains_normalized(&query, "AND value > {threshold:Float64}");
        assert_eq!(query_params.get("threshold"), Some(&"0.8".to_string()));
    }

    #[test]
    fn test_build_count_metric_feedbacks_query_curated_float_min() {
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Float,
            optimize: MetricConfigOptimize::Min,
            level: MetricConfigLevel::Inference,
        };

        let (query, query_params) = build_count_metric_feedbacks_query(
            "test_function",
            FunctionConfigType::Chat,
            "test_metric",
            &metric_config,
            Some(0.5),
        );

        assert_query_contains_normalized(&query, "AND value < {threshold:Float64}");
        assert_eq!(query_params.get("threshold"), Some(&"0.5".to_string()));
    }
}
