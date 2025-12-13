//! Functions endpoint for querying function-level information

use axum::extract::{Path, Query, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::config::Config;
use crate::db::feedback::{FeedbackQueries, MetricWithFeedback};
use crate::error::Error;
use crate::function::get_function;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the metrics endpoint
#[derive(Debug, Deserialize)]
pub struct MetricsQueryParams {
    /// Optional variant name to filter by
    pub variant_name: Option<String>,
}

/// Response containing metrics with feedback statistics
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct MetricsWithFeedbackResponse {
    /// Metrics with feedback statistics
    pub metrics: Vec<MetricWithFeedback>,
}

/// HTTP handler for getting metrics with feedback for a function
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_function_metrics_handler",
    skip_all,
    fields(
        function_name = %function_name,
    )
)]
pub async fn get_function_metrics_handler(
    State(app_state): AppState,
    Path(function_name): Path<String>,
    Query(params): Query<MetricsQueryParams>,
) -> Result<Json<MetricsWithFeedbackResponse>, Error> {
    let response = get_function_metrics(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        &function_name,
        params.variant_name.as_deref(),
    )
    .await?;
    Ok(Json(response))
}

/// Core business logic for getting metrics with feedback
pub async fn get_function_metrics(
    config: &Config,
    clickhouse: &impl FeedbackQueries,
    function_name: &str,
    variant_name: Option<&str>,
) -> Result<MetricsWithFeedbackResponse, Error> {
    // Get function config to determine the inference table
    let function_config = get_function(&config.functions, function_name)?;
    let inference_table = function_config.table_name();

    // Query metrics with feedback
    let metrics = clickhouse
        .query_metrics_with_feedback(function_name, inference_table, variant_name)
        .await?;
    Ok(MetricsWithFeedbackResponse { metrics })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ConfigFileGlob};
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_get_function_metrics_function_not_found() {
        let config = Config::default();
        // Empty config with no functions should return error
        // We can't actually test this without a real ClickHouse connection,
        // so we just test that get_function fails
        let result = get_function(&config.functions, "nonexistent_function");

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nonexistent_function"));
    }

    #[tokio::test]
    async fn test_get_function_determines_correct_inference_table_for_chat() {
        let config_str = r#"
            [functions.test_function]
            type = "chat"

            [functions.test_function.variants.variant_a]
            type = "chat_completion"
            model = "openai::gpt-4"
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let config = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests();

        // Test that we can get the function and the table name is correct
        let function_config = get_function(&config.functions, "test_function").unwrap();
        assert_eq!(function_config.table_name(), "ChatInference");
    }

    #[tokio::test]
    async fn test_get_function_determines_correct_inference_table_for_json() {
        let config_str = r#"
            [functions.test_function]
            type = "json"

            [functions.test_function.variants.variant_a]
            type = "chat_completion"
            model = "openai::gpt-4"
            json_mode = "implicit_tool"
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let config = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests();

        // Test that we can get the function and the table name is correct
        let function_config = get_function(&config.functions, "test_function").unwrap();
        assert_eq!(function_config.table_name(), "JsonInference");
    }
}
