//! Inference statistics endpoint for getting inference counts and feedback counts.

use axum::extract::{Path, Query, State};
use axum::{Json, debug_handler};
use futures::future::try_join;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::config::Config;
use crate::db::inference_stats::{
    CountByVariant, CountInferencesParams, CountInferencesWithDemonstrationFeedbacksParams,
    CountInferencesWithFeedbackParams, InferenceStatsQueries,
};
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the inference stats endpoint
#[derive(Debug, Deserialize)]
pub struct InferenceStatsQueryParams {
    /// Optional variant name to filter by
    pub variant_name: Option<String>,
    /// Optional grouping for the results
    pub group_by: Option<InferenceStatsGroupBy>,
}

/// Grouping options for inference statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum InferenceStatsGroupBy {
    /// Group by variant name
    Variant,
}

/// Response containing inference statistics
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct InferenceStatsResponse {
    /// The count of inferences for the function (and optionally variant)
    pub inference_count: u64,
    /// Counts grouped by variant (only present when group_by=variant)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats_by_variant: Option<Vec<InferenceStatsByVariant>>,
}

/// Inference stats for a variant
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct InferenceStatsByVariant {
    /// The variant name
    pub variant_name: String,
    /// Number of inferences for this variant
    pub inference_count: u64,
    /// ISO 8601 timestamp of the last inference
    pub last_used_at: String,
}

impl From<CountByVariant> for InferenceStatsByVariant {
    fn from(row: CountByVariant) -> Self {
        Self {
            variant_name: row.variant_name,
            inference_count: row.inference_count,
            last_used_at: row.last_used_at,
        }
    }
}

/// Query parameters for the feedback stats endpoint
#[derive(Debug, Deserialize)]
pub struct InferenceWithFeedbackStatsQueryParams {
    /// Optional threshold for curated inference filtering (float metrics only)
    #[serde(default)]
    pub threshold: f64,
}

/// Response containing inference stats with feedback statistics
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct InferenceWithFeedbackStatsResponse {
    /// Number of feedbacks for the metric
    pub feedback_count: u64,
    /// Number of inferences matching the metric threshold criteria
    pub inference_count: u64,
}

/// HTTP handler for the inference stats endpoint
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_inference_stats_handler",
    skip_all,
    fields(
        function_name = %function_name,
    )
)]
pub async fn get_inference_stats_handler(
    State(app_state): AppState,
    Path(function_name): Path<String>,
    Query(params): Query<InferenceStatsQueryParams>,
) -> Result<Json<InferenceStatsResponse>, Error> {
    Ok(Json(
        get_inference_stats(
            &app_state.config,
            &app_state.clickhouse_connection_info,
            &function_name,
            params,
        )
        .await?,
    ))
}

/// HTTP handler for the feedback stats endpoint
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_inference_with_feedback_stats_handler",
    skip_all,
    fields(
        function_name = %function_name,
        metric_name = %metric_name,
    )
)]
pub async fn get_inference_with_feedback_stats_handler(
    State(app_state): AppState,
    Path((function_name, metric_name)): Path<(String, String)>,
    Query(params): Query<InferenceWithFeedbackStatsQueryParams>,
) -> Result<Json<InferenceWithFeedbackStatsResponse>, Error> {
    Ok(Json(
        get_inference_with_feedback_stats(
            &app_state.config,
            &app_state.clickhouse_connection_info,
            function_name,
            metric_name,
            params,
        )
        .await?,
    ))
}

/// Core business logic for getting inference statistics
async fn get_inference_stats(
    config: &Config,
    clickhouse: &impl InferenceStatsQueries,
    function_name: &str,
    params: InferenceStatsQueryParams,
) -> Result<InferenceStatsResponse, Error> {
    // Get the function config to determine the function type
    let function = config.get_function(function_name)?;

    // If variant_name is provided, validate that it exists
    if let Some(ref variant_name) = params.variant_name
        && !function.variants().contains_key(variant_name)
    {
        return Err(ErrorDetails::UnknownVariant {
            name: variant_name.clone(),
        }
        .into());
    }

    // Standard count (optionally filtered by variant_name)
    let count_params = CountInferencesParams {
        function_name,
        function_type: function.config_type(),
        variant_name: params.variant_name.as_deref(),
    };

    // Handle group_by=variant case
    if let Some(InferenceStatsGroupBy::Variant) = params.group_by {
        let variant_rows = clickhouse.count_inferences_by_variant(count_params).await?;

        let inference_count = variant_rows.iter().map(|r| r.inference_count).sum();
        let stats_by_variant = variant_rows.into_iter().map(Into::into).collect();

        return Ok(InferenceStatsResponse {
            inference_count,
            stats_by_variant: Some(stats_by_variant),
        });
    }

    let inference_count = clickhouse
        .count_inferences_for_function(count_params)
        .await?;

    Ok(InferenceStatsResponse {
        inference_count,
        stats_by_variant: None,
    })
}

/// Core business logic for getting feedback statistics
async fn get_inference_with_feedback_stats(
    config: &Config,
    clickhouse_connection_info: &impl InferenceStatsQueries,
    function_name: String,
    metric_name: String,
    params: InferenceWithFeedbackStatsQueryParams,
) -> Result<InferenceWithFeedbackStatsResponse, Error> {
    // Get the function config (validates function exists)
    let function_config = config.get_function(&function_name)?;
    let function_type = function_config.config_type();

    // Demonstration feedbacks are simple
    // TODO(shuyangli): it's probably wrong that we're not distinguishing between the feedback type
    // ("demonstration") and the metric name, but we can fix it later.
    if metric_name == "demonstration" {
        let feedback_count = clickhouse_connection_info
            .count_inferences_with_demonstration_feedback(
                CountInferencesWithDemonstrationFeedbacksParams {
                    function_name: &function_name,
                    function_type,
                },
            )
            .await?;

        // Each inference has one demonstration feedback
        return Ok(InferenceWithFeedbackStatsResponse {
            feedback_count,
            inference_count: feedback_count,
        });
    }

    // Validate metric and get the metric info
    let metric_config = config.get_metric_or_err(&metric_name)?;

    // Get feedback and matching inference counts based on metric type
    let (feedback_count, inference_count) = try_join(
        clickhouse_connection_info.count_inferences_with_feedback(
            CountInferencesWithFeedbackParams {
                function_name: &function_name,
                function_type,
                metric_name: &metric_name,
                metric_config,
                metric_threshold: None,
            },
        ),
        clickhouse_connection_info.count_inferences_with_feedback(
            CountInferencesWithFeedbackParams {
                function_name: &function_name,
                function_type,
                metric_name: &metric_name,
                metric_config,
                metric_threshold: Some(params.threshold),
            },
        ),
    )
    .await?;

    Ok(InferenceWithFeedbackStatsResponse {
        feedback_count,
        inference_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ConfigFileGlob};
    use crate::db::clickhouse::MockClickHouseConnectionInfo;
    use crate::function::FunctionConfigType;
    use crate::testing::get_unit_test_gateway_handle;
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_get_inference_stats_function_not_found() {
        let config = Arc::new(Config::default());
        let gateway_handle = get_unit_test_gateway_handle(config);

        let params = InferenceStatsQueryParams {
            variant_name: None,
            group_by: None,
        };

        let result = get_inference_stats(
            &gateway_handle.app_state.config,
            &gateway_handle.app_state.clickhouse_connection_info,
            "nonexistent_function",
            params,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nonexistent_function"));
    }

    #[tokio::test]
    async fn test_get_inference_stats_variant_not_found() {
        // Create a config with a function but without the requested variant
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

        let gateway_handle = get_unit_test_gateway_handle(Arc::new(config));

        let params = InferenceStatsQueryParams {
            variant_name: Some("nonexistent_variant".to_string()),
            group_by: None,
        };

        let result = get_inference_stats(
            &gateway_handle.app_state.config,
            &gateway_handle.app_state.clickhouse_connection_info,
            "test_function",
            params,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nonexistent_variant"));
    }

    #[tokio::test]
    async fn test_get_inference_with_feedback_stats_unknown_function() {
        let config = Arc::new(Config::default());
        let gateway_handle = get_unit_test_gateway_handle(config);

        let params = InferenceWithFeedbackStatsQueryParams { threshold: 0.0 };

        let result = get_inference_with_feedback_stats(
            &gateway_handle.app_state.config,
            &gateway_handle.app_state.clickhouse_connection_info,
            "nonexistent_function".to_string(),
            "some_metric".to_string(),
            params,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nonexistent_function"));
    }

    #[tokio::test]
    async fn test_get_inference_with_feedback_stats_unknown_metric() {
        // Create a config with a function but no metrics
        let config_str = r#"
            [functions.test_function]
            type = "chat"

            [functions.test_function.variants.test_variant]
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

        let gateway_handle = get_unit_test_gateway_handle(Arc::new(config));

        let params = InferenceWithFeedbackStatsQueryParams { threshold: 0.0 };

        let result = get_inference_with_feedback_stats(
            &gateway_handle.app_state.config,
            &gateway_handle.app_state.clickhouse_connection_info,
            "test_function".to_string(),
            "nonexistent_metric".to_string(),
            params,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("nonexistent_metric"),
            "Error message should contain metric name, but got: {err}"
        );
    }

    #[tokio::test]
    async fn test_get_inference_stats_calls_clickhouse() {
        // Create a config with a function
        let config_str = r#"
            [functions.test_function]
            type = "chat"

            [functions.test_function.variants.test_variant]
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

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_stats_queries
            .expect_count_inferences_for_function()
            .withf(|params| {
                assert_eq!(params.function_name, "test_function");
                assert_eq!(params.function_type, FunctionConfigType::Chat);
                assert!(params.variant_name.is_none());
                true
            })
            .times(1)
            .returning(|_| Box::pin(async move { Ok(42) }));

        let params = InferenceStatsQueryParams {
            variant_name: None,
            group_by: None,
        };

        let result = get_inference_stats(&config, &mock_clickhouse, "test_function", params)
            .await
            .unwrap();

        assert_eq!(result.inference_count, 42);
        assert!(result.stats_by_variant.is_none());
    }
}
