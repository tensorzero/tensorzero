//! Inference count endpoint for getting inference counts and feedback counts.

use axum::extract::{Path, Query, State};
use axum::{Json, debug_handler};
use futures::future::try_join;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::config::Config;
use crate::db::TimeWindow;
use crate::db::inference_count::{
    CountByVariant, CountInferencesParams, CountInferencesWithDemonstrationFeedbacksParams,
    CountInferencesWithFeedbackParams, FunctionInferenceCount,
    GetFunctionThroughputByVariantParams, InferenceCountQueries, VariantThroughput,
};
use crate::error::{Error, ErrorDetails};
use crate::feature_flags::ENABLE_POSTGRES_READ;
use crate::function::DEFAULT_FUNCTION_NAME;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the inference count endpoint
#[derive(Debug, Deserialize)]
pub struct InferenceCountQueryParams {
    /// Optional variant name to filter by
    pub variant_name: Option<String>,
    /// Optional grouping for the results
    pub group_by: Option<InferenceCountGroupBy>,
}

/// Grouping options for inference count
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum InferenceCountGroupBy {
    /// Group by variant name
    Variant,
}

/// Response containing inference count
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct InferenceCountResponse {
    /// The count of inferences for the function (and optionally variant)
    pub inference_count: u64,
    /// Counts grouped by variant (only present when group_by=variant)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_by_variant: Option<Vec<InferenceCountByVariant>>,
}

/// Inference count for a variant
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct InferenceCountByVariant {
    /// The variant name
    pub variant_name: String,
    /// Number of inferences for this variant
    pub inference_count: u64,
    /// ISO 8601 timestamp of the last inference
    pub last_used_at: String,
}

impl From<CountByVariant> for InferenceCountByVariant {
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
pub struct InferenceWithFeedbackCountQueryParams {
    /// Optional threshold for curated inference filtering (float metrics only)
    #[serde(default)]
    pub threshold: f64,
}

/// Response containing inference count with feedback count
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct InferenceWithFeedbackCountResponse {
    /// Number of feedbacks for the metric
    pub feedback_count: u64,
    /// Number of inferences matching the metric threshold criteria
    pub inference_count: u64,
}

/// Query parameters for the function throughput by variant endpoint
#[derive(Debug, Deserialize)]
pub struct FunctionThroughputByVariantQueryParams {
    /// Time granularity for grouping throughput data
    pub time_window: TimeWindow,
    /// Maximum number of time periods to return (default: 10)
    #[serde(default = "default_max_periods")]
    pub max_periods: u32,
}

fn default_max_periods() -> u32 {
    10
}

/// Response containing function throughput data grouped by variant and time period
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetFunctionThroughputByVariantResponse {
    /// Throughput data for each (period, variant) combination
    pub throughput: Vec<VariantThroughput>,
}

/// Response containing all functions with their inference counts
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListFunctionsWithInferenceCountResponse {
    /// List of functions with their inference counts, ordered by most recent inference
    pub functions: Vec<FunctionInferenceCount>,
}

/// HTTP handler for the inference count endpoint
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_inference_count_handler",
    skip_all,
    fields(
        function_name = %function_name,
    )
)]
pub async fn get_inference_count_handler(
    State(app_state): AppState,
    Path(function_name): Path<String>,
    Query(params): Query<InferenceCountQueryParams>,
) -> Result<Json<InferenceCountResponse>, Error> {
    let database: &(dyn InferenceCountQueries + Sync) = if ENABLE_POSTGRES_READ.get() {
        &app_state.postgres_connection_info
    } else {
        &app_state.clickhouse_connection_info
    };

    let response = get_inference_count(&app_state.config, database, &function_name, params).await?;
    Ok(Json(response))
}

/// HTTP handler for the feedback stats endpoint
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_inference_with_feedback_count_handler",
    skip_all,
    fields(
        function_name = %function_name,
        metric_name = %metric_name,
    )
)]
pub async fn get_inference_with_feedback_count_handler(
    State(app_state): AppState,
    Path((function_name, metric_name)): Path<(String, String)>,
    Query(params): Query<InferenceWithFeedbackCountQueryParams>,
) -> Result<Json<InferenceWithFeedbackCountResponse>, Error> {
    let database: &(dyn InferenceCountQueries + Sync) = if ENABLE_POSTGRES_READ.get() {
        &app_state.postgres_connection_info
    } else {
        &app_state.clickhouse_connection_info
    };

    let response = get_inference_with_feedback_count(
        &app_state.config,
        database,
        function_name,
        metric_name,
        params,
    )
    .await?;
    Ok(Json(response))
}

/// Core business logic for getting inference count
async fn get_inference_count(
    config: &Config,
    database: &(dyn InferenceCountQueries + Sync),
    function_name: &str,
    params: InferenceCountQueryParams,
) -> Result<InferenceCountResponse, Error> {
    // Get the function config to determine the function type
    let function = config.get_function(function_name)?;

    // If variant_name is provided, validate that it exists
    // Skip validation for default function since its variants are dynamic
    if let Some(ref variant_name) = params.variant_name
        && function_name != DEFAULT_FUNCTION_NAME
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
    if let Some(InferenceCountGroupBy::Variant) = params.group_by {
        let variant_rows = database.count_inferences_by_variant(count_params).await?;

        let inference_count = variant_rows.iter().map(|r| r.inference_count).sum();
        let count_by_variant = variant_rows.into_iter().map(Into::into).collect();

        return Ok(InferenceCountResponse {
            inference_count,
            count_by_variant: Some(count_by_variant),
        });
    }

    let inference_count = database.count_inferences_for_function(count_params).await?;

    Ok(InferenceCountResponse {
        inference_count,
        count_by_variant: None,
    })
}

/// Core business logic for getting feedback count
async fn get_inference_with_feedback_count(
    config: &Config,
    database: &(dyn InferenceCountQueries + Sync),
    function_name: String,
    metric_name: String,
    params: InferenceWithFeedbackCountQueryParams,
) -> Result<InferenceWithFeedbackCountResponse, Error> {
    // Get the function config (validates function exists)
    let function_config = config.get_function(&function_name)?;
    let function_type = function_config.config_type();

    // Demonstration feedbacks are simple
    // TODO(shuyangli): it's probably wrong that we're not distinguishing between the feedback type
    // ("demonstration") and the metric name, but we can fix it later.
    if metric_name == "demonstration" {
        let feedback_count = database
            .count_inferences_with_demonstration_feedback(
                CountInferencesWithDemonstrationFeedbacksParams {
                    function_name: &function_name,
                    function_type,
                },
            )
            .await?;

        // Each inference has one demonstration feedback
        return Ok(InferenceWithFeedbackCountResponse {
            feedback_count,
            inference_count: feedback_count,
        });
    }

    // Validate metric and get the metric info
    let metric_config = config.get_metric_or_err(&metric_name)?;

    // Get feedback and matching inference counts based on metric type
    let (feedback_count, inference_count) = try_join(
        database.count_inferences_with_feedback(CountInferencesWithFeedbackParams {
            function_name: &function_name,
            function_type,
            metric_name: &metric_name,
            metric_config,
            metric_threshold: None,
        }),
        database.count_inferences_with_feedback(CountInferencesWithFeedbackParams {
            function_name: &function_name,
            function_type,
            metric_name: &metric_name,
            metric_config,
            metric_threshold: Some(params.threshold),
        }),
    )
    .await?;

    Ok(InferenceWithFeedbackCountResponse {
        feedback_count,
        inference_count,
    })
}

/// HTTP handler for the function throughput by variant endpoint
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_function_throughput_by_variant_handler",
    skip_all,
    fields(function_name = %function_name),
)]
pub async fn get_function_throughput_by_variant_handler(
    State(state): State<AppStateData>,
    Path(function_name): Path<String>,
    Query(params): Query<FunctionThroughputByVariantQueryParams>,
) -> Result<Json<GetFunctionThroughputByVariantResponse>, Error> {
    let database: &(dyn InferenceCountQueries + Sync) = if ENABLE_POSTGRES_READ.get() {
        &state.postgres_connection_info
    } else {
        &state.clickhouse_connection_info
    };

    let response =
        get_function_throughput_by_variant(&state.config, database, &function_name, params).await?;

    Ok(Json(response))
}

/// Core business logic for getting function throughput by variant.
/// Validates the function exists and returns throughput data grouped by variant and time period.
pub async fn get_function_throughput_by_variant(
    config: &Config,
    database: &(dyn InferenceCountQueries + Sync),
    function_name: &str,
    params: FunctionThroughputByVariantQueryParams,
) -> Result<GetFunctionThroughputByVariantResponse, Error> {
    // Validate function exists
    config.get_function(function_name)?;

    let throughput = database
        .get_function_throughput_by_variant(GetFunctionThroughputByVariantParams {
            function_name,
            time_window: params.time_window,
            max_periods: params.max_periods,
        })
        .await?;

    Ok(GetFunctionThroughputByVariantResponse { throughput })
}

/// HTTP handler for listing all functions with their inference counts
#[debug_handler(state = AppStateData)]
#[instrument(name = "list_functions_with_inference_count_handler", skip_all)]
pub async fn list_functions_with_inference_count_handler(
    State(state): State<AppStateData>,
) -> Result<Json<ListFunctionsWithInferenceCountResponse>, Error> {
    let database: &(dyn InferenceCountQueries + Sync) = if ENABLE_POSTGRES_READ.get() {
        &state.postgres_connection_info
    } else {
        &state.clickhouse_connection_info
    };

    let response = list_functions_with_inference_count(database).await?;
    Ok(Json(response))
}

/// Core business logic for listing all functions with their inference counts
async fn list_functions_with_inference_count(
    database: &(dyn InferenceCountQueries + Sync),
) -> Result<ListFunctionsWithInferenceCountResponse, Error> {
    let functions = database.list_functions_with_inference_count().await?;

    Ok(ListFunctionsWithInferenceCountResponse { functions })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ConfigFileGlob};
    use crate::db::clickhouse::MockClickHouseConnectionInfo;
    use crate::db::postgres::PostgresConnectionInfo;
    use crate::function::FunctionConfigType;
    use crate::testing::get_unit_test_gateway_handle;
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_get_inference_count_function_not_found() {
        let config = Arc::new(Config::default());
        let gateway_handle = get_unit_test_gateway_handle(config);

        let params = InferenceCountQueryParams {
            variant_name: None,
            group_by: None,
        };

        let result = get_inference_count(
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
    async fn test_get_inference_count_variant_not_found() {
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

        let params = InferenceCountQueryParams {
            variant_name: Some("nonexistent_variant".to_string()),
            group_by: None,
        };

        let result = get_inference_count(
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
    async fn test_get_inference_with_feedback_count_unknown_function() {
        let config = Arc::new(Config::default());
        let gateway_handle = get_unit_test_gateway_handle(config);

        let params = InferenceWithFeedbackCountQueryParams { threshold: 0.0 };

        let result = get_inference_with_feedback_count(
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
    async fn test_get_inference_with_feedback_count_unknown_metric() {
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

        let params = InferenceWithFeedbackCountQueryParams { threshold: 0.0 };

        let result = get_inference_with_feedback_count(
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
    async fn test_get_inference_count_calls_clickhouse() {
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
            .inference_count_queries
            .expect_count_inferences_for_function()
            .withf(|params| {
                assert_eq!(params.function_name, "test_function");
                assert_eq!(params.function_type, FunctionConfigType::Chat);
                assert!(params.variant_name.is_none());
                true
            })
            .times(1)
            .returning(|_| Box::pin(async move { Ok(42) }));

        let params = InferenceCountQueryParams {
            variant_name: None,
            group_by: None,
        };

        let result = get_inference_count(&config, &mock_clickhouse, "test_function", params)
            .await
            .unwrap();

        assert_eq!(result.inference_count, 42);
        assert!(result.count_by_variant.is_none());
    }

    #[tokio::test]
    async fn test_list_functions_with_inference_count_calls_clickhouse() {
        use chrono::Utc;

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_count_queries
            .expect_list_functions_with_inference_count()
            .times(1)
            .returning(|| {
                Box::pin(async move {
                    Ok(vec![
                        FunctionInferenceCount {
                            function_name: "write_haiku".to_string(),
                            last_inference_timestamp: Utc::now(),
                            inference_count: 150,
                        },
                        FunctionInferenceCount {
                            function_name: "extract_entities".to_string(),
                            last_inference_timestamp: Utc::now(),
                            inference_count: 75,
                        },
                    ])
                })
            });

        let result = list_functions_with_inference_count(&mock_clickhouse)
            .await
            .unwrap();

        assert_eq!(result.functions.len(), 2);
        assert_eq!(result.functions[0].function_name, "write_haiku");
        assert_eq!(result.functions[0].inference_count, 150);
        assert_eq!(result.functions[1].function_name, "extract_entities");
        assert_eq!(result.functions[1].inference_count, 75);
    }

    #[tokio::test]
    async fn test_list_functions_with_inference_count_empty() {
        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_count_queries
            .expect_list_functions_with_inference_count()
            .times(1)
            .returning(|| Box::pin(async move { Ok(vec![]) }));

        let result = list_functions_with_inference_count(&mock_clickhouse)
            .await
            .unwrap();

        assert!(result.functions.is_empty());
    }

    #[tokio::test]
    async fn test_get_inference_count_default_function_skips_variant_validation() {
        // Default config includes tensorzero::default which has no variants in config
        // but should allow any variant_name without returning UnknownVariant error
        let config = Config::default();

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_count_queries
            .expect_count_inferences_for_function()
            .withf(|params| {
                assert_eq!(params.function_name, DEFAULT_FUNCTION_NAME);
                assert_eq!(params.function_type, FunctionConfigType::Chat);
                assert_eq!(params.variant_name, Some("openai::gpt-5-mini"));
                true
            })
            .times(1)
            .returning(|_| Box::pin(async move { Ok(10) }));

        let params = InferenceCountQueryParams {
            variant_name: Some("openai::gpt-5-mini".to_string()),
            group_by: None,
        };

        // This should NOT return UnknownVariant error
        let result = get_inference_count(&config, &mock_clickhouse, DEFAULT_FUNCTION_NAME, params)
            .await
            .unwrap();

        assert_eq!(result.inference_count, 10);
    }

    // Tests for ENABLE_POSTGRES_READ flag dispatch
    // Note: The business logic functions take `&impl InferenceCountQueries` and are database-agnostic.
    // The flag only affects which connection the handlers pass to the business logic.
    // These tests verify the Postgres implementation can be invoked through the same interface.

    #[tokio::test]
    async fn test_get_inference_count_with_postgres_disabled_returns_error() {
        // When Postgres is disabled, calling the Postgres implementation should return an error

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

        let postgres = PostgresConnectionInfo::new_disabled();
        let params = InferenceCountQueryParams {
            variant_name: None,
            group_by: None,
        };

        let result = get_inference_count(&config, &postgres, "test_function", params).await;

        assert!(
            result.is_err(),
            "Should return error when Postgres is disabled"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("disabled"),
            "Error should indicate database is disabled: {err}"
        );
    }

    #[tokio::test]
    async fn test_list_functions_with_postgres_disabled_returns_error() {
        let postgres = PostgresConnectionInfo::new_disabled();
        let result = list_functions_with_inference_count(&postgres).await;

        assert!(
            result.is_err(),
            "Should return error when Postgres is disabled"
        );
    }
}
