//! Inference statistics endpoint for getting inference counts.

use axum::extract::{Path, Query, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::db::clickhouse::inference_stats::CountInferencesParams;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the inference stats endpoint
#[derive(Debug, Deserialize)]
pub struct InferenceStatsQueryParams {
    /// Optional variant name to filter by
    pub variant_name: Option<String>,
}

/// Response containing inference statistics
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct InferenceStatsResponse {
    /// The count of inferences for the function (and optionally variant)
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
        get_inference_stats(app_state, &function_name, params).await?,
    ))
}

/// Core business logic for getting inference statistics
async fn get_inference_stats(
    AppStateData {
        config,
        clickhouse_connection_info,
        ..
    }: AppStateData,
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

    let count_params = CountInferencesParams {
        function_name,
        function_type: function.config_type(),
        variant_name: params.variant_name.as_deref(),
    };

    let inference_count = clickhouse_connection_info
        .count_inferences_for_function(count_params)
        .await?;

    Ok(InferenceStatsResponse { inference_count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ConfigFileGlob};
    use crate::testing::get_unit_test_gateway_handle;
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_get_inference_stats_function_not_found() {
        let config = Arc::new(Config::default());
        let gateway_handle = get_unit_test_gateway_handle(config);

        let params = InferenceStatsQueryParams { variant_name: None };

        let result = get_inference_stats(
            gateway_handle.app_state.clone(),
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
        };

        let result =
            get_inference_stats(gateway_handle.app_state.clone(), "test_function", params).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nonexistent_variant"));
    }
}
