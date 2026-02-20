use std::collections::HashMap;

use axum::extract::{Path, Query, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::config::Config;
use crate::db::postgres::PostgresConnectionInfo;
use crate::error::{Error, ErrorDetails};
use crate::function::DEFAULT_FUNCTION_NAME;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the variant sampling probabilities endpoint
#[derive(Debug, Deserialize)]
pub struct GetVariantSamplingProbabilitiesParams {
    /// The name of the function to get probabilities for
    pub function_name: String,
}

/// Response containing variant sampling probabilities
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetVariantSamplingProbabilitiesResponse {
    /// Map of variant names to their sampling probabilities (0.0 to 1.0)
    /// Probabilities sum to 1.0
    pub probabilities: HashMap<String, f64>,
}

/// HTTP handler for the variant sampling probabilities endpoint (query-based)
#[debug_handler(state = AppStateData)]
pub async fn get_variant_sampling_probabilities_handler(
    State(app_state): AppState,
    Query(params): Query<GetVariantSamplingProbabilitiesParams>,
) -> Result<Json<GetVariantSamplingProbabilitiesResponse>, Error> {
    Ok(Json(
        get_variant_sampling_probabilities(
            &app_state.config,
            &app_state.postgres_connection_info,
            params,
        )
        .await?,
    ))
}

/// HTTP handler for the variant sampling probabilities endpoint (path-based)
#[debug_handler(state = AppStateData)]
pub async fn get_variant_sampling_probabilities_by_function_handler(
    State(app_state): AppState,
    Path(function_name): Path<String>,
) -> Result<Json<GetVariantSamplingProbabilitiesResponse>, Error> {
    let params = GetVariantSamplingProbabilitiesParams { function_name };
    Ok(Json(
        get_variant_sampling_probabilities(
            &app_state.config,
            &app_state.postgres_connection_info,
            params,
        )
        .await?,
    ))
}

/// Core business logic for getting variant sampling probabilities
#[instrument(
    name = "get_variant_sampling_probabilities",
    skip_all,
    fields(
        function_name = %params.function_name,
    )
)]
pub async fn get_variant_sampling_probabilities(
    config: &Config,
    postgres_connection_info: &PostgresConnectionInfo,
    params: GetVariantSamplingProbabilitiesParams,
) -> Result<GetVariantSamplingProbabilitiesResponse, Error> {
    let function_name = &params.function_name;

    // Default function has no variants, so return an empty response
    if function_name == DEFAULT_FUNCTION_NAME {
        return Ok(GetVariantSamplingProbabilitiesResponse {
            probabilities: HashMap::new(),
        });
    }

    // Get the function config
    let function = config.get_function(function_name)?;

    // If the function has no variants, return an error
    if function.variants().is_empty() {
        return Err(ErrorDetails::InvalidFunctionVariants {
            message: format!("Function `{function_name}` has no variants"),
        }
        .into());
    }

    // Get the current display probabilities from the experimentation config
    let probabilities = function
        .experimentation()
        .get_current_display_probabilities(
            function_name,
            function.variants(),
            postgres_connection_info,
        )?;

    // Convert HashMap<&str, f64> to HashMap<String, f64>
    let probabilities: HashMap<String, f64> = probabilities
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

    Ok(GetVariantSamplingProbabilitiesResponse { probabilities })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ConfigFileGlob};
    use crate::db::postgres::PostgresConnectionInfo;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_get_variant_sampling_probabilities_static_weights() {
        let config_str = r#"
            [functions.test_function]
            type = "chat"
            [functions.test_function.experimentation]
            type = "static_weights"
            candidate_variants = {"variant_a" = 0.7, "variant_b" = 0.3}
            fallback_variants = ["variant_a", "variant_b"]

            [functions.test_function.variants.variant_a]
            type = "chat_completion"
            model = "openai::gpt-4"

            [functions.test_function.variants.variant_b]
            type = "chat_completion"
            model = "anthropic::claude-sonnet-4-5"
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

        let postgres = PostgresConnectionInfo::new_mock(true);

        let params = GetVariantSamplingProbabilitiesParams {
            function_name: "test_function".to_string(),
        };

        let response = get_variant_sampling_probabilities(&config, &postgres, params)
            .await
            .unwrap();

        assert_eq!(response.probabilities.len(), 2);
        assert!(response.probabilities.contains_key("variant_a"));
        assert!(response.probabilities.contains_key("variant_b"));

        let prob_a = response.probabilities.get("variant_a").unwrap();
        let prob_b = response.probabilities.get("variant_b").unwrap();
        assert!((prob_a - 0.7).abs() < 1e-9);
        assert!((prob_b - 0.3).abs() < 1e-9);

        let sum: f64 = response.probabilities.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_get_variant_sampling_probabilities_default_function_returns_empty() {
        let config = Config::default();
        let postgres = PostgresConnectionInfo::new_mock(true);

        let params = GetVariantSamplingProbabilitiesParams {
            function_name: "tensorzero::default".to_string(),
        };

        let response = get_variant_sampling_probabilities(&config, &postgres, params)
            .await
            .unwrap();
        assert_eq!(response.probabilities.len(), 0);
    }

    #[tokio::test]
    async fn test_get_variant_sampling_probabilities_no_function() {
        let config = Config::default();
        let postgres = PostgresConnectionInfo::new_mock(true);

        let params = GetVariantSamplingProbabilitiesParams {
            function_name: "nonexistent_function".to_string(),
        };

        let result = get_variant_sampling_probabilities(&config, &postgres, params).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nonexistent_function"));
    }
}
