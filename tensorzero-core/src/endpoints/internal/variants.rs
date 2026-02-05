//! Variants endpoint for listing all configured variants with their usage statistics.

use std::collections::HashMap;

use axum::extract::{Path, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::db::inferences::{CountInferencesForFunctionParams, InferenceQueries};
use crate::error::Error;
use crate::feature_flags::ENABLE_POSTGRES_READ;
use crate::function::DEFAULT_FUNCTION_NAME;
use crate::utils::gateway::{AppState, AppStateData};

/// Statistics for a single variant
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct VariantStats {
    /// The variant name
    pub variant_name: String,
    /// The variant type (e.g., "chat_completion", "best_of_n_sampling")
    pub variant_type: String,
    /// The configured weight for this variant (None if not specified)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    /// Number of inferences for this variant
    pub inference_count: u64,
    /// ISO 8601 timestamp of the last inference, if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<String>,
}

/// Response containing all variants for a function with their statistics
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListVariantsResponse {
    /// List of all configured variants with their usage statistics
    pub variants: Vec<VariantStats>,
}

/// HTTP handler for listing variants with statistics
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "list_variants_handler",
    skip_all,
    fields(function_name = %function_name),
)]
pub async fn list_variants_handler(
    State(app_state): AppState,
    Path(function_name): Path<String>,
) -> Result<Json<ListVariantsResponse>, Error> {
    let database: &(dyn InferenceQueries + Sync) = if ENABLE_POSTGRES_READ.get() {
        &app_state.postgres_connection_info
    } else {
        &app_state.clickhouse_connection_info
    };

    let response = list_variants(&app_state, database, &function_name).await?;
    Ok(Json(response))
}

/// Core business logic for listing variants with statistics
async fn list_variants(
    app_state: &AppStateData,
    database: &(dyn InferenceQueries + Sync),
    function_name: &str,
) -> Result<ListVariantsResponse, Error> {
    let config = &app_state.config;

    // Get the function config (validates function exists)
    let function = config.get_function(function_name)?;

    // Query ClickHouse for inference counts by variant
    let count_params = CountInferencesForFunctionParams {
        function_name,
        function_type: function.config_type(),
        variant_name: None,
    };
    let variant_counts = database.count_inferences_by_variant(count_params).await?;

    // For the default function, variants are dynamic (model names from ClickHouse)
    // There's no config to use as source of truth, so we return observed variants only
    if function_name == DEFAULT_FUNCTION_NAME {
        let mut variants: Vec<VariantStats> = variant_counts
            .into_iter()
            .map(|row| VariantStats {
                variant_name: row.variant_name,
                variant_type: "chat_completion".to_string(),
                weight: None,
                inference_count: row.inference_count,
                last_used_at: Some(row.last_used_at),
            })
            .collect();
        variants.sort_by(|a, b| a.variant_name.cmp(&b.variant_name));
        return Ok(ListVariantsResponse { variants });
    }

    // Get all configured variants (source of truth)
    let configured_variants = function.variants();

    // Build a map of variant_name -> (inference_count, last_used_at) from ClickHouse data
    let counts_map: HashMap<String, (u64, String)> = variant_counts
        .into_iter()
        .map(|row| (row.variant_name, (row.inference_count, row.last_used_at)))
        .collect();

    // Merge config variants with ClickHouse data
    let mut variants: Vec<VariantStats> = configured_variants
        .iter()
        .map(|(name, info)| {
            let (inference_count, last_used_at) = counts_map
                .get(name)
                .map(|(count, last_used)| (*count, Some(last_used.clone())))
                .unwrap_or((0, None));

            VariantStats {
                variant_name: name.clone(),
                variant_type: info.inner.variant_type().to_string(),
                weight: info.inner.weight(),
                inference_count,
                last_used_at,
            }
        })
        .collect();

    // Sort by variant name for consistent ordering
    variants.sort_by(|a, b| a.variant_name.cmp(&b.variant_name));

    Ok(ListVariantsResponse { variants })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ConfigFileGlob};
    use crate::db::clickhouse::MockClickHouseConnectionInfo;
    use crate::db::inferences::CountByVariant;
    use crate::testing::get_unit_test_gateway_handle;
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_list_variants_merges_config_and_clickhouse() {
        // Create a config with two variants
        let config_str = r#"
            [functions.test_function]
            type = "chat"

            [functions.test_function.variants.variant_a]
            type = "chat_completion"
            model = "openai::gpt-4"
            weight = 0.7

            [functions.test_function.variants.variant_b]
            type = "chat_completion"
            model = "anthropic::claude-sonnet-4-5"
            weight = 0.3
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

        // Mock ClickHouse to return data for only variant_a
        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_queries
            .expect_count_inferences_by_variant()
            .times(1)
            .returning(|_| {
                Box::pin(async move {
                    Ok(vec![CountByVariant {
                        variant_name: "variant_a".to_string(),
                        inference_count: 100,
                        last_used_at: "2024-01-15T10:00:00Z".to_string(),
                    }])
                })
            });

        let result =
            list_variants(&gateway_handle.app_state, &mock_clickhouse, "test_function").await;

        assert!(result.is_ok(), "Expected success but got: {result:?}");
        let response = result.unwrap();

        assert_eq!(
            response.variants.len(),
            2,
            "Should return both configured variants"
        );

        // Find variant_a (has ClickHouse data)
        let variant_a = response
            .variants
            .iter()
            .find(|v| v.variant_name == "variant_a")
            .expect("variant_a should be present");
        assert_eq!(variant_a.inference_count, 100);
        assert_eq!(
            variant_a.last_used_at,
            Some("2024-01-15T10:00:00Z".to_string())
        );
        assert_eq!(variant_a.variant_type, "chat_completion");
        assert_eq!(variant_a.weight, Some(0.7));

        // Find variant_b (no ClickHouse data)
        let variant_b = response
            .variants
            .iter()
            .find(|v| v.variant_name == "variant_b")
            .expect("variant_b should be present");
        assert_eq!(
            variant_b.inference_count, 0,
            "Variant without inferences should have count 0"
        );
        assert_eq!(
            variant_b.last_used_at, None,
            "Variant without inferences should have None for last_used_at"
        );
        assert_eq!(variant_b.variant_type, "chat_completion");
        assert_eq!(variant_b.weight, Some(0.3));
    }

    #[tokio::test]
    async fn test_list_variants_default_function_returns_observed_variants() {
        let config = Arc::new(Config::default());
        let gateway_handle = get_unit_test_gateway_handle(config);

        // Mock ClickHouse to return observed model variants
        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_queries
            .expect_count_inferences_by_variant()
            .times(1)
            .returning(|_| {
                Box::pin(async move {
                    Ok(vec![
                        CountByVariant {
                            variant_name: "openai::gpt-4o".to_string(),
                            inference_count: 50,
                            last_used_at: "2024-01-15T10:00:00Z".to_string(),
                        },
                        CountByVariant {
                            variant_name: "anthropic::claude-sonnet-4-5".to_string(),
                            inference_count: 30,
                            last_used_at: "2024-01-14T08:00:00Z".to_string(),
                        },
                    ])
                })
            });

        let result = list_variants(
            &gateway_handle.app_state,
            &mock_clickhouse,
            DEFAULT_FUNCTION_NAME,
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(
            response.variants.len(),
            2,
            "Default function should return observed variants from ClickHouse"
        );

        // Variants should be sorted by name
        assert_eq!(
            response.variants[0].variant_name,
            "anthropic::claude-sonnet-4-5"
        );
        assert_eq!(response.variants[0].variant_type, "chat_completion");
        assert_eq!(response.variants[0].weight, None);
        assert_eq!(response.variants[0].inference_count, 30);

        assert_eq!(response.variants[1].variant_name, "openai::gpt-4o");
        assert_eq!(response.variants[1].variant_type, "chat_completion");
        assert_eq!(response.variants[1].weight, None);
        assert_eq!(response.variants[1].inference_count, 50);
    }

    #[tokio::test]
    async fn test_list_variants_unknown_function() {
        let config = Arc::new(Config::default());
        let gateway_handle = get_unit_test_gateway_handle(config);

        let mock_clickhouse = MockClickHouseConnectionInfo::new();

        let result =
            list_variants(&gateway_handle.app_state, &mock_clickhouse, "nonexistent").await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("nonexistent"),
            "Error should mention the function name"
        );
    }
}
