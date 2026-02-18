//! Endpoint for returning the gateway config to the UI.
//!
//! This endpoint returns a UI-safe subset of the Config for use by the TensorZero UI.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use serde::Serialize;

use crate::{
    config::snapshot::{ConfigSnapshot, SnapshotHash},
    config::{Config, MetricConfig, UninitializedConfig},
    db::ConfigQueries,
    db::delegating_connection::DelegatingDatabaseConnection,
    error::{Error, ErrorDetails},
    evaluations::EvaluationConfig,
    function::FunctionConfig,
    tool::StaticToolConfig,
    utils::gateway::{AppState, AppStateData},
};

/// Response type for GET /internal/ui_config
///
/// Contains only UI-safe fields from the gateway config, excluding sensitive
/// information like provider credentials, API keys, and internal settings.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct UiConfig {
    pub functions: HashMap<String, Arc<FunctionConfig>>,
    pub metrics: HashMap<String, MetricConfig>,
    pub tools: HashMap<String, Arc<StaticToolConfig>>,
    pub evaluations: HashMap<String, Arc<EvaluationConfig>>,
    pub model_names: Vec<String>,
    pub config_hash: String,
}

impl UiConfig {
    pub fn from_config(config: &Config) -> Self {
        Self {
            functions: config
                .functions
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(v)))
                .collect(),
            metrics: config.metrics.clone(),
            tools: config
                .tools
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(v)))
                .collect(),
            evaluations: config
                .evaluations
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(v)))
                .collect(),
            model_names: config.models.table.keys().map(|s| s.to_string()).collect(),
            config_hash: config.hash.to_string(),
        }
    }

    /// Creates a `UiConfig` from a historical config snapshot.
    ///
    /// This initializes only the parts needed by the UI (functions, tools, evaluations,
    /// metrics, model names), skipping heavy initialization like model credentials, HTTP
    /// clients, gateway config, object store, and rate limiting.
    pub fn from_snapshot(snapshot: ConfigSnapshot) -> Result<Self, Error> {
        let hash = snapshot.hash.to_string();
        let uninit_config: UninitializedConfig =
            snapshot.config.try_into().map_err(|e: &'static str| {
                Error::new(ErrorDetails::Config {
                    message: e.to_string(),
                })
            })?;

        let UninitializedConfig {
            models,
            embedding_models: _,
            functions,
            metrics,
            tools,
            evaluations,
            gateway: _,
            postgres: _,
            rate_limiting: _,
            object_storage: _,
            provider_types: _,
            optimizers: _,
        } = uninit_config;

        // Load functions (sync, no FS/network — file data embedded in ResolvedTomlPathData)
        let loaded_functions: HashMap<String, Arc<FunctionConfig>> = functions
            .into_iter()
            .map(|(name, func)| func.load(&name, &metrics).map(|c| (name, Arc::new(c))))
            .collect::<Result<_, _>>()?;

        // Load tools (sync, same reason)
        let loaded_tools: HashMap<String, Arc<StaticToolConfig>> = tools
            .into_iter()
            .map(|(name, tool)| tool.load(name.clone()).map(|c| (name, Arc::new(c))))
            .collect::<Result<_, _>>()?;

        // Load evaluations (sync, needs loaded functions)
        // Also collects generated evaluation functions and metrics
        let mut all_functions = loaded_functions;
        let mut all_metrics = metrics;
        let mut loaded_evaluations = HashMap::new();
        for (name, eval_config) in evaluations {
            let (eval, eval_functions, eval_metrics) = eval_config.load(&all_functions, &name)?;
            loaded_evaluations.insert(name, Arc::new(EvaluationConfig::Inference(eval)));
            all_functions.extend(eval_functions);
            all_metrics.extend(eval_metrics);
        }

        // Model names — just keys, no initialization (only inference models, matching from_config)
        let model_names: Vec<String> = models.keys().map(|s| s.to_string()).collect();

        Ok(Self {
            functions: all_functions,
            metrics: all_metrics,
            tools: loaded_tools,
            evaluations: loaded_evaluations,
            model_names,
            config_hash: hash,
        })
    }
}

/// Handler for GET /internal/ui_config
///
/// Returns a UI-safe subset of the Config.
#[expect(clippy::unused_async)]
pub async fn ui_config_handler(State(app_state): AppState) -> Json<UiConfig> {
    Json(UiConfig::from_config(&app_state.config))
}

/// Handler for GET /internal/ui_config/{hash}
///
/// Returns a UI-safe subset of the Config for a historical config snapshot.
#[axum::debug_handler(state = AppStateData)]
pub async fn ui_config_by_hash_handler(
    State(app_state): AppState,
    Path(hash): Path<String>,
) -> Result<Json<UiConfig>, Error> {
    let snapshot_hash: SnapshotHash = hash.parse().map_err(|_| {
        Error::new(ErrorDetails::ConfigSnapshotNotFound {
            snapshot_hash: hash.clone(),
        })
    })?;

    let db = DelegatingDatabaseConnection::new(
        app_state.clickhouse_connection_info.clone(),
        app_state.postgres_connection_info.clone(),
    );
    let snapshot = db.get_config_snapshot(snapshot_hash).await?;

    Ok(Json(UiConfig::from_snapshot(snapshot)?))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use axum::extract::State;

    use crate::config::{
        Config, MetricConfig, MetricConfigLevel, MetricConfigOptimize, MetricConfigType,
    };
    use crate::function::{FunctionConfig, FunctionConfigChat};
    use crate::testing::get_unit_test_gateway_handle;

    use super::*;

    #[tokio::test]
    async fn test_ui_config_handler_returns_ui_config_with_functions_and_metrics() {
        // Create a function config
        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: Default::default(),
            tools: vec![],
            tool_choice: Default::default(),
            parallel_tool_calls: None,
            description: Some("Test function".to_string()),
            experimentation: Default::default(),
            all_explicit_templates_names: Default::default(),
        });

        // Create a metric config
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Boolean,
            optimize: MetricConfigOptimize::Max,
            level: MetricConfigLevel::Inference,
            description: None,
        };

        // Build config with the function and metric
        let mut config = Config::default();
        config
            .functions
            .insert("test_function".to_string(), Arc::new(function_config));
        config
            .metrics
            .insert("test_metric".to_string(), metric_config);

        let config = Arc::new(config);
        let gateway_handle = get_unit_test_gateway_handle(config.clone());

        let response = ui_config_handler(State(gateway_handle.app_state.clone())).await;

        // Verify the returned UiConfig contains our function
        let ui_config = response.0;
        assert_eq!(ui_config.functions.len(), 1);
        assert!(ui_config.functions.contains_key("test_function"));
        let returned_function = ui_config.functions.get("test_function").unwrap();

        if let FunctionConfig::Chat(chat_config) = returned_function.as_ref() {
            assert_eq!(chat_config.description, Some("Test function".to_string()));
        } else {
            panic!("Expected Chat function config");
        }

        // Verify the returned UiConfig contains our metric
        assert_eq!(ui_config.metrics.len(), 1);
        assert!(ui_config.metrics.contains_key("test_metric"));
        let returned_metric = ui_config.metrics.get("test_metric").unwrap();
        assert_eq!(returned_metric.r#type, MetricConfigType::Boolean);
        assert_eq!(returned_metric.optimize, MetricConfigOptimize::Max);
        assert_eq!(returned_metric.level, MetricConfigLevel::Inference);

        // Verify model_names is empty (default config has no models)
        assert!(ui_config.model_names.is_empty());

        // Verify tools and evaluations are empty
        assert!(ui_config.tools.is_empty());
        assert!(ui_config.evaluations.is_empty());

        // Verify config_hash is present
        assert!(!ui_config.config_hash.is_empty());
    }

    #[test]
    fn test_ui_config_from_config_extracts_correct_fields() {
        // Create a function config
        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: Default::default(),
            tools: vec![],
            tool_choice: Default::default(),
            parallel_tool_calls: None,
            description: Some("My function".to_string()),
            experimentation: Default::default(),
            all_explicit_templates_names: Default::default(),
        });

        // Create a metric config
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Float,
            optimize: MetricConfigOptimize::Min,
            level: MetricConfigLevel::Episode,
            description: None,
        };

        let mut config = Config::default();
        config
            .functions
            .insert("my_function".to_string(), Arc::new(function_config));
        config
            .metrics
            .insert("my_metric".to_string(), metric_config);

        let ui_config = UiConfig::from_config(&config);

        // Verify functions are copied correctly
        assert_eq!(ui_config.functions.len(), 1);
        let func = ui_config.functions.get("my_function").unwrap();
        if let FunctionConfig::Chat(chat_config) = func.as_ref() {
            assert_eq!(chat_config.description, Some("My function".to_string()));
        } else {
            panic!("Expected Chat function config");
        }

        // Verify metrics are copied correctly
        assert_eq!(ui_config.metrics.len(), 1);
        let metric = ui_config.metrics.get("my_metric").unwrap();
        assert_eq!(metric.r#type, MetricConfigType::Float);
        assert_eq!(metric.optimize, MetricConfigOptimize::Min);
        assert_eq!(metric.level, MetricConfigLevel::Episode);

        // Verify config_hash is present
        assert!(!ui_config.config_hash.is_empty());
    }
}
