//! Endpoint for returning the gateway config to the UI.
//!
//! This endpoint returns a UI-safe subset of the Config for use by the TensorZero UI.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{Json, extract::State};
use serde::Serialize;

use crate::{
    config::{Config, MetricConfig},
    evaluations::EvaluationConfig,
    function::FunctionConfig,
    tool::StaticToolConfig,
    utils::gateway::AppState,
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
}

/// Handler for GET /internal/ui_config
///
/// Returns a UI-safe subset of the Config.
#[expect(clippy::unused_async)]
pub async fn ui_config_handler(State(app_state): AppState) -> Json<UiConfig> {
    Json(UiConfig::from_config(&app_state.config))
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
