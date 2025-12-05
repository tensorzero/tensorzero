//! Endpoint for returning the gateway config to the UI.
//!
//! This endpoint returns the serialized Config for use by the TensorZero UI.

use std::sync::Arc;

use axum::{Json, extract::State};

use crate::{config::Config, utils::gateway::AppState};

/// Handler for GET /internal/ui-config
///
/// Returns the full serialized Config for the UI to consume.
pub async fn ui_config_handler(State(app_state): AppState) -> Json<Arc<Config>> {
    Json(app_state.config)
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
    async fn test_ui_config_handler_returns_config_with_functions_and_metrics() {
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

        // Verify the returned config contains our function
        let returned_config = response.0;
        assert_eq!(returned_config.functions.len(), 1);
        assert!(returned_config.functions.contains_key("test_function"));
        let returned_function = returned_config.functions.get("test_function").unwrap();

        if let FunctionConfig::Chat(chat_config) = returned_function.as_ref() {
            assert_eq!(chat_config.description, Some("Test function".to_string()));
        } else {
            panic!("Expected Chat function config");
        }

        // Verify the returned config contains our metric
        assert_eq!(returned_config.metrics.len(), 1);
        assert!(returned_config.metrics.contains_key("test_metric"));
        let returned_metric = returned_config.metrics.get("test_metric").unwrap();
        assert_eq!(returned_metric.r#type, MetricConfigType::Boolean);
        assert_eq!(returned_metric.optimize, MetricConfigOptimize::Max);
        assert_eq!(returned_metric.level, MetricConfigLevel::Inference);
    }
}
