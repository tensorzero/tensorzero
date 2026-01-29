//! Models API endpoint for Anthropic-compatible API.
//!
//! This module provides the `/anthropic/v1/models` endpoint that returns a list of
//! configured models available in the TensorZero gateway. It follows the Anthropic
//! API specification for model listing.
//!
//! # Example
//!
//! ```rust,no_run
//! use reqwest::Client;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let response = Client::new()
//!     .get("http://localhost:3000/anthropic/v1/models")
//!     .send()
//!     .await?;
//!
//! let models = response.json::<serde_json::Value>().await?;
//! println!("Available models: {}", models);
//! # Ok(())
//! # }
//! ```
//!
//! # Model IDs
//!
//! Models are returned with prefixes indicating their type:
//! - `tensorzero::function_name::{name}`: TensorZero functions
//! - `tensorzero::model_name::{name}`: Direct provider models
//!
//! # Response Format
//!
//! ```json
//! {
//!   "data": [
//!     {
//!       "id": "tensorzero::function_name::my_function",
//!       "name": "my_function",
//!       "type": "model"
//!     }
//!   ],
//!   "object": "list"
//! }
//! ```

use axum::extract::State;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Prefix for function_name models
const FUNCTION_NAME_PREFIX: &str = "tensorzero::function_name::";

/// Prefix for model_name models
const MODEL_NAME_PREFIX: &str = "tensorzero::model_name::";

/// Individual model information for the Models API response
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub struct AnthropicModel {
    /// Unique identifier for the model (function name or model name)
    pub id: String,
    /// Display name for the model
    pub name: String,
    /// Type of resource (always "model")
    #[serde(rename = "type")]
    pub resource_type: String,
}

/// Response for GET /anthropic/v1/models
///
/// Returns a list of available models that can be used with the Messages API.
/// Models are returned in the format expected by the Anthropic API.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnthropicModelsResponse {
    /// List of available models (both functions and providers)
    pub data: Vec<AnthropicModel>,
    /// Object type (always "list" for consistency with Anthropic API)
    pub object: String,
}

/// Handler for `GET /anthropic/v1/models`
///
/// Returns all configured models (functions and providers) that can be used
/// with the Anthropic-compatible Messages API.
///
/// # Behavior
///
/// This endpoint:
/// 1. Iterates through all configured functions in the gateway
/// 2. Includes only functions that have at least one variant configured
/// 3. Iterates through all configured provider models
/// 4. Sorts all models alphabetically by ID for consistent ordering
/// 5. Returns models in Anthropic's format with appropriate prefixes
///
/// # Model ID Format
///
/// - Functions: `tensorzero::function_name::{function_name}`
/// - Providers: `tensorzero::model_name::{model_name}`
///
/// # Response Format
///
/// Returns models in Anthropic's format:
/// ```json
/// {
///   "data": [
///     {
///       "id": "tensorzero::function_name::my_function",
///       "name": "my_function",
///       "type": "model"
///     },
///     {
///       "id": "tensorzero::model_name::openai::gpt-4o-mini",
///       "name": "openai::gpt-4o-mini",
///       "type": "model"
///     }
///   ],
///   "object": "list"
/// }
/// ```
///
/// # Errors
///
/// This function will return an error if:
/// - The gateway state is unavailable (internal server error)
/// - Configuration cannot be accessed (internal server error)
///
/// # Example Usage
///
/// ```bash
/// curl http://localhost:3000/anthropic/v1/models
/// ```
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "anthropic_compatible.models", skip_all)]
#[allow(clippy::unused_async)]
pub async fn models_handler(
    State(app_state): AppState,
) -> Result<axum::Json<AnthropicModelsResponse>, Error> {
    let config = &app_state.config;

    let mut models = Vec::new();

    // Add function_name models (TensorZero functions)
    for (function_name, function_config) in &config.functions {
        // Only include functions that have at least one variant configured
        if !function_config.variants().is_empty() {
            models.push(AnthropicModel {
                id: format!("{FUNCTION_NAME_PREFIX}{function_name}"),
                name: function_name.clone(),
                resource_type: "model".to_string(),
            });
        }
    }

    // Add model_name models (direct provider models)
    for model_name in config.models.table.keys() {
        models.push(AnthropicModel {
            id: format!("{MODEL_NAME_PREFIX}{model_name}"),
            name: model_name.to_string(),
            resource_type: "model".to_string(),
        });
    }

    // Sort models by ID for consistent, predictable ordering
    models.sort_by(|a, b| a.id.cmp(&b.id));

    Ok(axum::Json(AnthropicModelsResponse {
        data: models,
        object: "list".to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_model_serialization() {
        let model = AnthropicModel {
            id: "claude-3-5-sonnet-20241022".to_string(),
            name: "Claude 3.5 Sonnet".to_string(),
            resource_type: "model".to_string(),
        };

        let json = serde_json::to_string(&model).unwrap();

        // Verify the JSON structure
        assert!(json.contains("\"id\""));
        assert!(json.contains("\"name\""));
        assert!(json.contains("\"type\""));
        assert!(json.contains("\"claude-3-5-sonnet-20241022\""));

        // Verify we can deserialize it back
        let parsed: AnthropicModel = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "claude-3-5-sonnet-20241022");
        assert_eq!(parsed.name, "Claude 3.5 Sonnet");
        assert_eq!(parsed.resource_type, "model");
    }

    #[test]
    fn test_anthropic_models_response_serialization() {
        let response = AnthropicModelsResponse {
            data: vec![
                AnthropicModel {
                    id: "tensorzero::function_name::test".to_string(),
                    name: "test".to_string(),
                    resource_type: "model".to_string(),
                },
                AnthropicModel {
                    id: "tensorzero::model_name::openai::gpt-4o-mini".to_string(),
                    name: "openai::gpt-4o-mini".to_string(),
                    resource_type: "model".to_string(),
                },
            ],
            object: "list".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        let parsed: Value = serde_json::from_str(&json).unwrap();

        // Verify top-level structure
        assert_eq!(parsed["object"], "list");
        assert!(parsed["data"].is_array());
        assert_eq!(parsed["data"].as_array().unwrap().len(), 2);

        // Verify first model
        assert_eq!(parsed["data"][0]["id"], "tensorzero::function_name::test");
        assert_eq!(parsed["data"][0]["name"], "test");
        assert_eq!(parsed["data"][0]["type"], "model");

        // Verify second model
        assert_eq!(
            parsed["data"][1]["id"],
            "tensorzero::model_name::openai::gpt-4o-mini"
        );
        assert_eq!(parsed["data"][1]["name"], "openai::gpt-4o-mini");
        assert_eq!(parsed["data"][1]["type"], "model");
    }

    #[test]
    fn test_anthropic_model_required_fields() {
        // Test that required fields are present and correctly typed
        let model = AnthropicModel {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            resource_type: "model".to_string(),
        };

        // Verify all fields are non-empty
        assert!(!model.id.is_empty());
        assert!(!model.name.is_empty());
        assert_eq!(model.resource_type, "model");
    }

    #[test]
    fn test_anthropic_models_response_object_type() {
        let response = AnthropicModelsResponse {
            data: vec![],
            object: "list".to_string(),
        };

        assert_eq!(response.object, "list");
        assert!(response.data.is_empty());
    }

    use serde_json::Value;
}
