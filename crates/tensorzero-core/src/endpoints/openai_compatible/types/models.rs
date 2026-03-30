//! Model types for OpenAI-compatible API.
//!
//! This module contains response types for the models endpoints,
//! matching OpenAI's API format.

use serde::{Deserialize, Serialize};

/// Represents an OpenAI model in the API response.
///
/// This matches OpenAI's response format:
/// ```json
/// {
///   "id": "gpt-4",
///   "object": "model",
///   "created": 1687882411,
///   "owned_by": "openai"
/// }
/// ```
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAIModel {
    /// The model identifier, which can be referenced in the API endpoints.
    pub id: String,
    /// The object type (always "model" for model objects).
    pub object: String,
    /// The Unix timestamp (in seconds) when the model was created.
    pub created: u64,
    /// The organization that owns the model.
    pub owned_by: String,
}

/// Response wrapper for listing models.
///
/// This matches OpenAI's response format:
/// ```json
/// {
///   "object": "list",
///   "data": [
///     { "id": "gpt-4", "object": "model", "created": 1687882411, "owned_by": "openai" }
///   ]
/// }
/// ```
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAIModelsListResponse {
    /// The object type (always "list" for list responses).
    pub object: String,
    /// The list of models.
    pub data: Vec<OpenAIModel>,
}

impl OpenAIModelsListResponse {
    /// Creates a new models list response.
    pub fn new(models: Vec<OpenAIModel>) -> Self {
        Self {
            object: "list".to_string(),
            data: models,
        }
    }
}

impl OpenAIModel {
    /// Creates a new model response.
    pub fn new(id: String, owned_by: String) -> Self {
        Self {
            id,
            object: "model".to_string(),
            // Use current Unix timestamp as a reasonable default
            // since TensorZero models don't have creation timestamps
            created: current_timestamp(),
            owned_by,
        }
    }

    /// Creates a new model response with a specific creation timestamp.
    pub fn new_with_timestamp(id: String, owned_by: String, created: u64) -> Self {
        Self {
            id,
            object: "model".to_string(),
            created,
            owned_by,
        }
    }
}

/// Returns the current Unix timestamp in seconds.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_model_serialization() {
        let model = OpenAIModel::new("gpt-4".to_string(), "openai".to_string());
        let json = serde_json::to_string(&model).unwrap();
        assert!(json.contains("\"id\":\"gpt-4\""));
        assert!(json.contains("\"object\":\"model\""));
        assert!(json.contains("\"owned_by\":\"openai\""));
    }

    #[test]
    fn test_openai_models_list_response() {
        let models = vec![
            OpenAIModel::new("gpt-4".to_string(), "openai".to_string()),
            OpenAIModel::new("claude-3-opus".to_string(), "anthropic".to_string()),
        ];
        let response = OpenAIModelsListResponse::new(models);
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"object\":\"list\""));
        assert!(json.contains("\"data\":"));
    }
}
