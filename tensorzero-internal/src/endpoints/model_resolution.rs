//! Model resolution module for OpenAI-compatible endpoints
//!
//! This module provides utilities to resolve model names based on authentication state,
//! supporting both authenticated and unauthenticated access patterns.

use axum::http::HeaderMap;
use std::borrow::Cow;

use crate::error::{Error, ErrorDetails};

const TENSORZERO_FUNCTION_NAME_PREFIX: &str = "tensorzero::function_name::";
const TENSORZERO_MODEL_NAME_PREFIX: &str = "tensorzero::model_name::";
const TENSORZERO_EMBEDDING_MODEL_NAME_PREFIX: &str = "tensorzero::embedding_model_name::";

/// Result of model resolution containing the resolved name and optional metadata
#[derive(Debug, Clone)]
pub struct ModelResolution<'a> {
    /// The function name to use (if this is a function-based request)
    pub function_name: Option<String>,
    /// The model name to use (if this is a direct model request)
    pub model_name: Option<String>,
    /// The original model name as provided by the user (for responses)
    pub original_model_name: Cow<'a, str>,
    /// Whether the request is authenticated
    pub is_authenticated: bool,
}

/// Resolves a model name based on authentication state and request format
///
/// This function handles the following scenarios:
/// 1. Authenticated requests with standard model names (e.g., "gpt-3.5-turbo")
/// 2. Unauthenticated requests with standard model names
/// 3. Legacy format with prefixes (for backward compatibility)
///
/// # Arguments
/// * `model` - The model name from the request
/// * `headers` - HTTP headers containing auth metadata
/// * `for_embedding` - Whether this is for an embedding endpoint
pub fn resolve_model_name<'a>(
    model: &'a str,
    headers: &HeaderMap,
    for_embedding: bool,
) -> Result<ModelResolution<'a>, Error> {
    // Check if we have authentication metadata
    let is_authenticated = headers.contains_key("x-tensorzero-endpoint-id");

    // Get the user-provided model name from header (set by auth middleware)
    let user_model_name = headers
        .get("x-tensorzero-model-name")
        .and_then(|v| v.to_str().ok());

    // Handle legacy prefixed format
    if let Some(function_name) = model.strip_prefix(TENSORZERO_FUNCTION_NAME_PREFIX) {
        // Function-based request with explicit prefix
        return Ok(ModelResolution {
            function_name: Some(function_name.to_string()),
            model_name: None,
            original_model_name: Cow::Borrowed(model),
            is_authenticated,
        });
    }

    if let Some(model_name) = model.strip_prefix(TENSORZERO_MODEL_NAME_PREFIX) {
        // Model-based request with explicit prefix
        return Ok(ModelResolution {
            function_name: None,
            model_name: Some(model_name.to_string()),
            original_model_name: user_model_name
                .map(|s| Cow::Owned(s.to_string()))
                .unwrap_or(Cow::Borrowed(model_name)),
            is_authenticated,
        });
    }

    if for_embedding {
        if let Some(model_name) = model.strip_prefix(TENSORZERO_EMBEDDING_MODEL_NAME_PREFIX) {
            // Embedding model with explicit prefix
            return Ok(ModelResolution {
                function_name: None,
                model_name: Some(model_name.to_string()),
                original_model_name: user_model_name
                    .map(|s| Cow::Owned(s.to_string()))
                    .unwrap_or(Cow::Borrowed(model_name)),
                is_authenticated,
            });
        }
    }

    // Handle deprecated "tensorzero::" prefix
    if let Some(function_name) = model.strip_prefix("tensorzero::") {
        tracing::warn!(
            function_name = function_name,
            "Deprecation Warning: Please set the `model` parameter to `tensorzero::function_name::your_function` instead of `tensorzero::your_function.` The latter will be removed in a future release."
        );
        return Ok(ModelResolution {
            function_name: Some(function_name.to_string()),
            model_name: None,
            original_model_name: Cow::Borrowed(model),
            is_authenticated,
        });
    }

    // Standard model name without prefix
    if is_authenticated {
        // For authenticated requests, we need to resolve the model using endpoint metadata
        let endpoint_id = headers
            .get("x-tensorzero-endpoint-id")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| {
                Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: "Missing endpoint metadata for authenticated request".to_string(),
                })
            })?;

        // The model name is the endpoint_id for authenticated requests
        Ok(ModelResolution {
            function_name: None,
            model_name: Some(endpoint_id.to_string()),
            original_model_name: Cow::Borrowed(model),
            is_authenticated,
        })
    } else {
        // For unauthenticated requests, use the model name directly
        // This assumes the model name in the config matches the user-facing name
        Ok(ModelResolution {
            function_name: None,
            model_name: Some(model.to_string()),
            original_model_name: Cow::Borrowed(model),
            is_authenticated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::header::{HeaderName, HeaderValue};

    #[test]
    fn test_resolve_model_name_unauthenticated() {
        let headers = HeaderMap::new();

        // Standard model name
        let result = resolve_model_name("gpt-3.5-turbo", &headers, false).unwrap();
        assert_eq!(result.model_name, Some("gpt-3.5-turbo".to_string()));
        assert_eq!(result.function_name, None);
        assert_eq!(result.original_model_name.as_ref(), "gpt-3.5-turbo");
        assert!(!result.is_authenticated);

        // Function name with prefix
        let result =
            resolve_model_name("tensorzero::function_name::my_function", &headers, false).unwrap();
        assert_eq!(result.function_name, Some("my_function".to_string()));
        assert_eq!(result.model_name, None);
        assert_eq!(
            result.original_model_name.as_ref(),
            "tensorzero::function_name::my_function"
        );
        assert!(!result.is_authenticated);

        // Model name with prefix
        let result =
            resolve_model_name("tensorzero::model_name::my_model", &headers, false).unwrap();
        assert_eq!(result.model_name, Some("my_model".to_string()));
        assert_eq!(result.function_name, None);
        assert_eq!(result.original_model_name.as_ref(), "my_model");
        assert!(!result.is_authenticated);
    }

    #[test]
    fn test_resolve_model_name_authenticated() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-tensorzero-endpoint-id"),
            HeaderValue::from_static("endpoint-123"),
        );
        headers.insert(
            HeaderName::from_static("x-tensorzero-model-name"),
            HeaderValue::from_static("gpt-3.5-turbo"),
        );

        // Standard model name with auth
        let result = resolve_model_name("gpt-3.5-turbo", &headers, false).unwrap();
        assert_eq!(result.model_name, Some("endpoint-123".to_string()));
        assert_eq!(result.function_name, None);
        assert_eq!(result.original_model_name.as_ref(), "gpt-3.5-turbo");
        assert!(result.is_authenticated);
    }

    #[test]
    fn test_resolve_embedding_model_name() {
        let headers = HeaderMap::new();

        // Embedding model with prefix
        let result = resolve_model_name(
            "tensorzero::embedding_model_name::text-embedding-ada-002",
            &headers,
            true,
        )
        .unwrap();
        assert_eq!(
            result.model_name,
            Some("text-embedding-ada-002".to_string())
        );
        assert_eq!(result.function_name, None);
        assert_eq!(
            result.original_model_name.as_ref(),
            "text-embedding-ada-002"
        );

        // Regular model prefix for embeddings
        let result = resolve_model_name(
            "tensorzero::model_name::text-embedding-ada-002",
            &headers,
            true,
        )
        .unwrap();
        assert_eq!(
            result.model_name,
            Some("text-embedding-ada-002".to_string())
        );
        assert_eq!(result.function_name, None);
        assert_eq!(
            result.original_model_name.as_ref(),
            "text-embedding-ada-002"
        );

        // Standard name for embeddings
        let result = resolve_model_name("text-embedding-ada-002", &headers, true).unwrap();
        assert_eq!(
            result.model_name,
            Some("text-embedding-ada-002".to_string())
        );
        assert_eq!(result.function_name, None);
        assert_eq!(
            result.original_model_name.as_ref(),
            "text-embedding-ada-002"
        );
    }

    #[test]
    fn test_deprecated_prefix() {
        let headers = HeaderMap::new();

        // Deprecated prefix
        let result = resolve_model_name("tensorzero::my_function", &headers, false).unwrap();
        assert_eq!(result.function_name, Some("my_function".to_string()));
        assert_eq!(result.model_name, None);
        assert_eq!(
            result.original_model_name.as_ref(),
            "tensorzero::my_function"
        );
    }

    #[test]
    fn test_authenticated_missing_metadata() {
        let mut headers = HeaderMap::new();
        // Has authentication but missing endpoint-id
        headers.insert(
            HeaderName::from_static("x-tensorzero-model-name"),
            HeaderValue::from_static("gpt-3.5-turbo"),
        );

        // This should work because we check for endpoint-id presence
        let result = resolve_model_name("gpt-3.5-turbo", &headers, false).unwrap();
        assert_eq!(result.model_name, Some("gpt-3.5-turbo".to_string()));
        assert!(!result.is_authenticated);
    }

    #[test]
    fn test_authenticated_with_prefix_still_works() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-tensorzero-endpoint-id"),
            HeaderValue::from_static("endpoint-123"),
        );

        // Even with auth, prefixed model names should work (backward compatibility)
        let result =
            resolve_model_name("tensorzero::model_name::my_model", &headers, false).unwrap();
        assert_eq!(result.model_name, Some("my_model".to_string()));
        assert_eq!(result.function_name, None);
        // Since user provided prefixed format, we don't have a separate original name
        assert_eq!(result.original_model_name.as_ref(), "my_model");
        assert!(result.is_authenticated);
    }

    #[test]
    fn test_edge_cases() {
        let headers = HeaderMap::new();

        // Empty function name after prefix
        let result = resolve_model_name("tensorzero::function_name::", &headers, false).unwrap();
        assert_eq!(result.function_name, Some("".to_string()));
        assert_eq!(result.model_name, None);

        // Empty model name after prefix
        let result = resolve_model_name("tensorzero::model_name::", &headers, false).unwrap();
        assert_eq!(result.model_name, Some("".to_string()));
        assert_eq!(result.function_name, None);

        // Model name that looks like a prefix but isn't
        let result = resolve_model_name("my-model-tensorzero::something", &headers, false).unwrap();
        assert_eq!(
            result.model_name,
            Some("my-model-tensorzero::something".to_string())
        );
        assert_eq!(result.function_name, None);
    }

    #[test]
    fn test_authenticated_with_all_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-tensorzero-endpoint-id"),
            HeaderValue::from_static("endpoint-123"),
        );
        headers.insert(
            HeaderName::from_static("x-tensorzero-model-name"),
            HeaderValue::from_static("claude-3-opus"),
        );
        headers.insert(
            HeaderName::from_static("x-tensorzero-project-id"),
            HeaderValue::from_static("project-456"),
        );
        headers.insert(
            HeaderName::from_static("x-tensorzero-model-id"),
            HeaderValue::from_static("model-789"),
        );

        // With full auth metadata
        let result = resolve_model_name("claude-3-opus", &headers, false).unwrap();
        assert_eq!(result.model_name, Some("endpoint-123".to_string()));
        assert_eq!(result.function_name, None);
        assert_eq!(result.original_model_name.as_ref(), "claude-3-opus");
        assert!(result.is_authenticated);
    }
}
