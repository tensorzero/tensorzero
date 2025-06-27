#[cfg(test)]
mod test_model_input_standardization {
    use axum::http::{HeaderMap, HeaderName, HeaderValue};
    use tensorzero_internal::endpoints::model_resolution;

    /// Test that model resolution works correctly for unauthenticated requests
    #[test]
    fn test_unauthenticated_standard_model_names() {
        let headers = HeaderMap::new();

        // Test standard model name without authentication
        let result =
            model_resolution::resolve_model_name("gpt-3.5-turbo", &headers, false).unwrap();
        assert_eq!(result.model_name, Some("gpt-3.5-turbo".to_string()));
        assert_eq!(result.function_name, None);
        assert_eq!(result.original_model_name.as_ref(), "gpt-3.5-turbo");
        assert!(!result.is_authenticated);

        // Test another standard model
        let result =
            model_resolution::resolve_model_name("claude-3-opus", &headers, false).unwrap();
        assert_eq!(result.model_name, Some("claude-3-opus".to_string()));
        assert!(!result.is_authenticated);
    }

    /// Test that model resolution works correctly for authenticated requests
    #[test]
    fn test_authenticated_standard_model_names() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-tensorzero-endpoint-id"),
            HeaderValue::from_static("endpoint-uuid-123"),
        );
        headers.insert(
            HeaderName::from_static("x-tensorzero-model-name"),
            HeaderValue::from_static("gpt-3.5-turbo"),
        );

        // Test standard model name with authentication
        let result =
            model_resolution::resolve_model_name("gpt-3.5-turbo", &headers, false).unwrap();
        assert_eq!(result.model_name, Some("endpoint-uuid-123".to_string()));
        assert_eq!(result.function_name, None);
        assert_eq!(result.original_model_name.as_ref(), "gpt-3.5-turbo");
        assert!(result.is_authenticated);
    }

    /// Test backward compatibility with prefixed model names
    #[test]
    fn test_backward_compatibility_prefixed_names() {
        let headers = HeaderMap::new();

        // Test model name with prefix (backward compatibility)
        let result = model_resolution::resolve_model_name(
            "tensorzero::model_name::gpt-3.5-turbo",
            &headers,
            false,
        )
        .unwrap();
        assert_eq!(result.model_name, Some("gpt-3.5-turbo".to_string()));
        assert_eq!(result.function_name, None);
        assert_eq!(result.original_model_name.as_ref(), "gpt-3.5-turbo");

        // Test function name with prefix
        let result = model_resolution::resolve_model_name(
            "tensorzero::function_name::my_function",
            &headers,
            false,
        )
        .unwrap();
        assert_eq!(result.function_name, Some("my_function".to_string()));
        assert_eq!(result.model_name, None);
        assert_eq!(
            result.original_model_name.as_ref(),
            "tensorzero::function_name::my_function"
        );
    }

    /// Test embedding model resolution
    #[test]
    fn test_embedding_model_resolution() {
        let headers = HeaderMap::new();

        // Test standard embedding model name
        let result = model_resolution::resolve_model_name(
            "text-embedding-ada-002",
            &headers,
            true, // for_embedding = true
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

        // Test embedding model with specific prefix
        let result = model_resolution::resolve_model_name(
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
    }

    /// Test mixed authentication scenarios
    #[test]
    fn test_mixed_auth_scenarios() {
        // Authenticated request with prefixed model name
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-tensorzero-endpoint-id"),
            HeaderValue::from_static("endpoint-456"),
        );

        let result = model_resolution::resolve_model_name(
            "tensorzero::model_name::custom-model",
            &headers,
            false,
        )
        .unwrap();
        // Even with auth headers, prefixed names should use the extracted model name
        assert_eq!(result.model_name, Some("custom-model".to_string()));
        assert!(result.is_authenticated);

        // Test function with auth (should still work with prefix)
        let result = model_resolution::resolve_model_name(
            "tensorzero::function_name::my_func",
            &headers,
            false,
        )
        .unwrap();
        assert_eq!(result.function_name, Some("my_func".to_string()));
        assert_eq!(result.model_name, None);
        assert!(result.is_authenticated);
    }

    /// Test edge cases and error scenarios
    #[test]
    fn test_edge_cases() {
        let headers = HeaderMap::new();

        // Model name that contains but doesn't start with prefix
        let result = model_resolution::resolve_model_name(
            "my-tensorzero::model_name::something",
            &headers,
            false,
        )
        .unwrap();
        assert_eq!(
            result.model_name,
            Some("my-tensorzero::model_name::something".to_string())
        );
        assert_eq!(result.function_name, None);

        // Empty string
        let result = model_resolution::resolve_model_name("", &headers, false).unwrap();
        assert_eq!(result.model_name, Some("".to_string()));
        assert_eq!(result.function_name, None);
    }

    /// Test authentication detection
    #[test]
    fn test_authentication_detection() {
        let mut headers = HeaderMap::new();

        // No auth headers
        let result = model_resolution::resolve_model_name("model", &headers, false).unwrap();
        assert!(!result.is_authenticated);

        // Only model name header (not enough for auth)
        headers.insert(
            HeaderName::from_static("x-tensorzero-model-name"),
            HeaderValue::from_static("model"),
        );
        let result = model_resolution::resolve_model_name("model", &headers, false).unwrap();
        assert!(!result.is_authenticated);

        // Add endpoint-id header (now authenticated)
        headers.insert(
            HeaderName::from_static("x-tensorzero-endpoint-id"),
            HeaderValue::from_static("endpoint-123"),
        );
        let result = model_resolution::resolve_model_name("model", &headers, false).unwrap();
        assert!(result.is_authenticated);
    }
}
