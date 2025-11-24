//! Helper functions for dealing with extra_body.rs and extra_headers.rs
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::extra_body::{DynamicExtraBody, UnfilteredInferenceExtraBody};
use crate::inference::types::extra_headers::{DynamicExtraHeader, UnfilteredInferenceExtraHeaders};
use crate::model::ModelTable;

/// Validate all filters in extra_body and extra_headers
pub async fn validate_inference_filters(
    extra_body: &UnfilteredInferenceExtraBody,
    extra_headers: &UnfilteredInferenceExtraHeaders,
    function: Option<&FunctionConfig>,
    models: &ModelTable,
) -> Result<(), Error> {
    // Validate extra_body filters
    for filter in extra_body.as_slice() {
        match filter {
            DynamicExtraBody::Variant { variant_name, .. }
            | DynamicExtraBody::VariantDelete { variant_name, .. } => {
                if let Some(func) = function {
                    validate_variant_filter(variant_name, func)?;
                }
            }
            #[expect(deprecated)]
            DynamicExtraBody::Provider {
                model_provider_name,
                ..
            }
            | DynamicExtraBody::ProviderDelete {
                model_provider_name,
                ..
            } => {
                validate_provider_filter(model_provider_name)?;
            }
            DynamicExtraBody::ModelProvider {
                model_name,
                provider_name,
                ..
            }
            | DynamicExtraBody::ModelProviderDelete {
                model_name,
                provider_name,
                ..
            } => {
                validate_model_provider_filter(model_name, provider_name.as_deref(), models)
                    .await?;
            }
            DynamicExtraBody::Always { .. } | DynamicExtraBody::AlwaysDelete { .. } => {
                // Always variant has no filter to validate
            }
        }
    }

    // Validate extra_headers filters
    for filter in extra_headers.as_slice() {
        match filter {
            DynamicExtraHeader::Variant { variant_name, .. }
            | DynamicExtraHeader::VariantDelete { variant_name, .. } => {
                if let Some(func) = function {
                    validate_variant_filter(variant_name, func)?;
                }
            }
            #[expect(deprecated)]
            DynamicExtraHeader::Provider {
                model_provider_name,
                ..
            }
            | DynamicExtraHeader::ProviderDelete {
                model_provider_name,
                ..
            } => {
                validate_provider_filter(model_provider_name)?;
            }
            DynamicExtraHeader::ModelProvider {
                model_name,
                provider_name,
                ..
            }
            | DynamicExtraHeader::ModelProviderDelete {
                model_name,
                provider_name,
                ..
            } => {
                validate_model_provider_filter(model_name, provider_name.as_deref(), models)
                    .await?;
            }
            DynamicExtraHeader::Always { .. } | DynamicExtraHeader::AlwaysDelete { .. } => {
                // Always variant has no filter to validate
            }
        }
    }

    Ok(())
}

/// Validate that variant filter references an existing variant in the function
fn validate_variant_filter(variant_name: &str, function: &FunctionConfig) -> Result<(), Error> {
    if !function.variants().contains_key(variant_name) {
        return Err(ErrorDetails::UnknownVariant {
            name: variant_name.to_string(),
        }
        .into());
    }
    Ok(())
}

/// Validate that provider filter references a valid provider
/// The provider_name in filters must be in fully qualified format:
/// "tensorzero::model_name::X::provider_name::Y"
///
/// Deprecated: Use separate `model_name` and `provider_name` fields instead.
#[deprecated]
fn validate_provider_filter(model_provider_name: &str) -> Result<(), Error> {
    tracing::warn!(
        "Deprecation Warning: Please provide `model_name` and `provider_name` fields instead of `model_provider_name` when specifying `extra_body` or `extra_headers` in the request. Alternatively, you can skip the filter altogether to match any model inference in your request."
    );

    // Check if it's a fully qualified name
    if !model_provider_name.starts_with("tensorzero::model_name::") {
        return Err(ErrorDetails::InvalidInferenceTarget {
            message: format!(
                "Invalid model_provider_name filter `{model_provider_name}`: must use fully qualified format 'tensorzero::model_name::{{model}}::provider_name::{{provider}}'"
            ),
        }
        .into());
    }

    Ok(())
}

/// Validate that model_provider filter references a valid model and provider
async fn validate_model_provider_filter(
    model_name: &str,
    provider_name: Option<&str>,
    models: &ModelTable,
) -> Result<(), Error> {
    // Check if the model exists in the table (supports shorthand notation)
    if let Some(model_config) = models.get(model_name).await? {
        // Check if the provider exists in that model (if provider_name is specified)
        if let Some(provider_name) = provider_name {
            if !model_config.providers.contains_key(provider_name) {
                return Err(ErrorDetails::InvalidInferenceTarget {
                    message: format!(
                        "Invalid model provider filter: provider `{provider_name}` not found in model `{model_name}`.",
                    ),
                }
                .into());
            }
        }
        Ok(())
    } else {
        Err(
            ErrorDetails::InvalidInferenceTarget {
                message: format!(
                    "Invalid model provider filter: model `{model_name}` does not exist.",
                ),
            }
            .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::{FunctionConfig, FunctionConfigChat};
    use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
    use crate::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
    use crate::model::ModelTable;
    use crate::variant::VariantInfo;
    use serde_json::json;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    fn create_test_function_config() -> FunctionConfig {
        let mut variants = HashMap::new();
        variants.insert(
            "variant1".to_string(),
            Arc::new(VariantInfo {
                timeouts: Default::default(),
                inner: crate::variant::VariantConfig::ChatCompletion(Default::default()),
            }),
        );
        variants.insert(
            "variant2".to_string(),
            Arc::new(VariantInfo {
                timeouts: Default::default(),
                inner: crate::variant::VariantConfig::ChatCompletion(Default::default()),
            }),
        );
        FunctionConfig::Chat(FunctionConfigChat {
            variants,
            schemas: Default::default(),
            tools: vec![],
            tool_choice: Default::default(),
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: Default::default(),
        })
    }

    fn create_test_model_table() -> ModelTable {
        use crate::config::provider_types::ProviderTypesConfig;
        use crate::model::{ModelConfig, ModelProvider, ProviderConfig};
        use crate::model_table::ProviderTypeDefaultCredentials;
        use crate::providers::dummy::DummyProvider;

        // Create a model table with one model ("test-model") and one provider ("test-provider")
        let map = HashMap::from([(
            "test-model".into(),
            ModelConfig {
                routing: vec!["test-provider".into()],
                providers: HashMap::from([(
                    "test-provider".into(),
                    ModelProvider {
                        name: "test-provider".into(),
                        config: ProviderConfig::Dummy(DummyProvider {
                            model_name: "test-model".into(),
                            ..Default::default()
                        }),
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                    },
                )]),
                timeouts: Default::default(),
            },
        )]);

        let provider_types = ProviderTypesConfig::default();
        ModelTable::new(
            map,
            ProviderTypeDefaultCredentials::new(&provider_types).into(),
            chrono::Duration::seconds(120),
        )
        .expect("Failed to create model table")
    }

    #[test]
    fn test_validate_variant_filter_valid() {
        let function = create_test_function_config();
        let result = validate_variant_filter("variant1", &function);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_variant_filter_invalid() {
        let function = create_test_function_config();
        let result = validate_variant_filter("nonexistent", &function);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_provider_filter_fully_qualified_valid() {
        #[expect(deprecated)]
        let result = validate_provider_filter(
            "tensorzero::model_name::test-model::provider_name::test-provider",
        );
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_inference_filters_function_context_valid_variant() {
        let function = create_test_function_config();
        let models = create_test_model_table();

        let extra_body: UnfilteredInferenceExtraBody = serde_json::from_value(json!([
            {
                "variant_name": "variant1",
                "pointer": "/test",
                "value": {"key": "value"}
            }
        ]))
        .unwrap();

        let result = validate_inference_filters(
            &extra_body,
            &UnfilteredInferenceExtraHeaders::default(),
            Some(&function),
            &models,
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_inference_filters_function_context_invalid_variant() {
        let function = create_test_function_config();
        let models = create_test_model_table();

        let extra_body: UnfilteredInferenceExtraBody = serde_json::from_value(json!([
            {
                "variant_name": "nonexistent",
                "pointer": "/test",
                "value": {"key": "value"}
            }
        ]))
        .unwrap();

        let result = validate_inference_filters(
            &extra_body,
            &UnfilteredInferenceExtraHeaders::default(),
            Some(&function),
            &models,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validate_inference_filters_function_context_invalid_provider() {
        let function = create_test_function_config();
        let models = create_test_model_table();

        let extra_headers: UnfilteredInferenceExtraHeaders = serde_json::from_value(json!([
            {
                "model_provider_name": "invalid::model",
                "name": "X-Custom-Header",
                "value": "value"
            }
        ]))
        .unwrap();

        let result = validate_inference_filters(
            &UnfilteredInferenceExtraBody::default(),
            &extra_headers,
            Some(&function),
            &models,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validate_inference_filters_model_context_invalid_provider() {
        let models = create_test_model_table();

        let extra_body: UnfilteredInferenceExtraBody = serde_json::from_value(json!([
            {
                "model_provider_name": "invalid::model",
                "pointer": "/test",
                "value": {"key": "value"}
            }
        ]))
        .unwrap();

        let result = validate_inference_filters(
            &extra_body,
            &UnfilteredInferenceExtraHeaders::default(),
            None,
            &models,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validate_inference_filters_always_variant_no_validation() {
        let function = create_test_function_config();
        let models = create_test_model_table();

        // Always variant should not trigger validation
        let extra_body: UnfilteredInferenceExtraBody = serde_json::from_value(json!([
            {
                "pointer": "/test",
                "value": {"key": "value"}
            }
        ]))
        .unwrap();

        let result = validate_inference_filters(
            &extra_body,
            &UnfilteredInferenceExtraHeaders::default(),
            Some(&function),
            &models,
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_inference_filters_empty() {
        let function = create_test_function_config();
        let models = create_test_model_table();

        let result = validate_inference_filters(
            &UnfilteredInferenceExtraBody::default(),
            &UnfilteredInferenceExtraHeaders::default(),
            Some(&function),
            &models,
        )
        .await;
        assert!(result.is_ok());
    }
}
