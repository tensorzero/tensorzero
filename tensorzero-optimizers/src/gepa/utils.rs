//! Utility functions for GEPA optimization
use std::collections::HashMap;

use tensorzero_core::{
    config::path::ResolvedTomlPath,
    variant::{
        chat_completion::{
            ChatCompletionConfig, UninitializedChatCompletionConfig, UninitializedChatTemplate,
        },
        VariantConfig, VariantInfo,
    },
};

/// Extract an `UninitializedChatCompletionConfig` from a `ChatCompletionConfig`
///
/// This is needed for GEPA to create new variant configurations at runtime.
/// Since `ChatCompletionConfig` is the loaded/initialized form with templates and schemas
/// resolved, we need to extract it back to the uninitialized form that can be serialized
/// and used to create new variants.
///
/// Note: This only handles new-style templates (defined via `templates` HashMap).
/// Legacy templates (system_template, user_template, assistant_template, input_wrappers)
/// are not extracted since `ChatCompletionConfig` has already unified them into `ChatTemplates`.
pub fn extract_chat_completion_config(
    config: &ChatCompletionConfig,
    retries: tensorzero_core::utils::retries::RetryConfig,
) -> UninitializedChatCompletionConfig {
    // Extract new-style templates from ChatTemplates
    // Since UninitializedChatTemplates has a private `inner` field with #[serde(flatten)],
    // we construct it via serde (serialize HashMap, then deserialize to UninitializedChatTemplates)
    let templates = {
        let mut inner = HashMap::new();

        // Get all explicit template names (filters out deprecated legacy templates with no schema)
        let template_names = config.templates().get_all_explicit_template_names();

        for template_name in template_names {
            if let Some(template_with_schema) =
                config.templates().get_named_template(&template_name)
            {
                // Create a new fake path with inline content to avoid duplicate path errors
                // when multiple variants use the same template files
                let fake_path = ResolvedTomlPath::new_fake_path(
                    format!("gepa_extracted/{template_name}"),
                    template_with_schema.template.contents.clone(),
                );

                inner.insert(template_name, UninitializedChatTemplate { path: fake_path });
            }
        }
        // Use serde to construct UninitializedChatTemplates from HashMap
        // The #[serde(flatten)] attribute makes this work correctly
        #[expect(clippy::unwrap_used)]
        {
            serde_json::from_value(serde_json::to_value(inner).unwrap()).unwrap()
        }
    };

    // Extract inference_params_v2 fields
    let reasoning_effort = config.reasoning_effort().cloned();
    let service_tier = config.service_tier().cloned();
    let thinking_budget_tokens = config.thinking_budget_tokens();
    let verbosity = config.verbosity().cloned();

    UninitializedChatCompletionConfig {
        weight: config.weight(),
        model: config.model().clone(),
        // Legacy fields - set to None since we only extract new-style templates
        system_template: None,
        user_template: None,
        assistant_template: None,
        input_wrappers: None,
        // New-style templates
        templates,
        // Simple config fields
        temperature: config.temperature(),
        top_p: config.top_p(),
        max_tokens: config.max_tokens(),
        presence_penalty: config.presence_penalty(),
        frequency_penalty: config.frequency_penalty(),
        seed: config.seed(),
        stop_sequences: config.stop_sequences().cloned(),
        // Inference params v2 fields (flattened)
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
        // Other config
        json_mode: config.json_mode().cloned(),
        retries,
        extra_body: config.extra_body().cloned(),
        extra_headers: config.extra_headers().cloned(),
    }
}

/// Extracts a ChatCompletion config from a VariantInfo
///
/// Returns `Some(config)` if the variant is a ChatCompletion variant,
/// or `None` if it's a different variant type (BestOfNSampling, Dicl, etc.).
///
/// This function combines variant type filtering with config extraction,
/// making it convenient for GEPA initialization where we want to extract
/// only ChatCompletion variants from a function's variant list.
pub fn extract_chat_completion_from_variant_info(
    variant_info: &VariantInfo,
    variant_name: &str,
    retries: tensorzero_core::utils::retries::RetryConfig,
) -> Option<UninitializedChatCompletionConfig> {
    match &variant_info.inner {
        VariantConfig::ChatCompletion(chat_config) => {
            // Use the pure conversion function to extract the config
            let uninitialized = extract_chat_completion_config(chat_config, retries);
            tracing::debug!(
                "Extracted ChatCompletion config for variant: {}",
                variant_name
            );
            Some(uninitialized)
        }
        VariantConfig::BestOfNSampling(_)
        | VariantConfig::Dicl(_)
        | VariantConfig::MixtureOfN(_)
        | VariantConfig::ChainOfThought(_) => {
            tracing::warn!(
                "Skipping non-ChatCompletion variant '{}' (GEPA only supports ChatCompletion variants)",
                variant_name
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorzero_core::utils::retries::RetryConfig;

    /// Create a default ChatCompletionConfig for testing
    /// Note: Due to private fields, we can only test with default configs from outside tensorzero-core
    fn create_test_config() -> ChatCompletionConfig {
        ChatCompletionConfig::default()
    }

    #[test]
    fn test_extract_default_config() {
        let config = create_test_config();
        let extracted = extract_chat_completion_config(
            &config,
            tensorzero_core::utils::retries::RetryConfig::default(),
        );

        // Verify basic extraction works
        // Since we can only use default configs from outside tensorzero-core,
        // we mainly verify the function runs without errors and returns valid data

        // Legacy fields should always be None
        assert_eq!(extracted.system_template, None);
        assert_eq!(extracted.user_template, None);
        assert_eq!(extracted.assistant_template, None);
        assert!(extracted.input_wrappers.is_none());

        // Default config should have inference_params_v2 fields as None
        assert_eq!(extracted.reasoning_effort, None);
        assert_eq!(extracted.service_tier, None);
        assert_eq!(extracted.thinking_budget_tokens, None);
        assert_eq!(extracted.verbosity, None);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = create_test_config();
        let extracted = extract_chat_completion_config(
            &config,
            tensorzero_core::utils::retries::RetryConfig::default(),
        );

        // Verify the extracted config can be serialized and deserialized
        let serialized = serde_json::to_string(&extracted).expect("Failed to serialize");
        let deserialized: UninitializedChatCompletionConfig =
            serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Verify key fields after roundtrip
        assert_eq!(deserialized.model, extracted.model);
        assert_eq!(deserialized.weight, extracted.weight);
        assert_eq!(deserialized.temperature, extracted.temperature);
        assert_eq!(deserialized.max_tokens, extracted.max_tokens);
    }

    #[test]
    fn test_template_extraction_with_empty_templates() {
        let config = create_test_config();
        let extracted = extract_chat_completion_config(
            &config,
            tensorzero_core::utils::retries::RetryConfig::default(),
        );

        // Empty templates should serialize/deserialize correctly
        let serialized = serde_json::to_string(&extracted.templates).expect("Failed to serialize");
        assert_eq!(serialized, "{}");
    }

    #[test]
    fn test_function_runs_without_error() {
        // Test that the extraction function runs without panicking or errors
        let config = create_test_config();
        let extracted = extract_chat_completion_config(
            &config,
            tensorzero_core::utils::retries::RetryConfig::default(),
        );

        // Just verify the result is valid - model field exists (may be empty for default)
        // The important thing is the function completed without errors
        let _ = extracted.model.as_ref();
    }

    #[test]
    fn test_extracted_config_has_default_retries() {
        let config = create_test_config();

        // Test with custom retries to verify override works
        let custom_retries = RetryConfig {
            num_retries: 3,
            max_delay_s: 5.0,
        };
        let extracted = extract_chat_completion_config(&config, custom_retries);

        // The extracted config should use the provided retries, not the parent's
        assert_eq!(extracted.retries.num_retries, 3);
        assert_eq!(extracted.retries.max_delay_s, 5.0);
    }
}
