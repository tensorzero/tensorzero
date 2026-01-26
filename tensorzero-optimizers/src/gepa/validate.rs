//! Validation functions for GEPA configuration and examples

use std::collections::HashMap;
use std::sync::Arc;

use serde::Serialize;
use tensorzero_core::{
    config::Config,
    error::{Error, ErrorDetails},
    evaluations::EvaluationConfig,
    function::{FunctionConfig, FunctionConfigChat, FunctionConfigJson},
    inference::types::{ContentBlockChatOutput, StoredInputMessageContent},
    optimization::gepa::GEPAConfig,
    stored_inference::{RenderedSample, StoredOutput},
    tool::StaticToolConfig,
    variant::{VariantConfig, VariantInfo, chat_completion::UninitializedChatCompletionConfig},
};

/// Minimum number of valid examples required for GEPA optimization
pub const MIN_EXAMPLES: usize = 4;

/// Function configuration with associated static tools for GEPA optimization
#[derive(Clone, Debug, Serialize)]
pub struct FunctionContext {
    pub function_config: Arc<FunctionConfig>,
    /// Static tools from Config.tools that are referenced by the function
    pub static_tools: Option<HashMap<String, Arc<StaticToolConfig>>>,
    pub evaluation_config: Arc<EvaluationConfig>,
}

/// Validates the GEPA configuration and checks that required resources exist
/// Returns the FunctionContext containing the function config, associated static tools, and evaluation config for the function being optimized
pub fn validate_gepa_config(
    config: &GEPAConfig,
    tensorzero_config: &Config,
) -> Result<FunctionContext, Error> {
    // Check that the function exists in the config
    let function_config = tensorzero_config
        .functions
        .get(&config.function_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "Function '{}' not found in configuration",
                    config.function_name
                ),
            })
        })?;

    // Extract the evaluation config from the TensorZero config
    let evaluation_name = &config.evaluation_name;
    let evaluation_config = tensorzero_config
        .evaluations
        .get(evaluation_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!("Evaluation '{evaluation_name}' not found in config"),
            })
        })?;

    // Validate initial_variants if specified
    if let Some(initial_variants) = &config.initial_variants {
        let function_variants = function_config.variants();

        // Check that all specified variants exist
        for variant_name in initial_variants {
            if !function_variants.contains_key(variant_name) {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "Variant '{}' specified in initial_variants not found in Function '{}'",
                        variant_name, config.function_name
                    ),
                }));
            }
        }

        // Ensure at least one variant exists after filtering
        if initial_variants.is_empty() {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "`initial_variants` is empty for Function '{}'",
                    config.function_name
                ),
            }));
        }
    }

    // Validate that specified initial_variants are ChatCompletion (GEPA requirement).
    // When initial_variants is None, we filter to only ChatCompletion variants below,
    // and get_uninitialized_variant_configs will error if none exist.
    let function_variants = function_config.variants();

    if let Some(initial_variants) = &config.initial_variants {
        for variant_name in initial_variants {
            if let Some(variant_info) = function_variants.get(variant_name) {
                match &variant_info.inner {
                    VariantConfig::ChatCompletion(_) => {
                        // Valid - ChatCompletion variant
                    }
                    _ => {
                        return Err(Error::new(ErrorDetails::Config {
                            message: format!(
                                "Variant '{}' in Function '{}' is not a ChatCompletion variant. GEPA only supports ChatCompletion variants.",
                                variant_name, config.function_name
                            ),
                        }));
                    }
                }
            }
        }
    }

    // Create filtered function config with only relevant variants.
    // This config is only used for serialization to show to the GEPA LLM for analysis/mutation,
    // not for performing actual inference.
    let filtered_function_config: Arc<FunctionConfig> = {
        let variants_to_include: Vec<&String> =
            if let Some(initial_variants) = &config.initial_variants {
                initial_variants.iter().collect()
            } else {
                function_config
                    .variants()
                    .iter()
                    .filter(|(_, info)| matches!(info.inner, VariantConfig::ChatCompletion(_)))
                    .map(|(name, _)| name)
                    .collect()
            };

        let filtered_variants: HashMap<String, Arc<VariantInfo>> = function_config
            .variants()
            .iter()
            .filter(|(name, _)| variants_to_include.contains(name))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        match &**function_config {
            FunctionConfig::Chat(chat_config) => {
                Arc::new(FunctionConfig::Chat(FunctionConfigChat {
                    variants: filtered_variants,
                    // Use defaults for non-cloneable fields - they're not critical for GEPA LLM context
                    schemas: Default::default(),
                    tools: chat_config.tools.clone(),
                    tool_choice: chat_config.tool_choice.clone(),
                    parallel_tool_calls: chat_config.parallel_tool_calls,
                    description: chat_config.description.clone(),
                    all_explicit_templates_names: Default::default(),
                    experimentation: Default::default(),
                }))
            }
            FunctionConfig::Json(json_config) => {
                Arc::new(FunctionConfig::Json(FunctionConfigJson {
                    variants: filtered_variants,
                    // Use defaults for non-cloneable fields - they're not critical for GEPA LLM context
                    schemas: Default::default(),
                    output_schema: json_config.output_schema.clone(),
                    json_mode_tool_call_config: json_config.json_mode_tool_call_config.clone(),
                    description: json_config.description.clone(),
                    all_explicit_template_names: Default::default(),
                    experimentation: Default::default(),
                }))
            }
        }
    };

    // Validate tools referenced by function exist in config and extract them
    let function_tool_names: Vec<String> = match &**function_config {
        FunctionConfig::Chat(chat_config) => chat_config.tools.clone(),
        FunctionConfig::Json(_) => {
            // JSON functions have implicit tool for schema, which is always available
            Vec::new()
        }
    };

    let static_tools = if function_tool_names.is_empty() {
        None
    } else {
        let mut tools = HashMap::new();
        for tool_name in &function_tool_names {
            if let Some(tool_config) = tensorzero_config.tools.get(tool_name) {
                tools.insert(tool_name.clone(), tool_config.clone());
            } else {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "Tool '{}' referenced by Function '{}' not found in configuration",
                        tool_name, config.function_name
                    ),
                }));
            }
        }
        Some(tools)
    };

    Ok(FunctionContext {
        function_config: filtered_function_config,
        static_tools,
        evaluation_config: evaluation_config.clone(),
    })
}

/// Validates the stored_output field of a RenderedSample
///
/// Returns Ok(()) if valid, Err(reason) if invalid
fn validate_stored_output(stored_output: Option<&StoredOutput>) -> Result<(), String> {
    // Check if stored_output is None
    if stored_output.is_none() {
        return Err("stored_output is None".to_string());
    }

    // Check if stored_output is JsonInferenceOutput with parsed is None
    if let Some(StoredOutput::Json(json_output)) = stored_output
        && json_output.parsed.is_none()
    {
        return Err("JsonInferenceOutput.parsed is None".to_string());
    }

    // Check if stored_output is a Chat output
    if let Some(StoredOutput::Chat(chat_output)) = stored_output {
        // Check if ChatInferenceOutput is empty
        if chat_output.is_empty() {
            return Err("stored_output (ChatInferenceOutput) has length 0".to_string());
        }

        // Validate each content block in the chat output
        for (content_idx, content_block) in chat_output.iter().enumerate() {
            match content_block {
                // Check Text block
                ContentBlockChatOutput::Text(text) => {
                    if text.text.is_empty() {
                        return Err(format!(
                            "stored_output[{content_idx}] Text block has empty text"
                        ));
                    }
                }
                // Check ToolCall block
                ContentBlockChatOutput::ToolCall(tool_call) => {
                    if tool_call.name.is_none() {
                        return Err(format!(
                            "stored_output[{content_idx}] ToolCall block has name as None"
                        ));
                    }
                }
                // Check Thought block
                // Destructure to cause compile error if new fields are added to Thought
                ContentBlockChatOutput::Thought(thought) => {
                    let tensorzero_core::inference::types::Thought {
                        text,
                        signature,
                        summary,
                        provider_type: _,
                        extra_data: _,
                    } = thought;
                    if text.is_none() && signature.is_none() && summary.is_none() {
                        return Err(format!(
                            "stored_output[{content_idx}] Thought block has text, signature, and summary all as None"
                        ));
                    }
                }
                // Unknown is fine (we'll let it through)
                ContentBlockChatOutput::Unknown(_) => {}
            }
        }
    }

    Ok(())
}

/// Validates the stored_input.messages field of a RenderedSample
///
/// Returns Ok(()) if valid, Err(reason) if invalid
fn validate_stored_input_messages(
    messages: &[tensorzero_core::inference::types::StoredInputMessage],
) -> Result<(), String> {
    for (msg_idx, message) in messages.iter().enumerate() {
        // Check if message has no content blocks
        if message.content.is_empty() {
            return Err(format!(
                "stored_input.messages[{msg_idx}] has no content blocks"
            ));
        }

        // Validate each content block in the message
        for (content_idx, content_block) in message.content.iter().enumerate() {
            match content_block {
                // Check for unsupported media types (File includes images/files)
                StoredInputMessageContent::File(_) => {
                    return Err(format!(
                        "stored_input.messages[{msg_idx}].content[{content_idx}] contains File (unsupported media type)"
                    ));
                }
                // Check Text block
                StoredInputMessageContent::Text(text) => {
                    if text.text.is_empty() {
                        return Err(format!(
                            "stored_input.messages[{msg_idx}].content[{content_idx}] Text block has empty text"
                        ));
                    }
                }
                // Check ToolCall block
                StoredInputMessageContent::ToolCall(tool_call) => {
                    if tool_call.name.is_empty() {
                        return Err(format!(
                            "stored_input.messages[{msg_idx}].content[{content_idx}] ToolCall block has name as empty"
                        ));
                    }
                }
                // Check Thought block
                // Destructure to cause compile error if new fields are added to Thought
                StoredInputMessageContent::Thought(thought) => {
                    let tensorzero_core::inference::types::Thought {
                        text,
                        signature,
                        summary,
                        provider_type: _,
                        extra_data: _,
                    } = thought;
                    if text.is_none() && signature.is_none() && summary.is_none() {
                        return Err(format!(
                            "stored_input.messages[{msg_idx}].content[{content_idx}] Thought block has text, signature, and summary all as None"
                        ));
                    }
                }
                // These are fine for GEPA
                StoredInputMessageContent::ToolResult(_) => {}
                StoredInputMessageContent::Template(_) => {}
                StoredInputMessageContent::RawText(_) => {}
                StoredInputMessageContent::Unknown(_) => {}
            }
        }
    }

    Ok(())
}

/// Validates and filters examples for GEPA optimization
///
/// Filters out examples with:
/// - stored_output is None
/// - stored_output is JsonInferenceOutput with parsed is None
/// - stored_output is ChatInferenceOutput with length 0
/// - Any message has no content blocks (empty content list)
/// - Any message contains a File block (StoredInputMessageContent::File(_))
/// - Any Text block has empty text (text.is_empty())
/// - Any ToolCall block has name as None/empty
/// - Any Thought block has text, signature, and summary all as None
///
/// Returns filtered list of valid examples, or error if all examples are dropped
pub fn validate_examples(examples: Vec<RenderedSample>) -> Result<Vec<RenderedSample>, Error> {
    if examples.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: "Cannot run GEPA optimization with zero examples".to_string(),
        }));
    }

    let total_examples = examples.len();
    let mut valid_examples = Vec::new();
    let mut drop_reasons: HashMap<String, usize> = HashMap::new();

    for sample in examples {
        // Validate stored_output and stored_input.messages
        let validation_result = validate_stored_output(sample.stored_output.as_ref())
            .and_then(|()| validate_stored_input_messages(&sample.stored_input.messages));

        match validation_result {
            Ok(()) => valid_examples.push(sample),
            Err(reason) => *drop_reasons.entry(reason).or_insert(0) += 1,
        }
    }

    // Log summary of dropped examples
    let total_dropped = total_examples - valid_examples.len();
    if total_dropped > 0 {
        tracing::warn!(
            "Dropped {}/{} examples during validation:",
            total_dropped,
            total_examples
        );

        // Sort by count (descending) for better readability
        let mut reasons_vec: Vec<_> = drop_reasons.iter().collect();
        reasons_vec.sort_by(|a, b| b.1.cmp(a.1));

        for (reason, count) in reasons_vec {
            tracing::warn!("  - {} examples: {}", count, reason);
        }
    }

    if valid_examples.len() < MIN_EXAMPLES {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "Insufficient valid examples for GEPA optimization: found {} but need at least {}. \
                {} examples provided, {} dropped during validation. \
                Reasons: {:?}",
                valid_examples.len(),
                MIN_EXAMPLES,
                total_examples,
                total_dropped,
                drop_reasons
            ),
        }));
    }

    Ok(valid_examples)
}

/// Extracts UninitializedVariantConfigs to initialize the Pareto frontier with existing
/// variants or provided initial variants
pub fn get_uninitialized_variant_configs(
    config: &GEPAConfig,
    function_context: &FunctionContext,
) -> Result<HashMap<String, UninitializedChatCompletionConfig>, Error> {
    let variants = function_context.function_config.variants();
    let mut frontier = HashMap::new();

    if let Some(initial_variant_names) = &config.initial_variants {
        // Use only the specified initial variants
        for variant_name in initial_variant_names {
            let variant_info = variants.get(variant_name).ok_or_else(|| {
                Error::new(ErrorDetails::Config {
                    message: format!(
                        "Variant '{}' not found in Function '{}'",
                        variant_name, config.function_name
                    ),
                })
            })?;

            // Note: All variants in initial_variants have been validated as ChatCompletion type
            // by validate_gepa_config, so this extraction should always succeed
            if let Some(uninitialized_chat_config) =
                extract_chat_completion_from_variant_info(variant_info, variant_name)
            {
                frontier.insert(variant_name.clone(), uninitialized_chat_config);
                tracing::info!("Using initial variant: {}", variant_name);
            }
        }

        tracing::info!(
            "Initialized Pareto frontier with {} specified initial variants from function '{}'",
            frontier.len(),
            config.function_name
        );
    } else {
        // Use all ChatCompletion variants from the function
        for (variant_name, variant_info) in variants {
            if let Some(uninitialized_chat_config) =
                extract_chat_completion_from_variant_info(variant_info, variant_name)
            {
                frontier.insert(variant_name.clone(), uninitialized_chat_config);
            }
        }

        tracing::info!(
            "Initialized Pareto frontier with {} ChatCompletion variants from function '{}'",
            frontier.len(),
            config.function_name
        );
    }

    if frontier.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "No ChatCompletion variants found for function '{}'. GEPA requires at least one ChatCompletion variant.",
                config.function_name
            ),
        }));
    }

    Ok(frontier)
}

/// Extracts a ChatCompletion config from a VariantInfo
///
/// Returns `Some(config)` if the variant is a ChatCompletion variant,
/// or `None` if it's a different variant type (BestOfNSampling, Dicl, etc.).
///
/// This function combines variant type filtering with config extraction,
/// making it convenient for GEPA initialization where we want to extract
/// only ChatCompletion variants from a function's variant list.
fn extract_chat_completion_from_variant_info(
    variant_info: &VariantInfo,
    variant_name: &str,
) -> Option<UninitializedChatCompletionConfig> {
    match &variant_info.inner {
        VariantConfig::ChatCompletion(chat_config) => {
            let uninitialized = chat_config.as_uninitialized();
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
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use tensorzero_core::{
        config::{Config, ErrorContext, SchemaData, TimeoutsConfig, path::ResolvedTomlPathData},
        evaluations::{EvaluationConfig, InferenceEvaluationConfig},
        experimentation::ExperimentationConfig,
        function::{FunctionConfig, FunctionConfigChat},
        inference::types::{
            ContentBlockChatOutput, ModelInput, ResolvedContentBlock, ResolvedRequestMessage, Role,
            StoredInput, StoredInputMessage, StoredInputMessageContent, System, Text, Unknown,
        },
        optimization::gepa::GEPAConfig,
        stored_inference::{RenderedSample, StoredOutput},
        tool::{DynamicToolParams, ToolChoice},
        utils::retries::RetryConfig,
        variant::{
            VariantConfig, VariantInfo,
            chat_completion::{
                ChatCompletionConfig, UninitializedChatCompletionConfig, UninitializedChatTemplates,
            },
        },
    };
    use uuid::Uuid;

    fn create_minimal_config() -> Config {
        // Create a minimal valid Config for testing
        // This would need to be adjusted based on Config's actual structure
        Config::default()
    }

    fn create_valid_rendered_sample() -> RenderedSample {
        RenderedSample {
            function_name: "test_function".to_string(),
            input: ModelInput {
                system: Some("Test system".to_string()),
                messages: vec![ResolvedRequestMessage {
                    role: Role::User,
                    content: vec![ResolvedContentBlock::Text(Text {
                        text: "Test message".to_string(),
                    })],
                }],
            },
            stored_input: StoredInput {
                system: Some(System::Text("Test system".to_string())),
                messages: vec![StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: "Test message".to_string(),
                    })],
                }],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "Test output".to_string(),
            })]),
            stored_output: Some(StoredOutput::Chat(vec![ContentBlockChatOutput::Text(
                Text {
                    text: "Test output".to_string(),
                },
            )])),
            episode_id: Some(Uuid::now_v7()),
            inference_id: Some(Uuid::now_v7()),
            tool_params: DynamicToolParams::default(),
            output_schema: None,
            dispreferred_outputs: vec![],
            tags: HashMap::new(),
        }
    }

    #[test]
    fn test_validate_examples_empty_input() {
        let examples: Vec<RenderedSample> = vec![];
        let result = validate_examples(examples);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("Cannot run GEPA optimization with zero examples")
        );
    }

    #[test]
    fn test_validate_examples_stored_output_none() {
        let mut sample = create_valid_rendered_sample();
        sample.stored_output = None;

        let examples = vec![sample; 15]; // Need at least MIN_EXAMPLES
        let result = validate_examples(examples);

        // All examples should be dropped, resulting in error
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Insufficient valid examples"));
    }

    #[test]
    fn test_validate_examples_chat_output_empty() {
        let mut sample = create_valid_rendered_sample();
        sample.stored_output = Some(StoredOutput::Chat(vec![]));

        let examples = vec![sample; 15];
        let result = validate_examples(examples);

        // All examples should be dropped
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_examples_chat_output_text_empty() {
        let mut sample = create_valid_rendered_sample();
        sample.stored_output = Some(StoredOutput::Chat(vec![ContentBlockChatOutput::Text(
            Text {
                text: String::new(),
            },
        )]));

        let examples = vec![sample; 15];
        let result = validate_examples(examples);

        // All examples should be dropped
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_examples_message_no_content() {
        let mut sample = create_valid_rendered_sample();
        sample.stored_input.messages[0].content = vec![];

        let examples = vec![sample; 15];
        let result = validate_examples(examples);

        // All examples should be dropped
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_examples_message_empty_text() {
        let mut sample = create_valid_rendered_sample();
        sample.stored_input.messages[0].content = vec![StoredInputMessageContent::Text(Text {
            text: String::new(),
        })];

        let examples = vec![sample; 15];
        let result = validate_examples(examples);

        // All examples should be dropped
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_examples_insufficient_valid() {
        // Create fewer than MIN_EXAMPLES valid samples
        let examples = vec![create_valid_rendered_sample(); MIN_EXAMPLES - 1];
        let result = validate_examples(examples);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Insufficient valid examples"));
    }

    #[test]
    fn test_validate_examples_minimum_valid() {
        // Create exactly MIN_EXAMPLES valid samples
        let examples = vec![create_valid_rendered_sample(); MIN_EXAMPLES];
        let result = validate_examples(examples);

        assert!(result.is_ok());
        let valid = result.unwrap();
        assert_eq!(valid.len(), MIN_EXAMPLES);
    }

    #[test]
    fn test_validate_examples_filters_and_keeps_valid() {
        // Create mix of valid and invalid samples
        let mut examples = vec![create_valid_rendered_sample(); MIN_EXAMPLES + 5];

        // Make 3 invalid
        examples[0].stored_output = None;
        examples[1].stored_output = None;
        examples[2].stored_output = Some(StoredOutput::Chat(vec![]));

        let result = validate_examples(examples);

        assert!(result.is_ok());
        let valid = result.unwrap();
        assert_eq!(valid.len(), MIN_EXAMPLES + 2); // Total - 3 invalid
    }

    #[test]
    fn test_validate_gepa_config_function_not_found() {
        let config = create_gepa_config("nonexistent_function", None, None);

        let tensorzero_config = create_minimal_config();
        let result = validate_gepa_config(&config, &tensorzero_config);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("Function 'nonexistent_function' not found")
        );
    }

    #[test]
    fn test_validate_gepa_config_evaluation_not_found() {
        // This test would need a proper Config with a function but no matching evaluation
        // Skipping detailed implementation as it requires full Config setup
    }

    #[test]
    fn test_validate_gepa_config_initial_variants_empty() {
        let _config = GEPAConfig {
            function_name: "test_function".to_string(),
            evaluation_name: "test_evaluation".to_string(),
            initial_variants: Some(vec![]), // Empty list
            variant_prefix: None,
            batch_size: 5,
            max_iterations: 1,
            max_concurrency: 10,
            analysis_model: "openai::gpt-5-mini".to_string(),
            mutation_model: "openai::gpt-5".to_string(),
            seed: None,
            timeout: 300,
            include_inference_for_mutation: true,
            retries: tensorzero_core::utils::retries::RetryConfig::default(),
            max_tokens: Some(16_384),
        };

        // This would fail during validation if we had a proper test config
        // The error would be "initial_variants is empty for function 'test_function'"
    }

    #[test]
    fn test_min_examples_constant() {
        // Verify the constant value is as expected
        assert_eq!(MIN_EXAMPLES, 4);
    }

    // ============================================================================
    // Helper function for validate_gepa_config tests
    // ============================================================================

    /// Creates a Config with the specified function and evaluation for testing validate_gepa_config
    fn create_config_with_function_and_evaluation(
        function_name: &str,
        evaluation_name: &str,
        variants: HashMap<String, Arc<VariantInfo>>,
    ) -> Config {
        let mut config = Config::default();

        // Add the function
        config
            .functions
            .insert(function_name.to_string(), create_function_config(variants));

        // Add the evaluation
        config.evaluations.insert(
            evaluation_name.to_string(),
            Arc::new(EvaluationConfig::Inference(InferenceEvaluationConfig {
                evaluators: HashMap::new(),
                function_name: function_name.to_string(),
                description: Some(evaluation_name.to_string()),
            })),
        );

        config
    }

    // ============================================================================
    // Unit tests for validate_gepa_config filtering
    // ============================================================================

    #[test]
    fn test_validate_gepa_config_filters_variants_with_initial_variants() {
        // Setup: function with variants v1, v2, v3
        let mut variants = HashMap::new();
        variants.insert(
            "v1".to_string(),
            create_variant_info("openai::gpt-4", Some("Prompt 1")),
        );
        variants.insert(
            "v2".to_string(),
            create_variant_info("openai::gpt-4", Some("Prompt 2")),
        );
        variants.insert(
            "v3".to_string(),
            create_variant_info("openai::gpt-4", Some("Prompt 3")),
        );

        let tensorzero_config = create_config_with_function_and_evaluation(
            "test_function",
            "test_evaluation",
            variants,
        );

        // Config: initial_variants = Some(["v1", "v3"])
        let gepa_config = create_gepa_config(
            "test_function",
            Some(vec!["v1".to_string(), "v3".to_string()]),
            None,
        );

        let result = validate_gepa_config(&gepa_config, &tensorzero_config);

        // Assert: returned function_config.variants() contains only v1, v3
        assert!(result.is_ok(), "validate_gepa_config should succeed");
        let function_context = result.unwrap();
        let filtered_variants = function_context.function_config.variants();

        assert_eq!(
            filtered_variants.len(),
            2,
            "Should contain only 2 variants (v1 and v3)"
        );
        assert!(
            filtered_variants.contains_key("v1"),
            "Should contain variant v1"
        );
        assert!(
            filtered_variants.contains_key("v3"),
            "Should contain variant v3"
        );
        assert!(
            !filtered_variants.contains_key("v2"),
            "Should NOT contain variant v2"
        );
    }

    #[test]
    fn test_validate_gepa_config_without_initial_variants_includes_all_chat_completion() {
        // Setup: function with only ChatCompletion variants (the only type we can easily construct)
        // This tests that when initial_variants is None, all ChatCompletion variants are included
        let mut variants = HashMap::new();
        variants.insert(
            "chat_v1".to_string(),
            create_variant_info("openai::gpt-4", Some("Prompt 1")),
        );
        variants.insert(
            "chat_v2".to_string(),
            create_variant_info("anthropic::claude-sonnet-4-5", Some("Prompt 2")),
        );
        variants.insert(
            "chat_v3".to_string(),
            create_variant_info("openai::gpt-3.5-turbo", Some("Prompt 3")),
        );

        let tensorzero_config = create_config_with_function_and_evaluation(
            "test_function",
            "test_evaluation",
            variants,
        );

        // Config: initial_variants = None (use all ChatCompletion variants)
        let gepa_config = create_gepa_config("test_function", None, None);

        let result = validate_gepa_config(&gepa_config, &tensorzero_config);

        // Assert: returned function_config.variants() contains all ChatCompletion variants
        assert!(result.is_ok(), "validate_gepa_config should succeed");
        let function_context = result.unwrap();
        let filtered_variants = function_context.function_config.variants();

        assert_eq!(
            filtered_variants.len(),
            3,
            "Should contain all 3 ChatCompletion variants"
        );
        assert!(
            filtered_variants.contains_key("chat_v1"),
            "Should contain chat_v1"
        );
        assert!(
            filtered_variants.contains_key("chat_v2"),
            "Should contain chat_v2"
        );
        assert!(
            filtered_variants.contains_key("chat_v3"),
            "Should contain chat_v3"
        );
    }

    // ============================================================================
    // Unit tests for validate_stored_output
    // ============================================================================

    #[test]
    fn test_validate_stored_output_none() {
        let result = validate_stored_output(None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "stored_output is None");
    }

    #[test]
    fn test_validate_stored_output_json_with_none_parsed() {
        use tensorzero_core::inference::types::JsonInferenceOutput;

        let output = Some(StoredOutput::Json(JsonInferenceOutput {
            raw: None,
            parsed: None,
        }));

        let result = validate_stored_output(output.as_ref());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "JsonInferenceOutput.parsed is None");
    }

    #[test]
    fn test_validate_stored_output_chat_empty() {
        let output = Some(StoredOutput::Chat(vec![]));

        let result = validate_stored_output(output.as_ref());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "stored_output (ChatInferenceOutput) has length 0"
        );
    }

    #[test]
    fn test_validate_stored_output_chat_text_empty() {
        let output = Some(StoredOutput::Chat(vec![ContentBlockChatOutput::Text(
            Text {
                text: String::new(),
            },
        )]));

        let result = validate_stored_output(output.as_ref());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Text block has empty text"));
    }

    #[test]
    fn test_validate_stored_output_chat_toolcall_no_name() {
        use tensorzero_core::tool::InferenceResponseToolCall;

        let output = Some(StoredOutput::Chat(vec![ContentBlockChatOutput::ToolCall(
            InferenceResponseToolCall {
                name: None,
                arguments: Some(serde_json::json!({})),
                raw_arguments: "{}".to_string(),
                raw_name: String::new(),
                id: "test".to_string(),
            },
        )]));

        let result = validate_stored_output(output.as_ref());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("ToolCall block has name as None")
        );
    }

    #[test]
    fn test_validate_stored_output_chat_thought_all_none() {
        use tensorzero_core::inference::types::Thought;

        let output = Some(StoredOutput::Chat(vec![ContentBlockChatOutput::Thought(
            Thought {
                text: None,
                summary: None,
                provider_type: None,
                signature: None,
                extra_data: None,
            },
        )]));

        let result = validate_stored_output(output.as_ref());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("Thought block has text, signature, and summary all as None")
        );
    }

    #[test]
    fn test_validate_stored_output_chat_thought_signature_only() {
        use tensorzero_core::inference::types::Thought;

        // Thought with only signature set should be valid
        let output = Some(StoredOutput::Chat(vec![ContentBlockChatOutput::Thought(
            Thought {
                text: None,
                summary: None,
                provider_type: None,
                signature: Some("encrypted_thinking_signature".to_string()),
                extra_data: None,
            },
        )]));

        let result = validate_stored_output(output.as_ref());
        assert!(
            result.is_ok(),
            "Thought with only signature should be valid"
        );
    }

    #[test]
    fn test_validate_stored_output_chat_valid() {
        let output = Some(StoredOutput::Chat(vec![ContentBlockChatOutput::Text(
            Text {
                text: "Valid text".to_string(),
            },
        )]));

        let result = validate_stored_output(output.as_ref());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_stored_output_chat_unknown_block() {
        let output = Some(StoredOutput::Chat(vec![ContentBlockChatOutput::Unknown(
            Unknown {
                data: serde_json::Value::Null,
                model_name: None,
                provider_name: None,
            },
        )]));

        // Unknown blocks should be accepted
        let result = validate_stored_output(output.as_ref());
        assert!(result.is_ok());
    }

    // ============================================================================
    // Unit tests for validate_stored_input_messages
    // ============================================================================

    #[test]
    fn test_validate_stored_input_messages_empty_message() {
        let messages = vec![StoredInputMessage {
            role: Role::User,
            content: vec![],
        }];

        let result = validate_stored_input_messages(&messages);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("has no content blocks"));
    }

    // Note: File content validation is covered by integration tests
    // File is an enum variant, not a struct, making unit testing complex

    #[test]
    fn test_validate_stored_input_messages_text_empty() {
        let messages = vec![StoredInputMessage {
            role: Role::User,
            content: vec![StoredInputMessageContent::Text(Text {
                text: String::new(),
            })],
        }];

        let result = validate_stored_input_messages(&messages);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Text block has empty text"));
    }

    #[test]
    fn test_validate_stored_input_messages_toolcall_empty_name() {
        use tensorzero_core::tool::ToolCall;

        let messages = vec![StoredInputMessage {
            role: Role::Assistant,
            content: vec![StoredInputMessageContent::ToolCall(ToolCall {
                name: String::new(),
                arguments: "{}".to_string(),
                id: "test".to_string(),
            })],
        }];

        let result = validate_stored_input_messages(&messages);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("ToolCall block has name as empty")
        );
    }

    #[test]
    fn test_validate_stored_input_messages_thought_all_none() {
        use tensorzero_core::inference::types::Thought;

        let messages = vec![StoredInputMessage {
            role: Role::Assistant,
            content: vec![StoredInputMessageContent::Thought(Thought {
                text: None,
                summary: None,
                provider_type: None,
                signature: None,
                extra_data: None,
            })],
        }];

        let result = validate_stored_input_messages(&messages);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("Thought block has text, signature, and summary all as None")
        );
    }

    #[test]
    fn test_validate_stored_input_messages_thought_signature_only() {
        use tensorzero_core::inference::types::Thought;

        // Thought with only signature set should be valid
        let messages = vec![StoredInputMessage {
            role: Role::Assistant,
            content: vec![StoredInputMessageContent::Thought(Thought {
                text: None,
                summary: None,
                provider_type: None,
                signature: Some("encrypted_thinking_signature".to_string()),
                extra_data: None,
            })],
        }];

        let result = validate_stored_input_messages(&messages);
        assert!(
            result.is_ok(),
            "Thought with only signature should be valid"
        );
    }

    #[test]
    fn test_validate_stored_input_messages_valid() {
        let messages = vec![StoredInputMessage {
            role: Role::User,
            content: vec![StoredInputMessageContent::Text(Text {
                text: "Valid message".to_string(),
            })],
        }];

        let result = validate_stored_input_messages(&messages);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_stored_input_messages_toolresult_allowed() {
        use tensorzero_core::tool::ToolResult;

        let messages = vec![StoredInputMessage {
            role: Role::User, // Tool role doesn't exist, use User
            content: vec![StoredInputMessageContent::ToolResult(ToolResult {
                name: "test_tool".to_string(),
                result: "success".to_string(),
                id: "tool_id".to_string(),
            })],
        }];

        // ToolResult is allowed
        let result = validate_stored_input_messages(&messages);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_stored_input_messages_multiple_messages() {
        let messages = vec![
            StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "First message".to_string(),
                })],
            },
            StoredInputMessage {
                role: Role::Assistant,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Second message".to_string(),
                })],
            },
        ];

        let result = validate_stored_input_messages(&messages);
        assert!(result.is_ok());
    }

    // Note: Early exit behavior and File content validation are covered by integration tests

    // ============================================================================
    // Helper Functions for get_uninitialized_variant_configs tests
    // ============================================================================

    /// Creates a minimal UninitializedChatCompletionConfig for testing
    fn create_uninitialized_chat_config(
        model: &str,
        template_content: Option<&str>,
    ) -> UninitializedChatCompletionConfig {
        let system_template = template_content.map(|content| {
            ResolvedTomlPathData::new_fake_path(
                "test_system.minijinja".to_string(),
                content.to_string(),
            )
        });

        UninitializedChatCompletionConfig {
            weight: Some(1.0),
            model: model.into(),
            system_template,
            user_template: None,
            assistant_template: None,
            input_wrappers: None,
            templates: UninitializedChatTemplates::default(),
            temperature: Some(0.7),
            top_p: None,
            max_tokens: Some(100),
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            stop_sequences: None,
            reasoning_effort: None,
            service_tier: None,
            thinking_budget_tokens: None,
            verbosity: None,
            json_mode: None,
            retries: RetryConfig::default(),
            extra_body: None,
            extra_headers: None,
        }
    }

    /// Creates a ChatCompletionConfig by loading an uninitialized config
    fn create_chat_completion_config(
        model: &str,
        template_content: Option<&str>,
    ) -> ChatCompletionConfig {
        let uninitialized = create_uninitialized_chat_config(model, template_content);
        let schemas = SchemaData::default();
        let error_context = ErrorContext {
            function_name: "test".to_string(),
            variant_name: "test".to_string(),
        };

        uninitialized
            .load(&schemas, &error_context)
            .expect("Failed to load chat completion config")
    }

    /// Creates a VariantInfo with ChatCompletion inner config
    fn create_variant_info(model: &str, template_content: Option<&str>) -> Arc<VariantInfo> {
        Arc::new(VariantInfo {
            inner: VariantConfig::ChatCompletion(create_chat_completion_config(
                model,
                template_content,
            )),
            timeouts: TimeoutsConfig::default(),
        })
    }

    /// Creates a minimal FunctionConfig for testing
    fn create_function_config(variants: HashMap<String, Arc<VariantInfo>>) -> Arc<FunctionConfig> {
        Arc::new(FunctionConfig::Chat(FunctionConfigChat {
            variants,
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        }))
    }

    /// Creates a minimal EvaluationConfig for testing
    fn create_evaluation_config() -> Arc<EvaluationConfig> {
        Arc::new(EvaluationConfig::Inference(InferenceEvaluationConfig {
            evaluators: HashMap::new(),
            function_name: "test_function".to_string(),
            description: Some("test_evaluation".to_string()),
        }))
    }

    /// Creates a FunctionContext for testing
    fn create_function_context(variants: HashMap<String, Arc<VariantInfo>>) -> FunctionContext {
        FunctionContext {
            function_config: create_function_config(variants),
            static_tools: None,
            evaluation_config: create_evaluation_config(),
        }
    }

    /// Creates a minimal GEPAConfig for testing
    fn create_gepa_config(
        function_name: &str,
        initial_variants: Option<Vec<String>>,
        retries: Option<RetryConfig>,
    ) -> GEPAConfig {
        GEPAConfig {
            function_name: function_name.to_string(),
            evaluation_name: "test_evaluation".to_string(),
            initial_variants,
            variant_prefix: Some("gepa_".to_string()),
            batch_size: 5,
            max_iterations: 1,
            max_concurrency: 10,
            analysis_model: "openai::gpt-4".to_string(),
            mutation_model: "openai::gpt-4".to_string(),
            seed: Some(42),
            timeout: 300,
            include_inference_for_mutation: true,
            retries: retries.unwrap_or_default(),
            max_tokens: Some(16_384),
        }
    }

    /// Creates a HashMap of test variants for pareto frontier tests
    fn create_test_variants(count: usize) -> HashMap<String, Arc<VariantInfo>> {
        let models = [
            "openai::gpt-4",
            "anthropic::claude-sonnet-4-5",
            "openai::gpt-3.5-turbo",
        ];
        (0..count)
            .map(|i| {
                let variant_name = format!("variant{}", i + 1);
                let model = models[i % models.len()];
                let prompt = format!("System prompt {}", i + 1);
                (variant_name, create_variant_info(model, Some(&prompt)))
            })
            .collect()
    }

    // ============================================================================
    // Tests for get_uninitialized_variant_configs
    // ============================================================================

    #[test]
    fn test_get_uninitialized_variant_configs_with_all_variants() {
        // Setup: Create function with multiple ChatCompletion variants
        let variants = create_test_variants(3);

        let function_context = create_function_context(variants);
        let config = create_gepa_config("test_function", None, None);

        // Test: Initialize without initial_variants (should use all variants)
        let result = get_uninitialized_variant_configs(&config, &function_context);

        assert!(result.is_ok());
        let frontier = result.unwrap();
        assert_eq!(frontier.len(), 3, "Should include all 3 variants");
        assert!(frontier.contains_key("variant1"));
        assert!(frontier.contains_key("variant2"));
        assert!(frontier.contains_key("variant3"));
    }

    #[test]
    fn test_get_uninitialized_variant_configs_with_initial_variants() {
        // Setup: Create function with multiple variants
        let variants = create_test_variants(3);

        let function_context = create_function_context(variants);

        // Test: Initialize with only specific variants
        let initial_variants = Some(vec!["variant1".to_string(), "variant3".to_string()]);
        let config = create_gepa_config("test_function", initial_variants, None);

        let result = get_uninitialized_variant_configs(&config, &function_context);

        assert!(result.is_ok());
        let frontier = result.unwrap();
        assert_eq!(frontier.len(), 2, "Should include only specified variants");
        assert!(frontier.contains_key("variant1"));
        assert!(frontier.contains_key("variant3"));
        assert!(
            !frontier.contains_key("variant2"),
            "Should not include variant2"
        );
    }

    #[test]
    fn test_get_uninitialized_variant_configs_nonexistent_variant() {
        // Setup: Create function with variants
        let mut variants = HashMap::new();
        variants.insert(
            "variant1".to_string(),
            create_variant_info("openai::gpt-4", Some("System prompt 1")),
        );

        let function_context = create_function_context(variants);

        // Test: Try to initialize with a nonexistent variant
        let initial_variants = Some(vec!["variant1".to_string(), "nonexistent".to_string()]);
        let config = create_gepa_config("test_function", initial_variants, None);

        let result = get_uninitialized_variant_configs(&config, &function_context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nonexistent"));
        assert!(err.to_string().contains("not found"));
    }

    // ============================================================================
    // Tests for extract_chat_completion_from_variant_info
    // ============================================================================

    #[test]
    fn test_extract_from_variant_info_chat_completion() {
        let variant_info = create_variant_info("openai::gpt-4", Some("Test"));

        let result = extract_chat_completion_from_variant_info(&variant_info, "test_variant");

        assert!(result.is_some());
        let extracted = result.unwrap();
        assert_eq!(extracted.model.as_ref(), "openai::gpt-4");
    }

    #[test]
    fn test_extract_from_variant_info_preserves_all_fields() {
        // Create a config with many fields set
        let mut uninitialized =
            create_uninitialized_chat_config("anthropic::claude-sonnet-4-5", Some("Test"));
        uninitialized.weight = Some(2.5);
        uninitialized.temperature = Some(0.9);
        uninitialized.top_p = Some(0.95);
        uninitialized.max_tokens = Some(2000);
        uninitialized.presence_penalty = Some(0.5);
        uninitialized.frequency_penalty = Some(0.3);
        uninitialized.seed = Some(12345);
        uninitialized.stop_sequences = Some(vec!["STOP".to_string()]);
        uninitialized.reasoning_effort = Some("medium".to_string());

        let schemas = SchemaData::default();
        let error_context = ErrorContext {
            function_name: "test".to_string(),
            variant_name: "test".to_string(),
        };
        let config = uninitialized.load(&schemas, &error_context).unwrap();

        let variant_info = Arc::new(VariantInfo {
            inner: VariantConfig::ChatCompletion(config),
            timeouts: TimeoutsConfig::default(),
        });

        let result = extract_chat_completion_from_variant_info(&variant_info, "test");
        assert!(result.is_some());

        let extracted = result.unwrap();

        // Verify all fields are preserved
        assert_eq!(extracted.weight, Some(2.5));
        assert_eq!(extracted.temperature, Some(0.9));
        assert_eq!(extracted.top_p, Some(0.95));
        assert_eq!(extracted.max_tokens, Some(2000));
        assert_eq!(extracted.presence_penalty, Some(0.5));
        assert_eq!(extracted.frequency_penalty, Some(0.3));
        assert_eq!(extracted.seed, Some(12345));
        assert_eq!(extracted.stop_sequences, Some(vec!["STOP".to_string()]));
        assert_eq!(extracted.reasoning_effort, Some("medium".to_string()));
    }
}
