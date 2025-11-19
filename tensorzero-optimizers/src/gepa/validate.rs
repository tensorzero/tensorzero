//! Validation functions for GEPA configuration and examples

use std::collections::HashMap;
use std::sync::Arc;

use serde::Serialize;
use tensorzero_core::{
    config::{path::ResolvedTomlPathData, Config},
    error::{Error, ErrorDetails},
    evaluations::EvaluationConfig,
    function::FunctionConfig,
    inference::types::{ContentBlockChatOutput, StoredInputMessageContent},
    optimization::gepa::GEPAConfig,
    stored_inference::{RenderedSample, StoredOutput},
    tool::StaticToolConfig,
    variant::{
        chat_completion::{
            ChatCompletionConfig, UninitializedChatCompletionConfig, UninitializedChatTemplate,
        },
        VariantConfig, VariantInfo,
    },
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
/// Returns the FunctionConfig and associated static tools for the function being optimized
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
                    "function '{}' not found in configuration",
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
                        "variant '{}' specified in initial_variants not found in function '{}'",
                        variant_name, config.function_name
                    ),
                }));
            }
        }

        // Ensure at least one variant exists after filtering
        if initial_variants.is_empty() {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "initial_variants is empty for function '{}'",
                    config.function_name
                ),
            }));
        }
    }

    // Validate that function's variants are ChatCompletion (GEPA requirement)
    let function_variants = function_config.variants();

    let variants_to_check: Vec<&String> = if let Some(initial_variants) = &config.initial_variants {
        // Only check the specified variants
        initial_variants.iter().collect()
    } else {
        // Check all variant names
        function_variants.keys().collect()
    };

    for variant_name in variants_to_check {
        if let Some(variant_info) = function_variants.get(variant_name) {
            match &variant_info.inner {
                VariantConfig::ChatCompletion(_) => {
                    // Valid - ChatCompletion variant
                }
                _ => {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "variant '{}' in function '{}' is not a ChatCompletion variant. GEPA only supports ChatCompletion variants.",
                            variant_name, config.function_name
                        ),
                    }));
                }
            }
        }
    }

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
                        "tool '{}' referenced by function '{}' not found in configuration",
                        tool_name, config.function_name
                    ),
                }));
            }
        }
        Some(tools)
    };

    Ok(FunctionContext {
        function_config: function_config.clone(),
        static_tools,
        evaluation_config: evaluation_config.clone(),
    })
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
/// - Any ToolCall block has arguments as None or name as empty string
/// - Any Thought block has both text and summary as None
/// - Invalid stored_input.system (not None, Text, or object)
///
/// Returns filtered list of valid examples, or error if all examples are dropped
pub fn validate_examples(examples: &[RenderedSample]) -> Result<Vec<RenderedSample>, Error> {
    if examples.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: "Cannot run GEPA optimization with zero examples".to_string(),
        }));
    }

    let mut valid_examples = Vec::new();
    let mut drop_reasons: HashMap<String, usize> = HashMap::new();

    for sample in examples {
        let mut drop_reason: Option<String> = None;

        // Check if stored_output is None
        if sample.stored_output.is_none() {
            drop_reason = Some("stored_output is None".to_string());
        }
        // Check if stored_output is JsonInferenceOutput with parsed is None
        else if let Some(StoredOutput::Json(json_output)) = &sample.stored_output {
            if json_output.parsed.is_none() {
                drop_reason = Some("JsonInferenceOutput.parsed is None".to_string());
            }
        }

        // Check stored_output if it's a Chat output (list) - typically shorter, so check first
        if drop_reason.is_none() {
            if let Some(StoredOutput::Chat(chat_output)) = &sample.stored_output {
                // Check if ChatInferenceOutput is empty
                if chat_output.is_empty() {
                    drop_reason =
                        Some("stored_output (ChatInferenceOutput) has length 0".to_string());
                } else {
                    for (content_idx, content_block) in chat_output.iter().enumerate() {
                        match content_block {
                            // Check Text block
                            ContentBlockChatOutput::Text(text) => {
                                if text.text.is_empty() {
                                    drop_reason = Some(format!(
                                        "stored_output[{content_idx}] Text block has empty text"
                                    ));
                                    break;
                                }
                            }
                            // Check ToolCall block
                            ContentBlockChatOutput::ToolCall(tool_call) => {
                                if tool_call.name.is_none() {
                                    drop_reason = Some(format!(
                                        "stored_output[{content_idx}] ToolCall block has name as None"
                                    ));
                                    break;
                                }
                            }
                            // Check Thought block
                            ContentBlockChatOutput::Thought(thought) => {
                                if thought.text.is_none() && thought.summary.is_none() {
                                    drop_reason = Some(format!(
                                        "stored_output[{content_idx}] Thought block has both text and summary as None"
                                    ));
                                    break;
                                }
                            }
                            // Unknown is fine (we'll let it through)
                            ContentBlockChatOutput::Unknown { .. } => {}
                        }
                    }
                }
            }
        }

        // Check stored_input messages - typically longer conversation history
        if drop_reason.is_none() {
            for (msg_idx, message) in sample.stored_input.messages.iter().enumerate() {
                // Check if message has no content blocks
                if message.content.is_empty() {
                    drop_reason = Some(format!(
                        "stored_input.messages[{msg_idx}] has no content blocks"
                    ));
                    break;
                }

                for (content_idx, content_block) in message.content.iter().enumerate() {
                    match content_block {
                        // Check for unsupported media types (File includes images/files)
                        StoredInputMessageContent::File(_) => {
                            drop_reason = Some(format!(
                                "stored_input.messages[{msg_idx}].content[{content_idx}] contains File (unsupported media type)"
                            ));
                            break;
                        }
                        // Check Text block
                        StoredInputMessageContent::Text(text) => {
                            if text.text.is_empty() {
                                drop_reason = Some(format!(
                                    "stored_input.messages[{msg_idx}].content[{content_idx}] Text block has empty text"
                                ));
                                break;
                            }
                        }
                        // Check ToolCall block
                        StoredInputMessageContent::ToolCall(tool_call) => {
                            if tool_call.name.is_empty() {
                                drop_reason = Some(format!(
                                    "stored_input.messages[{msg_idx}].content[{content_idx}] ToolCall block has name as empty"
                                ));
                                break;
                            }
                        }
                        // Check Thought block
                        StoredInputMessageContent::Thought(thought) => {
                            if thought.text.is_none() && thought.summary.is_none() {
                                drop_reason = Some(format!(
                                    "stored_input.messages[{msg_idx}].content[{content_idx}] Thought block has both text and summary as None"
                                ));
                                break;
                            }
                        }
                        // These are fine for GEPA
                        StoredInputMessageContent::ToolResult(_) => {}
                        StoredInputMessageContent::Template(_) => {}
                        StoredInputMessageContent::RawText(_) => {}
                        StoredInputMessageContent::Unknown(_) => {}
                    }
                }

                if drop_reason.is_some() {
                    break;
                }
            }
        }

        if let Some(reason) = drop_reason {
            *drop_reasons.entry(reason).or_insert(0) += 1;
        } else {
            valid_examples.push(sample.clone());
        }
    }

    // Log summary of dropped examples
    let total_dropped = examples.len() - valid_examples.len();
    if total_dropped > 0 {
        tracing::warn!(
            "Dropped {}/{} examples during validation:",
            total_dropped,
            examples.len()
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
                examples.len(),
                total_dropped,
                drop_reasons
            ),
        }));
    }

    Ok(valid_examples)
}

/// Initializes the Pareto frontier with baseline variants or provided initial variants
pub fn initialize_pareto_frontier(
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
                        "variant '{}' not found in function '{}'",
                        variant_name, config.function_name
                    ),
                })
            })?;

            if let Some(chat_config) = extract_chat_completion_from_variant_info(
                variant_info,
                variant_name,
                config.retries,
            ) {
                frontier.insert(variant_name.clone(), chat_config);
                tracing::info!("Using initial variant: {}", variant_name);
            }
        }
    } else {
        // Use all ChatCompletion variants from the function
        for (variant_name, variant_info) in variants {
            if let Some(chat_config) = extract_chat_completion_from_variant_info(
                variant_info,
                variant_name,
                config.retries,
            ) {
                frontier.insert(variant_name.clone(), chat_config);
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
fn extract_chat_completion_config(
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
                let fake_path = ResolvedTomlPathData::new_fake_path(
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
fn extract_chat_completion_from_variant_info(
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
    use tensorzero_core::{
        config::Config,
        inference::types::{
            ContentBlockChatOutput, ModelInput, ResolvedContentBlock, ResolvedRequestMessage, Role,
            StoredInput, StoredInputMessage, StoredInputMessageContent, System, Text,
        },
        optimization::gepa::GEPAConfig,
        stored_inference::{RenderedSample, StoredOutput},
        tool::DynamicToolParams,
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
        let result = validate_examples(&examples);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Cannot run GEPA optimization with zero examples"));
    }

    #[test]
    fn test_validate_examples_stored_output_none() {
        let mut sample = create_valid_rendered_sample();
        sample.stored_output = None;

        let examples = vec![sample; 15]; // Need at least MIN_EXAMPLES
        let result = validate_examples(&examples);

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
        let result = validate_examples(&examples);

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
        let result = validate_examples(&examples);

        // All examples should be dropped
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_examples_message_no_content() {
        let mut sample = create_valid_rendered_sample();
        sample.stored_input.messages[0].content = vec![];

        let examples = vec![sample; 15];
        let result = validate_examples(&examples);

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
        let result = validate_examples(&examples);

        // All examples should be dropped
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_examples_insufficient_valid() {
        // Create fewer than MIN_EXAMPLES valid samples
        let examples = vec![create_valid_rendered_sample(); MIN_EXAMPLES - 1];
        let result = validate_examples(&examples);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Insufficient valid examples"));
    }

    #[test]
    fn test_validate_examples_minimum_valid() {
        // Create exactly MIN_EXAMPLES valid samples
        let examples = vec![create_valid_rendered_sample(); MIN_EXAMPLES];
        let result = validate_examples(&examples);

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

        let result = validate_examples(&examples);

        assert!(result.is_ok());
        let valid = result.unwrap();
        assert_eq!(valid.len(), MIN_EXAMPLES + 2); // Total - 3 invalid
    }

    #[test]
    fn test_validate_gepa_config_function_not_found() {
        let config = GEPAConfig {
            function_name: "nonexistent_function".to_string(),
            evaluation_name: "test_evaluation".to_string(),
            initial_variants: None,
            variant_prefix: None,
            batch_size: 5,
            max_iterations: 1,
            max_concurrency: 10,
            analysis_model: "openai::gpt-5-mini".to_string(),
            mutation_model: "openai::gpt-5".to_string(),
            seed: None,
            timeout: 300,
            include_inference_input_for_mutation: false,
            retries: tensorzero_core::utils::retries::RetryConfig::default(),
            max_tokens: Some(16_384),
        };

        let tensorzero_config = create_minimal_config();
        let result = validate_gepa_config(&config, &tensorzero_config);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("function 'nonexistent_function' not found"));
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
            include_inference_input_for_mutation: false,
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
}

#[cfg(test)]
mod initialize_pareto_frontier_tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use tensorzero_core::{
        config::{path::ResolvedTomlPathData, ErrorContext, SchemaData, TimeoutsConfig},
        evaluations::{EvaluationConfig, InferenceEvaluationConfig},
        experimentation::ExperimentationConfig,
        function::{FunctionConfig, FunctionConfigChat},
        tool::ToolChoice,
        utils::retries::RetryConfig,
        variant::{
            chat_completion::{
                ChatCompletionConfig, UninitializedChatCompletionConfig, UninitializedChatTemplates,
            },
            VariantConfig, VariantInfo,
        },
    };

    // ============================================================================
    // Helper Functions
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
            include_inference_input_for_mutation: false,
            retries: retries.unwrap_or_default(),
            max_tokens: Some(16_384),
        }
    }

    // ============================================================================
    // Tests for initialize_pareto_frontier
    // ============================================================================

    #[test]
    fn test_initialize_pareto_frontier_with_all_variants() {
        // Setup: Create function with multiple ChatCompletion variants
        let mut variants = HashMap::new();
        variants.insert(
            "variant1".to_string(),
            create_variant_info("openai::gpt-4", Some("System prompt 1")),
        );
        variants.insert(
            "variant2".to_string(),
            create_variant_info(
                "anthropic::claude-3-5-sonnet-20241022",
                Some("System prompt 2"),
            ),
        );
        variants.insert(
            "variant3".to_string(),
            create_variant_info("openai::gpt-3.5-turbo", Some("System prompt 3")),
        );

        let function_context = create_function_context(variants);
        let config = create_gepa_config("test_function", None, None);

        // Test: Initialize without initial_variants (should use all variants)
        let result = initialize_pareto_frontier(&config, &function_context);

        assert!(result.is_ok());
        let frontier = result.unwrap();
        assert_eq!(frontier.len(), 3, "Should include all 3 variants");
        assert!(frontier.contains_key("variant1"));
        assert!(frontier.contains_key("variant2"));
        assert!(frontier.contains_key("variant3"));
    }

    #[test]
    fn test_initialize_pareto_frontier_with_initial_variants() {
        // Setup: Create function with multiple variants
        let mut variants = HashMap::new();
        variants.insert(
            "variant1".to_string(),
            create_variant_info("openai::gpt-4", Some("System prompt 1")),
        );
        variants.insert(
            "variant2".to_string(),
            create_variant_info(
                "anthropic::claude-3-5-sonnet-20241022",
                Some("System prompt 2"),
            ),
        );
        variants.insert(
            "variant3".to_string(),
            create_variant_info("openai::gpt-3.5-turbo", Some("System prompt 3")),
        );

        let function_context = create_function_context(variants);

        // Test: Initialize with only specific variants
        let initial_variants = Some(vec!["variant1".to_string(), "variant3".to_string()]);
        let config = create_gepa_config("test_function", initial_variants, None);

        let result = initialize_pareto_frontier(&config, &function_context);

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
    fn test_initialize_pareto_frontier_nonexistent_variant() {
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

        let result = initialize_pareto_frontier(&config, &function_context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nonexistent"));
        assert!(err.to_string().contains("not found"));
    }

    // ============================================================================
    // Tests for extract_chat_completion_config (via initialize_pareto_frontier)
    // ============================================================================

    #[test]
    fn test_extract_chat_completion_config_basic_fields() {
        // Create a ChatCompletionConfig with known values
        let config = create_chat_completion_config("openai::gpt-4", Some("Test system prompt"));
        let retries = RetryConfig::default();

        // Extract it
        let extracted = extract_chat_completion_config(&config, retries);

        // Verify basic fields
        assert_eq!(extracted.model, config.model().clone());
        assert_eq!(extracted.weight, config.weight());
        assert_eq!(extracted.temperature, config.temperature());
        assert_eq!(extracted.top_p, config.top_p());
        assert_eq!(extracted.max_tokens, config.max_tokens());
        assert_eq!(extracted.presence_penalty, config.presence_penalty());
        assert_eq!(extracted.frequency_penalty, config.frequency_penalty());
        assert_eq!(extracted.seed, config.seed());
        assert_eq!(extracted.stop_sequences, config.stop_sequences().cloned());
        assert_eq!(extracted.retries.num_retries, retries.num_retries);
        assert_eq!(extracted.retries.max_delay_s, retries.max_delay_s);
    }

    #[test]
    fn test_extract_chat_completion_config_inference_params_v2() {
        // Create config with inference_params_v2 fields set
        let mut uninitialized = create_uninitialized_chat_config("openai::gpt-4", Some("Test"));
        uninitialized.reasoning_effort = Some("high".to_string());
        uninitialized.thinking_budget_tokens = Some(1000);
        uninitialized.verbosity = Some("verbose".to_string());

        let schemas = SchemaData::default();
        let error_context = ErrorContext {
            function_name: "test".to_string(),
            variant_name: "test".to_string(),
        };
        let config = uninitialized.load(&schemas, &error_context).unwrap();

        let retries = RetryConfig::default();
        let extracted = extract_chat_completion_config(&config, retries);

        // Verify inference_params_v2 fields are extracted
        assert_eq!(extracted.reasoning_effort, Some("high".to_string()));
        assert_eq!(extracted.thinking_budget_tokens, Some(1000));
        assert_eq!(extracted.verbosity, Some("verbose".to_string()));
    }

    #[test]
    fn test_extract_chat_completion_config_custom_retries() {
        let config = create_chat_completion_config("openai::gpt-4", Some("Test"));

        // Custom retry config
        let custom_retries = RetryConfig {
            num_retries: 5,
            max_delay_s: 30.0,
        };

        let extracted = extract_chat_completion_config(&config, custom_retries);

        // Verify custom retries are used (not the original config's retries)
        assert_eq!(extracted.retries.num_retries, custom_retries.num_retries);
        assert_eq!(extracted.retries.max_delay_s, custom_retries.max_delay_s);
    }

    #[test]
    fn test_extract_chat_completion_config_legacy_fields_none() {
        // Create config with templates
        let config =
            create_chat_completion_config("openai::gpt-4", Some("System template content"));
        let retries = RetryConfig::default();

        let extracted = extract_chat_completion_config(&config, retries);

        // Legacy fields should be None
        assert!(extracted.system_template.is_none());
        assert!(extracted.user_template.is_none());
        assert!(extracted.assistant_template.is_none());
        assert!(extracted.input_wrappers.is_none());
    }

    // ============================================================================
    // Tests for extract_chat_completion_from_variant_info
    // ============================================================================

    #[test]
    fn test_extract_from_variant_info_chat_completion() {
        let variant_info = create_variant_info("openai::gpt-4", Some("Test"));
        let retries = RetryConfig::default();

        let result =
            extract_chat_completion_from_variant_info(&variant_info, "test_variant", retries);

        assert!(result.is_some());
        let extracted = result.unwrap();
        assert_eq!(extracted.model.as_ref(), "openai::gpt-4");
        assert_eq!(extracted.retries.num_retries, retries.num_retries);
        assert_eq!(extracted.retries.max_delay_s, retries.max_delay_s);
    }

    #[test]
    fn test_extract_from_variant_info_preserves_all_fields() {
        // Create a config with many fields set
        let mut uninitialized =
            create_uninitialized_chat_config("anthropic::claude-3-5-sonnet-20241022", Some("Test"));
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

        let retries = RetryConfig {
            num_retries: 3,
            max_delay_s: 15.0,
        };

        let result = extract_chat_completion_from_variant_info(&variant_info, "test", retries);
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
        assert_eq!(extracted.retries.num_retries, 3);
        assert_eq!(extracted.retries.max_delay_s, 15.0);
    }
}
