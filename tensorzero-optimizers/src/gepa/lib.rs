//! GEPA (Genetic Evolution with Pareto Analysis) optimizer implementation
//!
//! This module contains the core GEPA algorithm logic, separated from the trait
//! implementations in mod.rs for cleaner organization.

use std::collections::HashMap;

/// Minimum number of valid examples required for GEPA optimization
const MIN_EXAMPLES: usize = 10;

use tensorzero_core::{
    client::{Client, ClientBuilder, ClientBuilderMode},
    config::Config,
    db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo},
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    http::TensorzeroHttpClient,
    inference::types::{ContentBlockChatOutput, StoredInputMessageContent},
    optimization::gepa::GEPAConfig,
    stored_inference::{RenderedSample, StoredOutput},
    variant::{chat_completion::UninitializedChatCompletionConfig, VariantConfig},
};

// Re-export public types and functions from sibling modules
#[expect(unused_imports)]
pub use super::analyze::{analyze_inferences, InferenceWithAnalysis};
#[expect(unused_imports)]
pub use super::evaluate::{create_evaluation_dataset, evaluate_variants, EvaluationResults};
#[expect(unused_imports)]
pub use super::mutate::{create_mutated_variant, mutate_templates, MutateOutput};
#[expect(unused_imports)]
pub use super::pareto::{is_improvement, update_pareto_frontier};
#[expect(unused_imports)]
pub use super::sample::{random_sample, sample_by_frequency};

// Import utils module for config extraction
use super::utils;

/// Main GEPA optimization orchestration function
///
/// This function implements the GEPA algorithm:
/// 1. Initialize with baseline variants or provided initial variants
/// 2. For each iteration:
///    a. Analyze examples using the current variants
///    b. Identify Pareto-optimal variants based on multi-objective metrics
///    c. Generate new variants by mutating the Pareto frontier
///    d. Update the Pareto frontier with new variants
/// 3. Return the final Pareto frontier as variant configurations
pub async fn run_gepa_optimization(
    config: &GEPAConfig,
    client: &TensorzeroHttpClient,
    train_examples: Vec<RenderedSample>,
    val_examples: Option<Vec<RenderedSample>>,
    _credentials: &InferenceCredentials,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    tensorzero_config: std::sync::Arc<Config>,
) -> Result<HashMap<String, UninitializedChatCompletionConfig>, Error> {
    // Validate configuration and examples, get the function config
    let function_config = validate_gepa_config(config, &tensorzero_config)?;

    // Require validation examples for GEPA
    let val_examples = val_examples.ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: "val_examples are required for GEPA optimization (used for Pareto frontier filtering)".to_string(),
        })
    })?;

    // Validate both train and validation examples (this filters invalid examples)
    let train_examples = validate_examples(&train_examples)?;
    let val_examples = validate_examples(&val_examples)?;

    tracing::info!(
        "Starting GEPA optimization for function '{}' with {} train examples and {} val examples",
        config.function_name,
        train_examples.len(),
        val_examples.len()
    );

    // Build the gateway client ONCE for the entire optimization run
    // This avoids creating ~201 gateway instances (each with background tasks)
    let _gateway_client = ClientBuilder::new(ClientBuilderMode::FromComponents {
        config: tensorzero_config.clone(),
        clickhouse_connection_info: clickhouse_connection_info.clone(),
        postgres_connection_info: PostgresConnectionInfo::Disabled,
        http_client: client.clone(),
        timeout: None,
    })
    .build()
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to build gateway client for GEPA optimization: {e}"),
        })
    })?;

    tracing::info!("Gateway client built successfully for GEPA optimization");

    // Initialize the Pareto frontier with baseline or provided variants
    let pareto_frontier = initialize_pareto_frontier(config, function_config)?;

    // Create validation dataset for Pareto filtering
    let val_dataset_name = format!(
        "{}_gepa_val_{}",
        config.evaluation_name,
        uuid::Uuid::now_v7()
    );
    tracing::info!(
        "Creating validation dataset '{}' with {} examples",
        val_dataset_name,
        val_examples.len()
    );

    let _val_datapoint_ids = create_evaluation_dataset(
        &tensorzero_config,
        client,
        clickhouse_connection_info,
        &val_examples,
        &val_dataset_name,
    )
    .await?;

    tracing::info!("Validation dataset created successfully");

    // TODO: Evaluate initial variants on validation set and get val_scores
    // let val_scores = evaluate_variants(..., &val_examples, ...)?;

    // TODO: Initial Pareto frontier filtering
    // let (mut pareto_frontier, mut frequencies) = update_pareto_frontier(
    //     pareto_frontier,
    //     &val_scores,
    //     config,
    //     tensorzero_config,
    // )?;

    // Main GEPA loop
    for iteration in 0..config.max_iterations {
        tracing::info!("GEPA iteration {}/{}", iteration + 1, config.max_iterations);

        // TODO: 1. Sample mini-batch from train_examples
        // TODO: 2. Sample variant to mutate (proportional to frequency)
        // TODO: 3. Generate mutation using mutate() function (which does analysis internally)
        // TODO: 4. Evaluate mutation on batch
        // TODO: 5. If improvement, evaluate on validation set and add to val_scores
        // TODO: 6. Update Pareto frontier using instance-wise dominance on val_scores

        tracing::info!(
            "GEPA iteration {} complete. Pareto frontier size: {}",
            iteration + 1,
            pareto_frontier.len()
        );
    }

    // Return the final Pareto frontier
    tracing::info!(
        "GEPA optimization complete. Final Pareto frontier size: {}",
        pareto_frontier.len()
    );

    Ok(pareto_frontier)
}

/// Validates the GEPA configuration and checks that required resources exist
/// Returns the FunctionConfig for the function being optimized
fn validate_gepa_config<'a>(
    config: &GEPAConfig,
    tensorzero_config: &'a Config,
) -> Result<&'a FunctionConfig, Error> {
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

    // Check that the evaluation exists (fixed: should check evaluations, not metrics)
    if !tensorzero_config
        .evaluations
        .contains_key(&config.evaluation_name)
    {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "evaluation '{}' not found in configuration",
                config.evaluation_name
            ),
        }));
    }

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

    // Validate tools referenced by function exist in config
    let function_tool_names: Vec<String> = match &**function_config {
        FunctionConfig::Chat(chat_config) => chat_config.tools.clone(),
        FunctionConfig::Json(_) => {
            // JSON functions have implicit tool for schema, which is always available
            Vec::new()
        }
    };

    for tool_name in &function_tool_names {
        if !tensorzero_config.tools.contains_key(tool_name) {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "tool '{}' referenced by function '{}' not found in configuration",
                    tool_name, config.function_name
                ),
            }));
        }
    }

    Ok(function_config)
}

/// Validates and filters examples for GEPA optimization
///
/// Filters out examples with:
/// - stored_output is None
/// - stored_output is JsonInferenceOutput with parsed is None
/// - stored_output is ChatInferenceOutput with length 0
/// - Any message has no content blocks (empty content list)
/// - Any message contains FileUrl, ImageBase64, FileBase64, or ImageUrl
/// - Any Text block has both text and arguments as None
/// - Any ToolCall block has arguments is None or name is None
/// - Any Thought block has both text and summary as None
/// - Invalid stored_input.system (not None, Text, or object)
///
/// Returns filtered list of valid examples, or error if all examples are dropped
fn validate_examples(examples: &[RenderedSample]) -> Result<Vec<RenderedSample>, Error> {
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
fn initialize_pareto_frontier(
    config: &GEPAConfig,
    function_config: &FunctionConfig,
) -> Result<HashMap<String, UninitializedChatCompletionConfig>, Error> {
    let variants = function_config.variants();
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

            if let Some(chat_config) =
                utils::extract_chat_completion_from_variant_info(variant_info, variant_name)
            {
                frontier.insert(variant_name.clone(), chat_config);
                tracing::info!("Using initial variant: {}", variant_name);
            }
        }
    } else {
        // Use all ChatCompletion variants from the function
        for (variant_name, variant_info) in variants {
            if let Some(chat_config) =
                utils::extract_chat_completion_from_variant_info(variant_info, variant_name)
            {
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

/// Generate mutation of a variant using GEPA analysis and mutation
/// Returns None if mutation generation fails
///
/// Process:
/// 1. Run inference on batch_examples using candidate variant (with internal_dynamic_variant_config)
/// 2. Analyze each inference using tensorzero::optimization::gepa::analyze built-in function
///    (with internal_dynamic_variant_config for the analysis model)
/// 3. Generate mutation using tensorzero::optimization::gepa::mutate built-in function
///    (with internal_dynamic_variant_config for the mutation model)
/// 4. Return new variant config with updated templates
#[expect(dead_code)]
#[expect(clippy::too_many_arguments)]
async fn mutate(
    _gateway_client: &Client,
    _credentials: &InferenceCredentials,
    _function_name: &str,
    _variant_config: &UninitializedChatCompletionConfig,
    _batch_examples: &[RenderedSample],
    _analysis_model: &str,
    _mutation_model: &str,
    _job_id: &uuid::Uuid,
    _max_concurrency: u32,
) -> Option<UninitializedChatCompletionConfig> {
    // TODO: Implement mutation generation
    // 1. Run inference on batch_examples using variant_config
    // 2. For each inference result, call tensorzero::optimization::gepa::analyze
    // 3. Aggregate analysis results
    // 4. Call tensorzero::optimization::gepa::mutate with analysis and variant config
    // 5. Parse mutation response and create new UninitializedChatCompletionConfig
    // 6. Return new config (or None if mutation failed)

    None
}

/// Convert candidate variants to output format for the job handle
/// This is the final step before returning the Pareto frontier
#[expect(dead_code)]
fn convert_variants_to_output(
    variants: &HashMap<String, UninitializedChatCompletionConfig>,
) -> HashMap<String, UninitializedChatCompletionConfig> {
    // TODO: Implement conversion if needed
    // For now, the variants are already in the correct format (UninitializedChatCompletionConfig)
    // This function exists as a placeholder in case we need to do any transformations
    // or validation before returning the final results

    variants.clone()
}
