//! Template mutation functions for GEPA optimization
//!
//! This module provides functions for:
//! - Generating improved templates using the mutate function
//! - Creating mutated variant configurations
//! - Parsing mutation responses

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use tensorzero_core::{
    client::{
        Client, ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
        InferenceOutput,
    },
    config::{path::ResolvedTomlPath, UninitializedVariantConfig, UninitializedVariantInfo},
    endpoints::inference::InferenceResponse,
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    inference::types::{Role, TextKind},
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use super::analyze::InferenceWithAnalysis;

/// Output from the GEPA mutate function
///
/// Contains improved prompt templates generated based on aggregated analysis feedback.
/// Templates are only present if they existed in the original variant configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutateOutput {
    pub system_template: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assistant_template: Option<String>,
}

/// Build the input JSON for the mutate function
pub fn build_mutate_input(
    analyses: &[InferenceWithAnalysis],
    function_config: &FunctionConfig,
    parent_variant_config: &UninitializedChatCompletionConfig,
) -> Result<serde_json::Value, Error> {
    // Extract templates from parent variant config
    // For now, we'll use placeholder values - in production, we'd need to read the actual template content
    let system_template = ""; // TODO: Extract actual content from parent_variant_config.system_template
    let user_template: Option<String> = None; // TODO: Extract from parent_variant_config.user_template
    let assistant_template: Option<String> = None; // TODO: Extract from parent_variant_config.assistant_template

    // Extract schemas from function config
    let system_schema = function_config.system_schema().map(|s| s.value.clone());
    let user_schema = function_config.user_schema().map(|s| s.value.clone());
    let assistant_schema = function_config.assistant_schema().map(|s| s.value.clone());

    // Extract output schema for JSON functions
    let output_schema = match function_config {
        FunctionConfig::Json(params) => Some(params.output_schema.value.clone()),
        FunctionConfig::Chat(_) => None,
    };

    // Extract tools for Chat functions
    let tools = match function_config {
        FunctionConfig::Chat(params) => {
            if params.tools.is_empty() {
                None
            } else {
                Some(serde_json::json!(params.tools))
            }
        }
        FunctionConfig::Json(_) => None,
    };

    // Serialize analyses array
    let analyses_json = serde_json::to_value(analyses).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize analyses: {e}"),
        })
    })?;

    // Get function name from one of the analyses (they should all be for the same function)
    let function_name = if let Some(first_analysis) = analyses.first() {
        first_analysis.inference_output.variant_name()
    } else {
        "unknown"
    };

    // Build the input object
    let input = serde_json::json!({
        "function_name": function_name,
        "model": parent_variant_config.model,
        "system_template": system_template,
        "user_template": user_template,
        "assistant_template": assistant_template,
        "system_schema": system_schema,
        "user_schema": user_schema,
        "assistant_schema": assistant_schema,
        "output_schema": output_schema,
        "tools": tools,
        "analyses": analyses_json,
    });

    Ok(input)
}

/// Parse the mutate response and extract the MutateOutput
pub fn parse_mutate_response(response: &InferenceResponse) -> Result<MutateOutput, Error> {
    let json_response = match response {
        InferenceResponse::Json(json_response) => json_response,
        InferenceResponse::Chat(_) => {
            return Err(Error::new(ErrorDetails::Inference {
                message: "Expected Json response from mutate function, got Chat".to_string(),
            }))
        }
    };

    // Extract the parsed JSON output
    let parsed_output = json_response.output.parsed.clone().ok_or_else(|| {
        Error::new(ErrorDetails::Inference {
            message: "Mutate function returned no parsed output".to_string(),
        })
    })?;

    // Parse the JSON output into MutateOutput
    serde_json::from_value(parsed_output).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to parse mutate output: {e}"),
        })
    })
}

/// Generate improved templates using the GEPA mutate function
///
/// Takes aggregated analyses and calls the built-in `tensorzero::optimization::gepa::mutate`
/// function to synthesize improved prompt templates.
///
/// Returns MutateOutput with improved templates.
#[expect(dead_code)]
pub async fn mutate_templates(
    gateway_client: &Client,
    analyses: &[InferenceWithAnalysis],
    function_config: &FunctionConfig,
    parent_variant_config: &UninitializedChatCompletionConfig,
    mutation_model: &str,
) -> Result<MutateOutput, Error> {
    tracing::info!(
        "Generating improved templates using {} analyses with model '{}'",
        analyses.len(),
        mutation_model
    );

    // Build input JSON for the mutate function
    let input_data = build_mutate_input(analyses, function_config, parent_variant_config)?;

    // Create dynamic variant config for the mutate function
    let mutate_variant_config = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(UninitializedChatCompletionConfig {
            model: mutation_model.into(),
            weight: None,
            system_template: Some(ResolvedTomlPath::new_fake_path(
                "gepa/mutate/system.minijinja".to_string(),
                include_str!("config/functions/mutate/baseline/system_template.minijinja")
                    .to_string(),
            )),
            user_template: Some(ResolvedTomlPath::new_fake_path(
                "gepa/mutate/user.minijinja".to_string(),
                include_str!("config/functions/mutate/baseline/user_template.minijinja")
                    .to_string(),
            )),
            assistant_template: None,
            ..Default::default()
        }),
        timeouts: None,
    };

    // Create ClientInferenceParams for the mutate function
    let params = ClientInferenceParams {
        function_name: Some("tensorzero::optimization::gepa::mutate".to_string()),
        model_name: None,
        episode_id: None,
        input: ClientInput {
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: serde_json::to_string(&input_data).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!("Failed to serialize mutate input: {e}"),
                        })
                    })?,
                })],
            }],
            system: None,
        },
        stream: None,
        params: Default::default(),
        variant_name: None,
        dryrun: None,
        internal: true,
        tags: HashMap::new(),
        dynamic_tool_params: Default::default(),
        output_schema: None,
        credentials: HashMap::new(),
        cache_options: Default::default(),
        include_original_response: false,
        extra_body: Default::default(),
        extra_headers: Default::default(),
        internal_dynamic_variant_config: Some(mutate_variant_config),
        otlp_traces_extra_headers: HashMap::new(),
    };

    // Call the inference API
    let inference_output = gateway_client.inference(params).await.map_err(|e| {
        Error::new(ErrorDetails::Inference {
            message: format!("Failed to call mutate function: {e}"),
        })
    })?;

    // Extract the response
    let response = match inference_output {
        InferenceOutput::NonStreaming(response) => response,
        InferenceOutput::Streaming(_) => {
            return Err(Error::new(ErrorDetails::Inference {
                message: "Unexpected streaming response from mutate function".to_string(),
            }))
        }
    };

    // Parse the mutate output (JSON response)
    let mutate_output = parse_mutate_response(&response)?;

    Ok(mutate_output)
}

/// Create a new variant with mutated templates
///
/// Generates a new variant name and clones the parent variant config with updated templates.
#[expect(dead_code)]
pub fn create_mutated_variant(
    parent_config: &UninitializedChatCompletionConfig,
    mutated_templates: MutateOutput,
    iteration: usize,
    variant_prefix: &str,
    parent_name: &str,
) -> (String, UninitializedChatCompletionConfig) {
    // Generate variant name: {prefix}_iter{iteration}_{parent_name}
    let new_variant_name = format!("{variant_prefix}_iter{iteration}_{parent_name}");

    // Clone parent config
    let mut new_config = parent_config.clone();

    // Replace templates with mutated versions using fake paths
    // Fake paths are used because these templates are generated dynamically, not from files
    new_config.system_template = Some(ResolvedTomlPath::new_fake_path(
        format!("gepa_mutated/{new_variant_name}/system.minijinja"),
        mutated_templates.system_template,
    ));

    if let Some(user_template) = mutated_templates.user_template {
        new_config.user_template = Some(ResolvedTomlPath::new_fake_path(
            format!("gepa_mutated/{new_variant_name}/user.minijinja"),
            user_template,
        ));
    }

    if let Some(assistant_template) = mutated_templates.assistant_template {
        new_config.assistant_template = Some(ResolvedTomlPath::new_fake_path(
            format!("gepa_mutated/{new_variant_name}/assistant.minijinja"),
            assistant_template,
        ));
    }

    tracing::info!(
        "Created mutated variant '{}' from parent '{}'",
        new_variant_name,
        parent_name
    );

    (new_variant_name, new_config)
}
