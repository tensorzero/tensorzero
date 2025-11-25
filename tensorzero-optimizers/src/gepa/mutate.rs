//! Template mutation functions for GEPA optimization
//!
//! This module provides functions for:
//! - Generating improved templates using the mutate function
//! - Creating mutated variant configurations
//! - Parsing mutation responses

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{from_value, json, to_value, Map};

use tensorzero_core::{
    client::{
        Client, ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
        InferenceOutput,
    },
    config::{path::ResolvedTomlPathData, UninitializedVariantConfig, UninitializedVariantInfo},
    endpoints::inference::InferenceResponse,
    error::{Error, ErrorDetails},
    inference::types::{Arguments, Role, Template},
    optimization::gepa::GEPAConfig,
    variant::chat_completion::{UninitializedChatCompletionConfig, UninitializedChatTemplate},
};

use crate::gepa::{analyze::Analysis, validate::FunctionContext};

/// Helper struct to deserialize the JSON response that matches the output schema.
/// The schema defines templates as an array of objects with name and content fields.
#[derive(Debug, Deserialize)]
struct MutateResponse {
    templates: Vec<TemplateEntry>,
}

/// Individual template entry in the mutate response array.
#[derive(Debug, Deserialize)]
struct TemplateEntry {
    name: String,
    content: String,
}

/// Creates variant configuration for the mutate function.
///
/// Builds uninitialized chat completion config with embedded templates using GEPAConfig settings.
///
/// Returns configured UninitializedChatCompletionConfig with system and user templates.
fn create_mutate_variant_config(gepa_config: &GEPAConfig) -> UninitializedChatCompletionConfig {
    let mut mutate_config = UninitializedChatCompletionConfig {
        model: gepa_config.mutation_model.clone().into(),
        weight: None,
        retries: gepa_config.retries,
        max_tokens: gepa_config.max_tokens,
        ..Default::default()
    };

    mutate_config.templates.inner.insert(
        "system".to_string(),
        UninitializedChatTemplate {
            path: ResolvedTomlPathData::new_fake_path(
                "gepa/mutate/system.minijinja".to_string(),
                include_str!("functions/mutate/system_template.minijinja").to_string(),
            ),
        },
    );
    mutate_config.templates.inner.insert(
        "user".to_string(),
        UninitializedChatTemplate {
            path: ResolvedTomlPathData::new_fake_path(
                "gepa/mutate/user.minijinja".to_string(),
                include_str!("functions/mutate/user_template.minijinja").to_string(),
            ),
        },
    );

    mutate_config
}

/// Builds input JSON for the mutate function.
///
/// Passes high-level objects to the template for serialization.
///
/// Returns Arguments with template variables: function_config, static_tools, evaluation_config,
/// templates_map, and analyses.
///
/// Returns error if serialization fails.
pub fn build_mutate_input(
    analyses: &[Analysis],
    function_context: &FunctionContext,
    variant_config: &UninitializedChatCompletionConfig,
) -> Result<Arguments, Error> {
    // Extract fields from function context
    let FunctionContext {
        function_config,
        static_tools,
        evaluation_config,
    } = function_context;

    // Extract templates map from variant config
    let templates_map: HashMap<String, String> = variant_config
        .templates
        .inner
        .iter()
        .map(|(name, config)| (name.clone(), config.path.data().to_string()))
        .collect();

    // Build analyses map
    let analyses_json = serde_json::to_value(analyses).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize analyses: {e}"),
        })
    })?;

    // Build the input with high-level objects that will be serialized in the template
    let mut map = Map::new();
    map.insert("function_config".to_string(), to_value(function_config)?);
    if let Some(tools) = static_tools {
        map.insert("static_tools".to_string(), json!(tools));
    }
    map.insert(
        "evaluation_config".to_string(),
        to_value(evaluation_config)?,
    );
    map.insert("templates_map".to_string(), json!(templates_map));
    map.insert("analyses".to_string(), analyses_json);

    Ok(Arguments(map))
}

/// Generate improved templates using the GEPA mutate function.
///
/// Takes aggregated analyses and calls the built-in `tensorzero::optimization::gepa::mutate`
/// function to synthesize improved prompt templates.
///
/// Returns MutateOutput with improved templates.
///
/// Returns error only if mutation fails.
pub async fn mutate_variant(
    gateway_client: &Client,
    analyses: &[Analysis],
    function_context: &FunctionContext,
    parent: HashMap<&String, &UninitializedChatCompletionConfig>,
    gepa_config: &GEPAConfig,
    iteration: usize,
) -> Result<HashMap<String, UninitializedChatCompletionConfig>, Error> {
    // Extract the single parent from the HashMap
    // parent is HashMap<&String, &UninitializedChatCompletionConfig>, so .iter() gives (&&String, &&UninitializedChatCompletionConfig)
    let (parent_name, parent_config) =
        parent.iter().next().map(|(k, v)| (*k, *v)).ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: "parent HashMap must contain exactly one entry".to_string(),
            })
        })?;

    let mutation_model = &gepa_config.mutation_model;
    tracing::info!(
        "Generating improved templates using {} analyses with model '{}'",
        analyses.len(),
        mutation_model
    );

    // Create mutation variant configuration
    let mutate_variant_config = create_mutate_variant_config(gepa_config);
    let mutate_variant_info = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(mutate_variant_config),
        timeouts: None,
    };

    // Build input Arguments for the mutate function
    let arguments = build_mutate_input(analyses, function_context, parent_config)?;

    // Create ClientInferenceParams for the analyze function
    let params = ClientInferenceParams {
        function_name: Some("tensorzero::optimization::gepa::mutate".to_string()),
        input: ClientInput {
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments,
                })],
            }],
            system: None,
        },
        dryrun: Some(true), // Required when using internal_dynamic_variant_config
        internal: true,
        internal_dynamic_variant_config: Some(mutate_variant_info.clone()),
        ..Default::default()
    };

    // Call the inference API
    let inference_output = gateway_client.inference(params).await.map_err(|e| {
        Error::new(ErrorDetails::Inference {
            message: format!("Failed to call mutate function: {e}"),
        })
    })?;

    // Extract the response
    let InferenceOutput::NonStreaming(response) = inference_output else {
        return Err(Error::new(ErrorDetails::Inference {
            message: "Expected NonStreaming response but got Streaming".to_string(),
        }));
    };

    // Extract json content from the response
    let InferenceResponse::Json(json_response) = &response else {
        return Err(Error::new(ErrorDetails::Inference {
            message: "mutate function is defined as Json, cannot return Chat".to_string(),
        }));
    };

    // Try to get parsed output first, otherwise parse the raw output
    let output_value = json_response
        .output
        .parsed
        .clone()
        .or_else(|| {
            json_response
                .output
                .raw
                .as_ref()
                .and_then(|raw| serde_json::from_str(raw).ok())
        })
        .ok_or_else(|| {
            Error::new(ErrorDetails::Inference {
                message: "Mutate function returned no parsed or raw output".to_string(),
            })
        })?;

    // Deserialize the response which has the schema: {"templates": [{"name": "...", "content": "..."}, ...]}
    let response: MutateResponse = from_value(output_value).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to deserialize mutate output: {e}"),
        })
    })?;

    // Convert the array of template entries into a HashMap
    let templates: HashMap<String, String> = response
        .templates
        .into_iter()
        .map(|entry| (entry.name, entry.content))
        .collect();

    // Generate variant name: {prefix}-iter-{iteration}-{parent_name}
    let mutated_variant_name = format!(
        "{}-iter-{}-{}",
        gepa_config.variant_prefix.as_deref().unwrap_or("gepa"),
        iteration,
        parent_name
    );

    // Clone parent config
    let mut mutated_config = parent_config.clone();

    // Set retry configuration from GEPA config
    mutated_config.retries = gepa_config.retries;

    // Clear existing templates (to ensure clean state)
    mutated_config.templates.inner.clear();

    // Populate with mutated templates using fake paths
    // Fake paths are used because these templates are generated dynamically, not from files
    for (template_name, content) in templates {
        mutated_config.templates.inner.insert(
            template_name.clone(),
            UninitializedChatTemplate {
                path: ResolvedTomlPathData::new_fake_path(
                    format!("gepa_mutated/{mutated_variant_name}/{template_name}.minijinja"),
                    content,
                ),
            },
        );
    }

    let mut output = HashMap::new();
    output.insert(mutated_variant_name, mutated_config);
    Ok(output)
}
