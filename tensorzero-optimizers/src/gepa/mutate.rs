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
    variant::chat_completion::{UninitializedChatCompletionConfig, UninitializedChatTemplate},
};

use super::analyze::InferenceWithAnalysis;

/// Output from the GEPA mutate function
///
/// Contains improved prompt templates generated based on aggregated analysis feedback.
/// Uses the new template format with arbitrary template names (not just system/user/assistant).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutateOutput {
    pub templates: HashMap<String, String>,
}

/// Build the input JSON for the mutate function
pub fn build_mutate_input(
    analyses: &[InferenceWithAnalysis],
    function_config: &FunctionConfig,
    parent_variant_config: &UninitializedChatCompletionConfig,
) -> Result<serde_json::Value, Error> {
    // Extract templates from variant_config.templates.inner (new format only)
    let templates_map: HashMap<String, String> = parent_variant_config
        .templates
        .inner
        .iter()
        .map(|(name, config)| config.path.read().map(|content| (name.clone(), content)))
        .collect::<Result<_, _>>()?;

    // Error if empty templates
    if templates_map.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: "Cannot mutate variant with no templates".to_string(),
        }));
    }

    // Extract schemas from function_config.schemas.inner (matching analyze.rs pattern)
    let schemas_map: HashMap<String, serde_json::Value> = function_config
        .schemas()
        .inner
        .iter()
        .map(|(name, schema_with_metadata)| {
            (name.clone(), schema_with_metadata.schema.value.clone())
        })
        .collect();

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
        "templates": templates_map,
        "schemas": if schemas_map.is_empty() { None } else { Some(schemas_map) },
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
#[cfg_attr(not(test), expect(dead_code))]
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

    // Clear existing templates (to ensure clean state)
    new_config.templates.inner.clear();

    // Populate with mutated templates using fake paths
    // Fake paths are used because these templates are generated dynamically, not from files
    for (template_name, content) in mutated_templates.templates {
        new_config.templates.inner.insert(
            template_name.clone(),
            UninitializedChatTemplate {
                path: ResolvedTomlPath::new_fake_path(
                    format!("gepa_mutated/{new_variant_name}/{template_name}.minijinja"),
                    content,
                ),
            },
        );
    }

    // Ensure legacy fields remain None (don't mix old and new formats)
    new_config.system_template = None;
    new_config.user_template = None;
    new_config.assistant_template = None;

    tracing::info!(
        "Created mutated variant '{}' from parent '{}'",
        new_variant_name,
        parent_name
    );

    (new_variant_name, new_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorzero_core::config::path::ResolvedTomlPath;

    #[test]
    fn test_mutate_output_serde() {
        // Create MutateOutput with templates HashMap
        let mut templates = HashMap::new();
        templates.insert("system".to_string(), "Improved system template".to_string());
        templates.insert("user".to_string(), "Improved user template".to_string());

        let mutate_output = MutateOutput { templates };

        // Verify JSON serialization works
        let serialized = serde_json::to_string(&mutate_output).expect("Failed to serialize");
        let deserialized: MutateOutput =
            serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Verify roundtrip
        assert_eq!(deserialized.templates.len(), 2);
        assert_eq!(
            deserialized.templates.get("system").unwrap(),
            "Improved system template"
        );
        assert_eq!(
            deserialized.templates.get("user").unwrap(),
            "Improved user template"
        );
    }

    #[test]
    fn test_mutate_output_from_json() {
        // Test parsing from JSON (as returned by LLM)
        let json = r#"{"templates": {"system": "Test system", "user": "Test user"}}"#;
        let mutate_output: MutateOutput = serde_json::from_str(json).expect("Failed to parse");

        assert_eq!(mutate_output.templates.len(), 2);
        assert_eq!(
            mutate_output.templates.get("system").unwrap(),
            "Test system"
        );
        assert_eq!(mutate_output.templates.get("user").unwrap(), "Test user");
    }

    #[test]
    fn test_create_mutated_variant_uses_new_format() {
        // Create parent config with templates.inner
        let mut parent_config = UninitializedChatCompletionConfig {
            model: "test-model".into(),
            ..Default::default()
        };

        // Add some templates to parent
        parent_config.templates.inner.insert(
            "system".to_string(),
            UninitializedChatTemplate {
                path: ResolvedTomlPath::new_fake_path(
                    "old_system.minijinja".to_string(),
                    "Old system template".to_string(),
                ),
            },
        );

        // Create MutateOutput with improved templates
        let mut improved_templates = HashMap::new();
        improved_templates.insert("system".to_string(), "New system template".to_string());
        improved_templates.insert("user".to_string(), "New user template".to_string());

        let mutate_output = MutateOutput {
            templates: improved_templates,
        };

        // Call create_mutated_variant
        let (variant_name, new_config) =
            create_mutated_variant(&parent_config, mutate_output, 1, "gepa", "baseline");

        // Verify variant name
        assert_eq!(variant_name, "gepa_iter1_baseline");

        // Verify new_config.templates.inner is populated
        assert_eq!(new_config.templates.inner.len(), 2);
        assert!(new_config.templates.inner.contains_key("system"));
        assert!(new_config.templates.inner.contains_key("user"));

        // Verify legacy fields are None
        assert!(new_config.system_template.is_none());
        assert!(new_config.user_template.is_none());
        assert!(new_config.assistant_template.is_none());

        // Verify template content
        let system_template = new_config.templates.inner.get("system").unwrap();
        let system_content = system_template.path.read().expect("Failed to read");
        assert_eq!(system_content, "New system template");

        let user_template = new_config.templates.inner.get("user").unwrap();
        let user_content = user_template.path.read().expect("Failed to read");
        assert_eq!(user_content, "New user template");
    }

    #[test]
    fn test_create_mutated_variant_clears_old_templates() {
        // Create parent config with multiple templates
        let mut parent_config = UninitializedChatCompletionConfig {
            model: "test-model".into(),
            ..Default::default()
        };

        parent_config.templates.inner.insert(
            "system".to_string(),
            UninitializedChatTemplate {
                path: ResolvedTomlPath::new_fake_path(
                    "system.minijinja".to_string(),
                    "Old system".to_string(),
                ),
            },
        );
        parent_config.templates.inner.insert(
            "old_template".to_string(),
            UninitializedChatTemplate {
                path: ResolvedTomlPath::new_fake_path(
                    "old.minijinja".to_string(),
                    "Old template to be removed".to_string(),
                ),
            },
        );

        // Create MutateOutput with only one template
        let mut improved_templates = HashMap::new();
        improved_templates.insert("system".to_string(), "New system".to_string());

        let mutate_output = MutateOutput {
            templates: improved_templates,
        };

        // Call create_mutated_variant
        let (_variant_name, new_config) =
            create_mutated_variant(&parent_config, mutate_output, 1, "gepa", "baseline");

        // Verify parent_config was NOT mutated (defensive check for immutability)
        assert_eq!(
            parent_config.templates.inner.len(),
            2,
            "parent_config should still have 2 templates (not mutated)"
        );
        assert!(
            parent_config.templates.inner.contains_key("system"),
            "parent_config should still contain 'system' template"
        );
        assert!(
            parent_config.templates.inner.contains_key("old_template"),
            "parent_config should still contain 'old_template' template"
        );

        // Verify parent_config template contents unchanged
        let parent_system = parent_config.templates.inner.get("system").unwrap();
        let parent_system_content = parent_system
            .path
            .read()
            .expect("Failed to read parent system");
        assert_eq!(
            parent_system_content, "Old system",
            "parent_config system template content should be unchanged"
        );

        let parent_old = parent_config.templates.inner.get("old_template").unwrap();
        let parent_old_content = parent_old
            .path
            .read()
            .expect("Failed to read parent old_template");
        assert_eq!(
            parent_old_content, "Old template to be removed",
            "parent_config old_template content should be unchanged"
        );

        // Verify only the new template is present (old_template should be gone)
        assert_eq!(new_config.templates.inner.len(), 1);
        assert!(new_config.templates.inner.contains_key("system"));
        assert!(!new_config.templates.inner.contains_key("old_template"));

        // Verify new_config has the mutated content (not the old content)
        let new_system = new_config.templates.inner.get("system").unwrap();
        let new_system_content = new_system.path.read().expect("Failed to read new system");
        assert_eq!(
            new_system_content, "New system",
            "new_config system template should contain the mutated content"
        );
    }
}
