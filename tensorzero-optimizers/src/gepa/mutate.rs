//! Template mutation functions for GEPA optimization
//!
//! This module provides functions for:
//! - Generating improved templates using the mutate function
//! - Creating mutated variant configurations
//! - Parsing mutation responses

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::json;

use tensorzero_core::{
    client::{
        Client, ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
        InferenceOutput,
    },
    config::{path::ResolvedTomlPath, UninitializedVariantConfig, UninitializedVariantInfo},
    endpoints::inference::InferenceResponse,
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    inference::types::{Arguments, Role, Template},
    tool::ToolCallConfigDatabaseInsert,
    utils::retries::RetryConfig,
    variant::chat_completion::{UninitializedChatCompletionConfig, UninitializedChatTemplate},
};

use super::analyze::InferenceWithAnalysis;
use super::{
    utils::{extract_templates_map, serialize_to_value},
    validate::FunctionConfigAndTools,
};

/// Output from the GEPA mutate function
///
/// Contains improved prompt templates generated based on aggregated analysis feedback.
/// Uses the new template format with arbitrary template names (not just system/user/assistant).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutateOutput {
    pub templates: HashMap<String, String>,
}

/// Template entry from mutate function output (array format for OpenAI strict mode)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TemplateEntry {
    name: String,
    content: String,
}

/// Raw output from mutate function (array format)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MutateOutputRaw {
    templates: Vec<TemplateEntry>,
}

/// Build the input Arguments for the mutate function
pub fn build_mutate_input(
    analyses: &[InferenceWithAnalysis],
    config_and_tools: &FunctionConfigAndTools,
    parent_variant_config: &UninitializedChatCompletionConfig,
) -> Result<Arguments, Error> {
    // Extract templates using helper function
    let templates_map = extract_templates_map(parent_variant_config)?;

    // Error if empty templates
    if templates_map.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: "Cannot mutate variant with no templates".to_string(),
        }));
    }

    // Extract schemas using method on FunctionConfigAndTools
    let schemas_map = config_and_tools.extract_schemas_map();

    // Extract output schema for JSON functions
    let output_schema = match &*config_and_tools.function_config {
        FunctionConfig::Json(params) => Some(params.output_schema.value.clone()),
        FunctionConfig::Chat(_) => None,
    };

    // Extract and serialize tool configuration as ToolCallConfig
    let tools = {
        let empty_tools = HashMap::new();
        // Use default ToolCallConfigDatabaseInsert (no datapoint tool_params in mutate)
        let tool_params = ToolCallConfigDatabaseInsert::default();

        let tool_config = tool_params.into_tool_call_config(
            &config_and_tools.function_config,
            config_and_tools
                .static_tools
                .as_ref()
                .unwrap_or(&empty_tools),
        )?;

        // Serialize full ToolCallConfig
        tool_config
            .map(|config| serialize_to_value(&config, "tool config"))
            .transpose()?
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

    // Build the input Arguments using Map pattern (matching analyze.rs)
    let mut map = serde_json::Map::new();
    map.insert("function_name".to_string(), json!(function_name));
    map.insert(
        "model".to_string(),
        json!(parent_variant_config.model.as_ref()),
    );
    map.insert("templates".to_string(), json!(templates_map));

    // Always include optional fields (even if empty/null) so templates can check them
    map.insert(
        "schemas".to_string(),
        if schemas_map.is_empty() {
            json!(null)
        } else {
            json!(schemas_map)
        },
    );
    map.insert("output_schema".to_string(), json!(output_schema));
    map.insert("tools".to_string(), json!(tools));
    map.insert("analyses".to_string(), analyses_json);

    Ok(Arguments(map))
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

    // Try to get parsed output first, otherwise parse the raw output
    let output_value = if let Some(parsed) = json_response.output.parsed.clone() {
        parsed
    } else if let Some(raw) = json_response.output.raw.as_ref() {
        serde_json::from_str(raw).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to parse raw mutate output as JSON: {e}"),
            })
        })?
    } else {
        return Err(Error::new(ErrorDetails::Inference {
            message: "Mutate function returned no parsed or raw output".to_string(),
        }));
    };

    // Parse the JSON output from array format to HashMap
    let raw_output: MutateOutputRaw = serde_json::from_value(output_value).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to deserialize mutate output: {e}"),
        })
    })?;

    // Convert array format to HashMap
    let templates = raw_output
        .templates
        .into_iter()
        .map(|entry| (entry.name, entry.content))
        .collect();

    Ok(MutateOutput { templates })
}

/// Generate improved templates using the GEPA mutate function
///
/// Takes aggregated analyses and calls the built-in `tensorzero::optimization::gepa::mutate`
/// function to synthesize improved prompt templates.
///
/// Returns MutateOutput with improved templates.
pub async fn mutate_templates(
    gateway_client: &Client,
    analyses: &[InferenceWithAnalysis],
    config_and_tools: &FunctionConfigAndTools,
    parent_variant_config: &UninitializedChatCompletionConfig,
    gepa_config: &tensorzero_core::optimization::gepa::GEPAConfig,
) -> Result<MutateOutput, Error> {
    let mutation_model = &gepa_config.mutation_model;

    tracing::info!(
        "Generating improved templates using {} analyses with model '{}'",
        analyses.len(),
        mutation_model
    );

    // Build input Arguments for the mutate function
    let arguments = build_mutate_input(analyses, config_and_tools, parent_variant_config)?;

    // Create dynamic variant config for the mutate function using new template format
    let mut mutate_config = UninitializedChatCompletionConfig {
        model: mutation_model.clone().into(),
        weight: None,
        max_tokens: Some(gepa_config.max_tokens),
        retries: gepa_config.retries,
        ..Default::default()
    };

    // Populate templates.inner with the mutate function's templates
    mutate_config.templates.inner.insert(
        "system".to_string(),
        UninitializedChatTemplate {
            path: ResolvedTomlPath::new_fake_path(
                "gepa/mutate/system.minijinja".to_string(),
                include_str!("config/functions/mutate/baseline/system_template.minijinja")
                    .to_string(),
            ),
        },
    );
    mutate_config.templates.inner.insert(
        "user".to_string(),
        UninitializedChatTemplate {
            path: ResolvedTomlPath::new_fake_path(
                "gepa/mutate/user.minijinja".to_string(),
                include_str!("config/functions/mutate/baseline/user_template.minijinja")
                    .to_string(),
            ),
        },
    );

    let mutate_variant_config = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(mutate_config),
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
                content: vec![ClientInputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments,
                })],
            }],
            system: None,
        },
        stream: None,
        params: Default::default(),
        variant_name: None,
        dryrun: Some(true),
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
pub fn create_mutated_variant(
    parent_config: &UninitializedChatCompletionConfig,
    mutated_templates: MutateOutput,
    iteration: usize,
    variant_prefix: &str,
    parent_name: &str,
    retries: RetryConfig,
) -> (String, UninitializedChatCompletionConfig) {
    // Generate variant name: {prefix}_iter{iteration}_{parent_name}
    let new_variant_name = format!("{variant_prefix}_iter{iteration}_{parent_name}");

    // Clone parent config
    let mut new_config = parent_config.clone();

    // Set retry configuration from GEPA config
    new_config.retries = retries;

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
        let (variant_name, new_config) = create_mutated_variant(
            &parent_config,
            mutate_output,
            1,
            "gepa",
            "baseline",
            RetryConfig::default(),
        );

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
        let (_variant_name, new_config) = create_mutated_variant(
            &parent_config,
            mutate_output,
            1,
            "gepa",
            "baseline",
            RetryConfig::default(),
        );

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

    #[test]
    fn test_build_mutate_input_with_schemas() {
        use crate::gepa::validate::FunctionConfigAndTools;
        use std::sync::Arc;
        use tensorzero_core::{
            config::SchemaData,
            function::{FunctionConfig, FunctionConfigChat},
            jsonschema_util::{SchemaWithMetadata, StaticJSONSchema},
        };

        // Create function config with schemas
        let system_schema = StaticJSONSchema::from_value(serde_json::json!({
            "type": "object",
            "properties": {
                "greeting": {"type": "string"}
            }
        }))
        .expect("Failed to create system schema");

        let mut schema_inner = HashMap::new();
        schema_inner.insert(
            "system".to_string(),
            SchemaWithMetadata {
                schema: system_schema,
                legacy_definition: false,
            },
        );

        let schemas = SchemaData {
            inner: schema_inner,
        };

        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas,
            tools: vec![],
            tool_choice: tensorzero_core::tool::ToolChoice::None,
            parallel_tool_calls: None,
            description: Some("Test function with schemas".to_string()),
            all_explicit_templates_names: std::collections::HashSet::new(),
            experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
        });

        // Create variant config with templates
        let mut variant_config = UninitializedChatCompletionConfig {
            model: "test-model".into(),
            ..Default::default()
        };

        variant_config.templates.inner.insert(
            "system".to_string(),
            UninitializedChatTemplate {
                path: ResolvedTomlPath::new_fake_path(
                    "system.minijinja".to_string(),
                    "Test system template".to_string(),
                ),
            },
        );

        // Create empty analyses
        let analyses: Vec<InferenceWithAnalysis> = vec![];

        // Wrap function_config in FunctionConfigAndTools
        let config_and_tools = FunctionConfigAndTools {
            function_config: Arc::new(function_config),
            static_tools: None,
        };

        // Call build_mutate_input
        let result = build_mutate_input(&analyses, &config_and_tools, &variant_config);

        // Assert: Should succeed
        assert!(result.is_ok(), "build_mutate_input should succeed");
        let arguments = result.unwrap();

        // Verify schemas are included in the Arguments map
        assert!(
            arguments.0.contains_key("schemas"),
            "Arguments should contain schemas key"
        );

        // Verify schema structure
        let schemas_value = arguments.0.get("schemas").unwrap();
        assert!(
            schemas_value.is_object(),
            "schemas value should be an object"
        );
        let schemas_obj = schemas_value.as_object().unwrap();
        assert!(
            schemas_obj.contains_key("system"),
            "schemas should contain system schema"
        );
    }

    #[test]
    fn test_build_mutate_input_empty_templates_error() {
        use crate::gepa::validate::FunctionConfigAndTools;
        use std::sync::Arc;
        use tensorzero_core::function::{FunctionConfig, FunctionConfigChat};

        // Create function config
        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: tensorzero_core::config::SchemaData::default(),
            tools: vec![],
            tool_choice: tensorzero_core::tool::ToolChoice::None,
            parallel_tool_calls: None,
            description: Some("Test function".to_string()),
            all_explicit_templates_names: std::collections::HashSet::new(),
            experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
        });

        // Create variant config with EMPTY templates.inner
        let variant_config = UninitializedChatCompletionConfig {
            model: "test-model".into(),
            ..Default::default()
        };

        // Verify templates.inner is empty
        assert!(
            variant_config.templates.inner.is_empty(),
            "templates.inner should be empty for this test"
        );

        // Create empty analyses
        let analyses: Vec<InferenceWithAnalysis> = vec![];

        // Wrap function_config in FunctionConfigAndTools
        let config_and_tools = FunctionConfigAndTools {
            function_config: Arc::new(function_config),
            static_tools: None,
        };

        // Call build_mutate_input
        let result = build_mutate_input(&analyses, &config_and_tools, &variant_config);

        // Assert: Should return Config error
        assert!(
            result.is_err(),
            "build_mutate_input should error with empty templates"
        );
        let error = result.unwrap_err();
        let error_msg = error.to_string();
        assert!(
            error_msg.contains("Cannot mutate variant with no templates"),
            "Error message should indicate empty templates, got: {error_msg}"
        );
    }

    #[test]
    fn test_parse_mutate_response_valid_json() {
        use tensorzero_core::{
            endpoints::inference::{InferenceResponse, JsonInferenceResponse},
            inference::types::{JsonInferenceOutput, Usage},
        };
        use uuid::Uuid;

        // Create valid JSON response with MutateOutput structure (array format)
        let output_json = serde_json::json!({
            "templates": [
                {"name": "system", "content": "Improved system template"},
                {"name": "user", "content": "Improved user template"}
            ]
        });

        let json_response = JsonInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            output: JsonInferenceOutput {
                raw: Some("{}".to_string()),
                parsed: Some(output_json.clone()),
            },
            usage: Usage::default(),
            original_response: None,
            finish_reason: None,
        };

        let response = InferenceResponse::Json(json_response);

        // Call parse_mutate_response
        let result = parse_mutate_response(&response);

        // Assert: Should succeed
        assert!(result.is_ok(), "parse_mutate_response should succeed");
        let mutate_output = result.unwrap();

        // Verify templates were extracted correctly
        assert_eq!(mutate_output.templates.len(), 2, "Should have 2 templates");
        assert_eq!(
            mutate_output.templates.get("system").unwrap(),
            "Improved system template"
        );
        assert_eq!(
            mutate_output.templates.get("user").unwrap(),
            "Improved user template"
        );
    }

    #[test]
    fn test_parse_mutate_response_chat_response_error() {
        use tensorzero_core::{
            endpoints::inference::{ChatInferenceResponse, InferenceResponse},
            inference::types::{ContentBlockChatOutput, Text, Usage},
        };
        use uuid::Uuid;

        // Create Chat response (wrong type - should be Json)
        let chat_response = ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            content: vec![ContentBlockChatOutput::Text(Text {
                text: "This is chat text, not JSON".to_string(),
            })],
            usage: Usage::default(),
            original_response: None,
            finish_reason: None,
        };

        let response = InferenceResponse::Chat(chat_response);

        // Call parse_mutate_response
        let result = parse_mutate_response(&response);

        // Assert: Should return Inference error
        assert!(
            result.is_err(),
            "parse_mutate_response should error with Chat response"
        );
        let error = result.unwrap_err();
        let error_msg = error.to_string();
        assert!(
            error_msg.contains("Expected Json response") || error_msg.contains("got Chat"),
            "Error message should indicate wrong response type, got: {error_msg}"
        );
    }
}
