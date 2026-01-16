//! Template mutation for GEPA optimization.
//!
//! This module implements the mutation step of the GEPA optimization algorithm.
//!
//! The sole public function is [`mutate_variant`], which:
//! - Takes aggregated analyses from the GEPA analyze step
//! - Calls the built-in `tensorzero::optimization::gepa::mutate` function to generate improved templates
//! - Validates that returned templates exactly match the parent's template structure
//! - Returns a new child variant with the improved templates
//!
//! The child variant inherits the parent's configuration but with improved templates based on
//! the analyses. Template validation ensures the LLM doesn't hallucinate missing or extra templates.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{Map, from_value, json, to_value};

use tensorzero_core::{
    client::{
        Client, ClientInferenceParams, InferenceOutput, Input, InputMessage, InputMessageContent,
    },
    config::{UninitializedVariantConfig, UninitializedVariantInfo, path::ResolvedTomlPathData},
    endpoints::inference::InferenceResponse,
    error::{Error, ErrorDetails},
    inference::types::{Arguments, Role, Template},
    optimization::gepa::GEPAConfig,
    variant::chat_completion::{UninitializedChatCompletionConfig, UninitializedChatTemplate},
};

use crate::gepa::{GEPAVariant, analyze::Analysis, validate::FunctionContext};

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

/// Builds input Arguments for the mutate function.
///
/// Constructs an Arguments struct containing high-level objects that will be serialized
/// to JSON during template rendering.
///
/// Returns Arguments with template variables: function_config, static_tools, evaluation_config,
/// templates_map, and analyses.
///
/// Returns error if serialization fails.
fn build_mutate_input(
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

    // Serialize analyses to JSON
    // Note: Thought signatures in inference outputs are already conditionally stripped
    // in analyze_inference based on response type (Chat only, not Json)
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
/// Returns a GEPAVariant containing the improved templates.
/// The variant name follows the pattern: `{prefix}-iter-{iteration}-{parent_name}`.
///
/// Returns error if mutation fails (LLM call fails, invalid response format, etc.).
pub async fn mutate_variant(
    gateway_client: &Client,
    analyses: &[Analysis],
    function_context: &FunctionContext,
    parent: &GEPAVariant,
    gepa_config: &GEPAConfig,
    iteration: usize,
) -> Result<GEPAVariant, Error> {
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
    let arguments = build_mutate_input(analyses, function_context, &parent.config)?;

    // Create ClientInferenceParams for the mutate function
    let params = ClientInferenceParams {
        function_name: Some("tensorzero::optimization::gepa::mutate".to_string()),
        input: Input {
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Template(Template {
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

    // Extract JSON content from the response
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

    // Check for duplicate template names before converting to HashMap
    let template_names: Vec<&str> = response.templates.iter().map(|t| t.name.as_str()).collect();
    let unique_names: std::collections::HashSet<&str> = template_names.iter().copied().collect();
    if template_names.len() != unique_names.len() {
        return Err(Error::new(ErrorDetails::Inference {
            message: format!(
                "Mutate function returned duplicate template names. Expected {} unique templates, got {} entries",
                unique_names.len(),
                template_names.len()
            ),
        }));
    }

    // Convert the array of template entries into a HashMap
    let templates: HashMap<String, String> = response
        .templates
        .into_iter()
        .map(|entry| (entry.name, entry.content))
        .collect();

    // Validate that all parent templates are present
    for parent_template_name in parent.config.templates.inner.keys() {
        if !templates.contains_key(parent_template_name) {
            return Err(Error::new(ErrorDetails::Inference {
                message: format!(
                    "Mutate function did not return template '{parent_template_name}' present in parent"
                ),
            }));
        }
    }

    // Validate that no extra templates were added
    for template_name in templates.keys() {
        if !parent.config.templates.inner.contains_key(template_name) {
            return Err(Error::new(ErrorDetails::Inference {
                message: format!(
                    "Mutate function returned unexpected template '{template_name}' not present in parent"
                ),
            }));
        }
    }

    // Log the generated mutated templates at debug level
    let template_names: Vec<&String> = templates.keys().collect();
    tracing::debug!(
        "Generated {} mutated templates: {:?}",
        templates.len(),
        template_names
    );
    for (template_name, content) in &templates {
        tracing::debug!("Mutated template '{}' content:\n{}", template_name, content);
    }

    // Generate variant name: {prefix}-iter-{iteration}-{parent_name}
    let mutated_variant_name = format!(
        "{}-iter-{}-{}",
        gepa_config.variant_prefix.as_deref().unwrap_or("gepa"),
        iteration,
        parent.name
    );

    // Clone parent config
    let mut mutated_config = parent.config.clone();

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

    Ok(GEPAVariant {
        name: mutated_variant_name,
        config: mutated_config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tensorzero_core::{
        config::{SchemaData, path::ResolvedTomlPathData},
        evaluations::{EvaluationConfig, InferenceEvaluationConfig},
        function::{FunctionConfig, FunctionConfigChat},
        inference::types::{Input, StoredInput},
        jsonschema_util::{JSONSchema, SchemaWithMetadata},
        tool::StaticToolConfig,
    };

    use crate::gepa::{analyze::Analysis, validate::FunctionContext};

    // ============================================================================
    // Test Helper Functions
    // ============================================================================

    /// Create a minimal Chat FunctionConfig for testing
    fn create_test_function_config() -> FunctionConfig {
        FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: tensorzero_core::tool::ToolChoice::None,
            parallel_tool_calls: None,
            description: Some("Test function".to_string()),
            all_explicit_templates_names: std::collections::HashSet::new(),
            experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
        })
    }

    /// Create a Chat FunctionConfig with schemas
    fn create_test_function_config_with_schemas() -> FunctionConfig {
        let system_schema = JSONSchema::from_value(json!({
            "type": "object",
            "properties": {
                "greeting": {"type": "string"}
            }
        }))
        .unwrap();

        let user_schema = JSONSchema::from_value(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }))
        .unwrap();

        let mut schema_inner = HashMap::new();
        schema_inner.insert(
            "system".to_string(),
            SchemaWithMetadata {
                schema: system_schema,
                legacy_definition: false,
            },
        );
        schema_inner.insert(
            "user".to_string(),
            SchemaWithMetadata {
                schema: user_schema,
                legacy_definition: false,
            },
        );

        let schemas = SchemaData {
            inner: schema_inner,
        };

        FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas,
            tools: vec![],
            tool_choice: tensorzero_core::tool::ToolChoice::None,
            parallel_tool_calls: None,
            description: Some("Test function with schemas".to_string()),
            all_explicit_templates_names: std::collections::HashSet::new(),
            experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
        })
    }

    /// Create a test UninitializedChatCompletionConfig using templates.inner
    fn create_test_variant_config() -> UninitializedChatCompletionConfig {
        let mut config = UninitializedChatCompletionConfig {
            model: "test-model".into(),
            weight: None,
            ..Default::default()
        };

        config.templates.inner.insert(
            "system".to_string(),
            UninitializedChatTemplate {
                path: ResolvedTomlPathData::new_fake_path(
                    "system.minijinja".to_string(),
                    "You are a helpful assistant.".to_string(),
                ),
            },
        );
        config.templates.inner.insert(
            "user".to_string(),
            UninitializedChatTemplate {
                path: ResolvedTomlPathData::new_fake_path(
                    "user.minijinja".to_string(),
                    "User: {{input}}".to_string(),
                ),
            },
        );

        config
    }

    /// Create a test EvaluationConfig with realistic evaluators
    fn create_test_evaluation_config() -> EvaluationConfig {
        use tensorzero_core::evaluations::{EvaluatorConfig, ExactMatchConfig};

        let mut evaluators = HashMap::new();

        // Add an exact match evaluator
        evaluators.insert(
            "exact_match".to_string(),
            EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }),
        );

        EvaluationConfig::Inference(InferenceEvaluationConfig {
            evaluators,
            function_name: "test_function".to_string(),
            description: Some("evaluation".to_string()),
        })
    }

    /// Create a test Analysis with optional inference
    fn create_test_analysis(include_inference: bool) -> Analysis {
        let inference = if include_inference {
            let input = Input {
                messages: vec![],
                system: None,
            };
            let stored_input: StoredInput =
                serde_json::from_value(serde_json::to_value(&input).unwrap()).unwrap();

            Some(crate::gepa::analyze::Inference {
                input: stored_input,
                output: json!({"content": "test output"}),
            })
        } else {
            None
        };

        Analysis {
            inference,
            analysis: "Test analysis feedback".to_string(),
        }
    }

    /// Helper to render the mutate user template with given arguments
    fn render_template(arguments: &Arguments) -> Result<String, String> {
        // Initialize MiniJinja environment
        // Note: tojson filter is built-in to minijinja when json+builtins features are enabled
        let mut env = minijinja::Environment::new();

        // Load the actual user template
        let template_source = include_str!("functions/mutate/user_template.minijinja");
        env.add_template("user", template_source)
            .map_err(|e| format!("Failed to add template: {e}"))?;

        let template = env
            .get_template("user")
            .map_err(|e| format!("Failed to get template: {e}"))?;

        // Convert Arguments to minijinja context
        let context_value = serde_json::to_value(&arguments.0)
            .map_err(|e| format!("Failed to serialize arguments: {e}"))?;
        let minijinja_value = minijinja::Value::from_serialize(&context_value);

        // Render the template
        template
            .render(minijinja_value)
            .map_err(|e| format!("Template rendering failed: {e}"))
    }

    // ============================================================================
    // Unit Tests for build_mutate_input
    // ============================================================================

    #[test]
    fn test_build_mutate_input_comprehensive() {
        // Test with minimal config (no schemas, no static_tools)
        let analyses = vec![create_test_analysis(false)];
        let function_config = create_test_function_config();
        let static_tools = None;
        let variant_config = create_test_variant_config();
        let eval_config = create_test_evaluation_config();

        let function_context = FunctionContext {
            function_config: Arc::new(function_config),
            static_tools,
            evaluation_config: Arc::new(eval_config),
        };

        let result = build_mutate_input(&analyses, &function_context, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        // Check all required high-level fields are present
        assert!(input.0.get("function_config").is_some());
        assert!(input.0.get("static_tools").is_none()); // static_tools is None, so should not be present
        assert!(input.0.get("evaluation_config").is_some());
        assert!(input.0.get("templates_map").is_some());
        assert!(input.0.get("analyses").is_some());

        // Verify function_config is properly serialized and preserves structure
        let func_config = input.0.get("function_config").unwrap();
        assert!(func_config.is_object());
        let func_config_obj = func_config.as_object().unwrap();
        assert!(func_config_obj.contains_key("description"));
        assert_eq!(
            func_config_obj.get("description").unwrap().as_str(),
            Some("Test function")
        );

        // Verify evaluation_config is properly serialized and preserves structure
        let eval_config_value = input.0.get("evaluation_config").unwrap();
        assert!(eval_config_value.is_object());
        let eval_config_obj = eval_config_value.as_object().unwrap();
        assert!(eval_config_obj.contains_key("function_name"));
        assert_eq!(
            eval_config_obj.get("function_name").unwrap().as_str(),
            Some("test_function")
        );

        // Verify analyses is an array
        let analyses_value = input.0.get("analyses").unwrap();
        assert!(analyses_value.is_array());
        let analyses_arr = analyses_value.as_array().unwrap();
        assert_eq!(analyses_arr.len(), 1);

        // Verify templates_map content is extracted correctly
        let templates_map = input.0.get("templates_map").unwrap();
        let templates_obj = templates_map.as_object().unwrap();
        assert_eq!(templates_obj.len(), 2, "Should have 2 templates");

        // Verify exact template content
        let system_template = templates_obj.get("system").unwrap();
        assert_eq!(
            system_template.as_str().unwrap(),
            "You are a helpful assistant."
        );

        let user_template = templates_obj.get("user").unwrap();
        assert_eq!(user_template.as_str().unwrap(), "User: {{input}}");

        // Render template and verify evaluators and function description appear
        let rendered = render_template(&input).expect("Template rendering should succeed");
        assert!(
            rendered.contains("exact_match"),
            "Rendered template should contain 'exact_match' evaluator"
        );
        assert!(
            rendered.contains("Test function"),
            "Rendered template should contain function description"
        );
        // Verify static_tools section is not present (since None)
        assert!(
            !rendered.contains("<tool_schemas>"),
            "Rendered template should not have tool_schemas section when no tools"
        );
        // Verify template content appears in rendered output
        assert!(
            rendered.contains("You are a helpful assistant."),
            "Rendered template should contain system template content"
        );
        assert!(
            rendered.contains("User: {{input}}"),
            "Rendered template should contain user template content with variable placeholder"
        );
        // Verify template names appear in the rendered output
        assert!(
            rendered.contains("\"system\"") || rendered.contains("'system'"),
            "Rendered template should mention 'system' template name"
        );
        assert!(
            rendered.contains("\"user\"") || rendered.contains("'user'"),
            "Rendered template should mention 'user' template name"
        );

        // Test with schemas
        let function_config_with_schemas = create_test_function_config_with_schemas();
        let static_tools_schemas = None;
        let eval_config_schemas = create_test_evaluation_config();
        let function_context_with_schemas = FunctionContext {
            function_config: Arc::new(function_config_with_schemas),
            static_tools: static_tools_schemas,
            evaluation_config: Arc::new(eval_config_schemas),
        };
        let result_with_schemas =
            build_mutate_input(&analyses, &function_context_with_schemas, &variant_config);

        assert!(result_with_schemas.is_ok());
        let input_with_schemas = result_with_schemas.unwrap();

        // Verify schemas are included in serialized function_config
        let func_config_with_schemas = input_with_schemas.0.get("function_config").unwrap();
        let func_config_obj = func_config_with_schemas.as_object().unwrap();
        assert!(func_config_obj.contains_key("schemas"));

        // Render template with schemas and verify schemas appear
        let rendered_with_schemas =
            render_template(&input_with_schemas).expect("Template rendering should succeed");
        assert!(
            rendered_with_schemas.contains("schemas"),
            "Rendered template should contain schemas when present"
        );
        assert!(
            rendered_with_schemas.contains("Test function with schemas"),
            "Rendered template should contain function description with schemas"
        );
    }

    #[test]
    fn test_build_mutate_input_static_tools() {
        // Create a static tool config
        let tool_config = Arc::new(StaticToolConfig {
            name: "test_tool".to_string(),
            key: "test_tool".to_string(),
            description: "Test tool".to_string(),
            parameters: JSONSchema::from_value(json!({
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                }
            }))
            .unwrap(),
            strict: false,
        });

        let mut static_tools = HashMap::new();
        static_tools.insert("test_tool".to_string(), tool_config);

        let analyses = vec![create_test_analysis(false)];
        let function_config = create_test_function_config();
        let variant_config = create_test_variant_config();
        let eval_config = create_test_evaluation_config();

        let function_context = FunctionContext {
            function_config: Arc::new(function_config),
            static_tools: Some(static_tools),
            evaluation_config: Arc::new(eval_config),
        };

        let result = build_mutate_input(&analyses, &function_context, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        // Verify static_tools is serialized
        let tools = input.0.get("static_tools").unwrap();
        assert!(!tools.is_null());

        // Verify the tool is present in the serialized data
        let tools_obj = tools.as_object().unwrap();
        assert!(tools_obj.contains_key("test_tool"));

        // Verify tool has description
        let test_tool = tools_obj.get("test_tool").unwrap();
        let tool_obj = test_tool.as_object().unwrap();
        assert!(tool_obj.contains_key("description"));
        assert_eq!(
            tool_obj.get("description").unwrap().as_str().unwrap(),
            "Test tool"
        );

        // Render template and verify tool information appears
        let rendered = render_template(&input).expect("Template rendering should succeed");
        assert!(
            rendered.contains("test_tool"),
            "Rendered template should contain tool name"
        );
        assert!(
            rendered.contains("Test tool"),
            "Rendered template should contain tool description"
        );
        assert!(
            rendered.contains("<tool_schemas>"),
            "Rendered template should have tool_schemas section when tools present"
        );
        assert!(
            rendered.contains("param1"),
            "Rendered template should contain tool parameter name"
        );
    }

    #[test]
    fn test_build_mutate_input_analyses_serialization() {
        // Create analyses with different structures
        let analyses = vec![
            create_test_analysis(false), // Without inference
            create_test_analysis(true),  // With inference
        ];

        let function_config = create_test_function_config();
        let static_tools = None;
        let variant_config = create_test_variant_config();
        let eval_config = create_test_evaluation_config();

        let function_context = FunctionContext {
            function_config: Arc::new(function_config),
            static_tools,
            evaluation_config: Arc::new(eval_config),
        };

        let result = build_mutate_input(&analyses, &function_context, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        // Verify analyses array is properly serialized
        let analyses_value = input.0.get("analyses").unwrap();
        assert!(analyses_value.is_array());
        let analyses_arr = analyses_value.as_array().unwrap();
        assert_eq!(analyses_arr.len(), 2, "Should have 2 analyses");

        // Check first analysis (without inference)
        let first_analysis = analyses_arr[0].as_object().unwrap();
        assert!(first_analysis.contains_key("analysis"));
        assert_eq!(
            first_analysis.get("analysis").unwrap().as_str().unwrap(),
            "Test analysis feedback"
        );
        // inference field should not be present (skipped if None)
        assert!(!first_analysis.contains_key("inference"));

        // Check second analysis (with inference - flattened)
        let second_analysis = analyses_arr[1].as_object().unwrap();
        assert!(second_analysis.contains_key("analysis"));
        // inference should be flattened, so input/output appear at top level
        assert!(second_analysis.contains_key("input"));
        assert!(second_analysis.contains_key("output"));

        // Render template and verify analyses information appears
        let rendered = render_template(&input).expect("Template rendering should succeed");
        // The analysis text should appear (potentially multiple times if repeated)
        let analysis_count = rendered.matches("Test analysis feedback").count();
        assert!(
            analysis_count >= 2,
            "Rendered template should contain analysis feedback at least twice (once per analysis), found {analysis_count}"
        );
        // Verify inference context appears for the second analysis
        assert!(
            rendered.contains("test output"),
            "Rendered template should contain inference output from second analysis"
        );
        // Verify the analyses section exists
        assert!(
            rendered.contains("<analyses>"),
            "Rendered template should have analyses section"
        );
        assert!(
            rendered.contains("<example n=\"1\">"),
            "Rendered template should have numbered example tags (example 1)"
        );
        assert!(
            rendered.contains("<example n=\"2\">"),
            "Rendered template should have numbered example tags (example 2)"
        );
    }
}
