//! Built-in TensorZero functions
//!
//! This module defines built-in functions that are automatically available
//! in all TensorZero deployments. These functions are prefixed with `tensorzero::`
//! and cannot be overridden by user-defined functions.
//!
//! Built-in functions are variant-less - they have empty variants HashMaps
//! but are fully defined FunctionConfigs with all necessary fields.
//!
//! ## Supported Variant Types
//!
//! Built-in functions currently only support simple ChatCompletion variants
//! provided via `internal_dynamic_variant_config`. Complex variant types
//! (Dicl, MixtureOfN, BestOfNSampling) that require predefined variants
//! are not supported.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Error;
use crate::function::FunctionConfig;

use crate::config::SchemaData;
use crate::experimentation::ExperimentationConfig;
use crate::function::FunctionConfigChat;
#[cfg(feature = "e2e_tests")]
use crate::function::FunctionConfigJson;
use crate::jsonschema_util::{SchemaWithMetadata, StaticJSONSchema};
#[cfg(feature = "e2e_tests")]
use crate::tool::create_json_mode_tool_call_config;
use crate::tool::ToolChoice;
use std::collections::HashSet;

/// Returns the `tensorzero::hello_chat` function configuration.
///
/// This is a simple Chat function that serves as a basic
/// example of a built-in chat function with template variable support.
///
/// Supports system template with a `greeting` variable.
#[cfg(feature = "e2e_tests")]
fn get_hello_chat_function() -> Result<Arc<FunctionConfig>, Error> {
    // Define a simple schema that accepts a "greeting" variable
    let system_schema = StaticJSONSchema::from_value(serde_json::json!({
        "type": "object",
        "properties": {
            "greeting": {
                "type": "string",
                "description": "A greeting message"
            }
        },
        "additionalProperties": false
    }))?;

    let mut inner = HashMap::new();
    inner.insert(
        "system".to_string(),
        SchemaWithMetadata {
            schema: system_schema,
            legacy_definition: true,
        },
    );

    let schemas = SchemaData { inner };

    Ok(Arc::new(FunctionConfig::Chat(FunctionConfigChat {
        variants: HashMap::new(),
        schemas,
        tools: vec![],
        tool_choice: ToolChoice::None,
        parallel_tool_calls: None,
        description: Some("Built-in hello chat function - a simple greeting function with template variable support".to_string()),
        all_explicit_templates_names: HashSet::new(),
        experimentation: ExperimentationConfig::default(),
    })))
}

/// Returns the `tensorzero::hello_json` function configuration.
///
/// This is a simple variant-less JSON function that serves as a basic
/// example of a built-in JSON function.
#[cfg(feature = "e2e_tests")]
fn get_hello_json_function() -> Arc<FunctionConfig> {
    // Use default schema (no validation)
    let output_schema = StaticJSONSchema::default();
    let json_mode_tool_call_config = create_json_mode_tool_call_config(output_schema.clone());

    Arc::new(FunctionConfig::Json(FunctionConfigJson {
        variants: HashMap::new(),
        schemas: SchemaData::default(),
        output_schema,
        json_mode_tool_call_config,
        description: Some(
            "Built-in hello JSON function - a simple JSON response function".to_string(),
        ),
        all_explicit_template_names: HashSet::new(),
        experimentation: ExperimentationConfig::default(),
    }))
}

/// Returns the `tensorzero::optimization::gepa::analyze` function configuration.
///
/// This is a Chat function that analyzes inference outputs and provides
/// structured feedback for the GEPA optimization algorithm.
///
/// The function outputs XML in one of three formats:
/// - report_error: For critical failures in the inference output
/// - report_improvement: For suboptimal but technically correct outputs
/// - report_optimal: For high-quality aspects worth preserving
fn get_gepa_analyze_function() -> Result<Arc<FunctionConfig>, Error> {
    // Define user schema inline
    let user_schema_json = serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["function_config", "evaluation_config", "templates_map", "datapoint", "output", "evaluation_scores"],
        "additionalProperties": false,
        "properties": {
            "function_config": {
                "type": "object",
                "description": "Complete function configuration including schemas, tools, variants, and other metadata"
            },
            "static_tools": {
                "type": ["object", "null"],
                "description": "Map of tool names to their StaticToolConfig definitions from the config. Omitted when the function has no static tools configured.",
                "additionalProperties": {
                    "type": "object"
                }
            },
            "evaluation_config": {
                "type": "object",
                "description": "Evaluation configuration including all evaluator definitions",
                "properties": {
                    "evaluators": {
                        "type": "object",
                        "description": "Map of evaluator names to their full configurations"
                    },
                    "function_name": {
                        "type": "string"
                    }
                }
            },
            "templates_map": {
                "type": "object",
                "description": "Map of template names to their contents (extracted from variant config)",
                "additionalProperties": {
                    "type": "string"
                }
            },
            "datapoint": {
                "type": "object",
                "description": "Complete datapoint including input, tags, episode_id, function_name, and other metadata"
            },
            "output": {
                "description": "The inference response (ChatInferenceResponse or JsonInferenceResponse)"
            },
            "evaluation_scores": {
                "type": "object",
                "description": "Evaluation scores for this inference (numeric values 0-1, boolean values, or null if failed)",
                "additionalProperties": {
                    "type": ["number", "boolean", "null"]
                }
            }
        }
    });
    let user_schema = StaticJSONSchema::from_value(user_schema_json)?;

    let mut inner = HashMap::new();
    inner.insert(
        "user".to_string(),
        SchemaWithMetadata {
            schema: user_schema,
            legacy_definition: true,
        },
    );

    let schemas = SchemaData { inner };

    Ok(Arc::new(FunctionConfig::Chat(FunctionConfigChat {
        variants: HashMap::new(),
        schemas,
        tools: vec![],
        tool_choice: ToolChoice::None,
        parallel_tool_calls: None,
        description: Some(
            "Built-in GEPA analyze function - analyzes inference outputs and provides structured XML feedback for optimization".to_string(),
        ),
        all_explicit_templates_names: HashSet::new(),
        experimentation: ExperimentationConfig::default(),
    })))
}

/// Returns all built-in functions as a HashMap.
///
/// The keys are function names (e.g., "tensorzero::hello_chat")
/// and the values are Arc-wrapped FunctionConfigs.
///
/// **Note**: If you add or remove built-in functions here, you must also update
/// the UI e2e test in `ui/e2e_tests/homePage.spec.ts` which checks the total
/// function count displayed on the homepage.
pub fn get_all_built_in_functions() -> Result<HashMap<String, Arc<FunctionConfig>>, Error> {
    let mut functions = HashMap::new();

    #[cfg(feature = "e2e_tests")]
    {
        functions.insert(
            "tensorzero::hello_chat".to_string(),
            get_hello_chat_function()?,
        );
        functions.insert(
            "tensorzero::hello_json".to_string(),
            get_hello_json_function(),
        );
    }
    // GEPA built-in functions (always available, not feature-gated)
    functions.insert(
        "tensorzero::optimization::gepa::analyze".to_string(),
        get_gepa_analyze_function()?,
    );
    Ok(functions)
}

#[cfg(all(test, feature = "e2e_tests"))]
mod tests {
    use super::*;

    #[test]
    fn test_get_hello_chat_function() {
        let hello_chat = get_hello_chat_function().unwrap();
        match &*hello_chat {
            FunctionConfig::Chat(config) => {
                assert!(config.variants.is_empty());
                assert!(config.tools.is_empty());
                assert_eq!(config.tool_choice, ToolChoice::None);
                assert!(config.description.is_some());
                assert!(config.all_explicit_templates_names.is_empty());
            }
            FunctionConfig::Json(_) => panic!("Expected Chat function"),
        }
    }

    #[test]
    fn test_get_hello_json_function() {
        let hello_json = get_hello_json_function();
        match &*hello_json {
            FunctionConfig::Json(config) => {
                assert!(config.variants.is_empty());
                assert!(config.description.is_some());
                assert!(config.all_explicit_template_names.is_empty());
            }
            FunctionConfig::Chat(_) => panic!("Expected JSON function"),
        }
    }

    #[test]
    fn test_get_all_built_in_functions() {
        let functions = get_all_built_in_functions().unwrap();
        assert_eq!(functions.len(), 3);
        assert!(functions.contains_key("tensorzero::hello_chat"));
        assert!(functions.contains_key("tensorzero::hello_json"));
        assert!(functions.contains_key("tensorzero::optimization::gepa::analyze"));
    }
}
