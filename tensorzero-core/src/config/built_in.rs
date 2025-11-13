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
use crate::function::{FunctionConfigChat, FunctionConfigJson};
use crate::jsonschema_util::{SchemaWithMetadata, StaticJSONSchema};
use crate::tool::{create_implicit_tool_call_config, ToolChoice};
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
    let implicit_tool_call_config = create_implicit_tool_call_config(output_schema.clone());

    Arc::new(FunctionConfig::Json(FunctionConfigJson {
        variants: HashMap::new(),
        schemas: SchemaData::default(),
        output_schema,
        implicit_tool_call_config,
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
/// TODO: support new templates/schemas
fn get_gepa_analyze_function() -> Result<Arc<FunctionConfig>, Error> {
    // Define user schema inline
    let user_schema_json = serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["function_name", "model", "templates", "input", "output"],
        "additionalProperties": false,
        "properties": {
            "function_name": {
                "type": "string",
                "description": "The name of the function you are analyzing"
            },
            "model": {
                "type": "string",
                "description": "The the name of the LLM used"
            },
            "templates": {
                "type": "object",
                "description": "Map of template names to their contents (e.g., 'system', 'user', 'assistant', or custom template names)",
                "additionalProperties": {
                    "type": "string"
                }
            },
            "output_schema": {
                "type": ["object", "null"],
                "description": "Optional reference JSON schema for the function output (if function type is json)"
            },
            "schemas": {
                "type": ["object", "null"],
                "description": "Optional map of schema names to their JSON schemas (e.g., 'system', 'user', 'assistant', or custom schema names)",
                "additionalProperties": {
                    "type": "object",
                    "description": "JSON schema for the named template"
                }
            },
            "tools": {
                "type": ["object", "null"],
                "description": "Optional dictionary of tool names to tool schemas",
                "additionalProperties": {
                    "type": "object",
                    "description": "Tool schema"
                }
            },
            "input": {
                "description": "The input messages leading up to the assistant's response"
            },
            "output": {
                "description": "The assistant's response"
            },
            "tags": {
                "type": ["object", "null"],
                "description": "Tags associated with the inference"
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

/// Returns the `tensorzero::optimization::gepa::mutate` function configuration.
///
/// This is a JSON function that generates improved prompt templates based on
/// analysis feedback from the GEPA optimization algorithm.
fn get_gepa_mutate_function() -> Result<Arc<FunctionConfig>, Error> {
    // Define user schema inline
    let user_schema_json = serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["function_name", "model", "templates", "analyses"],
        "additionalProperties": false,
        "properties": {
            "function_name": {
                "type": "string",
                "description": "The name of the function you are improving"
            },
            "model": {
                "type": "string",
                "description": "The target model for the optimized templates"
            },
            "templates": {
                "type": "object",
                "description": "Map of template names to their contents",
                "additionalProperties": {
                    "type": "string"
                }
            },
            "schemas": {
                "type": ["object", "null"],
                "description": "Optional map of schema names to their JSON schemas",
                "additionalProperties": {
                    "type": "object"
                }
            },
            "output_schema": {
                "type": ["object", "null"],
                "description": "Optional reference JSON schema for the function output (if function type is json)"
            },
            "tools": {
                "type": ["object", "null"],
                "description": "Optional dictionary of tool names to tool schemas",
                "additionalProperties": {
                    "type": "object",
                    "description": "Tool schema"
                }
            },
            "analyses": {
                "type": "array",
                "description": "Array of inference examples with their corresponding analyses/feedback",
                "items": {
                    "type": "object",
                    "required": ["inference_output", "analysis"],
                    "properties": {
                        "inference_output": {
                            "description": "The LLM inference output analyzed"
                        },
                        "analysis": {
                            "type": "array",
                            "description": "The analysis/feedback for this inference output (from gepa_generate_analysis: error report, improvement suggestion, or optimal pattern)"
                        }
                    }
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

    // Define output schema inline
    let output_schema_json = serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "templates": {
                "type": "object",
                "description": "Map of template names to improved template contents",
                "additionalProperties": {
                    "type": "string"
                }
            }
        },
        "required": ["templates"],
        "additionalProperties": false
    });
    let output_schema = StaticJSONSchema::from_value(output_schema_json)?;

    let implicit_tool_call_config = create_implicit_tool_call_config(output_schema.clone());

    Ok(Arc::new(FunctionConfig::Json(FunctionConfigJson {
        variants: HashMap::new(),
        schemas,
        output_schema,
        implicit_tool_call_config,
        description: Some(
            "Built-in GEPA mutate function - generates improved prompt templates based on analysis feedback".to_string(),
        ),
        all_explicit_template_names: HashSet::new(),
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
    functions.insert(
        "tensorzero::optimization::gepa::mutate".to_string(),
        get_gepa_mutate_function()?,
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
        assert_eq!(functions.len(), 4);
        assert!(functions.contains_key("tensorzero::hello_chat"));
        assert!(functions.contains_key("tensorzero::hello_json"));
    }
}
