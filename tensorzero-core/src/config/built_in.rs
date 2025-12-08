//! Built-in TensorZero functions
//!
//! This module defines built-in functions that are automatically available
//! in all TensorZero deployments. These functions are prefixed with `tensorzero::`
//! and cannot be overridden by user-defined functions.
//!
//! Built-in functions are variant-less - they have empty variants HashMaps
//! but are fully defined function configs with all necessary fields.
//!
//! ## Supported Variant Types
//!
//! Built-in functions currently only support simple ChatCompletion variants
//! provided via `internal_dynamic_variant_config`. Complex variant types
//! (Dicl, MixtureOfN, BestOfNSampling) that require predefined variants
//! are not supported.

use std::collections::HashMap;

use crate::config::path::ResolvedTomlPathData;
use crate::config::{
    UninitializedFunctionConfig, UninitializedFunctionConfigChat, UninitializedFunctionConfigJson,
    UninitializedSchemas,
};
use crate::error::{Error, ErrorDetails};
use crate::tool::ToolChoice;

/// Creates an inline schema path for built-in functions.
///
/// This uses `ResolvedTomlPathData::new_fake_path()` to create a fake path
/// that contains the serialized JSON schema data inline, rather than referencing
/// an actual file on disk.
fn create_inline_schema_path(
    function_name: &str,
    schema_name: &str,
    schema_value: &serde_json::Value,
) -> Result<ResolvedTomlPathData, Error> {
    let serialized = serde_json::to_string(schema_value).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize schema for {function_name}::{schema_name}: {e}"),
        })
    })?;
    Ok(ResolvedTomlPathData::new_fake_path(
        format!("tensorzero::builtin::{function_name}::{schema_name}.json"),
        serialized,
    ))
}

/// Returns the `tensorzero::hello_chat` function configuration.
///
/// This is a simple Chat function that serves as a basic
/// example of a built-in chat function with template variable support.
///
/// Supports system template with a `greeting` variable.
#[cfg(feature = "e2e_tests")]
fn get_hello_chat_function() -> Result<UninitializedFunctionConfig, Error> {
    let system_schema_value = serde_json::json!({
        "type": "object",
        "properties": {
            "greeting": {
                "type": "string",
                "description": "A greeting message"
            }
        },
        "additionalProperties": false
    });

    let system_schema_path =
        create_inline_schema_path("hello_chat", "system", &system_schema_value)?;

    Ok(UninitializedFunctionConfig::Chat(
        UninitializedFunctionConfigChat {
            variants: HashMap::new(),
            system_schema: Some(system_schema_path),
            user_schema: None,
            assistant_schema: None,
            schemas: UninitializedSchemas::default(),
            tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            description: Some(
                "Built-in hello chat function - a simple greeting function with template variable support".to_string(),
            ),
            experimentation: None,
        },
    ))
}

/// Returns the `tensorzero::hello_json` function configuration.
///
/// This is a simple variant-less JSON function that serves as a basic
/// example of a built-in JSON function.
#[cfg(feature = "e2e_tests")]
fn get_hello_json_function() -> UninitializedFunctionConfig {
    // output_schema is None - will default to {} during load
    UninitializedFunctionConfig::Json(UninitializedFunctionConfigJson {
        variants: HashMap::new(),
        system_schema: None,
        user_schema: None,
        assistant_schema: None,
        schemas: UninitializedSchemas::default(),
        output_schema: None,
        description: Some(
            "Built-in hello JSON function - a simple JSON response function".to_string(),
        ),
        experimentation: None,
    })
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
fn get_gepa_analyze_function() -> Result<UninitializedFunctionConfig, Error> {
    let user_schema_value = serde_json::json!({
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

    let user_schema_path =
        create_inline_schema_path("optimization::gepa::analyze", "user", &user_schema_value)?;

    Ok(UninitializedFunctionConfig::Chat(
        UninitializedFunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: Some(user_schema_path),
            assistant_schema: None,
            schemas: UninitializedSchemas::default(),
            tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            description: Some(
                "Built-in GEPA analyze function - analyzes inference outputs and provides structured XML feedback for optimization".to_string(),
            ),
            experimentation: None,
        },
    ))
}

/// Returns the `tensorzero::optimization::gepa::mutate` function configuration.
///
/// This is a JSON function that generates improved prompt templates based on
/// analysis feedback from the GEPA optimization algorithm.
fn get_gepa_mutate_function() -> Result<UninitializedFunctionConfig, Error> {
    let user_schema_value = serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["function_config", "evaluation_config", "templates_map", "analyses"],
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
                "description": "Map of template names to their contents. Your task is to improve the contents",
                "additionalProperties": {
                    "type": "string"
                }
            },
            "analyses": {
                "type": "array",
                "description": "Array of analyses to inform how you can improve the templates",
                "items": {
                    "type": "object",
                    "required": ["analysis"],
                    "properties": {
                        "inference": {
                            "type": ["object", "null"],
                            "description": "The input and output of the LLM inference analyzed",
                            "properties": {
                                "input": {
                                    "type": "object",
                                    "description": "The inference input"
                                },
                                "output": {
                                    "description": "The inference output"
                                }
                            },
                            "required": ["input", "output"],
                            "additionalProperties": false
                        },
                        "analysis": {
                            "type": "string",
                            "description": "Analysis of an LLM inference to guide your template improvement"
                        }
                    }
                }
            }
        }
    });

    let user_schema_path =
        create_inline_schema_path("optimization::gepa::mutate", "user", &user_schema_value)?;

    // Define output schema inline
    // Note: Using array format instead of additionalProperties to support OpenAI strict mode
    let output_schema_value = serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "templates": {
                "type": "array",
                "description": "Array of improved templates with their names",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The template name (e.g., 'system', 'user', 'assistant')"
                        },
                        "content": {
                            "type": "string",
                            "description": "The improved template content"
                        }
                    },
                    "required": ["name", "content"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["templates"],
        "additionalProperties": false
    });

    let output_schema_path =
        create_inline_schema_path("optimization::gepa::mutate", "output", &output_schema_value)?;

    Ok(UninitializedFunctionConfig::Json(
        UninitializedFunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: Some(user_schema_path),
            assistant_schema: None,
            schemas: UninitializedSchemas::default(),
            output_schema: Some(output_schema_path),
            description: Some(
                "Built-in GEPA mutate function - generates improved message templates based on analysis feedback".to_string(),
            ),
            experimentation: None,
        },
    ))
}

/// Returns all built-in functions as UninitializedFunctionConfigs.
///
/// The keys are function names (e.g., "tensorzero::hello_chat")
/// and the values are UninitializedFunctionConfigs that can be loaded
/// via `UninitializedFunctionConfig::load()`.
///
/// **Note**: If you add or remove built-in functions here, you must also update
/// the UI e2e test in `ui/e2e_tests/homePage.spec.ts` which checks the total
/// function count displayed on the homepage.
pub fn get_all_built_in_functions() -> Result<HashMap<String, UninitializedFunctionConfig>, Error> {
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
        match hello_chat {
            UninitializedFunctionConfig::Chat(config) => {
                assert!(config.variants.is_empty());
                assert!(config.tools.is_empty());
                assert_eq!(config.tool_choice, ToolChoice::None);
                assert!(config.description.is_some());
                assert!(config.system_schema.is_some());
            }
            UninitializedFunctionConfig::Json(_) => panic!("Expected Chat function"),
        }
    }

    #[test]
    fn test_get_hello_json_function() {
        let hello_json = get_hello_json_function();
        match hello_json {
            UninitializedFunctionConfig::Json(config) => {
                assert!(config.variants.is_empty());
                assert!(config.description.is_some());
                assert!(config.output_schema.is_none()); // Will default during load
            }
            UninitializedFunctionConfig::Chat(_) => panic!("Expected JSON function"),
        }
    }

    #[test]
    fn test_get_all_built_in_functions() {
        let functions = get_all_built_in_functions().unwrap();
        assert_eq!(functions.len(), 4);
        assert!(functions.contains_key("tensorzero::hello_chat"));
        assert!(functions.contains_key("tensorzero::hello_json"));
        assert!(functions.contains_key("tensorzero::optimization::gepa::analyze"));
        assert!(functions.contains_key("tensorzero::optimization::gepa::mutate"));
    }

    #[test]
    fn test_built_in_functions_load_successfully() {
        // Test that built-in functions can be loaded via UninitializedFunctionConfig::load()
        let functions = get_all_built_in_functions().unwrap();
        let metrics = std::collections::HashMap::new();

        for (name, config) in functions {
            let loaded = config.load(&name, &metrics);
            assert!(
                loaded.is_ok(),
                "Failed to load built-in function {name}: {:?}",
                loaded.err()
            );
        }
    }
}
