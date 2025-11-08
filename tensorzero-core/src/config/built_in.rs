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
/// This is a Chat function with tools that analyzes inference outputs and provides
/// structured feedback for the GEPA optimization algorithm.
///
/// The function has three tools:
/// - report_error: For critical failures in the inference output
/// - report_improvement: For suboptimal but technically correct outputs
/// - report_optimal: For high-quality aspects worth preserving
fn get_gepa_analyze_function() -> Result<Arc<FunctionConfig>, Error> {
    // Load user schema from embedded JSON file
    let user_schema_json: serde_json::Value = serde_json::from_str(include_str!(
        "../optimization/gepa/config/functions/analyze/user_schema.json"
    ))?;
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

    // TODO: Add tool support for built-in functions
    // For now, GEPA analyze function uses empty tools - tools will be provided
    // via internal_dynamic_variant_config when the actual GEPA implementation
    // in tensorzero-optimizers is complete
    //
    // Tool definitions exist in:
    // - ../optimization/gepa/config/tools/report_error.json
    // - ../optimization/gepa/config/tools/report_improvement.json
    // - ../optimization/gepa/config/tools/report_optimal.json

    Ok(Arc::new(FunctionConfig::Chat(FunctionConfigChat {
        variants: HashMap::new(),
        schemas,
        tools: vec![],
        tool_choice: ToolChoice::None,
        parallel_tool_calls: None,
        description: Some(
            "Built-in GEPA analyze function - analyzes inference outputs and provides structured feedback for optimization".to_string(),
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
    // Load user schema from embedded JSON file
    let user_schema_json: serde_json::Value = serde_json::from_str(include_str!(
        "../optimization/gepa/config/functions/mutate/user_schema.json"
    ))?;
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

    // Load output schema from embedded JSON file
    let output_schema_json: serde_json::Value = serde_json::from_str(include_str!(
        "../optimization/gepa/config/functions/mutate/output_schema.json"
    ))?;
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
        assert_eq!(functions.len(), 2);
        assert!(functions.contains_key("tensorzero::hello_chat"));
        assert!(functions.contains_key("tensorzero::hello_json"));
    }
}
