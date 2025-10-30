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

#[cfg(feature = "e2e_tests")]
use crate::config::SchemaData;
#[cfg(feature = "e2e_tests")]
use crate::experimentation::ExperimentationConfig;
#[cfg(feature = "e2e_tests")]
use crate::function::{FunctionConfigChat, FunctionConfigJson};
#[cfg(feature = "e2e_tests")]
use crate::jsonschema_util::{SchemaWithMetadata, StaticJSONSchema};
#[cfg(feature = "e2e_tests")]
use crate::tool::{create_implicit_tool_call_config, ToolChoice};
#[cfg(feature = "e2e_tests")]
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

/// Returns all built-in functions as a HashMap.
///
/// The keys are function names (e.g., "tensorzero::hello_chat")
/// and the values are Arc-wrapped FunctionConfigs.
///
/// **Note**: If you add or remove built-in functions here, you must also update
/// the UI e2e test in `ui/e2e_tests/homePage.spec.ts` which checks the total
/// function count displayed on the homepage.
pub fn get_all_built_in_functions() -> Result<HashMap<String, Arc<FunctionConfig>>, Error> {
    #[cfg_attr(not(feature = "e2e_tests"), allow(unused_mut))]
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
