//! Built-in TensorZero functions
//!
//! This module defines built-in functions that are automatically available
//! in all TensorZero deployments. These functions are prefixed with `tensorzero::`
//! and cannot be overridden by user-defined functions.
//!
//! Built-in functions are variant-less - they have empty variants HashMaps
//! but are fully defined FunctionConfigs with all necessary fields.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::config::SchemaData;
use crate::experimentation::ExperimentationConfig;
use crate::function::{FunctionConfig, FunctionConfigChat};
use crate::tool::ToolChoice;

/// Returns the `tensorzero::hello` function configuration.
///
/// This is a simple variant-less Chat function that serves as a basic
/// example of a built-in function.
fn get_hello_function() -> Arc<FunctionConfig> {
    Arc::new(FunctionConfig::Chat(FunctionConfigChat {
        variants: HashMap::new(),
        schemas: SchemaData::default(),
        tools: vec![],
        tool_choice: ToolChoice::None,
        parallel_tool_calls: None,
        description: Some("Built-in hello function - a simple greeting function".to_string()),
        all_explicit_templates_names: HashSet::new(),
        experimentation: ExperimentationConfig::default(),
    }))
}

/// Returns all built-in functions as a HashMap.
///
/// The keys are function names (e.g., "tensorzero::hello")
/// and the values are Arc-wrapped FunctionConfigs.
pub fn get_all_built_in_functions() -> HashMap<String, Arc<FunctionConfig>> {
    let mut functions = HashMap::new();
    functions.insert("tensorzero::hello".to_string(), get_hello_function());
    functions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_hello_function() {
        let hello = get_hello_function();
        match &*hello {
            FunctionConfig::Chat(config) => {
                assert!(config.variants.is_empty());
                assert!(config.tools.is_empty());
                assert_eq!(config.tool_choice, ToolChoice::None);
                assert!(config.description.is_some());
            }
            FunctionConfig::Json(_) => panic!("Expected Chat function"),
        }
    }

    #[test]
    fn test_get_all_built_in_functions() {
        let functions = get_all_built_in_functions();
        assert_eq!(functions.len(), 1);
        assert!(functions.contains_key("tensorzero::hello"));
    }
}
