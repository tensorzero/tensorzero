//! Static analysis utilities for MiniJinja templates.
//!
//! This crate analyzes MiniJinja templates to extract all template dependencies
//! (includes, imports, extends) by parsing the template syntax and walking the AST.
//!
//! ## Core Concept
//!
//! The analyzer distinguishes between **static** and **dynamic** template loading:
//!
//! - **Static loads**: Template names are string literals that can be determined at parse time
//!   - `{% include 'header.html' %}` ✓
//!   - `{% extends 'base.html' %}` ✓
//!   - `{% include ['a.html', 'b.html'] %}` ✓
//!   - `{% include 'optional.html' if condition else 'default.html' %}` ✓
//!
//! - **Dynamic loads**: Template names depend on runtime values and cannot be statically determined
//!   - `{% include template_var %}` ✗ (variable)
//!   - `{% include 'optional.html' if condition %}` ✗ (conditional without else)
//!   - `{% include get_template() %}` ✗ (function call)
//!
//! ## Main Function
//!
//! [`collect_all_template_paths()`] recursively analyzes templates starting from a root template
//! and returns all discovered template paths. It returns an error if any dynamic loads are found.
//!
//! ## Example
//!
//! ```rust
//! use minijinja::Environment;
//! use minijinja_utils::collect_all_template_paths;
//!
//! let mut env = Environment::new();
//! env.add_template("main.html", "{% include 'header.html' %}Content").unwrap();
//! env.add_template("header.html", "Header").unwrap();
//!
//! // Collect all template dependencies
//! let paths = collect_all_template_paths(&env, "main.html").unwrap();
//! assert_eq!(paths.len(), 2); // main.html and header.html
//! ```
//!
//! ## Error Handling
//!
//! When dynamic loads are detected, [`AnalysisError::DynamicLoadsFound`] provides detailed
//! location information including line numbers, column positions, and source quotes to help
//! identify the problematic template expressions.
//!
//! ## Implementation Note
//!
//! This crate uses MiniJinja's `unstable_machinery` feature to access the AST. This API
//! may change between MiniJinja versions.

mod error;
mod loading;

pub use error::{AnalysisError, DynamicLoadLocation, LoadKind};
pub use loading::collect_all_template_paths;

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::Environment;
    use std::path::PathBuf;

    #[test]
    fn test_simple_static_includes() {
        let mut env = Environment::new();
        env.add_template(
            "main.html",
            "{% include 'header.html' %}Content{% include 'footer.html' %}",
        )
        .unwrap();
        env.add_template("header.html", "Header").unwrap();
        env.add_template("footer.html", "Footer").unwrap();

        let paths = collect_all_template_paths(&env, "main.html").unwrap();

        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&PathBuf::from("main.html")));
        assert!(paths.contains(&PathBuf::from("header.html")));
        assert!(paths.contains(&PathBuf::from("footer.html")));
    }

    #[test]
    fn test_nested_includes() {
        let mut env = Environment::new();
        env.add_template("main.html", "{% include 'partial.html' %}")
            .unwrap();
        env.add_template("partial.html", "{% include 'nested.html' %}")
            .unwrap();
        env.add_template("nested.html", "Nested content").unwrap();

        let paths = collect_all_template_paths(&env, "main.html").unwrap();

        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&PathBuf::from("main.html")));
        assert!(paths.contains(&PathBuf::from("partial.html")));
        assert!(paths.contains(&PathBuf::from("nested.html")));
    }

    #[test]
    fn test_extends_and_blocks() {
        let mut env = Environment::new();
        env.add_template("base.html", "{% block content %}{% endblock %}")
            .unwrap();
        env.add_template(
            "child.html",
            "{% extends 'base.html' %}{% block content %}Child content{% endblock %}",
        )
        .unwrap();

        let paths = collect_all_template_paths(&env, "child.html").unwrap();

        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&PathBuf::from("child.html")));
        assert!(paths.contains(&PathBuf::from("base.html")));
    }

    #[test]
    fn test_import_statements() {
        let mut env = Environment::new();
        env.add_template("macros.html", "{% macro test() %}Macro{% endmacro %}")
            .unwrap();
        env.add_template("main.html", "{% import 'macros.html' as m %}{{ m.test() }}")
            .unwrap();

        let paths = collect_all_template_paths(&env, "main.html").unwrap();

        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&PathBuf::from("main.html")));
        assert!(paths.contains(&PathBuf::from("macros.html")));
    }

    #[test]
    fn test_dynamic_include_error() {
        let mut env = Environment::new();
        env.add_template("main.html", "{% include template_name %}")
            .unwrap();

        let result = collect_all_template_paths(&env, "main.html");

        assert!(result.is_err());
        match result {
            Err(AnalysisError::DynamicLoadsFound(locations)) => {
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].template_name, "main.html");
                assert_eq!(locations[0].reason, "variable");
                assert!(matches!(locations[0].load_kind, LoadKind::Include { .. }));
                assert!(locations[0].source_quote.contains("template_name"));
            }
            _ => panic!("Expected DynamicLoadsFound error"),
        }
    }

    #[test]
    fn test_conditional_without_else_error() {
        let mut env = Environment::new();
        env.add_template("main.html", "{% include 'static.html' if condition %}")
            .unwrap();
        env.add_template("static.html", "Static content").unwrap();

        let result = collect_all_template_paths(&env, "main.html");

        assert!(result.is_err());
        match result {
            Err(AnalysisError::DynamicLoadsFound(locations)) => {
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].reason, "conditional without else");
            }
            Err(e) => panic!("Expected DynamicLoadsFound error, got: {e:?}"),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_static_list_includes() {
        let mut env = Environment::new();
        env.add_template("main.html", "{% include ['first.html', 'second.html'] %}")
            .unwrap();
        env.add_template("first.html", "First").unwrap();
        env.add_template("second.html", "Second").unwrap();

        let paths = collect_all_template_paths(&env, "main.html").unwrap();

        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&PathBuf::from("main.html")));
        assert!(paths.contains(&PathBuf::from("first.html")));
        assert!(paths.contains(&PathBuf::from("second.html")));
    }

    #[test]
    fn test_mixed_list_with_dynamic_error() {
        let mut env = Environment::new();
        env.add_template("main.html", "{% include ['static.html', dynamic_var] %}")
            .unwrap();
        env.add_template("static.html", "Static content").unwrap();

        let result = collect_all_template_paths(&env, "main.html");

        assert!(result.is_err());
        match result {
            Err(AnalysisError::DynamicLoadsFound(locations)) => {
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].reason, "variable");
            }
            Err(e) => panic!("Expected DynamicLoadsFound error, got: {e:?}"),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_error_contains_line_and_column() {
        let mut env = Environment::new();
        env.add_template("main.html", "Line 1\nLine 2\n{% include variable %}")
            .unwrap();

        let result = collect_all_template_paths(&env, "main.html");

        assert!(result.is_err());
        match result {
            Err(AnalysisError::DynamicLoadsFound(locations)) => {
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].line, 3);
                assert!(locations[0].column > 0);
            }
            _ => panic!("Expected DynamicLoadsFound error"),
        }
    }

    #[test]
    fn test_circular_dependency_handled() {
        let mut env = Environment::new();
        env.add_template("a.html", "{% include 'b.html' %}")
            .unwrap();
        env.add_template("b.html", "{% include 'a.html' %}")
            .unwrap();

        let paths = collect_all_template_paths(&env, "a.html").unwrap();

        // Should handle circular dependency without infinite loop
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&PathBuf::from("a.html")));
        assert!(paths.contains(&PathBuf::from("b.html")));
    }
}
