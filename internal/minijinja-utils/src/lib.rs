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
//!   - `{% include 'optional.html' if condition %}` ✓
//!
//! - **Dynamic loads**: Template names depend on runtime values and cannot be statically determined
//!   - `{% include template_var %}` ✗ (variable)
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
mod tests;
