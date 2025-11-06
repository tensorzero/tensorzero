//! Proc macro for automatic JSON schema generation
//!
//! This crate provides the `#[export_schema]` attribute macro that automatically
//! generates tests to export JSON schemas for types that derive `JsonSchema`.
//!
//! ## Usage
//!
//! ```ignore
//! use schemars::JsonSchema;
//! use tensorzero_schema_generation::export_schema;
//!
//! #[derive(JsonSchema)]
//! #[export_schema]
//! pub struct MyType {
//!     field: String,
//! }
//! ```
//!
//! When you run `cargo test`, this will automatically generate a JSON schema
//! file in the `schemas/` directory.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

/// Attribute macro to automatically export JSON schemas during tests.
///
/// This macro generates a test function that writes the JSON schema for the
/// annotated type to a file in the `schemas/` directory.
///
/// The macro works similarly to ts-rs's `#[ts(export)]` - when you run tests,
/// the schemas are automatically generated.
///
/// ## Example
///
/// ```ignore
/// #[derive(JsonSchema)]
/// #[export_schema]
/// pub struct DynamicToolParams {
///     // ...
/// }
/// ```
///
/// This generates a test that will create `schemas/DynamicToolParams.json`
/// when you run `cargo test`.
#[proc_macro_attribute]
pub fn export_schema(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();

    // Generate a unique test name based on the type name
    let test_name = syn::Ident::new(
        &format!("export_schema_{}", name_str.to_lowercase()),
        name.span(),
    );

    // Generate the test function
    let expanded = quote! {
        // Keep the original item
        #input

        // Generate a test that exports the schema
        #[cfg(test)]
        #[test]
        fn #test_name() {
            use schemars::schema_for;
            use std::fs;
            use std::path::Path;

            // Get the schema for this type
            let schema = schema_for!(#name);

            // Serialize to pretty JSON
            let json = serde_json::to_string_pretty(&schema)
                .expect(&format!("Failed to serialize schema for {}", #name_str));

            // Create the schemas directory if it doesn't exist
            let schema_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("Failed to find repository root")
                .join("schemas");

            fs::create_dir_all(&schema_dir)
                .expect("Failed to create schemas directory");

            // Write the schema file
            let file_path = schema_dir.join(format!("{}.json", #name_str));
            fs::write(&file_path, json)
                .expect(&format!("Failed to write schema for {}", #name_str));

            println!("✓ Generated schema: {}", file_path.display());
        }
    };

    TokenStream::from(expanded)
}
