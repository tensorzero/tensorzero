//! Generate JSON schemas for all types with JsonSchema derive
//!
//! Run with: cargo run --example generate_schemas
//!
//! This will create a `schemas/` directory with separate JSON files for each type.
//!
//! To add a new type to schema generation, simply add it to the `generate_all_schemas!` macro below.

use schemars::schema_for;
use std::fs;
use std::path::Path;

// Import types from tensorzero-core
use tensorzero_core::endpoints::datasets::v1::types::{
    DatapointMetadataUpdate, JsonDatapointOutputUpdate, UpdateChatDatapointRequest,
    UpdateDatapointRequest, UpdateDatapointsRequest, UpdateDatapointsResponse,
    UpdateJsonDatapointRequest,
};
use tensorzero_core::inference::types::{Arguments, ContentBlockChatOutput, Input, System};
use tensorzero_core::tool::{DynamicToolParams, ToolCallChunk, ToolChoice};

/// Macro to automatically generate schemas for a list of types
macro_rules! generate_all_schemas {
    (
        output_dir: $output_dir:expr,
        $(
            $category:literal => [
                $( $type:ty => $name:literal ),* $(,)?
            ]
        ),* $(,)?
    ) => {
        {
            let mut count = 0;
            $(
                println!("\nGenerating {} schemas:", $category);
                $(
                    generate_and_save::<$type>($name, $output_dir);
                    count += 1;
                )*
            )*
            count
        }
    };
}

fn main() {
    println!("Generating JSON schemas for tensorzero-core types...\n");

    let output_dir = Path::new("schemas");

    // Create output directory if it doesn't exist
    if !output_dir.exists() {
        fs::create_dir(output_dir).expect("Failed to create schemas directory");
        println!("Created output directory: {:?}\n", output_dir);
    }

    // Auto-generate schemas for all registered types
    // To add a new type: just add a line like: TypeName => "TypeName",
    let count = generate_all_schemas! {
        output_dir: output_dir,

        "tool type" => [
            DynamicToolParams => "DynamicToolParams",
            ToolChoice => "ToolChoice",
            ToolCallChunk => "ToolCallChunk",
        ],

        "inference type" => [
            Input => "Input",
            Arguments => "Arguments",
            System => "System",
            ContentBlockChatOutput => "ContentBlockChatOutput",
        ],

        "dataset API type" => [
            UpdateDatapointsRequest => "UpdateDatapointsRequest",
            UpdateDatapointRequest => "UpdateDatapointRequest",
            UpdateChatDatapointRequest => "UpdateChatDatapointRequest",
            UpdateJsonDatapointRequest => "UpdateJsonDatapointRequest",
            JsonDatapointOutputUpdate => "JsonDatapointOutputUpdate",
            DatapointMetadataUpdate => "DatapointMetadataUpdate",
            UpdateDatapointsResponse => "UpdateDatapointsResponse",
        ],
    };

    println!(
        "\n✓ Schema generation complete! {} files saved in {:?}",
        count,
        output_dir.canonicalize().unwrap_or(output_dir.to_path_buf())
    );
}

fn generate_and_save<T: schemars::JsonSchema>(type_name: &str, output_dir: &Path) {
    let schema = schema_for!(T);
    let json = serde_json::to_string_pretty(&schema)
        .unwrap_or_else(|e| panic!("Failed to serialize schema for {}: {}", type_name, e));

    let file_path = output_dir.join(format!("{}.json", type_name));
    fs::write(&file_path, json)
        .unwrap_or_else(|e| panic!("Failed to write schema for {}: {}", type_name, e));

    println!("  ✓ Generated: {}.json", type_name);
}
