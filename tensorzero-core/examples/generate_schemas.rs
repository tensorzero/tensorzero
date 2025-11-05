//! Generate JSON schemas for all types with JsonSchema derive
//!
//! Run with: cargo run --example generate_schemas
//!
//! This will create a `schemas/` directory with separate JSON files for each type.

use schemars::schema_for;
use std::fs;
use std::path::Path;

// Import types from tensorzero-core
use tensorzero_core::endpoints::datasets::v1::types::{
    DatapointMetadataUpdate, JsonDatapointOutputUpdate, UpdateChatDatapointRequest,
    UpdateDatapointRequest, UpdateDatapointsRequest, UpdateDatapointsResponse,
    UpdateJsonDatapointRequest,
};
use tensorzero_core::inference::types::{
    Arguments, ContentBlockChatOutput, Input, System,
};
use tensorzero_core::tool::{DynamicToolParams, ToolCallChunk, ToolChoice};

fn main() {
    println!("Generating JSON schemas for tensorzero-core types...\n");

    let output_dir = Path::new("schemas");

    // Create output directory if it doesn't exist
    if !output_dir.exists() {
        fs::create_dir(output_dir).expect("Failed to create schemas directory");
        println!("Created output directory: {:?}\n", output_dir);
    }

    let mut count = 0;

    // Generate schemas for tool types
    println!("Generating tool type schemas:");
    generate_and_save::<DynamicToolParams>("DynamicToolParams", output_dir);
    count += 1;
    generate_and_save::<ToolChoice>("ToolChoice", output_dir);
    count += 1;
    generate_and_save::<ToolCallChunk>("ToolCallChunk", output_dir);
    count += 1;

    // Generate schemas for inference types
    println!("\nGenerating inference type schemas:");
    generate_and_save::<Input>("Input", output_dir);
    count += 1;
    generate_and_save::<Arguments>("Arguments", output_dir);
    count += 1;
    generate_and_save::<System>("System", output_dir);
    count += 1;
    generate_and_save::<ContentBlockChatOutput>("ContentBlockChatOutput", output_dir);
    count += 1;

    // Generate schemas for dataset API types
    println!("\nGenerating dataset API type schemas:");
    generate_and_save::<UpdateDatapointsRequest>("UpdateDatapointsRequest", output_dir);
    count += 1;
    generate_and_save::<UpdateDatapointRequest>("UpdateDatapointRequest", output_dir);
    count += 1;
    generate_and_save::<UpdateChatDatapointRequest>(
        "UpdateChatDatapointRequest",
        output_dir,
    );
    count += 1;
    generate_and_save::<UpdateJsonDatapointRequest>(
        "UpdateJsonDatapointRequest",
        output_dir,
    );
    count += 1;
    generate_and_save::<JsonDatapointOutputUpdate>("JsonDatapointOutputUpdate", output_dir);
    count += 1;
    generate_and_save::<DatapointMetadataUpdate>("DatapointMetadataUpdate", output_dir);
    count += 1;
    generate_and_save::<UpdateDatapointsResponse>("UpdateDatapointsResponse", output_dir);
    count += 1;

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
