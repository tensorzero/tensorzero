#![expect(clippy::print_stdout)]

use anyhow::Result;
use cursorzero::tree_export::export_all_trees;
use std::path::Path;

fn main() -> Result<()> {
    // Create tree_exports directory in the current working directory
    let output_dir = Path::new("tree_exports");

    println!("Exporting trees from integration tests to JSON files...");
    export_all_trees(output_dir)?;

    println!("Tree export completed successfully!");
    println!("Files created in: {}", output_dir.display());

    Ok(())
}
