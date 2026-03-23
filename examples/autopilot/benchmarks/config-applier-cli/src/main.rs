//! CLI wrapper around the config-applier crate.
//!
//! Reads a JSON array of EditPayload objects from stdin, applies each edit to
//! the config files matched by --config-glob, and prints the written file paths
//! as a JSON array to stdout.

use std::io::Read;
use std::process;

use config_applier::{ConfigApplier, EditPayload};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    let config_glob = match parse_args(&args) {
        Some(glob) => glob,
        None => {
            eprintln!("Usage: config-applier-cli --config-glob <GLOB>");
            eprintln!("  Reads JSON array of edits from stdin.");
            eprintln!("  Writes JSON array of written file paths to stdout.");
            eprintln!();
            eprintln!("Example:");
            eprintln!(
                r#"  echo '[{{"operation": "upsert_variant", ...}}]' | config-applier-cli --config-glob "config/**/*.toml""#
            );
            process::exit(1);
        }
    };

    // Read edits JSON from stdin
    let mut input = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut input) {
        eprintln!("Error reading stdin: {e}");
        process::exit(1);
    }

    let edits: Vec<EditPayload> = match serde_json::from_str(&input) {
        Ok(edits) => edits,
        Err(e) => {
            eprintln!("Error parsing JSON input: {e}");
            eprintln!("Input was: {input}");
            process::exit(1);
        }
    };

    // Create the config applier
    let mut applier = match ConfigApplier::new(&config_glob).await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error loading config files: {e}");
            process::exit(1);
        }
    };

    // Apply each edit and collect written paths
    let mut all_paths = Vec::new();
    for (i, edit) in edits.iter().enumerate() {
        match applier.apply_edit(edit).await {
            Ok(paths) => {
                for path in paths {
                    all_paths.push(path.display().to_string());
                }
            }
            Err(e) => {
                eprintln!("Error applying edit {i}: {e}");
                process::exit(1);
            }
        }
    }

    // Output written paths as JSON array
    let output = serde_json::to_string(&all_paths).expect("failed to serialize paths");
    println!("{output}");
}

fn parse_args(args: &[String]) -> Option<String> {
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--config-glob" {
            if i + 1 < args.len() {
                return Some(args[i + 1].clone());
            }
            return None;
        }
        i += 1;
    }
    None
}
