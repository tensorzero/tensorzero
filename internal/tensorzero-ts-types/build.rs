#![expect(clippy::expect_used)]
#![expect(clippy::panic)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Root types for which we generate TypeScript bundles.
/// Each entry is (const_name, type_name) where:
/// - const_name: The Rust constant name (e.g., "INFERENCE_TOOL_PARAMS")
/// - type_name: The TypeScript type name matching the .ts filename (e.g., "InferenceToolParams")
const ROOT_TYPES: &[(&str, &str)] = &[
    // LlmParams types
    ("AUTO_REJECT_TOOL_CALL_PARAMS", "AutoRejectToolCallParams"),
    (
        "CREATE_DATAPOINTS_TOOL_PARAMS",
        "CreateDatapointsToolParams",
    ),
    (
        "CREATE_DATAPOINTS_FROM_INFERENCES_TOOL_PARAMS",
        "CreateDatapointsFromInferencesToolParams",
    ),
    (
        "DELETE_DATAPOINTS_TOOL_PARAMS",
        "DeleteDatapointsToolParams",
    ),
    ("FEEDBACK_TOOL_PARAMS", "FeedbackToolParams"),
    ("GET_CONFIG_TOOL_PARAMS", "GetConfigToolParams"),
    ("GET_DATAPOINTS_TOOL_PARAMS", "GetDatapointsToolParams"),
    (
        "GET_FEEDBACK_BY_VARIANT_TOOL_PARAMS",
        "GetFeedbackByVariantToolParams",
    ),
    ("GET_INFERENCES_TOOL_PARAMS", "GetInferencesToolParams"),
    (
        "GET_LATEST_FEEDBACK_BY_METRIC_TOOL_PARAMS",
        "GetLatestFeedbackByMetricToolParams",
    ),
    ("INFERENCE_TOOL_PARAMS", "InferenceToolParams"),
    (
        "LAUNCH_OPTIMIZATION_WORKFLOW_TOOL_PARAMS",
        "LaunchOptimizationWorkflowToolParams",
    ),
    ("LIST_DATAPOINTS_TOOL_PARAMS", "ListDatapointsToolParams"),
    ("LIST_DATASETS_TOOL_PARAMS", "ListDatasetsToolParams"),
    ("LIST_INFERENCES_TOOL_PARAMS", "ListInferencesToolParams"),
    ("RUN_EVALUATION_TOOL_PARAMS", "RunEvaluationToolParams"),
    (
        "UPDATE_DATAPOINTS_TOOL_PARAMS",
        "UpdateDatapointsToolParams",
    ),
    ("WRITE_CONFIG_TOOL_PARAMS", "WriteConfigToolParams"),
    // Output types
    ("CREATE_DATAPOINTS_RESPONSE", "CreateDatapointsResponse"),
    ("DELETE_DATAPOINTS_RESPONSE", "DeleteDatapointsResponse"),
    ("FEEDBACK_RESPONSE", "FeedbackResponse"),
    ("GET_DATAPOINTS_RESPONSE", "GetDatapointsResponse"),
    ("GET_INFERENCES_RESPONSE", "GetInferencesResponse"),
    ("INFERENCE_RESPONSE", "InferenceResponse"),
    (
        "LAUNCH_OPTIMIZATION_WORKFLOW_TOOL_OUTPUT",
        "LaunchOptimizationWorkflowToolOutput",
    ),
    (
        "LATEST_FEEDBACK_ID_BY_METRIC_RESPONSE",
        "LatestFeedbackIdByMetricResponse",
    ),
    ("LIST_DATASETS_RESPONSE", "ListDatasetsResponse"),
    ("RUN_EVALUATION_RESPONSE", "RunEvaluationResponse"),
    ("UPDATE_DATAPOINTS_RESPONSE", "UpdateDatapointsResponse"),
    ("WRITE_CONFIG_RESPONSE", "WriteConfigResponse"),
    ("GET_CONFIG_RESPONSE", "GetConfigResponse"),
    ("FEEDBACK_BY_VARIANT", "FeedbackByVariant"),
];

/// Parsed information from a single .ts file.
struct TsFile {
    /// The type name (derived from filename).
    type_name: String,
    /// Names of types this file imports.
    imports: Vec<String>,
    /// The declaration body (with imports and `export` keyword stripped).
    declaration: String,
}

fn main() {
    let bindings_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tensorzero-node")
        .join("lib")
        .join("bindings");

    // Tell Cargo to re-run if any .ts file changes
    println!("cargo::rerun-if-changed={}", bindings_dir.display());

    // Parse all .ts files
    let ts_files = parse_all_ts_files(&bindings_dir);

    // Build a lookup map: type_name -> TsFile
    let file_map: HashMap<&str, &TsFile> =
        ts_files.iter().map(|f| (f.type_name.as_str(), f)).collect();

    // Generate bundles for each root type
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| ".".to_string()));
    let out_path = out_dir.join("ts_bundles.rs");

    let mut output = String::new();
    let mut bundles_for_validation: Vec<(String, String)> = Vec::new();

    for &(const_name, type_name) in ROOT_TYPES {
        if !file_map.contains_key(type_name) {
            // Type doesn't have a .ts file yet - generate an empty placeholder
            println!(
                "cargo::warning=tensorzero-ts-types: No .ts file found for `{type_name}`, generating empty bundle for `{const_name}`"
            );
            output.push_str(&format!(
                "pub const {const_name}: TsTypeBundle = TsTypeBundle(\"\");\n",
            ));
            continue;
        }

        let bundle = generate_bundle(type_name, &file_map);
        bundles_for_validation.push((const_name.to_string(), bundle.clone()));
        output.push_str(&format!(
            "pub const {const_name}: TsTypeBundle = TsTypeBundle({bundle});\n",
            bundle = quote_rust_string(&bundle),
        ));
    }

    fs::write(&out_path, output).expect("Failed to write generated ts_bundles.rs");

    // Validate that all bundles compile as TypeScript
    validate_bundles_with_tsc(&out_dir, &bundles_for_validation);
}

/// Parse all .ts files in the bindings directory (recursively).
fn parse_all_ts_files(dir: &Path) -> Vec<TsFile> {
    let mut files = Vec::new();
    parse_ts_files_recursive(dir, &mut files);
    files
}

fn parse_ts_files_recursive(dir: &Path, files: &mut Vec<TsFile>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            println!(
                "cargo::warning=tensorzero-ts-types: Could not read bindings directory {}: {e}",
                dir.display()
            );
            return;
        }
    };

    for entry in entries {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        if path.is_dir() {
            parse_ts_files_recursive(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "ts")
            && path
                .file_name()
                .is_none_or(|n| n.to_string_lossy() != "index.ts")
            && let Some(ts_file) = parse_ts_file(&path)
        {
            files.push(ts_file);
        }
    }
}

/// Parse a single .ts file to extract type name, imports, and declaration.
fn parse_ts_file(path: &Path) -> Option<TsFile> {
    let type_name = path.file_stem()?.to_string_lossy().to_string();
    let content = fs::read_to_string(path).ok()?;

    let mut imports = Vec::new();
    let mut declaration_lines = Vec::new();

    for line in content.lines() {
        if line.starts_with("import type") || line.starts_with("import {") {
            // Parse import: `import type { X } from "./X";`
            // or `import type { X, Y } from "./X";`
            if let Some(start) = line.find('{')
                && let Some(end) = line.find('}')
            {
                let names_str = &line[start + 1..end];
                for name in names_str.split(',') {
                    let name = name.trim();
                    if !name.is_empty() {
                        imports.push(name.to_string());
                    }
                }
            }
        } else if line.starts_with("// This file was generated by") {
            // Skip the ts-rs header comment
            continue;
        } else {
            // Strip `export ` prefix from declarations
            let stripped = line.strip_prefix("export ").unwrap_or(line);
            declaration_lines.push(stripped.to_string());
        }
    }

    // Join and trim
    let declaration = declaration_lines.join("\n").trim().to_string();

    Some(TsFile {
        type_name,
        imports,
        declaration,
    })
}

/// Generate a complete TypeScript bundle for a root type by resolving
/// its transitive dependency graph and concatenating declarations in
/// topological order.
fn generate_bundle(root_type: &str, file_map: &HashMap<&str, &TsFile>) -> String {
    // BFS to find transitive closure of dependencies
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();

    queue.push_back(root_type.to_string());
    visited.insert(root_type.to_string());

    // Track edges for topological sort
    // dependencies[A] = set of types that A depends on
    let mut dependencies: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    while let Some(current) = queue.pop_front() {
        let deps = if let Some(file) = file_map.get(current.as_str()) {
            file.imports
                .iter()
                .filter(|imp| file_map.contains_key(imp.as_str()))
                .cloned()
                .collect::<BTreeSet<_>>()
        } else {
            BTreeSet::new()
        };

        for dep in &deps {
            if visited.insert(dep.clone()) {
                queue.push_back(dep.clone());
            }
        }

        dependencies.insert(current, deps);
    }

    // Topological sort (Kahn's algorithm)
    // dependencies[A] = types that A imports (A depends on them).
    // We want: if A depends on B, B comes first.
    // in_degree[A] = number of A's dependencies that haven't been placed yet.
    let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
    for (node, deps) in &dependencies {
        *in_degree.entry(node.clone()).or_insert(0) = deps
            .iter()
            .filter(|d| dependencies.contains_key(d.as_str()))
            .count();
    }

    let mut queue: VecDeque<String> = VecDeque::new();
    for (node, &deg) in &in_degree {
        if deg == 0 {
            queue.push_back(node.clone());
        }
    }

    let mut sorted = Vec::new();
    while let Some(node) = queue.pop_front() {
        sorted.push(node.clone());
        // For every type that depends on `node`, decrease its in-degree
        for (other, deps) in &dependencies {
            if deps.contains(&node)
                && let Some(deg) = in_degree.get_mut(other)
            {
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(other.clone());
                }
            }
        }
    }

    // Build the bundle by concatenating declarations in topological order
    let mut parts = Vec::new();
    for type_name in &sorted {
        if let Some(file) = file_map.get(type_name.as_str())
            && !file.declaration.is_empty()
        {
            parts.push(file.declaration.as_str());
        }
    }

    parts.join("\n\n")
}

/// Validate that all generated TypeScript bundles compile by shelling out to `tsc`.
///
/// Each bundle is written to a separate `.ts` file (with `export {}` to make it a module
/// and prevent cross-file type conflicts). If `tsc` is not found in PATH, a warning is
/// emitted and validation is skipped.
fn validate_bundles_with_tsc(out_dir: &Path, bundles: &[(String, String)]) {
    let tsc_dir = out_dir.join("tsc_check");
    let _ = fs::remove_dir_all(&tsc_dir);
    fs::create_dir_all(&tsc_dir).expect("Failed to create directory for tsc check");

    let mut ts_files = Vec::new();
    for (name, bundle) in bundles {
        if bundle.is_empty() {
            continue;
        }
        let file_path = tsc_dir.join(format!("{name}.ts"));
        // Prefix with `export {}` so each file is treated as a module,
        // preventing duplicate type errors across bundles that share dependencies.
        let content = format!("export {{}}\n\n{bundle}");
        fs::write(&file_path, content).expect("Failed to write TypeScript file for tsc check");
        ts_files.push(file_path);
    }

    if ts_files.is_empty() {
        return;
    }

    // Resolve `tsc` from the workspace's node_modules/.bin (installed via pnpm)
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let tsc = workspace_root.join("node_modules").join(".bin").join("tsc");

    let result = Command::new(&tsc)
        .args(["--noEmit", "--skipLibCheck"])
        .args(&ts_files)
        .output();

    match result {
        Ok(output) if !output.status.success() => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            panic!("TypeScript bundle compilation check failed!\n{stdout}\n{stderr}");
        }
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            assert!(
                std::env::var("TENSORZERO_CI").as_deref() != Ok("1"),
                "`tsc` not found at {} but TENSORZERO_CI=1 requires it. Run `pnpm install` to install.",
                tsc.display()
            );
            println!(
                "cargo::warning=tensorzero-ts-types: `tsc` not found at {}, skipping TypeScript bundle validation. Run `pnpm install` to enable.",
                tsc.display()
            );
        }
        Err(e) => {
            assert!(
                std::env::var("TENSORZERO_CI").as_deref() != Ok("1"),
                "Failed to run `tsc`: {e} but TENSORZERO_CI=1 requires it"
            );
            println!(
                "cargo::warning=tensorzero-ts-types: Failed to run `tsc`: {e}, skipping TypeScript bundle validation"
            );
        }
    }
}

/// Quote a string as a Rust string literal, escaping special characters.
fn quote_rust_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
