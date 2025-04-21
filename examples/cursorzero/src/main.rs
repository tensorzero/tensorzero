use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use cursorzero::{
    clickhouse::get_inferences_in_time_range,
    cursor::parse_cursor_output,
    git::{
        find_paths_in_repo, get_commit_timestamp_and_parent_timestamp, get_diff_by_file,
        get_last_commit_from_repo,
    },
    parsing::parse_hunk,
};
use git2::Repository;
use tensorzero_internal::clickhouse::ClickHouseConnectionInfo;
use tree_sitter::Tree;
use uuid::Uuid;

#[derive(Parser, Debug)]
struct Cli {
    #[clap(short, long, default_value = ".")]
    path: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();
    let repo = Repository::discover(args.path)?;
    let commit = get_last_commit_from_repo(&repo)?;
    let commit_interval = get_commit_timestamp_and_parent_timestamp(&commit)?;
    println!(
        "Found last commit with message: {}",
        commit.message().unwrap()
    );
    let diffs = get_diff_by_file(&repo, &commit)?;
    let mut diff_trees: HashMap<String, Vec<Tree>> = HashMap::new();

    for (file, diffs) in diffs {
        println!("File: {}", file);
        for diff in diffs {
            println!("  {}", diff.content);
            let tree = parse_hunk(&diff.content, &file.split('.').last().unwrap()).ok();
            if let Some(tree) = tree {
                println!("  {:?}", tree);
                diff_trees.entry(file.clone()).or_default().push(tree);
            } else {
                println!("  Failed to parse");
            }
        }
    }
    let clickhouse_url = std::env::var("CURSORZERO_CLICKHOUSE_URL")?;
    let clickhouse = ClickHouseConnectionInfo::new(&clickhouse_url).await?;
    let inferences = get_inferences_in_time_range(&clickhouse, commit_interval).await?;
    let mut inference_trees: HashMap<Uuid, Vec<InferenceTreeInfo>> = HashMap::new();
    for inference in inferences {
        println!("Inference: {}", inference.id);
        println!("  Input: {:?}", inference.input);
        println!("  Output: {:?}", inference.output);
        let code_blocks =
            parse_cursor_output(&inference.input, &inference.output).with_context(|| {
                format!("Error parsing cursor output for inference {}", inference.id)
            })?;
        println!("Code blocks: {:?}", code_blocks);
        for code_block in code_blocks {
            let tree = parse_hunk(&code_block.content, &code_block.language_extension).ok();
            println!("Inference: {}", inference.id);
            println!("  Tree: {:?}", tree);
            if let Some(tree) = tree {
                inference_trees
                    .entry(inference.id)
                    .or_default()
                    .push(InferenceTreeInfo {
                        path: PathBuf::from(code_block.path),
                        tree,
                    });
            }
        }
    }
    // Map all the InferenceTreeInfo to NormalizedInferenceTreeInfo
    let mut normalized_inference_trees: HashMap<Uuid, Vec<NormalizedInferenceTreeInfo>> =
        HashMap::new();
    for (inference_id, inference_tree_info) in inference_trees {
        for tree_info in inference_tree_info {
            let path = find_paths_in_repo(&repo, &tree_info.path)?;
            normalized_inference_trees
                .entry(inference_id)
                .or_default()
                .push(NormalizedInferenceTreeInfo {
                    path,
                    tree: tree_info.tree,
                });
        }
    }
    // TODOs:
    // - For each InferenceTreeInfo in the map, find the git-relative file path and find the diff tree that corresponds to it.
    // - Compute for each diff tree the minimum edit distance to the inference tree.
    // - Print the results.
    // - Send feedback to TensorZero for that inference.
    Ok(())
}

#[derive(Debug)]
struct InferenceTreeInfo {
    path: PathBuf, // VSCode workspace relative path
    tree: Tree,
}

#[derive(Debug)]
struct NormalizedInferenceTreeInfo {
    path: Vec<PathBuf>, // git-relative paths that might be the right path for this inference
    tree: Tree,
}
