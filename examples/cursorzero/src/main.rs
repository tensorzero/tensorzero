use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use cursorzero::ted::minimum_ted;
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
use serde_json::json;
use tensorzero::{ClientBuilder, ClientBuilderMode, FeedbackParams};
use tensorzero_internal::clickhouse::ClickHouseConnectionInfo;
use tree_sitter::Tree;
use url::Url;
use uuid::Uuid;

#[derive(Parser, Debug)]
struct Cli {
    #[clap(short, long, default_value = ".")]
    path: String,
    #[clap(long, default_value = "http://localhost:6900")]
    gateway_url: Url,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();
    let repo = Repository::discover(args.path)?;
    let commit = get_last_commit_from_repo(&repo)?;
    let commit_interval = get_commit_timestamp_and_parent_timestamp(&commit)?;
    let diffs = get_diff_by_file(&repo, &commit)?;
    let mut diff_trees: HashMap<PathBuf, Vec<Tree>> = HashMap::new();
    let client = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
        url: args.gateway_url,
    })
    .build()
    .await?;
    for (file, diffs) in diffs {
        for diff in diffs {
            let Some(ext) = file.extension().and_then(|ext| ext.to_str()) else {
                continue;
            };
            let tree = parse_hunk(&diff.content, ext).ok();
            let Some(tree) = tree else {
                continue;
            };
            diff_trees.entry(file.clone()).or_default().push(tree);
        }
    }
    let clickhouse_url = std::env::var("CURSORZERO_CLICKHOUSE_URL")?;
    let clickhouse = ClickHouseConnectionInfo::new(&clickhouse_url).await?;
    let inferences = get_inferences_in_time_range(&clickhouse, commit_interval).await?;
    let mut inference_trees: HashMap<Uuid, Vec<InferenceTreeInfo>> = HashMap::new();
    for inference in inferences {
        let code_blocks =
            parse_cursor_output(&inference.input, &inference.output).with_context(|| {
                format!("Error parsing cursor output for inference {}", inference.id)
            })?;
        for code_block in code_blocks {
            let tree = parse_hunk(&code_block.content, &code_block.language_extension).ok();
            if let Some(tree) = tree {
                inference_trees
                    .entry(inference.id)
                    .or_default()
                    .push(InferenceTreeInfo {
                        path: code_block.path,
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
                    paths: path,
                    tree: tree_info.tree,
                });
        }
    }

    for (inference_id, inference_tree_info) in normalized_inference_trees {
        for tree_info in inference_tree_info {
            // Get all the diff trees for the paths in this NormalizedInferenceTreeInfo
            let mut all_diff_trees = Vec::new();
            for path in tree_info.paths {
                let Some(trees) = diff_trees.get(&path) else {
                    continue;
                };
                all_diff_trees.extend(trees);
            }
            let mut best_ted_info = None;
            for diff_tree in all_diff_trees {
                // TODO: do all the DFS once for each tree (maybe lazily) and memoize the results.
                // TODO: skip all checks here or in the inner loop where the tree size difference is larger than the minimum TED
                // already found.
                let ted = minimum_ted(&tree_info.tree.root_node(), &diff_tree.root_node());
                if best_ted_info.is_none() {
                    best_ted_info = Some(ted);
                } else if let Some(ted_info) = best_ted_info.as_ref() {
                    if ted.min_ted < ted_info.min_ted {
                        best_ted_info = Some(ted);
                    }
                }
            }
            let Some(best_ted_info) = best_ted_info.as_ref() else {
                continue;
            };
            // Send the minimum TED to TensorZero as feedback.
            client
                .feedback(FeedbackParams {
                    inference_id: Some(inference_id),
                    metric_name: "min_ted".to_string(),
                    value: json!(best_ted_info.min_ted),
                    tags: HashMap::new(),
                    episode_id: None,
                    internal: false,
                    dryrun: None,
                })
                .await?;
            client
                .feedback(FeedbackParams {
                    inference_id: Some(inference_id),
                    metric_name: "ted_ratio".to_string(),
                    value: json!(best_ted_info.ted_ratio),
                    tags: HashMap::new(),
                    episode_id: None,
                    internal: false,
                    dryrun: None,
                })
                .await?;

            // TODO: Send a demonstration of the matching snippet if it's sufficiently good.
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
    paths: Vec<PathBuf>, // git-relative paths that might be the right path for this inference
    tree: Tree,
}
