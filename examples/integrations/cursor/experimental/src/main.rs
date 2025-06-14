//! This file is intended to run as a post-commit hook for the cursorzero application.
//!
//! Steps performed in `main`:
//! 1. Parse command-line arguments (repository path and gateway URL).
//! 2. Discover the Git repository at the given path.
//! 3. Retrieve the latest commit and its parent's timestamp interval.
//! 4. Generate diffs for each file in the commit.
//! 5. For each diff hunk, parse its content into a syntax tree.
//! 6. Compute tree-edit-distance metrics and collect inference data.
//! 7. Send collected inferences to an external service via HTTP gateway.
//!
//! By running automatically after each commit, this hook enables continuous
//! code-change analysis and integration with TensorZero.

use std::{collections::HashMap, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use cursorzero::clickhouse::InferenceInfo;
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
use rayon::prelude::*;
use serde_json::json;
use tensorzero::{ClientBuilder, ClientBuilderMode, FeedbackParams};
use tensorzero_internal::clickhouse::ClickHouseConnectionInfo;
use tensorzero_internal::inference::types::ContentBlockChatOutput;
use tracing_subscriber::{EnvFilter, FmtSubscriber};
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
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .finish();
    #[expect(clippy::expect_used)]
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set global default subscriber");

    let args = Cli::parse();
    let repo = Repository::discover(args.path)?;
    let commit = get_last_commit_from_repo(&repo)?;
    let commit_interval = get_commit_timestamp_and_parent_timestamp(&commit)?;
    let diffs = get_diff_by_file(&repo, &commit)?;
    let client = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
        url: args.gateway_url,
    })
    .build()
    .await?;
    let diff_trees: HashMap<PathBuf, Vec<TreeInfo>> = diffs
        .into_par_iter()
        .map(|(file, diffs)| {
            let tree_infos = diffs
                .into_par_iter()
                .filter_map(|diff| {
                    let ext = file.extension().and_then(|ext| ext.to_str())?;
                    let tree = parse_hunk(&diff.content, ext).ok()?;
                    Some(TreeInfo {
                        path: file.clone(),
                        tree,
                        src: diff.content.into(),
                    })
                })
                .collect::<Vec<_>>();
            (file, tree_infos)
        })
        .collect();

    let clickhouse_url = std::env::var("CURSORZERO_CLICKHOUSE_URL")?;
    let clickhouse = ClickHouseConnectionInfo::new(&clickhouse_url).await?;
    let inferences = get_inferences_in_time_range(&clickhouse, commit_interval).await?;
    let inferences_by_id: HashMap<Uuid, &InferenceInfo> =
        inferences.iter().map(|i| (i.id, i)).collect();
    let mut inference_trees: HashMap<Uuid, Vec<TreeInfo>> = HashMap::new();
    let inference_results: Vec<_> = inferences
        .par_iter()
        .filter_map(|inference| {
            // If the parsing fails (meaning the inference doesn't look like something we recognize),
            // we log a warning and skip the inference.
            let code_blocks = match parse_cursor_output(&inference.input, &inference.output) {
                Ok(code_blocks) => code_blocks,
                Err(e) => {
                    tracing::warn!(
                        "Skipping inference {} because it doesn't look like a cursor response: {e}",
                        inference.id
                    );
                    return None;
                }
            };
            let tree_infos: Vec<TreeInfo> = code_blocks
                .into_iter()
                .filter_map(|code_block| {
                    let tree =
                        parse_hunk(&code_block.content, &code_block.language_extension).ok()?;
                    Some(TreeInfo {
                        path: code_block.path,
                        tree,
                        src: code_block.content.into(),
                    })
                })
                .collect();
            if tree_infos.is_empty() {
                None
            } else {
                Some((inference.id, tree_infos))
            }
        })
        .collect();

    for (inference_id, tree_infos) in inference_results {
        inference_trees
            .entry(inference_id)
            .or_default()
            .extend(tree_infos);
    }
    // Map all the InferenceTreeInfo to NormalizedInferenceTreeInfo
    let mut normalized_inference_trees: HashMap<Uuid, Vec<NormalizedInferenceTreeInfo>> =
        HashMap::new();
    for (inference_id, inference_tree_info) in inference_trees {
        for tree_info in inference_tree_info {
            let paths = find_paths_in_repo(&repo, &tree_info.path)?;
            normalized_inference_trees
                .entry(inference_id)
                .or_default()
                .push(NormalizedInferenceTreeInfo {
                    paths,
                    tree: tree_info.tree,
                    src: tree_info.src,
                });
        }
    }
    let mut num_feedbacks_sent = 0;
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
            let best_ted_info = all_diff_trees
                .par_iter()
                .map(|diff_tree| {
                    // TODO: do all the DFS once for each tree (maybe lazily) and memoize the results.
                    // TODO: skip all checks here or in the inner loop where the tree size difference is larger than the minimum TED
                    // already found.
                    minimum_ted(
                        &tree_info.tree.root_node(),
                        &tree_info.src,
                        &diff_tree.tree.root_node(),
                        &diff_tree.src,
                    )
                })
                .min_by_key(|ted| ted.min_ted);
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

            if let Some(min_ted_source) = &best_ted_info.min_ted_source {
                client
                    .feedback(FeedbackParams {
                        inference_id: Some(inference_id),
                        metric_name: "demonstration".to_string(),
                        value: serde_json::Value::String(min_ted_source.clone()),
                        tags: HashMap::from_iter([(
                            "cursorzero_demonstration_kind".to_string(),
                            "raw".to_string(),
                        )]),
                        episode_id: None,
                        internal: false,
                        dryrun: None,
                    })
                    .await?;

                let original_inference = inferences_by_id.get(&inference_id).ok_or_else(|| {
                    anyhow::anyhow!("Inference not found in inferences_by_id: {inference_id}")
                })?;
                let output_text = match original_inference.output.as_slice() {
                    [ContentBlockChatOutput::Text(t)] => &t.text,
                    _ => {
                        return Err(anyhow::anyhow!("Output is not a single text block"));
                    }
                };

                // Inside of the original cursor response, replace the old tree with the closest tree we found in the git diff.
                // This produces a demonstration with the proper context (e.g. the '// Start of Selection' comment).
                let updated_cursor_output =
                    output_text.replace(&String::from_utf8(tree_info.src)?, min_ted_source);

                client
                    .feedback(FeedbackParams {
                        inference_id: Some(inference_id),
                        metric_name: "demonstration".to_string(),
                        value: serde_json::Value::String(updated_cursor_output),
                        tags: HashMap::from_iter([(
                            "cursorzero_demonstration_kind".to_string(),
                            "replaced".to_string(),
                        )]),
                        episode_id: None,
                        internal: false,
                        dryrun: None,
                    })
                    .await?;
            }
            num_feedbacks_sent += 1;
        }
    }
    #[expect(clippy::print_stdout)]
    {
        println!("Number of feedbacks sent: {num_feedbacks_sent}");
    }
    Ok(())
}

#[derive(Debug)]
struct TreeInfo {
    path: PathBuf, // VSCode workspace relative path for inferences, git-relative path for diffs
    tree: Tree,
    src: Vec<u8>,
}

#[derive(Debug)]
struct NormalizedInferenceTreeInfo {
    paths: Vec<PathBuf>, // git-relative paths that might be the right path for this inference
    tree: Tree,
    src: Vec<u8>,
}
