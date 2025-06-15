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
use cursorzero::ted::{minimum_ted, TedInfo};
use cursorzero::util::{generate_demonstration, NormalizedInferenceTreeInfo, TreeInfo};
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
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use url::Url;
use uuid::Uuid;

#[derive(Parser, Debug)]
struct Cli {
    #[clap(short, long, default_value = ".")]
    path: String,
    #[clap(long, default_value = "http://localhost:6900")]
    gateway_url: Url,
    #[clap(long)]
    user: Option<String>,
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
    let inferences = get_inferences_in_time_range(&clickhouse, commit_interval, args.user).await?;
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
    // Map all the InferenceTreeInfo to NormalizedInferenceTreeInfo by figuring out the paths from the repo root
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
            let mut inference_diff_trees = Vec::new();
            for path in tree_info.paths {
                let Some(trees) = diff_trees.get(&path) else {
                    continue;
                };
                inference_diff_trees.extend(trees);
            }
            let min_teds: Vec<TedInfo> = inference_diff_trees
                .par_iter()
                .map(|diff_tree| {
                    // TODO(optimization): do all the DFS once for each tree (maybe lazily) and memoize the results.
                    // TODO(optimization): skip all checks here or in the inner loop where the tree size difference is larger than the minimum TED
                    // already found.
                    minimum_ted(
                        &tree_info.tree.root_node(),
                        &tree_info.src,
                        &diff_tree.tree.root_node(),
                        &diff_tree.src,
                    )
                })
                .collect();

            // Compute the average TED ratio for each diff tree
            let sum_teds = min_teds.iter().map(|m| m.min_ted).sum::<u64>();
            let sum_sizes = min_teds.iter().map(|m| m.size).sum::<usize>();
            if sum_sizes == 0 {
                // This should never happen if there were any trees in the inference.
                tracing::warn!("No trees found for inference {inference_id}, skipping feedbacks");
                continue;
            }
            let average_ted_ratio = sum_teds as f64 / sum_sizes as f64;

            // Send the average TED ratio to TensorZero as feedback.
            client
                .feedback(FeedbackParams {
                    inference_id: Some(inference_id),
                    metric_name: "average_ted_ratio".to_string(),
                    value: json!(average_ted_ratio),
                    tags: HashMap::new(),
                    episode_id: None,
                    internal: false,
                    dryrun: None,
                })
                .await?;
            // Send the total tree size to TensorZero as feedback.
            client
                .feedback(FeedbackParams {
                    inference_id: Some(inference_id),
                    metric_name: "total_tree_size".to_string(),
                    value: json!(sum_sizes),
                    tags: HashMap::new(),
                    episode_id: None,
                    internal: false,
                    dryrun: None,
                })
                .await?;
            // Send the number of code blocks to TensorZero as feedback.
            client
                .feedback(FeedbackParams {
                    inference_id: Some(inference_id),
                    metric_name: "num_code_blocks".to_string(),
                    value: json!(inference_diff_trees.len()),
                    tags: HashMap::new(),
                    episode_id: None,
                    internal: false,
                    dryrun: None,
                })
                .await?;

            let original_inference = inferences_by_id.get(&inference_id).ok_or_else(|| {
                anyhow::anyhow!("Inference not found in inferences_by_id: {inference_id}")
            })?;
            let demonstration = generate_demonstration(
                original_inference,
                &min_teds,
                inference_diff_trees.as_slice(),
            )?;
            client
                .feedback(FeedbackParams {
                    inference_id: Some(inference_id),
                    metric_name: "demonstration".to_string(),
                    value: serde_json::Value::String(demonstration),
                    tags: HashMap::new(),
                    episode_id: None,
                    internal: false,
                    dryrun: None,
                })
                .await?;
            num_feedbacks_sent += 1;
        }
    }
    #[expect(clippy::print_stdout)]
    {
        println!("Number of feedbacks sent: {num_feedbacks_sent}");
    }
    Ok(())
}
