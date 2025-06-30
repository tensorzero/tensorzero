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
use std::sync::Arc;
use tensorzero::{ClientBuilder, ClientBuilderMode, FeedbackParams};
use tensorzero_core::clickhouse::ClickHouseConnectionInfo;
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use url::Url;
use uuid::Uuid;

#[derive(Parser, Debug)]
struct Cli {
    #[clap(short, long, default_value = ".")]
    path: String,
    #[clap(long, default_value = "http://localhost:13000")]
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
    // Compute the tree-sitter trees for each diff.
    let diff_trees: Arc<HashMap<PathBuf, Vec<TreeInfo>>> = Arc::new(
        diffs
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
            .collect(),
    );

    let clickhouse_url = std::env::var("CURSORZERO_CLICKHOUSE_URL")?;
    let clickhouse = ClickHouseConnectionInfo::new(&clickhouse_url).await?;
    let inferences = get_inferences_in_time_range(&clickhouse, commit_interval, args.user).await?;
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
                tracing::warn!("Inference {} has no trees, skipping", inference.id);
                None
            } else {
                Some((inference.id, tree_infos, inference.clone()))
            }
        })
        .collect();

    // Grab all the trees for each inference and normalize the paths to the repo root.
    // We will want the inference ID mapping to the parsed inference trees as well as the raw inference info.
    #[expect(clippy::type_complexity)]
    let normalized_inference_trees: Result<
        HashMap<Uuid, (Vec<NormalizedInferenceTreeInfo>, Arc<InferenceInfo>)>,
    > = inference_results
        .into_iter()
        .map(|(inference_id, tree_infos, inference)| {
            let normalized_tree_infos: Result<Vec<_>> = tree_infos
                .into_iter()
                .map(|tree_info| {
                    let paths = find_paths_in_repo(&repo, &tree_info.path)?;
                    // There could be zero or more paths in the repo for a given path suffix.
                    Ok(NormalizedInferenceTreeInfo {
                        paths,
                        tree: tree_info.tree,
                        src: tree_info.src,
                    })
                })
                .collect();

            Ok((inference_id, (normalized_tree_infos?, inference)))
        })
        .collect();

    let normalized_inference_trees = normalized_inference_trees?;

    // Run TED calculations and send feedbacks in parallel.
    // Each will run independently so an error will only affect the inference that caused it.
    let feedback_results: Vec<Result<Vec<FeedbackParams>>> = normalized_inference_trees
        .into_par_iter()
        .flat_map(|(inference_id, (inference_tree_info, inference_info))| {
            let diff_trees = diff_trees.clone();
            inference_tree_info.into_par_iter().map(move |tree_info| {
                process_tree_info(tree_info, inference_id, &inference_info, &diff_trees)
            })
        })
        .collect();
    let mut all_feedback_params = Vec::new();
    let mut inferences_with_feedback: usize = 0;
    for feedback_result in feedback_results {
        match feedback_result {
            Ok(feedbacks) => {
                all_feedback_params.extend(feedbacks);
                inferences_with_feedback += 1;
            }
            Err(e) => tracing::error!("Failed to process inference, skipping: {e}"),
        }
    }
    // Send all the feedbacks in parallel using futures::future::join_all
    use futures::future::join_all;
    let feedback_futures = all_feedback_params
        .into_iter()
        .map(|feedback_params| client.feedback(feedback_params));
    let feedback_results = join_all(feedback_futures).await;
    let mut num_feedbacks_sent: usize = 0;
    for feedback_result in feedback_results {
        match feedback_result {
            Ok(_) => num_feedbacks_sent += 1,
            Err(e) => tracing::error!("Failed to send feedback: {e}"),
        }
    }

    #[expect(clippy::print_stdout)]
    {
        println!("Number of feedbacks sent: {num_feedbacks_sent}");
        println!("Number of inferences with feedback: {inferences_with_feedback}");
    }
    Ok(())
}

/// Processes a single tree info by computing TED metrics and sending feedback to TensorZero.
/// Returns the number of feedbacks sent (1 if successful, 0 if skipped).
fn process_tree_info(
    tree_info: NormalizedInferenceTreeInfo,
    inference_id: Uuid,
    inference_info: &InferenceInfo,
    diff_trees: &HashMap<PathBuf, Vec<TreeInfo>>,
) -> Result<Vec<FeedbackParams>> {
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
        return Ok(vec![]);
    }
    let average_ted_ratio = sum_teds as f64 / sum_sizes as f64;
    let mut feedbacks = vec![
        FeedbackParams {
            inference_id: Some(inference_id),
            metric_name: "average_ted_ratio".to_string(),
            value: json!(average_ted_ratio),
            tags: HashMap::new(),
            episode_id: None,
            internal: false,
            dryrun: None,
        },
        FeedbackParams {
            inference_id: Some(inference_id),
            metric_name: "num_code_blocks".to_string(),
            value: json!(inference_diff_trees.len()),
            tags: HashMap::new(),
            episode_id: None,
            internal: false,
            dryrun: None,
        },
    ];
    let demonstration =
        generate_demonstration(inference_info, &min_teds, inference_diff_trees.as_slice());
    match demonstration {
        Ok(demonstration) => {
            feedbacks.push(FeedbackParams {
                inference_id: Some(inference_id),
                metric_name: "demonstration".to_string(),
                value: serde_json::Value::String(demonstration),
                tags: HashMap::new(),
                episode_id: None,
                internal: false,
                dryrun: None,
            });
        }
        Err(e) => {
            tracing::warn!(
                "Failed to generate demonstration for inference {inference_id}, skipping: {e}"
            );
        }
    }

    Ok(feedbacks)
}
