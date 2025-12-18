pub mod clickhouse;
pub mod cursor;
pub mod git;
pub mod parsing;
pub mod ted;
pub mod util;

use anyhow::Result;
use git2::Repository;
use rayon::prelude::*;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tensorzero::FeedbackParams;
use tensorzero_core::{config::BatchWritesConfig, db::clickhouse::ClickHouseConnectionInfo};
use uuid::Uuid;

use crate::{
    clickhouse::{InferenceInfo, get_inferences_in_time_range},
    cursor::parse_cursor_output,
    git::{CommitInterval, DiffAddition, find_paths_in_repo},
    parsing::{InferenceWithTrees, parse_hunk},
    ted::{TedInfo, minimum_ted},
    util::{NormalizedInferenceTreeInfo, TreeInfo, generate_demonstration},
};

/// Parses git diff additions into AST trees for edit distance analysis
pub fn process_diffs(
    diffs: HashMap<PathBuf, Vec<DiffAddition>>,
) -> Result<Arc<HashMap<PathBuf, Vec<TreeInfo>>>> {
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
    Ok(diff_trees)
}

/// Fetches AI inferences from ClickHouse and parses them into normalized tree structures
pub async fn process_inferences(
    repo: &Repository,
    commit_interval: CommitInterval,
    user: Option<String>,
) -> Result<HashMap<Uuid, InferenceWithTrees>> {
    let clickhouse_url = std::env::var("CURSORZERO_CLICKHOUSE_URL")?;
    // Don't use batching here
    let clickhouse =
        ClickHouseConnectionInfo::new(&clickhouse_url, BatchWritesConfig::default()).await?;
    let inferences = get_inferences_in_time_range(&clickhouse, commit_interval, user).await?;

    let inference_results: Vec<_> = inferences
        .par_iter()
        .filter_map(|inference| {
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

    let normalized_inference_trees: Result<HashMap<Uuid, InferenceWithTrees>> = inference_results
        .into_iter()
        .map(|(inference_id, tree_infos, inference)| {
            let normalized_tree_infos: Result<Vec<_>> = tree_infos
                .into_iter()
                .map(|tree_info| {
                    let paths = find_paths_in_repo(repo, &tree_info.path)?;
                    Ok(NormalizedInferenceTreeInfo {
                        paths,
                        tree: tree_info.tree,
                        src: tree_info.src,
                    })
                })
                .collect();

            let inference_with_trees = InferenceWithTrees::new(normalized_tree_infos?, inference);
            Ok((inference_id, inference_with_trees))
        })
        .collect();

    normalized_inference_trees
}

/// Computes edit distances between inferences and diffs, sending feedback to TensorZero
pub async fn process_and_send_feedback(
    normalized_inference_trees: HashMap<Uuid, InferenceWithTrees>,
    diff_trees: &Arc<HashMap<PathBuf, Vec<TreeInfo>>>,
    client: &tensorzero::Client,
) -> Result<(usize, usize)> {
    let feedback_results: Vec<Result<Vec<FeedbackParams>>> = normalized_inference_trees
        .into_par_iter()
        .flat_map(|(inference_id, inference_with_trees)| {
            let diff_trees = diff_trees.clone();
            let inference_info = inference_with_trees.inference.clone();
            inference_with_trees
                .trees
                .into_par_iter()
                .map(move |tree_info| {
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

    Ok((num_feedbacks_sent, inferences_with_feedback))
}

/// Calculates tree edit distances and generates feedback metrics for a single inference tree
pub fn process_tree_info(
    tree_info: NormalizedInferenceTreeInfo,
    inference_id: Uuid,
    inference_info: &InferenceInfo,
    diff_trees: &HashMap<PathBuf, Vec<TreeInfo>>,
) -> Result<Vec<FeedbackParams>> {
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
            minimum_ted(
                &tree_info.tree.root_node(),
                &tree_info.src,
                &diff_tree.tree.root_node(),
                &diff_tree.src,
            )
        })
        .collect();

    let sum_teds = min_teds.iter().map(|m| m.min_ted).sum::<u64>();
    let sum_sizes = min_teds.iter().map(|m| m.size).sum::<usize>();
    if sum_sizes == 0 {
        tracing::warn!("No trees found for inference {inference_id}, skipping feedbacks");
        return Ok(vec![]);
    }
    let average_ted_ratio = sum_teds as f64 / sum_sizes as f64;
    let mut feedbacks = vec![
        FeedbackParams {
            inference_id: Some(inference_id),
            metric_name: "average_ted_ratio".to_string(),
            value: serde_json::json!(average_ted_ratio),
            tags: HashMap::new(),
            episode_id: None,
            internal: false,
            dryrun: None,
        },
        FeedbackParams {
            inference_id: Some(inference_id),
            metric_name: "num_code_blocks".to_string(),
            value: serde_json::json!(inference_diff_trees.len()),
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
