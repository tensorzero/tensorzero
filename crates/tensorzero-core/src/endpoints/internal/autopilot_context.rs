//! Computes a deployment context summary for autopilot sessions.
//!
//! This module pre-fetches configuration, feedback, and dataset information
//! so that the autopilot agent starts each session with rich deployment context
//! instead of spending early rounds on orientation tool calls.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;
use std::time::Duration;

use futures::future::join_all;
use tokio::time::timeout;

use crate::config::Config;
use crate::db::datasets::{DatasetQueries, GetDatasetMetadataParams};
use crate::db::feedback::{FeedbackByVariant, FeedbackQueries};
use crate::error::Error;
use crate::function::FunctionConfig;
use crate::utils::gateway::ResolvedAppStateData;

/// Maximum time to spend fetching feedback data across all functions.
const FEEDBACK_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum token budget for the deployment context (estimated at ~4 chars/token).
const MAX_CONTEXT_CHARS: usize = 32_000; // ~8K tokens

/// Compute a structured markdown deployment context from the current app state.
///
/// This is called once per new session and the result is attached to the
/// `CreateEventRequest` so the autopilot agent has immediate knowledge of the
/// deployment's functions, variants, metrics, feedback performance, and datasets.
pub async fn compute_deployment_context(app_state: &ResolvedAppStateData) -> Result<String, Error> {
    let config = &app_state.config;
    let database = app_state.get_delegating_database();

    // 1. Build config summary (synchronous, from in-memory config)
    let config_section = build_config_section(config);

    // 2. Fetch feedback by variant for each function+metric pair (with timeout)
    let feedback_section = build_feedback_section(config, &database).await;

    // 3. Fetch dataset inventory
    let dataset_section = build_dataset_section(&database).await;

    // 4. Assemble and truncate
    let mut context = String::with_capacity(8192);
    context.push_str("## Deployment Context\n\n");
    context.push_str(&config_section);
    context.push_str(&feedback_section);
    context.push_str(&dataset_section);

    Ok(truncate_deployment_context(context, MAX_CONTEXT_CHARS))
}

/// Build the config summary section from in-memory config.
fn build_config_section(config: &Arc<Config>) -> String {
    let mut out = String::new();

    // Functions table
    out.push_str("### Functions\n\n");
    if config.functions.is_empty() {
        out.push_str("No functions configured.\n\n");
    } else {
        out.push_str("| Function | Type | Variants |\n");
        out.push_str("| --- | --- | --- |\n");
        let mut functions: Vec<_> = config.functions.iter().collect();
        functions.sort_by_key(|(name, _)| *name);
        for (func_name, func_config) in &functions {
            let func_type = match func_config.as_ref() {
                FunctionConfig::Chat(_) => "chat",
                FunctionConfig::Json(_) => "json",
            };
            let variants = func_config.variants();
            let mut variant_strs: Vec<String> = Vec::new();
            let mut variant_names: Vec<_> = variants.keys().collect();
            variant_names.sort();
            for variant_name in variant_names {
                let variant_info = &variants[variant_name];
                let models = variant_info.inner.direct_model_names();
                let model_str = if models.is_empty() {
                    String::new()
                } else {
                    format!(
                        " ({})",
                        models
                            .iter()
                            .map(|m| m.as_ref())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
                variant_strs.push(format!("`{variant_name}`{model_str}"));
            }
            let variants_cell = if variant_strs.is_empty() {
                "none".to_string()
            } else {
                variant_strs.join(", ")
            };
            writeln!(out, "| `{func_name}` | {func_type} | {variants_cell} |").ok();
        }
        out.push('\n');
    }

    // Metrics table
    out.push_str("### Metrics\n\n");
    if config.metrics.is_empty() {
        out.push_str("No metrics configured.\n\n");
    } else {
        out.push_str("| Metric | Type | Optimize | Level |\n");
        out.push_str("| --- | --- | --- | --- |\n");
        let mut metrics: Vec<_> = config.metrics.iter().collect();
        metrics.sort_by_key(|(name, _)| *name);
        for (metric_name, metric_config) in &metrics {
            let metric_type = format!("{:?}", metric_config.r#type).to_lowercase();
            let optimize = format!("{:?}", metric_config.optimize).to_lowercase();
            let level = format!("{:?}", metric_config.level).to_lowercase();
            writeln!(
                out,
                "| `{metric_name}` | {metric_type} | {optimize} | {level} |"
            )
            .ok();
        }
        out.push('\n');
    }

    // Evaluations
    if !config.evaluations.is_empty() {
        out.push_str("### Evaluations\n\n");
        out.push_str("| Evaluation | Function | Evaluators |\n");
        out.push_str("| --- | --- | --- |\n");
        let mut evals: Vec<_> = config.evaluations.iter().collect();
        evals.sort_by_key(|(name, _)| *name);
        for (eval_name, eval_config) in &evals {
            let crate::evaluations::EvaluationConfig::Inference(inference_eval) =
                eval_config.as_ref();
            let evaluator_names: Vec<_> = inference_eval
                .evaluators
                .keys()
                .map(|e| format!("`{e}`"))
                .collect();
            writeln!(
                out,
                "| `{eval_name}` | `{}` | {} |",
                inference_eval.function_name,
                evaluator_names.join(", ")
            )
            .ok();
        }
        out.push('\n');
    }

    out
}

/// Feedback query result for a single function+metric pair.
struct FeedbackResult {
    metric_name: String,
    variants: Vec<FeedbackByVariant>,
}

/// Build the feedback-by-variant section by querying the database.
///
/// Uses a timeout to avoid blocking session creation if the database is slow.
/// Any functions that don't complete in time are noted as skipped.
async fn build_feedback_section(
    config: &Arc<Config>,
    database: &(dyn FeedbackQueries + Send + Sync),
) -> String {
    if config.functions.is_empty() || config.metrics.is_empty() {
        return String::new();
    }

    // Build list of (function_name, metric_name) pairs to query
    let mut queries: Vec<(String, String)> = Vec::new();
    for func_name in config.functions.keys() {
        for metric_name in config.metrics.keys() {
            queries.push((func_name.clone(), metric_name.clone()));
        }
    }

    // Execute all queries in parallel with a timeout
    let futures: Vec<_> = queries
        .iter()
        .map(|(func_name, metric_name)| {
            let func_name = func_name.clone();
            let metric_name = metric_name.clone();
            async move {
                let result = database
                    .get_feedback_by_variant(&metric_name, &func_name, None, None, None)
                    .await;
                (func_name, metric_name, result)
            }
        })
        .collect();

    let results = match timeout(FEEDBACK_TIMEOUT, join_all(futures)).await {
        Ok(results) => results,
        Err(_) => {
            tracing::warn!("Feedback queries timed out during deployment context computation");
            return "### Feedback by Variant\n\n_Feedback data timed out._\n\n".to_string();
        }
    };

    // Group results by function
    let mut by_function: HashMap<String, Vec<FeedbackResult>> = HashMap::new();
    for (func_name, metric_name, result) in results {
        match result {
            Ok(variants) if !variants.is_empty() => {
                by_function
                    .entry(func_name.clone())
                    .or_default()
                    .push(FeedbackResult {
                        metric_name,
                        variants,
                    });
            }
            Ok(_) => {} // Empty results, skip
            Err(e) => {
                tracing::debug!(
                    function = %func_name,
                    metric = %metric_name,
                    error = %e,
                    "Failed to fetch feedback by variant"
                );
            }
        }
    }

    if by_function.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("### Feedback by Variant\n\n");
    let mut func_names: Vec<_> = by_function.keys().collect();
    func_names.sort();
    for func_name in func_names {
        let results = &by_function[func_name];
        writeln!(out, "#### `{func_name}`\n").ok();
        out.push_str("| Variant | Metric | Count | Mean | StdErr |\n");
        out.push_str("| --- | --- | --- | --- | --- |\n");
        let mut sorted_results: Vec<_> = results.iter().collect();
        sorted_results.sort_by(|a, b| a.metric_name.cmp(&b.metric_name));
        for result in sorted_results {
            let mut sorted_variants: Vec<_> = result.variants.iter().collect();
            sorted_variants.sort_by(|a, b| b.count.cmp(&a.count));
            for fbv in sorted_variants {
                let stderr = fbv
                    .variance
                    .map(|v| {
                        if fbv.count > 1 {
                            format!("{:.4}", (v / fbv.count as f32).sqrt())
                        } else {
                            "—".to_string()
                        }
                    })
                    .unwrap_or_else(|| "—".to_string());
                writeln!(
                    out,
                    "| `{}` | `{}` | {} | {:.4} | {stderr} |",
                    fbv.variant_name, result.metric_name, fbv.count, fbv.mean
                )
                .ok();
            }
        }
        out.push('\n');
    }

    out
}

/// Build the dataset inventory section.
async fn build_dataset_section(database: &(dyn DatasetQueries + Send + Sync)) -> String {
    let params = GetDatasetMetadataParams {
        function_name: None,
        limit: Some(100), // Cap at 100 datasets
        offset: None,
    };

    match database.get_dataset_metadata(&params).await {
        Ok(datasets) if !datasets.is_empty() => {
            let mut out = String::new();
            out.push_str("### Datasets\n\n");
            out.push_str("| Dataset | Datapoints |\n");
            out.push_str("| --- | --- |\n");
            for ds in &datasets {
                writeln!(out, "| `{}` | {} |", ds.dataset_name, ds.count).ok();
            }
            out.push('\n');
            out
        }
        Ok(_) => String::new(), // No datasets
        Err(e) => {
            tracing::debug!(error = %e, "Failed to fetch dataset metadata");
            String::new()
        }
    }
}

/// Truncate deployment context to fit within the token budget.
///
/// Sections are removed in reverse priority order:
/// 1. Datasets
/// 2. Feedback tables (truncated to top functions by count)
/// 3. Config (reduced to function names only)
fn truncate_deployment_context(context: String, max_chars: usize) -> String {
    if context.len() <= max_chars {
        return context;
    }

    // Strategy: find section boundaries and drop from the end
    // Drop datasets section first
    if let Some(datasets_start) = context.find("### Datasets\n") {
        let truncated = &context[..datasets_start];
        if truncated.len() <= max_chars {
            return truncated.to_string();
        }
    }

    // Drop feedback section
    if let Some(feedback_start) = context.find("### Feedback by Variant\n") {
        let truncated = &context[..feedback_start];
        if truncated.len() <= max_chars {
            return truncated.to_string();
        }
    }

    // Last resort: hard truncate
    let mut truncated: String = context.chars().take(max_chars).collect();
    truncated.push_str("\n\n_[Context truncated]_\n");
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_within_budget() {
        let ctx = "short context".to_string();
        assert_eq!(truncate_deployment_context(ctx.clone(), 1000), ctx);
    }

    #[test]
    fn test_truncate_drops_datasets() {
        let ctx = "## Deployment Context\n\n### Functions\n\nstuff\n\n### Datasets\n\nlots of data here\n".to_string();
        let result = truncate_deployment_context(ctx, 50);
        assert!(!result.contains("### Datasets"));
        assert!(result.contains("### Functions"));
    }

    #[test]
    fn test_truncate_drops_feedback_then_datasets() {
        let ctx = "## Deployment Context\n\n### Functions\n\nstuff\n\n### Feedback by Variant\n\nlots\n\n### Datasets\n\nmore\n".to_string();
        let result = truncate_deployment_context(ctx, 60);
        assert!(!result.contains("### Feedback"));
        assert!(result.contains("### Functions"));
    }
}
