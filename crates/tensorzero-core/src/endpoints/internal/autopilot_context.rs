//! Computes a deployment context summary for autopilot sessions.
//!
//! This module pre-fetches configuration, feedback, and dataset information
//! so that the autopilot agent starts each session with rich deployment context
//! instead of spending early rounds on orientation tool calls.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use autopilot_client::{
    DeploymentContext, DeploymentContextDataset, DeploymentContextEvaluation,
    DeploymentContextFunction, DeploymentContextFunctionFeedback, DeploymentContextMetric,
    DeploymentContextMetricFeedback, DeploymentContextVariant, DeploymentContextVariantFeedback,
};
use futures::future::join_all;
use tokio::time::timeout;

use crate::config::Config;
use crate::db::datasets::{DatasetQueries, GetDatasetMetadataParams};
use crate::db::feedback::{FeedbackByVariant, FeedbackQueries};
use crate::error::Error;
use crate::evaluations::EvaluationConfig;
use crate::function::FunctionConfig;
use crate::utils::gateway::ResolvedAppStateData;

/// Maximum time to spend fetching feedback data across all functions.
const FEEDBACK_TIMEOUT: Duration = Duration::from_secs(5);

/// Compute a structured deployment context from the current app state.
///
/// This is called once per new session and the result is attached to the
/// `CreateEventRequest` so the autopilot agent has immediate knowledge of the
/// deployment's functions, variants, metrics, feedback performance, and datasets.
pub async fn compute_deployment_context(
    app_state: &ResolvedAppStateData,
) -> Result<DeploymentContext, Error> {
    let config = &app_state.config;
    let database = app_state.get_delegating_database();

    let (functions, metrics, evaluations) = build_config_data(config);
    let feedback_by_function = build_feedback_data(config, &database).await;
    let datasets = build_dataset_data(&database).await;

    Ok(DeploymentContext {
        functions,
        metrics,
        evaluations,
        feedback_by_function,
        datasets,
    })
}

/// Build config data from in-memory config.
fn build_config_data(
    config: &Arc<Config>,
) -> (
    Vec<DeploymentContextFunction>,
    Vec<DeploymentContextMetric>,
    Vec<DeploymentContextEvaluation>,
) {
    // Functions
    let mut functions: Vec<DeploymentContextFunction> = config
        .functions
        .iter()
        .map(|(func_name, func_config)| {
            let func_type = match func_config.as_ref() {
                FunctionConfig::Chat(_) => "chat",
                FunctionConfig::Json(_) => "json",
            };
            let variants_map = func_config.variants();
            let mut variant_names: Vec<_> = variants_map.keys().collect();
            variant_names.sort();
            let variants = variant_names
                .into_iter()
                .map(|variant_name| {
                    let variant_info = &variants_map[variant_name];
                    let model_names = variant_info
                        .inner
                        .direct_model_names()
                        .into_iter()
                        .map(|m| m.to_string())
                        .collect();
                    DeploymentContextVariant {
                        name: variant_name.to_string(),
                        model_names,
                    }
                })
                .collect();
            DeploymentContextFunction {
                name: func_name.to_string(),
                r#type: func_type.to_string(),
                variants,
            }
        })
        .collect();
    functions.sort_by(|a, b| a.name.cmp(&b.name));

    // Metrics
    let mut metrics: Vec<DeploymentContextMetric> = config
        .metrics
        .iter()
        .map(|(metric_name, metric_config)| DeploymentContextMetric {
            name: metric_name.to_string(),
            r#type: format!("{:?}", metric_config.r#type).to_lowercase(),
            optimize: format!("{:?}", metric_config.optimize).to_lowercase(),
            level: format!("{:?}", metric_config.level).to_lowercase(),
        })
        .collect();
    metrics.sort_by(|a, b| a.name.cmp(&b.name));

    // Evaluations
    let mut evaluations: Vec<DeploymentContextEvaluation> = config
        .evaluations
        .iter()
        .map(|(eval_name, eval_config)| {
            let EvaluationConfig::Inference(inference_eval) = eval_config.as_ref();
            DeploymentContextEvaluation {
                name: eval_name.to_string(),
                function_name: inference_eval.function_name.to_string(),
                evaluator_names: inference_eval
                    .evaluators
                    .keys()
                    .map(|e: &String| e.to_string())
                    .collect(),
            }
        })
        .collect();
    evaluations.sort_by(|a, b| a.name.cmp(&b.name));

    (functions, metrics, evaluations)
}

/// Feedback query result for a single function+metric pair.
struct FeedbackResult {
    metric_name: String,
    variants: Vec<FeedbackByVariant>,
}

/// Build feedback-by-variant data by querying the database.
///
/// Uses a timeout to avoid blocking session creation if the database is slow.
async fn build_feedback_data(
    config: &Arc<Config>,
    database: &(dyn FeedbackQueries + Send + Sync),
) -> Vec<DeploymentContextFunctionFeedback> {
    if config.functions.is_empty() || config.metrics.is_empty() {
        return Vec::new();
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
            return Vec::new();
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

    let mut func_names: Vec<_> = by_function.keys().cloned().collect();
    func_names.sort();

    func_names
        .into_iter()
        .map(|func_name| {
            let results = by_function.remove(&func_name).unwrap_or_default();
            let mut sorted_results = results;
            sorted_results.sort_by(|a, b| a.metric_name.cmp(&b.metric_name));

            let metrics = sorted_results
                .into_iter()
                .map(|result| {
                    let mut sorted_variants = result.variants;
                    sorted_variants.sort_by(|a, b| b.count.cmp(&a.count));

                    let variants = sorted_variants
                        .into_iter()
                        .map(|fbv| DeploymentContextVariantFeedback {
                            variant_name: fbv.variant_name,
                            mean: fbv.mean,
                            variance: fbv.variance,
                            count: fbv.count,
                        })
                        .collect();

                    DeploymentContextMetricFeedback {
                        metric_name: result.metric_name,
                        variants,
                    }
                })
                .collect();

            DeploymentContextFunctionFeedback {
                function_name: func_name,
                metrics,
            }
        })
        .collect()
}

/// Build the dataset inventory.
async fn build_dataset_data(
    database: &(dyn DatasetQueries + Send + Sync),
) -> Vec<DeploymentContextDataset> {
    let params = GetDatasetMetadataParams {
        function_name: None,
        limit: Some(100),
        offset: None,
    };

    match database.get_dataset_metadata(&params).await {
        Ok(datasets) => datasets
            .into_iter()
            .map(|ds| DeploymentContextDataset {
                name: ds.dataset_name,
                count: ds.count as i64,
            })
            .collect(),
        Err(e) => {
            tracing::debug!(error = %e, "Failed to fetch dataset metadata");
            Vec::new()
        }
    }
}
