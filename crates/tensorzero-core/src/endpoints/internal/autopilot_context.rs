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
    DeploymentContextFunction, DeploymentContextFunctionFeedback,
    DeploymentContextFunctionInferenceCount, DeploymentContextMetric,
    DeploymentContextMetricFeedback, DeploymentContextVariant, DeploymentContextVariantFeedback,
};
use futures::future::join_all;
use tokio::time::timeout;

use crate::config::Config;
use crate::db::datasets::{DatasetQueries, GetDatasetMetadataParams};
use crate::db::feedback::{FeedbackByVariant, FeedbackQueries};
use crate::db::inferences::InferenceQueries;
use crate::error::Error;
use crate::evaluations::EvaluationConfig;
use crate::function::FunctionConfig;
use crate::utils::gateway::ResolvedAppStateData;

/// Maximum time to spend on any single category of DB prefetch queries.
const PREFETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum number of functions to include in feedback prefetch queries.
/// Caps the Cartesian product of functions × metrics to avoid query fan-out.
const MAX_FEEDBACK_FUNCTIONS: usize = 10;

/// Maximum number of metrics to include in feedback prefetch queries.
const MAX_FEEDBACK_METRICS: usize = 10;

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

    compute_deployment_context_inner(config, &database, &database, &database).await
}

async fn compute_deployment_context_inner(
    config: &Arc<Config>,
    feedback_db: &(dyn FeedbackQueries + Send + Sync),
    dataset_db: &(dyn DatasetQueries + Send + Sync),
    inference_db: &(dyn InferenceQueries + Send + Sync),
) -> Result<DeploymentContext, Error> {
    let (functions, metrics, evaluations) = build_config_data(config);
    let (feedback_by_function, datasets, inference_counts_by_function) = tokio::join!(
        build_feedback_data(config, feedback_db),
        build_dataset_data(dataset_db),
        build_inference_count_data(inference_db),
    );

    Ok(DeploymentContext {
        functions,
        metrics,
        evaluations,
        feedback_by_function,
        datasets,
        inference_counts_by_function,
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

    // Build list of (function_name, metric_name) pairs to query.
    // HACK: We truncate to MAX_FEEDBACK_FUNCTIONS × MAX_FEEDBACK_METRICS to avoid
    // unbounded query fan-out on large configs. A proper fix would be a single batched
    // query, but this is good enough for now.
    let func_names: Vec<_> = config
        .functions
        .keys()
        .take(MAX_FEEDBACK_FUNCTIONS)
        .collect();
    let metric_names: Vec<_> = config.metrics.keys().take(MAX_FEEDBACK_METRICS).collect();
    let mut queries: Vec<(String, String)> = Vec::new();
    for func_name in &func_names {
        for metric_name in &metric_names {
            queries.push(((*func_name).clone(), (*metric_name).clone()));
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

    let results = match timeout(PREFETCH_TIMEOUT, join_all(futures)).await {
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

    let result = match timeout(PREFETCH_TIMEOUT, database.get_dataset_metadata(&params)).await {
        Ok(result) => result,
        Err(_) => {
            tracing::warn!(
                "Dataset metadata query timed out during deployment context computation"
            );
            return Vec::new();
        }
    };

    match result {
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

/// Build inference counts per function.
async fn build_inference_count_data(
    database: &(dyn InferenceQueries + Send + Sync),
) -> Vec<DeploymentContextFunctionInferenceCount> {
    let result = match timeout(
        PREFETCH_TIMEOUT,
        database.list_functions_with_inference_count(),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => {
            tracing::warn!("Inference count query timed out during deployment context computation");
            return Vec::new();
        }
    };

    match result {
        Ok(counts) => counts
            .into_iter()
            .map(|c| DeploymentContextFunctionInferenceCount {
                function_name: c.function_name,
                inference_count: c.inference_count,
                last_inference_timestamp: c.last_inference_timestamp,
            })
            .collect(),
        Err(e) => {
            tracing::debug!(error = %e, "Failed to fetch inference counts by function");
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::Utc;
    use googletest::prelude::*;

    use crate::config::{MetricConfig, MetricConfigLevel, MetricConfigOptimize, MetricConfigType};
    use crate::db::datasets::{DatasetMetadata, MockDatasetQueries};
    use crate::db::feedback::MockFeedbackQueries;
    use crate::db::inferences::{FunctionInferenceCount, MockInferenceQueries};
    use crate::error::ErrorDetails;
    use crate::evaluations::{EvaluatorConfig, ExactMatchConfig, InferenceEvaluationConfig};
    use crate::function::FunctionConfig;
    use crate::function::FunctionConfigChat;
    use crate::variant::chat_completion::ChatCompletionConfig;
    use crate::variant::{VariantConfig, VariantInfo};

    // =========================================================================
    // build_config_data tests
    // =========================================================================

    #[gtest]
    fn test_build_config_data_empty_config() {
        let config = Arc::new(Config::default());
        let (functions, metrics, evaluations) = build_config_data(&config);
        expect_that!(functions, is_empty());
        expect_that!(metrics, is_empty());
        expect_that!(evaluations, is_empty());
    }

    #[gtest]
    fn test_build_config_data_with_functions_and_metrics() {
        let mut config = Config::default();

        // Add a chat function with one variant
        let mut variants = HashMap::new();
        variants.insert(
            "variant_a".to_string(),
            Arc::new(VariantInfo {
                inner: VariantConfig::ChatCompletion(ChatCompletionConfig::default()),
                timeouts: Default::default(),
                namespace: None,
            }),
        );
        config.functions.insert(
            "my_chat_fn".to_string(),
            Arc::new(FunctionConfig::Chat(FunctionConfigChat {
                variants,
                ..Default::default()
            })),
        );

        // Add metrics
        config.metrics.insert(
            "accuracy".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Boolean,
                optimize: MetricConfigOptimize::Max,
                level: MetricConfigLevel::Inference,
                description: None,
            },
        );
        config.metrics.insert(
            "cost".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Float,
                optimize: MetricConfigOptimize::Min,
                level: MetricConfigLevel::Episode,
                description: None,
            },
        );

        let config = Arc::new(config);
        let (functions, metrics, _evaluations) = build_config_data(&config);

        // Verify functions
        expect_that!(functions, len(eq(1)));
        expect_that!(functions[0].name, eq("my_chat_fn"));
        expect_that!(functions[0].r#type, eq("chat"));
        expect_that!(functions[0].variants, len(eq(1)));
        expect_that!(functions[0].variants[0].name, eq("variant_a"));

        // Verify metrics are sorted by name
        expect_that!(metrics, len(eq(2)));
        expect_that!(metrics[0].name, eq("accuracy"));
        expect_that!(metrics[0].r#type, eq("boolean"));
        expect_that!(metrics[0].optimize, eq("max"));
        expect_that!(metrics[0].level, eq("inference"));
        expect_that!(metrics[1].name, eq("cost"));
        expect_that!(metrics[1].r#type, eq("float"));
        expect_that!(metrics[1].optimize, eq("min"));
        expect_that!(metrics[1].level, eq("episode"));
    }

    #[gtest]
    fn test_build_config_data_with_evaluations() {
        let mut config = Config::default();

        let mut evaluators = HashMap::new();
        #[expect(deprecated)]
        let exact_match = EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None });
        evaluators.insert("exact_match".to_string(), exact_match);

        config.evaluations.insert(
            "eval_1".to_string(),
            Arc::new(EvaluationConfig::Inference(InferenceEvaluationConfig {
                function_name: "test_function".to_string(),
                evaluators,
                description: None,
            })),
        );

        let config = Arc::new(config);
        let (_functions, _metrics, evaluations) = build_config_data(&config);

        expect_that!(evaluations, len(eq(1)));
        expect_that!(evaluations[0].name, eq("eval_1"));
        expect_that!(evaluations[0].function_name, eq("test_function"));
        expect_that!(evaluations[0].evaluator_names, len(eq(1)));
        expect_that!(evaluations[0].evaluator_names, contains(eq("exact_match")));
    }

    // =========================================================================
    // build_feedback_data tests
    // =========================================================================

    #[gtest]
    #[tokio::test]
    async fn test_build_feedback_data_empty_config() {
        let config = Arc::new(Config::default());
        let mock_db = MockFeedbackQueries::new();
        let result = build_feedback_data(&config, &mock_db).await;
        expect_that!(result, is_empty());
    }

    #[gtest]
    #[tokio::test]
    async fn test_build_feedback_data_with_results() {
        let mut config = Config::default();
        config.functions.insert(
            "fn_a".to_string(),
            Arc::new(FunctionConfig::Chat(FunctionConfigChat::default())),
        );
        config.metrics.insert(
            "metric_1".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Boolean,
                optimize: MetricConfigOptimize::Max,
                level: MetricConfigLevel::Inference,
                description: None,
            },
        );
        let config = Arc::new(config);

        let mut mock_db = MockFeedbackQueries::new();
        mock_db.expect_get_feedback_by_variant().returning(
            |_metric, _func, _variants, _ns, _max| {
                Box::pin(async {
                    Ok(vec![
                        FeedbackByVariant {
                            variant_name: "v1".to_string(),
                            mean: 0.8,
                            variance: Some(0.1),
                            count: 100,
                        },
                        FeedbackByVariant {
                            variant_name: "v2".to_string(),
                            mean: 0.6,
                            variance: None,
                            count: 50,
                        },
                    ])
                })
            },
        );

        let result = build_feedback_data(&config, &mock_db).await;
        expect_that!(result, len(eq(1)));
        expect_that!(result[0].function_name, eq("fn_a"));
        expect_that!(result[0].metrics, len(eq(1)));
        expect_that!(result[0].metrics[0].metric_name, eq("metric_1"));
        // Variants sorted by count descending
        expect_that!(result[0].metrics[0].variants, len(eq(2)));
        expect_that!(result[0].metrics[0].variants[0].variant_name, eq("v1"));
        expect_that!(result[0].metrics[0].variants[0].count, eq(100));
        expect_that!(result[0].metrics[0].variants[1].variant_name, eq("v2"));
        expect_that!(result[0].metrics[0].variants[1].count, eq(50));
    }

    #[gtest]
    #[tokio::test]
    async fn test_build_feedback_data_db_error() {
        let mut config = Config::default();
        config.functions.insert(
            "fn_a".to_string(),
            Arc::new(FunctionConfig::Chat(FunctionConfigChat::default())),
        );
        config.metrics.insert(
            "metric_1".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Boolean,
                optimize: MetricConfigOptimize::Max,
                level: MetricConfigLevel::Inference,
                description: None,
            },
        );
        let config = Arc::new(config);

        let mut mock_db = MockFeedbackQueries::new();
        mock_db.expect_get_feedback_by_variant().returning(
            |_metric, _func, _variants, _ns, _max| {
                Box::pin(async {
                    Err(Error::new(ErrorDetails::ClickHouseConnection {
                        message: "connection refused".to_string(),
                    }))
                })
            },
        );

        let result = build_feedback_data(&config, &mock_db).await;
        expect_that!(result, is_empty());
    }

    #[gtest]
    #[tokio::test]
    async fn test_build_feedback_data_empty_results() {
        let mut config = Config::default();
        config.functions.insert(
            "fn_a".to_string(),
            Arc::new(FunctionConfig::Chat(FunctionConfigChat::default())),
        );
        config.metrics.insert(
            "metric_1".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Boolean,
                optimize: MetricConfigOptimize::Max,
                level: MetricConfigLevel::Inference,
                description: None,
            },
        );
        let config = Arc::new(config);

        let mut mock_db = MockFeedbackQueries::new();
        mock_db
            .expect_get_feedback_by_variant()
            .returning(|_metric, _func, _variants, _ns, _max| Box::pin(async { Ok(vec![]) }));

        let result = build_feedback_data(&config, &mock_db).await;
        expect_that!(result, is_empty());
    }

    // =========================================================================
    // build_dataset_data tests
    // =========================================================================

    #[gtest]
    #[tokio::test]
    async fn test_build_dataset_data_success() {
        let mut mock_db = MockDatasetQueries::new();
        mock_db.expect_get_dataset_metadata().returning(|_params| {
            Box::pin(async {
                Ok(vec![
                    DatasetMetadata {
                        dataset_name: "train_set".to_string(),
                        count: 1000,
                        last_updated: "2024-01-01T00:00:00Z".to_string(),
                    },
                    DatasetMetadata {
                        dataset_name: "eval_set".to_string(),
                        count: 200,
                        last_updated: "2024-01-02T00:00:00Z".to_string(),
                    },
                ])
            })
        });

        let result = build_dataset_data(&mock_db).await;
        expect_that!(result, len(eq(2)));
        expect_that!(result[0].name, eq("train_set"));
        expect_that!(result[0].count, eq(1000));
        expect_that!(result[1].name, eq("eval_set"));
        expect_that!(result[1].count, eq(200));
    }

    #[gtest]
    #[tokio::test]
    async fn test_build_dataset_data_error() {
        let mut mock_db = MockDatasetQueries::new();
        mock_db.expect_get_dataset_metadata().returning(|_params| {
            Box::pin(async {
                Err(Error::new(ErrorDetails::ClickHouseConnection {
                    message: "connection refused".to_string(),
                }))
            })
        });

        let result = build_dataset_data(&mock_db).await;
        expect_that!(result, is_empty());
    }

    // =========================================================================
    // build_inference_count_data tests
    // =========================================================================

    #[gtest]
    #[tokio::test]
    async fn test_build_inference_count_data_success() {
        let now = Utc::now();
        let mut mock_db = MockInferenceQueries::new();
        mock_db
            .expect_list_functions_with_inference_count()
            .returning(move || {
                Box::pin(async move {
                    Ok(vec![
                        FunctionInferenceCount {
                            function_name: "fn_a".to_string(),
                            inference_count: 500,
                            last_inference_timestamp: now,
                        },
                        FunctionInferenceCount {
                            function_name: "fn_b".to_string(),
                            inference_count: 42,
                            last_inference_timestamp: now,
                        },
                    ])
                })
            });

        let result = build_inference_count_data(&mock_db).await;
        expect_that!(result, len(eq(2)));
        expect_that!(result[0].function_name, eq("fn_a"));
        expect_that!(result[0].inference_count, eq(500));
        expect_that!(result[1].function_name, eq("fn_b"));
        expect_that!(result[1].inference_count, eq(42));
    }

    #[gtest]
    #[tokio::test]
    async fn test_build_inference_count_data_error() {
        let mut mock_db = MockInferenceQueries::new();
        mock_db
            .expect_list_functions_with_inference_count()
            .returning(|| {
                Box::pin(async {
                    Err(Error::new(ErrorDetails::ClickHouseConnection {
                        message: "connection refused".to_string(),
                    }))
                })
            });

        let result = build_inference_count_data(&mock_db).await;
        expect_that!(result, is_empty());
    }

    // =========================================================================
    // compute_deployment_context_inner (end-to-end) tests
    // =========================================================================

    #[gtest]
    #[tokio::test]
    async fn test_compute_deployment_context_inner_empty_config() {
        let config = Arc::new(Config::default());
        let mock_feedback = MockFeedbackQueries::new();
        // No feedback expectations needed — empty config short-circuits

        let mut mock_dataset = MockDatasetQueries::new();
        mock_dataset
            .expect_get_dataset_metadata()
            .returning(|_| Box::pin(async { Ok(vec![]) }));

        let mut mock_inference = MockInferenceQueries::new();
        mock_inference
            .expect_list_functions_with_inference_count()
            .returning(|| Box::pin(async { Ok(vec![]) }));

        let result = compute_deployment_context_inner(
            &config,
            &mock_feedback,
            &mock_dataset,
            &mock_inference,
        )
        .await
        .expect("should succeed with empty config");

        expect_that!(result.functions, is_empty());
        expect_that!(result.metrics, is_empty());
        expect_that!(result.evaluations, is_empty());
        expect_that!(result.feedback_by_function, is_empty());
        expect_that!(result.datasets, is_empty());
        expect_that!(result.inference_counts_by_function, is_empty());
    }

    #[gtest]
    #[tokio::test]
    async fn test_compute_deployment_context_inner_with_all_data() {
        let now = Utc::now();
        let mut config = Config::default();

        // Add a function with a variant
        let mut variants = HashMap::new();
        variants.insert(
            "variant_a".to_string(),
            Arc::new(VariantInfo {
                inner: VariantConfig::ChatCompletion(ChatCompletionConfig::default()),
                timeouts: Default::default(),
                namespace: None,
            }),
        );
        config.functions.insert(
            "my_fn".to_string(),
            Arc::new(FunctionConfig::Chat(FunctionConfigChat {
                variants,
                ..Default::default()
            })),
        );

        // Add a metric
        config.metrics.insert(
            "accuracy".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Boolean,
                optimize: MetricConfigOptimize::Max,
                level: MetricConfigLevel::Inference,
                description: None,
            },
        );

        let config = Arc::new(config);

        // Mock feedback
        let mut mock_feedback = MockFeedbackQueries::new();
        mock_feedback.expect_get_feedback_by_variant().returning(
            |_metric, _func, _variants, _ns, _max| {
                Box::pin(async {
                    Ok(vec![FeedbackByVariant {
                        variant_name: "variant_a".to_string(),
                        mean: 0.9,
                        variance: Some(0.05),
                        count: 200,
                    }])
                })
            },
        );

        // Mock datasets
        let mut mock_dataset = MockDatasetQueries::new();
        mock_dataset.expect_get_dataset_metadata().returning(|_| {
            Box::pin(async {
                Ok(vec![DatasetMetadata {
                    dataset_name: "train".to_string(),
                    count: 500,
                    last_updated: "2024-01-01T00:00:00Z".to_string(),
                }])
            })
        });

        // Mock inference counts
        let mut mock_inference = MockInferenceQueries::new();
        mock_inference
            .expect_list_functions_with_inference_count()
            .returning(move || {
                Box::pin(async move {
                    Ok(vec![FunctionInferenceCount {
                        function_name: "my_fn".to_string(),
                        inference_count: 1000,
                        last_inference_timestamp: now,
                    }])
                })
            });

        let result = compute_deployment_context_inner(
            &config,
            &mock_feedback,
            &mock_dataset,
            &mock_inference,
        )
        .await
        .expect("should succeed");

        // Config data
        expect_that!(result.functions, len(eq(1)));
        expect_that!(result.functions[0].name, eq("my_fn"));
        expect_that!(result.metrics, len(eq(1)));
        expect_that!(result.metrics[0].name, eq("accuracy"));

        // Feedback data
        expect_that!(result.feedback_by_function, len(eq(1)));
        expect_that!(result.feedback_by_function[0].function_name, eq("my_fn"));
        expect_that!(
            result.feedback_by_function[0].metrics[0].variants[0].count,
            eq(200)
        );

        // Dataset data
        expect_that!(result.datasets, len(eq(1)));
        expect_that!(result.datasets[0].name, eq("train"));
        expect_that!(result.datasets[0].count, eq(500));

        // Inference count data
        expect_that!(result.inference_counts_by_function, len(eq(1)));
        expect_that!(
            result.inference_counts_by_function[0].function_name,
            eq("my_fn")
        );
        expect_that!(
            result.inference_counts_by_function[0].inference_count,
            eq(1000)
        );
    }

    #[gtest]
    #[tokio::test]
    async fn test_compute_deployment_context_inner_partial_db_failures() {
        let mut config = Config::default();
        config.functions.insert(
            "fn_a".to_string(),
            Arc::new(FunctionConfig::Chat(FunctionConfigChat::default())),
        );
        config.metrics.insert(
            "m1".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Float,
                optimize: MetricConfigOptimize::Min,
                level: MetricConfigLevel::Episode,
                description: None,
            },
        );
        let config = Arc::new(config);

        // Feedback errors
        let mut mock_feedback = MockFeedbackQueries::new();
        mock_feedback.expect_get_feedback_by_variant().returning(
            |_metric, _func, _variants, _ns, _max| {
                Box::pin(async {
                    Err(Error::new(ErrorDetails::ClickHouseConnection {
                        message: "down".to_string(),
                    }))
                })
            },
        );

        // Datasets errors
        let mut mock_dataset = MockDatasetQueries::new();
        mock_dataset.expect_get_dataset_metadata().returning(|_| {
            Box::pin(async {
                Err(Error::new(ErrorDetails::ClickHouseConnection {
                    message: "down".to_string(),
                }))
            })
        });

        // Inference counts succeed
        let now = Utc::now();
        let mut mock_inference = MockInferenceQueries::new();
        mock_inference
            .expect_list_functions_with_inference_count()
            .returning(move || {
                Box::pin(async move {
                    Ok(vec![FunctionInferenceCount {
                        function_name: "fn_a".to_string(),
                        inference_count: 42,
                        last_inference_timestamp: now,
                    }])
                })
            });

        let result = compute_deployment_context_inner(
            &config,
            &mock_feedback,
            &mock_dataset,
            &mock_inference,
        )
        .await
        .expect("should succeed even with partial DB failures");

        // Config data still present
        expect_that!(result.functions, len(eq(1)));
        expect_that!(result.metrics, len(eq(1)));

        // Failed queries degrade gracefully to empty
        expect_that!(result.feedback_by_function, is_empty());
        expect_that!(result.datasets, is_empty());

        // Successful query still returns data
        expect_that!(result.inference_counts_by_function, len(eq(1)));
        expect_that!(
            result.inference_counts_by_function[0].inference_count,
            eq(42)
        );
    }
}
