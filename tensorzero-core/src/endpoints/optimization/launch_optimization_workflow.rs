use std::{collections::HashMap, sync::Arc};

use rand::seq::SliceRandom;

use crate::{
    config::Config,
    db::clickhouse::ClickHouseConnectionInfo,
    db::datasets::{DatasetQueries, GetDatapointsParams},
    endpoints::{inference::InferenceCredentials, stored_inference::render_samples},
    error::{Error, ErrorDetails},
    http::TensorzeroHttpClient,
    optimization::{OptimizationJobHandle, Optimizer},
};

use super::types::{
    LaunchOptimizationWorkflowParams as LaunchOptimizationWorkflowInternalParams, OptimizationData,
};

/// FOR NOW: INTERNAL ONLY / UI (NAPI EXPORT)
///
/// Launches an optimization workflow from either inferences or datasets.
///
/// This function is the main entry point for starting fine-tuning jobs.
/// It supports two data sources:
/// 1. `experimental_list_inferences`: Queries inferences from the inference table
/// 2. `list_datapoints`: Queries datapoints from a dataset
///
/// Steps:
/// 1. Query samples based on data source type
/// 2. Render samples with the specified variant
/// 3. Filter out samples with no output
/// 4. Split into train/val sets
/// 5. Launch the optimizer
pub async fn launch_optimization_workflow(
    http_client: &TensorzeroHttpClient,
    config: Arc<Config>,
    clickhouse: &ClickHouseConnectionInfo,
    params: LaunchOptimizationWorkflowInternalParams,
) -> Result<OptimizationJobHandle, Error> {
    let LaunchOptimizationWorkflowInternalParams {
        render_variant_name,
        data: data_source,
        val_fraction,
        format,
        optimizer_config,
    } = params;

    // Step 1: Query samples and render based on data source type
    // We need to handle each type separately because render_samples requires concrete types
    let rendered_samples = match data_source {
        OptimizationData::ExperimentalListInferences(mut source) => {
            let function_name = source.function_name.clone();

            // Set the format from the outer params
            source.format = format;

            let inferences = clickhouse.list_inferences(&config, &source).await?;

            // Step 2: Render samples with the template variant
            let variants = HashMap::from([(function_name, render_variant_name.clone())]);
            render_samples(config.clone(), inferences, variants).await?
        }
        OptimizationData::ListDatapoints(source) => {
            let function_name = source.request.function_name.clone().ok_or_else(|| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: "function_name is required for optimization workflows".to_string(),
                })
            })?;

            let datapoints = clickhouse
                .get_datapoints(&GetDatapointsParams {
                    dataset_name: Some(source.dataset_name),
                    function_name: Some(function_name.clone()),
                    ids: None,
                    page_size: source.request.page_size.unwrap_or(u32::MAX),
                    offset: source.request.offset.unwrap_or(0),
                    allow_stale: false,
                    filter: source.request.filter,
                })
                .await?;

            // Step 2: Render samples with the template variant
            let variants = HashMap::from([(function_name, render_variant_name.clone())]);
            render_samples(config.clone(), datapoints, variants).await?
        }
    };

    // Step 3: Filter out examples with no output
    let rendered_samples: Vec<_> = rendered_samples
        .into_iter()
        .filter(|example| example.output.is_some())
        .collect();

    // Step 4: Split into train/val sets
    let (train_examples, val_examples) = split_examples(rendered_samples, val_fraction)?;

    // Step 5: Launch the optimizer
    let default_credentials = &config.models.default_credentials;
    optimizer_config
        .load(default_credentials)
        .await?
        .launch(
            http_client,
            train_examples,
            val_examples,
            &InferenceCredentials::default(),
            clickhouse,
            &config,
        )
        .await
}

/// Randomly split examples into train and val sets.
/// Returns a tuple of (train_examples, val_examples).
/// val_examples is None if val_fraction is None.
fn split_examples<T>(
    stored_inferences: Vec<T>,
    val_fraction: Option<f64>,
) -> Result<(Vec<T>, Option<Vec<T>>), Error> {
    if let Some(val_fraction) = val_fraction {
        if val_fraction <= 0.0 || val_fraction >= 1.0 {
            // If val_fraction is not in (0, 1), treat as no split
            return Err(Error::new(ErrorDetails::InvalidValFraction {
                val_fraction,
            }));
        }
        let mut rng = rand::rng();
        let mut examples = stored_inferences;
        let n = examples.len();
        let n_val = ((n as f64) * val_fraction).round() as usize;
        // Shuffle the examples
        examples.as_mut_slice().shuffle(&mut rng);

        // Split examples into val and train sets
        let mut val = Vec::with_capacity(n_val);
        let mut train = Vec::with_capacity(n - n_val);

        // Move elements from examples into val and train
        for (i, example) in examples.into_iter().enumerate() {
            if i < n_val {
                val.push(example);
            } else {
                train.push(example);
            }
        }

        Ok((train, Some(val)))
    } else {
        Ok((stored_inferences, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_deserialize_experimental_list_inferences() {
        let json = json!({
            "render_variant_name": "my_variant",
            "data": {
                "type": "experimental_list_inferences",
                "function_name": "my_function",
                "output_source": "Inference",
                "limit": 100
            },
            "format": "JsonEachRow",
            "optimizer_config": {
                "type": "openai_sft",
                "model": "gpt-4o-2024-08-06"
            }
        });

        let params: LaunchOptimizationWorkflowInternalParams =
            serde_json::from_value(json).unwrap();
        match params.data {
            OptimizationData::ExperimentalListInferences(source) => {
                assert_eq!(source.function_name, "my_function");
                assert_eq!(source.limit, Some(100));
            }
            OptimizationData::ListDatapoints(_) => {
                panic!("Expected ExperimentalListInferences")
            }
        }
    }

    #[test]
    fn test_deserialize_list_datapoints() {
        let json = json!({
            "render_variant_name": "my_variant",
            "data": {
                "type": "list_datapoints",
                "dataset_name": "my_dataset",
                "function_name": "my_function",
                "page_size": 50
            },
            "format": "JsonEachRow",
            "optimizer_config": {
                "type": "fireworks_sft",
                "model": "llama-3.2-3b-instruct",
                "account_id": "test_account"
            }
        });

        let params: LaunchOptimizationWorkflowInternalParams =
            serde_json::from_value(json).unwrap();
        match params.data {
            OptimizationData::ListDatapoints(source) => {
                assert_eq!(source.dataset_name, "my_dataset");
                assert_eq!(
                    source.request.function_name,
                    Some("my_function".to_string())
                );
                assert_eq!(source.request.page_size, Some(50));
            }
            OptimizationData::ExperimentalListInferences(_) => {
                panic!("Expected ListDatapoints")
            }
        }
    }
}
