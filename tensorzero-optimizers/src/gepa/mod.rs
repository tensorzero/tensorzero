//! GEPA optimizer implementation
//!
//! This module provides the trait implementations for the GEPA optimizer.
//! The actual GEPA algorithm logic is in the `lib` submodule.

mod analyze;
mod evaluate;
mod lib;
mod mutate;
mod pareto;
mod sample;
mod utils;
mod validate;

// Re-export public functions and types for testing
pub use analyze::{analyze_inferences, InferenceWithAnalysis};
pub use mutate::mutate_templates;

use async_trait::async_trait;

use tensorzero_core::{
    config::{Config, UninitializedVariantConfig},
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceCredentials,
    error::Error,
    http::TensorzeroHttpClient,
    model_table::ProviderTypeDefaultCredentials,
    optimization::{
        gepa::{GEPAConfig, GEPAJobHandle},
        OptimizationJobInfo, OptimizerOutput,
    },
    stored_inference::RenderedSample,
};

use crate::{JobHandle, Optimizer};

#[async_trait]
impl Optimizer for GEPAConfig {
    type Handle = GEPAJobHandle;

    async fn launch(
        &self,
        client: &TensorzeroHttpClient,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        _credentials: &InferenceCredentials,
        clickhouse_connection_info: &ClickHouseConnectionInfo,
        config: std::sync::Arc<Config>,
    ) -> Result<Self::Handle, Error> {
        // Run the GEPA optimization algorithm
        let result = lib::run_gepa_optimization(
            self,
            client,
            train_examples,
            val_examples,
            clickhouse_connection_info,
            config,
        )
        .await;

        // Convert the result to store either success or failure
        // For synchronous optimizers, we store the result in the handle
        // rather than returning an error from launch()
        let handle_result = result.map_err(|e| e.to_string());

        // Return a job handle containing the result (success or failure)
        Ok(GEPAJobHandle {
            result: handle_result,
        })
    }
}

#[async_trait]
impl JobHandle for GEPAJobHandle {
    async fn poll(
        &self,
        _client: &TensorzeroHttpClient,
        _credentials: &InferenceCredentials,
        _default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OptimizationJobInfo, Error> {
        // GEPA optimization is synchronous, so the result is available immediately
        // Check if optimization succeeded or failed
        match &self.result {
            Ok(variant_configs) => {
                // Return the Pareto frontier of variant configurations
                Ok(OptimizationJobInfo::Completed {
                    output: OptimizerOutput::Variants(
                        variant_configs
                            .iter()
                            .map(|(k, v)| {
                                (
                                    k.clone(),
                                    Box::new(UninitializedVariantConfig::ChatCompletion(v.clone())),
                                )
                            })
                            .collect(),
                    ),
                })
            }
            Err(error_message) => {
                // Return failure status with the error message
                Ok(OptimizationJobInfo::Failed {
                    message: error_message.clone(),
                    error: None,
                })
            }
        }
    }
}
