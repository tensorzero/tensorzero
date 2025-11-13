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
        credentials: &InferenceCredentials,
        clickhouse_connection_info: &ClickHouseConnectionInfo,
        config: std::sync::Arc<Config>,
    ) -> Result<Self::Handle, Error> {
        // Run the GEPA optimization algorithm
        let variant_configs = lib::run_gepa_optimization(
            self,
            client,
            train_examples,
            val_examples,
            credentials,
            clickhouse_connection_info,
            config,
        )
        .await?;

        // Return a job handle containing the Pareto frontier of variants
        Ok(GEPAJobHandle { variant_configs })
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
        // GEPA optimization is synchronous, so it's always complete once launched
        // Return the Pareto frontier of variant configurations
        Ok(OptimizationJobInfo::Completed {
            output: OptimizerOutput::Variants(
                self.variant_configs
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
}
