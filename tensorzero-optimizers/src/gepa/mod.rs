//! GEPA optimizer implementation
//!
//! This module provides the trait implementations for the GEPA optimizer.
//! The actual GEPA algorithm will be implemented here.

use async_trait::async_trait;
use std::{collections::HashMap, time::Duration};

use tensorzero_core::{
    client::{ClientBuilder, ClientBuilderMode},
    config::{Config, UninitializedVariantConfig},
    db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo},
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    http::TensorzeroHttpClient,
    model_table::ProviderTypeDefaultCredentials,
    optimization::{
        gepa::{GEPAConfig, GEPAJobHandle},
        OptimizationJobInfo, OptimizerOutput,
    },
    stored_inference::RenderedSample,
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use crate::{JobHandle, Optimizer};

mod pareto;
// TDOD: do not re-export
pub use pareto::{is_improvement, update_pareto_frontier};

#[async_trait]
impl Optimizer for GEPAConfig {
    type Handle = GEPAJobHandle;

    async fn launch(
        &self,
        client: &TensorzeroHttpClient,
        _train_examples: Vec<RenderedSample>,
        _val_examples: Option<Vec<RenderedSample>>,
        _credentials: &InferenceCredentials,
        clickhouse_connection_info: &ClickHouseConnectionInfo,
        config: std::sync::Arc<Config>,
    ) -> Result<Self::Handle, Error> {
        // Build the gateway client once for the entire optimization run
        let _gateway_client = ClientBuilder::new(ClientBuilderMode::FromComponents {
            config: config.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            http_client: client.clone(),
            timeout: Some(Duration::from_secs(self.timeout)),
        })
        .build()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to build gateway client for GEPA optimization: {e}"),
            })
        })?;

        tracing::info!("Gateway client built successfully for GEPA optimization");

        tracing::warn!("GEPA algorithm is not yet implemented - returning placeholder result");

        // TODO: Implement actual GEPA algorithm here
        // For now, return a placeholder result
        let mut result = HashMap::new();
        result.insert(
            "dummy".to_string(),
            UninitializedChatCompletionConfig::default(),
        );

        // For synchronous optimizers, we store the result in the handle
        // rather than returning an error from launch()
        Ok(GEPAJobHandle { result: Ok(result) })
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
