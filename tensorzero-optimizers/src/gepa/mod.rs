//! GEPA optimizer implementation
//!
//! This module provides the trait implementations for the GEPA optimizer.
//! The actual GEPA algorithm will be implemented here.

use async_trait::async_trait;
use std::time::Duration;

use tensorzero_core::{
    client::{ClientBuilder, ClientBuilderMode},
    config::{snapshot::SnapshotHash, Config, UninitializedVariantConfig},
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
};

use crate::{JobHandle, Optimizer};

mod analyze;
mod evaluate;
mod validate;

pub use analyze::{analyze_inferences, Analysis};
pub use evaluate::create_evaluation_dataset;
pub use validate::FunctionContext;
use validate::{initialize_pareto_frontier, validate_examples, validate_gepa_config};

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
        snapshot_hash: SnapshotHash,
    ) -> Result<Self::Handle, Error> {
        // Validate configuration and examples, get the FunctionContext (function_config, static_tools, and evaluation_config)
        let function_context = validate_gepa_config(self, &config)?;

        // Require validation examples for GEPA Pareto filtering
        // TODO[#4772]: Random split from train_examples if None
        let val_examples = val_examples.ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: "`val_examples` are required for GEPA optimization (used for Pareto frontier filtering)".to_string(),
            })
        })?;

        // Validate both train and validation examples (this filters invalid examples)
        let train_examples = validate_examples(train_examples)?;
        let val_examples = validate_examples(val_examples)?;

        tracing::info!(
            "Starting GEPA optimization for function '{}' with {} train examples and {} val examples",
            self.function_name,
            train_examples.len(),
            val_examples.len()
        );

        // Build the gateway client once for the entire optimization run
        let _gateway_client = ClientBuilder::new(ClientBuilderMode::FromComponents {
            config: config.clone(),
            snapshot_hash,
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

        // Initialize the Pareto frontier with baseline or provided variants
        let pareto_frontier_variants = initialize_pareto_frontier(self, &function_context)?;

        // Track original variant names to filter them out at the end
        let original_variant_names: std::collections::HashSet<String> =
            pareto_frontier_variants.keys().cloned().collect();

        tracing::info!(
            "Initialized with {} baseline variants: {:?}",
            original_variant_names.len(),
            original_variant_names
        );

        tracing::warn!("GEPA algorithm is not yet implemented - returning placeholder result");

        // TODO: Implement actual GEPA algorithm here
        // For now, return initial variants

        // For synchronous optimizers, we store the result in the handle
        // rather than returning an error from launch()
        Ok(GEPAJobHandle {
            result: Ok(pareto_frontier_variants),
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
