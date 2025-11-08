//! GEPA optimizer implementation (stub)
//!
//! This is a placeholder module that will be replaced with the actual GEPA implementation
//! in later commits. For now, it just provides stub trait implementations to allow compilation.

use async_trait::async_trait;
use std::sync::Arc;

use tensorzero_core::{
    config::Config,
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    http::TensorzeroHttpClient,
    model_table::ProviderTypeDefaultCredentials,
    optimization::{
        gepa::{GEPAConfig, GEPAJobHandle},
        OptimizationJobInfo,
    },
    stored_inference::RenderedSample,
};

use crate::{JobHandle, Optimizer};

#[async_trait]
impl Optimizer for GEPAConfig {
    type Handle = GEPAJobHandle;

    async fn launch(
        &self,
        _client: &TensorzeroHttpClient,
        _train_examples: Vec<RenderedSample>,
        _val_examples: Option<Vec<RenderedSample>>,
        _credentials: &InferenceCredentials,
        _clickhouse_connection_info: &ClickHouseConnectionInfo,
        _config: Arc<Config>,
    ) -> Result<Self::Handle, Error> {
        Err(Error::new(ErrorDetails::InternalError {
            message: "GEPA optimizer is not yet implemented (stub)".to_string(),
        }))
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
        Err(Error::new(ErrorDetails::InternalError {
            message: "GEPA job handle polling is not yet implemented (stub)".to_string(),
        }))
    }
}
