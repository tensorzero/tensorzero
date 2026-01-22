// Required to compile large async/instrumented futures pulled in from tensorzero-core (e.g., AWS Bedrock client types)
#![recursion_limit = "256"]
//! TensorZero Optimizer Implementations
//!
//! This crate provides optimizer trait definitions, implementations, and HTTP endpoints.
//! Config types live in `tensorzero-core`, while behavior and handlers live here.
//!
//! This crate was extracted from `tensorzero-core` to avoid circular dependencies when
//! optimizers need to depend on the `evaluations` crate.

use std::future::Future;
use std::sync::Arc;
use tensorzero_core::{
    config::{Config, provider_types::ProviderTypesConfig},
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceCredentials,
    error::Error,
    http::TensorzeroHttpClient,
    model_table::ProviderTypeDefaultCredentials,
    optimization::{OptimizationJobHandle, OptimizationJobInfo, OptimizerConfig, OptimizerInfo},
    stored_inference::RenderedSample,
};

pub mod dicl;
pub mod endpoints;
pub mod fireworks_sft;
pub mod gcp_vertex_gemini_sft;
pub mod gepa;
pub mod openai;
pub mod openai_rft;
pub mod openai_sft;
pub mod together_sft;

// Re-export core types for convenience
pub use tensorzero_core::optimization::OptimizerOutput;

impl JobHandle for OptimizationJobHandle {
    async fn poll(
        &self,
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        default_credentials: &ProviderTypeDefaultCredentials,
        provider_types: &ProviderTypesConfig,
    ) -> Result<OptimizationJobInfo, Error> {
        match self {
            OptimizationJobHandle::Dicl(job_handle) => {
                job_handle
                    .poll(client, credentials, default_credentials, provider_types)
                    .await
            }
            OptimizationJobHandle::OpenAISFT(job_handle) => {
                job_handle
                    .poll(client, credentials, default_credentials, provider_types)
                    .await
            }
            OptimizationJobHandle::OpenAIRFT(job_handle) => {
                job_handle
                    .poll(client, credentials, default_credentials, provider_types)
                    .await
            }
            OptimizationJobHandle::FireworksSFT(job_handle) => {
                job_handle
                    .poll(client, credentials, default_credentials, provider_types)
                    .await
            }
            OptimizationJobHandle::GCPVertexGeminiSFT(job_handle) => {
                job_handle
                    .poll(client, credentials, default_credentials, provider_types)
                    .await
            }
            OptimizationJobHandle::TogetherSFT(job_handle) => {
                job_handle
                    .poll(client, credentials, default_credentials, provider_types)
                    .await
            }
            OptimizationJobHandle::GEPA(job_handle) => {
                job_handle
                    .poll(client, credentials, default_credentials, provider_types)
                    .await
            }
        }
    }
}

pub trait JobHandle {
    fn poll(
        &self,
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        default_credentials: &ProviderTypeDefaultCredentials,
        provider_types: &ProviderTypesConfig,
    ) -> impl Future<Output = Result<OptimizationJobInfo, Error>> + Send;
}

pub trait Optimizer {
    type Handle: JobHandle;

    fn launch(
        &self,
        client: &TensorzeroHttpClient,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
        clickhouse_connection_info: &ClickHouseConnectionInfo,
        config: Arc<Config>,
    ) -> impl Future<Output = Result<Self::Handle, Error>> + Send;
}

impl Optimizer for OptimizerInfo {
    type Handle = OptimizationJobHandle;

    async fn launch(
        &self,
        client: &TensorzeroHttpClient,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
        clickhouse_connection_info: &ClickHouseConnectionInfo,
        config: Arc<Config>,
    ) -> Result<Self::Handle, Error> {
        match &self.inner {
            OptimizerConfig::Dicl(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    clickhouse_connection_info,
                    config.clone(),
                )
                .await
                .map(OptimizationJobHandle::Dicl),
            OptimizerConfig::OpenAISFT(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    clickhouse_connection_info,
                    config.clone(),
                )
                .await
                .map(OptimizationJobHandle::OpenAISFT),
            OptimizerConfig::OpenAIRFT(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    clickhouse_connection_info,
                    config.clone(),
                )
                .await
                .map(OptimizationJobHandle::OpenAIRFT),
            OptimizerConfig::FireworksSFT(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    clickhouse_connection_info,
                    config.clone(),
                )
                .await
                .map(OptimizationJobHandle::FireworksSFT),
            OptimizerConfig::GCPVertexGeminiSFT(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    clickhouse_connection_info,
                    config.clone(),
                )
                .await
                .map(OptimizationJobHandle::GCPVertexGeminiSFT),
            OptimizerConfig::TogetherSFT(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    clickhouse_connection_info,
                    config.clone(),
                )
                .await
                .map(OptimizationJobHandle::TogetherSFT),
            OptimizerConfig::GEPA(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    clickhouse_connection_info,
                    config.clone(),
                )
                .await
                .map(OptimizationJobHandle::GEPA),
        }
    }
}
