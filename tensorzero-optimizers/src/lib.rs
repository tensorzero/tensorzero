// Required to compile large async/instrumented futures pulled in from tensorzero-core (e.g., AWS Bedrock client types)
#![recursion_limit = "256"]
//! TensorZero Optimizer Implementations
//!
//! This crate provides optimizer trait definitions, implementations, and HTTP endpoints.
//! Config types live in `tensorzero-core`, while behavior and handlers live here.
//!
//! This crate was extracted from `tensorzero-core` to avoid circular dependencies when
//! optimizers need to depend on the `evaluations` crate.

use durable_tools_spawn::SpawnClient;
use std::future::Future;
use std::sync::Arc;
use tensorzero_core::{
    config::{Config, provider_types::ProviderTypesConfig},
    db::delegating_connection::DelegatingDatabaseQueries,
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
pub mod postgres;
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
        spawn_client: Option<&SpawnClient>,
    ) -> Result<OptimizationJobInfo, Error> {
        match self {
            OptimizationJobHandle::Dicl(job_handle) => {
                job_handle
                    .poll(
                        client,
                        credentials,
                        default_credentials,
                        provider_types,
                        spawn_client,
                    )
                    .await
            }
            OptimizationJobHandle::OpenAISFT(job_handle) => {
                job_handle
                    .poll(
                        client,
                        credentials,
                        default_credentials,
                        provider_types,
                        spawn_client,
                    )
                    .await
            }
            OptimizationJobHandle::OpenAIRFT(job_handle) => {
                job_handle
                    .poll(
                        client,
                        credentials,
                        default_credentials,
                        provider_types,
                        spawn_client,
                    )
                    .await
            }
            OptimizationJobHandle::FireworksSFT(job_handle) => {
                job_handle
                    .poll(
                        client,
                        credentials,
                        default_credentials,
                        provider_types,
                        spawn_client,
                    )
                    .await
            }
            OptimizationJobHandle::GCPVertexGeminiSFT(job_handle) => {
                job_handle
                    .poll(
                        client,
                        credentials,
                        default_credentials,
                        provider_types,
                        spawn_client,
                    )
                    .await
            }
            OptimizationJobHandle::TogetherSFT(job_handle) => {
                job_handle
                    .poll(
                        client,
                        credentials,
                        default_credentials,
                        provider_types,
                        spawn_client,
                    )
                    .await
            }
            OptimizationJobHandle::GEPA(job_handle) => {
                job_handle
                    .poll(
                        client,
                        credentials,
                        default_credentials,
                        provider_types,
                        spawn_client,
                    )
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
        spawn_client: Option<&SpawnClient>,
    ) -> impl Future<Output = Result<OptimizationJobInfo, Error>> + Send;
}

pub trait Optimizer {
    type Handle: JobHandle;

    #[expect(clippy::too_many_arguments)]
    fn launch(
        &self,
        client: &TensorzeroHttpClient,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
        db: &Arc<dyn DelegatingDatabaseQueries + Send + Sync>,
        config: Arc<Config>,
        spawn_client: Option<&SpawnClient>,
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
        db: &Arc<dyn DelegatingDatabaseQueries + Send + Sync>,
        config: Arc<Config>,
        spawn_client: Option<&SpawnClient>,
    ) -> Result<Self::Handle, Error> {
        match &self.inner {
            OptimizerConfig::Dicl(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    db,
                    config.clone(),
                    spawn_client,
                )
                .await
                .map(OptimizationJobHandle::Dicl),
            OptimizerConfig::OpenAISFT(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    db,
                    config.clone(),
                    spawn_client,
                )
                .await
                .map(OptimizationJobHandle::OpenAISFT),
            OptimizerConfig::OpenAIRFT(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    db,
                    config.clone(),
                    spawn_client,
                )
                .await
                .map(OptimizationJobHandle::OpenAIRFT),
            OptimizerConfig::FireworksSFT(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    db,
                    config.clone(),
                    spawn_client,
                )
                .await
                .map(OptimizationJobHandle::FireworksSFT),
            OptimizerConfig::GCPVertexGeminiSFT(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    db,
                    config.clone(),
                    spawn_client,
                )
                .await
                .map(OptimizationJobHandle::GCPVertexGeminiSFT),
            OptimizerConfig::TogetherSFT(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    db,
                    config.clone(),
                    spawn_client,
                )
                .await
                .map(OptimizationJobHandle::TogetherSFT),
            OptimizerConfig::GEPA(optimizer_config) => optimizer_config
                .launch(
                    client,
                    train_examples,
                    val_examples,
                    credentials,
                    db,
                    config.clone(),
                    spawn_client,
                )
                .await
                .map(OptimizationJobHandle::GEPA),
        }
    }
}
