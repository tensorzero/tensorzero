use std::{collections::HashMap, sync::Arc};

use axum::Extension;
use serde::Deserialize;
use tokio_util::task::TaskTracker;
use tracing::instrument;

use crate::{
    cache::CacheParamsOptions,
    config::Config,
    db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo},
    embeddings::{Embedding, EmbeddingEncodingFormat, EmbeddingInput, EmbeddingRequest},
    endpoints::{inference::InferenceClients, RequestApiKeyExtension},
    error::{Error, ErrorDetails},
    http::TensorzeroHttpClient,
    inference::types::Usage,
    rate_limiting::ScopeInfo,
};

#[cfg(test)]
use crate::http::DEFAULT_HTTP_CLIENT_TIMEOUT;

use super::inference::InferenceCredentials;

#[derive(Debug, Clone, Deserialize)]
pub struct Params {
    pub input: EmbeddingInput,
    pub model_name: String,
    pub dimensions: Option<u32>,
    pub encoding_format: EmbeddingEncodingFormat,
    // if true, the embedding will not be stored
    pub dryrun: Option<bool>,
    #[serde(default)]
    pub credentials: InferenceCredentials,
    #[serde(default)]
    pub cache_options: CacheParamsOptions,
}

#[instrument(name = "embeddings", skip_all, fields(model, num_inputs))]
pub async fn embeddings(
    config: Arc<Config>,
    http_client: &TensorzeroHttpClient,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    postgres_connection_info: PostgresConnectionInfo,
    deferred_tasks: TaskTracker,
    params: Params,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
) -> Result<EmbeddingResponse, Error> {
    let span = tracing::Span::current();
    span.record("model", &params.model_name);
    span.record("num_inputs", params.input.num_inputs());
    let embedding_model = config
        .embedding_models
        .get(&params.model_name)
        .await?
        .ok_or_else(|| {
            Error::new(ErrorDetails::ModelNotFound {
                model_name: params.model_name.clone(),
            })
        })?;
    if let EmbeddingInput::Batch(array) = &params.input {
        if array.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Input cannot be empty".to_string(),
            }));
        }
    }

    let request = EmbeddingRequest {
        input: params.input,
        dimensions: params.dimensions,
        encoding_format: params.encoding_format,
    };

    // NOTE: we do not support tags for embeddings yet
    // we should fix this once the tags are implemented
    let tags = Arc::new(HashMap::default());
    let dryrun = params.dryrun.unwrap_or(false);
    let clients = InferenceClients {
        http_client: http_client.clone(),
        credentials: Arc::new(params.credentials.clone()),
        cache_options: (params.cache_options, dryrun).into(),
        clickhouse_connection_info: clickhouse_connection_info.clone(),
        postgres_connection_info: postgres_connection_info.clone(),
        tags: tags.clone(),
        rate_limiting_config: Arc::new(config.rate_limiting.clone()),
        otlp_config: config.gateway.export.otlp.clone(),
        deferred_tasks,
        scope_info: ScopeInfo::new(tags.clone(), api_key_ext),
    };
    let response = embedding_model
        .embed(&request, &params.model_name, &clients)
        .await?;
    let usage = response.usage_considering_cached();
    Ok(EmbeddingResponse {
        embeddings: response.embeddings,
        usage,
        model: params.model_name,
    })
}

pub struct EmbeddingResponse {
    pub embeddings: Vec<Embedding>,
    pub usage: Usage,
    pub model: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::provider_types::ProviderTypesConfig;
    use crate::config::Config;
    use crate::embeddings::{EmbeddingModelConfig, EmbeddingProviderConfig, EmbeddingProviderInfo};
    use crate::model_table::ProviderTypeDefaultCredentials;
    use crate::providers::dummy::DummyProvider;
    use std::collections::HashMap;
    #[tokio::test]
    async fn test_no_warning_when_model_exists() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Create a config with a valid embedding model
        let dummy_provider = EmbeddingProviderConfig::Dummy(DummyProvider {
            model_name: "good-model".into(),
            ..Default::default()
        });
        let provider_info = EmbeddingProviderInfo {
            inner: dummy_provider,
            timeout_ms: None,
            provider_name: Arc::from("dummy"),
            extra_body: None,
        };
        let embedding_model = EmbeddingModelConfig {
            routing: vec!["dummy".to_string().into()],
            providers: HashMap::from([("dummy".to_string().into(), provider_info)]),
            timeout_ms: None,
        };

        // Create a minimal config with just the embedding model
        let mut embedding_models = HashMap::new();
        embedding_models.insert("test-model".to_string().into(), embedding_model);

        let provider_types = ProviderTypesConfig::default();
        let config = Config {
            embedding_models: Arc::new(
                crate::embeddings::EmbeddingModelTable::new(
                    embedding_models,
                    Arc::new(ProviderTypeDefaultCredentials::new(&provider_types)),
                    DEFAULT_HTTP_CLIENT_TIMEOUT,
                )
                .unwrap(),
            ),
            ..Default::default()
        };

        let config = Arc::new(config);

        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let params = Params {
            input: EmbeddingInput::Single("test input".to_string()),
            model_name: "test-model".to_string(),
            dimensions: None,
            encoding_format: EmbeddingEncodingFormat::Float,
            dryrun: None,
            credentials: InferenceCredentials::default(),
            cache_options: CacheParamsOptions::default(),
        };

        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();

        let result = embeddings(
            config,
            &http_client,
            clickhouse_connection_info,
            PostgresConnectionInfo::Disabled,
            tokio_util::task::TaskTracker::new(),
            params,
            None,
        )
        .await;

        // The function should succeed
        assert!(result.is_ok());

        // Check that no warnings were logged for model not found
        assert!(!logs_contain("Model not found"));
    }
    #[tokio::test]
    async fn test_warning_when_model_not_found() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Create an empty config with no embedding models
        let config = Arc::new(Config::default());

        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let params = Params {
            input: EmbeddingInput::Single("test input".to_string()),
            model_name: "nonexistent-model".to_string(),
            dimensions: None,
            encoding_format: EmbeddingEncodingFormat::Float,
            dryrun: None,
            credentials: InferenceCredentials::default(),
            cache_options: CacheParamsOptions::default(),
        };

        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();

        let result = embeddings(
            config,
            &http_client,
            clickhouse_connection_info,
            PostgresConnectionInfo::Disabled,
            tokio_util::task::TaskTracker::new(),
            params,
            None,
        )
        .await;

        // The function should fail with ModelNotFound
        assert!(result.is_err());

        // Check that a warning was logged for model not found
        assert!(logs_contain("Model not found: nonexistent-model"));
    }
}
