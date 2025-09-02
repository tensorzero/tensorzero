use std::sync::Arc;

use serde::Deserialize;
use tracing::instrument;

use crate::{
    cache::CacheOptions,
    config::Config,
    db::clickhouse::ClickHouseConnectionInfo,
    embeddings::{Embedding, EmbeddingEncodingFormat, EmbeddingInput, EmbeddingRequest},
    endpoints::inference::InferenceClients,
    error::{Error, ErrorDetails},
    inference::types::Usage,
};

use super::inference::InferenceCredentials;

#[derive(Debug, Clone, Deserialize)]
pub struct Params {
    pub input: EmbeddingInput,
    pub model_name: String,
    pub dimensions: Option<u32>,
    pub encoding_format: EmbeddingEncodingFormat,
    #[serde(default)]
    pub credentials: InferenceCredentials,
}

#[instrument(
    name = "embeddings",
    skip(config, http_client, params),
    fields(model, num_inputs)
)]
pub async fn embeddings(
    config: Arc<Config>,
    http_client: &reqwest::Client,
    params: Params,
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
    // Caching and clickhouse writes are disabled in the embeddings endpoint for now
    let cache_options = CacheOptions::default();
    let clickhouse = ClickHouseConnectionInfo::Disabled;
    let clients = InferenceClients {
        http_client,
        credentials: &params.credentials,
        cache_options: &cache_options,
        clickhouse_connection_info: &clickhouse,
    };
    let response = embedding_model
        .embed(&request, &params.model_name, &clients)
        .await?;
    Ok(EmbeddingResponse {
        embeddings: response.embeddings,
        usage: response.usage,
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
    use crate::config::Config;
    use crate::config::TimeoutsConfig;
    use crate::embeddings::{EmbeddingModelConfig, EmbeddingProviderConfig, EmbeddingProviderInfo};
    use crate::providers::dummy::DummyProvider;
    use std::collections::HashMap;
    use tracing_test::traced_test;

    #[traced_test]
    #[tokio::test]
    async fn test_no_warning_when_model_exists() {
        // Create a config with a valid embedding model
        let dummy_provider = EmbeddingProviderConfig::Dummy(DummyProvider {
            model_name: "good-model".into(),
            ..Default::default()
        });
        let provider_info = EmbeddingProviderInfo {
            inner: dummy_provider,
            timeouts: TimeoutsConfig::default(),
            provider_name: Arc::from("dummy"),
            extra_body: None,
        };
        let embedding_model = EmbeddingModelConfig {
            routing: vec!["dummy".to_string().into()],
            providers: HashMap::from([("dummy".to_string().into(), provider_info)]),
            timeouts: TimeoutsConfig::default(),
        };

        // Create a minimal config with just the embedding model
        let mut embedding_models = HashMap::new();
        embedding_models.insert("test-model".to_string().into(), embedding_model);

        let config = Config {
            embedding_models: embedding_models.try_into().unwrap(),
            ..Default::default()
        };

        let config = Arc::new(config);

        let http_client = reqwest::Client::new();
        let params = Params {
            input: EmbeddingInput::Single("test input".to_string()),
            model_name: "test-model".to_string(),
            dimensions: None,
            encoding_format: EmbeddingEncodingFormat::Float,
            credentials: InferenceCredentials::default(),
        };

        let result = embeddings(config, &http_client, params).await;

        // The function should succeed
        assert!(result.is_ok());

        // Check that no warnings were logged for model not found
        assert!(!logs_contain("Model not found"));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_warning_when_model_not_found() {
        // Create an empty config with no embedding models
        let config = Arc::new(Config::default());

        let http_client = reqwest::Client::new();
        let params = Params {
            input: EmbeddingInput::Single("test input".to_string()),
            model_name: "nonexistent-model".to_string(),
            dimensions: None,
            encoding_format: EmbeddingEncodingFormat::Float,
            credentials: InferenceCredentials::default(),
        };

        let result = embeddings(config, &http_client, params).await;

        // The function should fail with ModelNotFound
        assert!(result.is_err());

        // Check that a warning was logged for model not found
        assert!(logs_contain("Model not found: nonexistent-model"));
    }
}
