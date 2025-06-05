use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use tracing::instrument;

use crate::error::{Error, ErrorDetails};
use crate::model::{ModelTable, UninitializedModelConfig};
use crate::config_parser::ProviderTypesConfig;
use crate::gateway_util::AppStateData;

pub struct RedisClient {
    redis_url: String,
    app_state: AppStateData
}

impl RedisClient {
    pub fn new(url: &str, app_state: AppStateData) -> Self {
        Self { redis_url: url.to_string(), app_state }
    }

    async fn parse_models(json: &str, provider_types: &ProviderTypesConfig) -> Result<ModelTable, Error> {
        let raw: HashMap<String, UninitializedModelConfig> =
            serde_json::from_str(json).map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to parse models from redis: {e}"),
                })
            })?;
        
        let models = raw
            .into_iter()
            .map(|(name, config)| {
                config
                    .load(&name, provider_types)
                    .map(|c| (Arc::<str>::from(name), c))
            })
            .collect::<Result<HashMap<_, _>, Error>>()?;
        
        models.try_into().map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to load models: {e}"),
            })
        })
    }

    #[instrument(skip(self))]
    pub async fn start(self) -> Result<(), Error> {
        let client = redis::Client::open(self.redis_url).map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to connect to Redis: {e}"),
            })
        })?;

        // Get a connection for pubsub
        let mut pubsub_conn = client
            .get_async_pubsub()
            .await
            .map_err(|e| Error::new(ErrorDetails::Config { message: format!("Failed to connect to redis: {e}") }))?;
        // let mut pubsub = pubsub_conn.into_pubsub();

        pubsub_conn
            .subscribe("model_table_updates")
            .await
            .map_err(|e| Error::new(ErrorDetails::Config { message: format!("Failed to subscribe to redis: {e}") }))?;

        let app_state = self.app_state.clone();

        tokio::spawn(async move {
            let mut stream = pubsub_conn.on_message();
            while let Some(msg) = stream.next().await {
                let payload: String = match msg.get_payload() {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::error!("Failed to decode redis message: {e}");
                        continue;
                    }
                };

                match Self::parse_models(&payload, &app_state.config.provider_types).await {
                    Ok(models) => app_state.update_model_table(models).await,
                    Err(e) => tracing::error!("Failed to parse models from redis: {e}")
                }
            }
        });

        Ok(())
    }

}