use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use redis::AsyncCommands;
use tracing::instrument;

use crate::error::{Error, ErrorDetails};
use crate::model::{ModelTable, UninitializedModelConfig};
use crate::config_parser::ProviderTypesConfig;
use crate::gateway_util::AppStateData;
use crate::auth::{Auth, APIConfig};

pub struct RedisClient {
    redis_url: String,
    app_state: AppStateData,
    auth: Option<Auth>,
}

impl RedisClient {
    pub fn new(url: &str, app_state: AppStateData) -> Self {
        Self { 
            redis_url: url.to_string(), 
            app_state,
            auth: None,
        }
    }

    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = Some(auth);
        self
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

    async fn parse_api_keys(json: &str) -> Result<HashMap<String, APIConfig>, Error> {
        serde_json::from_str(json).map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to parse API keys from redis: {e}"),
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

        // Initial fetch: fetch all model_table_* and api_keys_* keys
        if let Ok(mut conn) = client.get_multiplexed_async_connection().await {
            // Fetch all model_table_* keys
            if let Ok(model_keys) = conn.keys::<_, Vec<String>>("model_table_*").await {
                for key in model_keys {
                    if let Ok(json) = conn.get::<&str, String>(key.as_str()).await {
                        match Self::parse_models(&json, &self.app_state.config.provider_types).await {
                            Ok(models) => self.app_state.update_model_table(models).await,
                            Err(e) => tracing::error!("Failed to parse initial model table from redis (key: {key}): {e}"),
                        }
                    }
                }
            }
            // Fetch all api_keys_* keys
            if let Ok(api_keys_keys) = conn.keys::<_, Vec<String>>("api_keys_*").await {
                for key in api_keys_keys {
                    if let Ok(json) = conn.get::<&str, String>(key.as_str()).await {
                        match Self::parse_api_keys(&json).await {
                            Ok(keys) => {
                                if let Some(ref auth) = self.auth {
                                    for (api_key, config) in keys {
                                        auth.update_api_keys(&api_key, config);
                                    }
                                }
                            },
                            Err(e) => tracing::error!("Failed to parse initial api keys from redis (key: {key}): {e}"),
                        }
                    }
                }
            }
        }

        // Get a connection for pubsub
        let mut pubsub_conn = client
            .get_async_pubsub()
            .await
            .map_err(|e| Error::new(ErrorDetails::Config { message: format!("Failed to connect to redis: {e}") }))?;

        // Subscribe to model updates
        pubsub_conn
            .subscribe("model_table_updates")
            .await
            .map_err(|e| Error::new(ErrorDetails::Config { message: format!("Failed to subscribe to model_table_updates: {e}") }))?;

        // Subscribe to API key updates if auth is available
        if self.auth.is_some() {
            pubsub_conn
                .subscribe("api_keys_updates")
                .await
                .map_err(|e| Error::new(ErrorDetails::Config { message: format!("Failed to subscribe to api_keys_updates: {e}") }))?;
        }

        let app_state = self.app_state.clone();
        let auth = self.auth.clone();

        tokio::spawn(async move {
            let mut stream = pubsub_conn.on_message();
            while let Some(msg) = stream.next().await {
                let channel: String = match msg.get_channel_name() {
                    s => s.to_string()
                };

                let payload: String = match msg.get_payload() {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::error!("Failed to decode redis message: {e}");
                        continue;
                    }
                };

                match channel.as_str() {
                    "model_table_updates" => {
                        match Self::parse_models(&payload, &app_state.config.provider_types).await {
                            Ok(models) => app_state.update_model_table(models).await,
                            Err(e) => tracing::error!("Failed to parse models from redis: {e}")
                        }
                    }
                    "api_keys_updates" => {
                        if let Some(ref auth_instance) = auth {
                            match Self::parse_api_keys(&payload).await {
                                Ok(api_keys) => {
                                    // Update all API keys (replace the entire HashMap)
                                    for (key, config) in api_keys {
                                        auth_instance.update_api_keys(&key, config);
                                    }
                                    tracing::info!("Updated API keys from Redis");
                                }
                                Err(e) => tracing::error!("Failed to parse API keys from redis: {e}")
                            }
                        } else {
                            tracing::warn!("Received API keys update but no Auth instance available");
                        }
                    }
                    _ => {
                        tracing::warn!("Received message from unknown channel: {channel}");
                    }
                }
            }
        });

        Ok(())
    }
}