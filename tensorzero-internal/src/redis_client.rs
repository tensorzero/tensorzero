use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use redis::aio::MultiplexedConnection;
use redis::AsyncCommands;
use tracing::instrument;

use crate::auth::{APIConfig, Auth};
use crate::config_parser::ProviderTypesConfig;
use crate::error::{Error, ErrorDetails};
use crate::gateway_util::AppStateData;
use crate::model::{ModelTable, UninitializedModelConfig};

const MODEL_TABLE_KEY_PREFIX: &str = "model_table:";
const API_KEY_KEY_PREFIX: &str = "api_key:";

pub struct RedisClient {
    client: redis::Client,
    conn: MultiplexedConnection,
    app_state: AppStateData,
    auth: Auth,
}

impl RedisClient {
    pub async fn new(url: &str, app_state: AppStateData, auth: Auth) -> Self {
        #[expect(clippy::expect_used)]
        let (client, conn) = Self::init_conn(url)
            .await
            .map_err(|e| {
                tracing::error!("Failed to connect to Redis: {e}");
                e
            })
            .expect("Redis connection is required for operation");
        Self {
            client,
            conn,
            app_state,
            auth,
        }
    }

    async fn init_conn(url: &str) -> Result<(redis::Client, MultiplexedConnection), Error> {
        let client = redis::Client::open(url).map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to create Redis client: {e}"),
            })
        })?;
        let conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to get Redis connection: {e}"),
                })
            })?;

        Ok((client, conn))
    }

    async fn parse_models(
        json: &str,
        provider_types: &ProviderTypesConfig,
    ) -> Result<ModelTable, Error> {
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

    async fn handle_set_key_event(
        key: &str,
        conn: &mut MultiplexedConnection,
        app_state: &AppStateData,
        auth: &Auth,
    ) -> Result<(), Error> {
        match key {
            k if k.starts_with(API_KEY_KEY_PREFIX) => {
                let value = conn.get::<_, String>(key).await.map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Failed to get value for key {key} from Redis: {e}"),
                    })
                })?;

                match Self::parse_api_keys(&value).await {
                    Ok(keys) => {
                        for (api_key, config) in keys {
                            auth.update_api_keys(&api_key, config);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to parse API keys from redis (key: {key}): {e}")
                    }
                }
            }
            k if k.starts_with(MODEL_TABLE_KEY_PREFIX) => {
                let value = conn.get::<_, String>(key).await.map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Failed to get value for key {key} from Redis: {e}"),
                    })
                })?;

                match Self::parse_models(&value, &app_state.config.provider_types).await {
                    Ok(models) => app_state.update_model_table(models).await,
                    Err(e) => {
                        tracing::error!("Failed to parse models from redis (key: {key}): {e}")
                    }
                }
            }
            _ => {
                tracing::info!("Received message from unknown key pattern: {key}");
            }
        }

        Ok(())
    }

    async fn handle_del_key_event(
        key: &str,
        app_state: &AppStateData,
        auth: &Auth,
    ) -> Result<(), Error> {
        match key {
            k if k.starts_with(API_KEY_KEY_PREFIX) => {
                if let Some(api_key) = key.rsplit(':').next() {
                    auth.delete_api_key(api_key);
                } else {
                    tracing::error!("Invalid API key format: {key}");
                }
                tracing::info!("Deleted API key");
            }
            k if k.starts_with(MODEL_TABLE_KEY_PREFIX) => {
                if let Some(model_name) = key.rsplit(':').next() {
                    app_state.remove_model_table(model_name).await;
                } else {
                    tracing::error!("Invalid model table key format: {key}");
                }
                tracing::info!("Deleted model table: {key}");
            }
            _ => {
                tracing::info!("Received message from unknown key pattern: {key}");
            }
        }

        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn start(mut self) -> Result<(), Error> {
        // Initial fetch: fetch all model_table:* and api_key:* keys
        // Fetch all model_table:* keys
        if let Ok(model_keys) = self
            .conn
            .keys::<_, Vec<String>>(format!("{MODEL_TABLE_KEY_PREFIX}*"))
            .await
        {
            for key in model_keys {
                if let Ok(json) = self.conn.get::<_, String>(&key).await {
                    match Self::parse_models(&json, &self.app_state.config.provider_types).await {
                        Ok(models) => self.app_state.update_model_table(models).await,
                        Err(e) => tracing::error!(
                            "Failed to parse initial model table from redis (key: {key}): {e}"
                        ),
                    }
                }
            }
        }
        // Fetch all api_key:* keys
        if let Ok(api_keys_keys) = self
            .conn
            .keys::<_, Vec<String>>(format!("{API_KEY_KEY_PREFIX}*"))
            .await
        {
            for key in api_keys_keys {
                if let Ok(json) = self.conn.get::<_, String>(&key).await {
                    match Self::parse_api_keys(&json).await {
                        Ok(keys) => {
                            for (api_key, config) in keys {
                                self.auth.update_api_keys(&api_key, config);
                            }
                        }
                        Err(e) => tracing::error!(
                            "Failed to parse initial api keys from redis (key: {key}): {e}"
                        ),
                    }
                }
            }
        }

        // Get a connection for pubsub
        let mut pubsub_conn = self.client.get_async_pubsub().await.map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to connect to redis: {e}"),
            })
        })?;

        pubsub_conn
            .psubscribe("__keyevent@*__:set")
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to subscribe to redis: {e}"),
                })
            })?;

        pubsub_conn
            .psubscribe("__keyevent@*__:del")
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to subscribe to redis: {e}"),
                })
            })?;

        pubsub_conn
            .psubscribe("__keyevent@*__:expired")
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to subscribe to redis: {e}"),
                })
            })?;

        let app_state = self.app_state.clone();
        let auth = self.auth.clone();
        let mut conn = self.conn.clone();

        tokio::spawn(async move {
            let mut stream = pubsub_conn.on_message();
            while let Some(msg) = stream.next().await {
                let channel: String = msg.get_channel_name().to_string();

                let payload: String = match msg.get_payload() {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::error!("Failed to decode redis message: {e}");
                        continue;
                    }
                };

                match channel.as_str() {
                    c if c.ends_with("__:set") => {
                        if let Err(e) = Self::handle_set_key_event(
                            payload.as_str(),
                            &mut conn,
                            &app_state,
                            &auth,
                        )
                        .await
                        {
                            tracing::error!("Failed to handle set key event: {e}");
                        }
                    }
                    c if c.ends_with("__:del") => {
                        if let Err(e) =
                            Self::handle_del_key_event(payload.as_str(), &app_state, &auth).await
                        {
                            tracing::error!("Failed to handle del key event: {e}");
                        }
                    }
                    c if c.ends_with("__:expired") => {
                        if let Err(e) =
                            Self::handle_del_key_event(payload.as_str(), &app_state, &auth).await
                        {
                            tracing::error!("Failed to handle expired key event: {e}");
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
