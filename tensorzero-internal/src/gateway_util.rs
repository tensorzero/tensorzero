use std::sync::Arc;

use axum::extract::{rejection::JsonRejection, FromRequest, Json, Request};
use reqwest::Client;
use serde::de::DeserializeOwned;
use tracing::instrument;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::clickhouse_migration_manager;
use crate::config_parser::Config;
use crate::error::{Error, ErrorDetails};

/// State for the API
#[derive(Clone)]
pub struct AppStateData {
    pub config: Arc<Config<'static>>,
    pub http_client: Client,
    pub clickhouse_connection_info: ClickHouseConnectionInfo,
}
pub type AppState = axum::extract::State<AppStateData>;

impl AppStateData {
    pub async fn new(config: Arc<Config<'static>>) -> Result<Self, Error> {
        let clickhouse_url = std::env::var("TENSORZERO_CLICKHOUSE_URL")
            .ok()
            .or_else(|| {
                std::env::var("CLICKHOUSE_URL").ok().inspect(|_| {
                    tracing::warn!("Deprecation Warning: The environment variable \"CLICKHOUSE_URL\" has been renamed to \"TENSORZERO_CLICKHOUSE_URL\" and will be removed in a future version. Please update your environment to use \"TENSORZERO_CLICKHOUSE_URL\" instead.");
                })
            });
        let clickhouse_connection_info = setup_clickhouse(&config, clickhouse_url).await?;

        let http_client = Client::new();

        Ok(Self {
            config,
            http_client,
            clickhouse_connection_info,
        })
    }
}

pub async fn setup_clickhouse(
    config: &Config<'static>,
    clickhouse_url: Option<String>,
) -> Result<ClickHouseConnectionInfo, Error> {
    let clickhouse_connection_info = match (config.gateway.observability.enabled, clickhouse_url) {
        // Observability disabled by config
        (Some(false), _) => ClickHouseConnectionInfo::new_disabled(),
        // Observability enabled but no ClickHouse URL
        (Some(true), None) => {
            return Err(ErrorDetails::AppState {
                message: "Missing environment variable TENSORZERO_CLICKHOUSE_URL".to_string(),
            }
            .into())
        }
        // Observability enabled and ClickHouse URL provided
        (Some(true), Some(clickhouse_url)) => {
            ClickHouseConnectionInfo::new(&clickhouse_url).await?
        }
        // Observability default and no ClickHouse URL
        (None, None) => {
            tracing::warn!("Observability not explicitly specified in config and no ClickHouse URL provided. Disabling ClickHouse.");
            ClickHouseConnectionInfo::new_disabled()
        }
        // Observability default and ClickHouse URL provided
        (None, Some(clickhouse_url)) => ClickHouseConnectionInfo::new(&clickhouse_url).await?,
    };

    // Run ClickHouse migrations (if any) if we have a production ClickHouse connection
    if let ClickHouseConnectionInfo::Production { .. } = &clickhouse_connection_info {
        clickhouse_migration_manager::run(&clickhouse_connection_info).await?;
    }
    Ok(clickhouse_connection_info)
}

/// Custom Axum extractor that validates the JSON body and deserializes it into a custom type
///
/// When this extractor is present, we don't check if the `Content-Type` header is `application/json`,
/// and instead simply assume that the request body is a JSON object.
pub struct StructuredJson<T>(pub T);

#[axum::async_trait]
impl<S, T> FromRequest<S> for StructuredJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
    T: Send + Sync + DeserializeOwned,
{
    type Rejection = Error;

    #[instrument(skip_all, level = "trace", name = "StructuredJson::from_request")]
    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        // Retrieve the request body as Bytes before deserializing it
        let bytes = bytes::Bytes::from_request(req, state).await.map_err(|e| {
            Error::new(ErrorDetails::JsonRequest {
                message: format!("{} ({})", e, e.status()),
            })
        })?;

        // Convert the entire body into `serde_json::Value`
        let value = Json::<serde_json::Value>::from_bytes(&bytes)
            .map_err(|e| {
                Error::new(ErrorDetails::JsonRequest {
                    message: format!("{} ({})", e, e.status()),
                })
            })?
            .0;

        // Now use `serde_path_to_error::deserialize` to attempt deserialization into `T`
        let deserialized: T = serde_path_to_error::deserialize(&value).map_err(|e| {
            Error::new(ErrorDetails::JsonRequest {
                message: e.to_string(),
            })
        })?;

        Ok(StructuredJson(deserialized))
    }
}

#[cfg(test)]
mod tests {
    use tracing_test::traced_test;

    use super::*;
    use crate::config_parser::{GatewayConfig, ObservabilityConfig};

    #[tokio::test]
    #[traced_test]
    async fn test_setup_clickhouse() {
        // Disabled observability
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(false),
            },
            bind_address: None,
            debug: false,
        };

        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));

        let clickhouse_connection_info = setup_clickhouse(config, None).await.unwrap();
        assert!(matches!(
            clickhouse_connection_info,
            ClickHouseConnectionInfo::Disabled
        ));
        assert!(!logs_contain(
            "Missing environment variable TENSORZERO_CLICKHOUSE_URL"
        ));

        // Default observability and no ClickHouse URL
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig { enabled: None },
            ..Default::default()
        };
        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));
        let clickhouse_connection_info = setup_clickhouse(config, None).await.unwrap();
        assert!(matches!(
            clickhouse_connection_info,
            ClickHouseConnectionInfo::Disabled
        ));
        assert!(!logs_contain(
            "Missing environment variable TENSORZERO_CLICKHOUSE_URL"
        ));

        // We do not test the case where a ClickHouse URL is provided but observability is default,
        // as this would require a working ClickHouse and we don't have one in unit tests.

        // Observability enabled but ClickHouse URL is missing
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(true),
            },
            bind_address: None,
            debug: false,
        };

        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));

        let err = setup_clickhouse(config, None).await.unwrap_err();
        assert!(err
            .to_string()
            .contains("Missing environment variable TENSORZERO_CLICKHOUSE_URL"));

        // Bad URL
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(true),
            },
            bind_address: None,
            debug: false,
        };
        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));
        setup_clickhouse(config, Some("bad_url".to_string()))
            .await
            .expect_err("ClickHouse setup should fail given a bad URL");
        assert!(logs_contain("Invalid ClickHouse database URL"));
    }

    #[tokio::test]
    #[traced_test]
    async fn test_unhealthy_clickhouse() {
        // Sensible URL that doesn't point to ClickHouse
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(true),
            },
            bind_address: None,
            debug: false,
        };
        let config = Config {
            gateway: gateway_config,
            ..Default::default()
        };
        setup_clickhouse(&config, Some("https://tensorzero.com:8123".to_string()))
            .await
            .expect_err(
                "ClickHouse setup should fail given a URL that doesn't point to ClickHouse",
            );
        assert!(logs_contain(
            "Error connecting to ClickHouse: ClickHouse is not healthy"
        ));
        // We do not test the case where a ClickHouse URL is provided and observability is on,
        // as this would require a working ClickHouse and we don't have one in unit tests.
    }
}
