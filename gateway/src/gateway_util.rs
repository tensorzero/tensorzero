use axum::extract::{rejection::JsonRejection, FromRequest, Json, Request};
use reqwest::Client;
use serde::de::DeserializeOwned;
use tracing::instrument;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::Config;
use crate::error::{Error, ResultExt};

/// State for the API
#[derive(Clone)]
pub struct AppStateData {
    pub config: &'static Config<'static>,
    pub http_client: Client,
    pub clickhouse_connection_info: ClickHouseConnectionInfo,
}
pub type AppState = axum::extract::State<AppStateData>;

impl AppStateData {
    pub async fn new(config: &'static Config<'static>) -> Self {
        let clickhouse_url = std::env::var("CLICKHOUSE_URL")
            .map_err(|_| Error::AppState {
                message: "Missing environment variable CLICKHOUSE_URL".to_string(),
            })
            .ok_or_log()
            .unwrap_or_else(|| "".to_string());
        let clickhouse_connection_info = setup_clickhouse(config, &clickhouse_url).await;

        let http_client = Client::new();

        Self {
            config,
            http_client,
            clickhouse_connection_info,
        }
    }
}

async fn setup_clickhouse(
    config: &'static Config<'static>,
    clickhouse_url: &str,
) -> ClickHouseConnectionInfo {
    if config.gateway.disable_observability {
        ClickHouseConnectionInfo::new_disabled()
    } else {
        ClickHouseConnectionInfo::new(clickhouse_url)
            .await
            .ok_or_log()
            .unwrap_or_else(ClickHouseConnectionInfo::new_disabled)
    }
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
        let bytes =
            bytes::Bytes::from_request(req, state)
                .await
                .map_err(|e| Error::JsonRequest {
                    message: format!("{} ({})", e, e.status()),
                })?;

        // Convert the entire body into `serde_json::Value`
        let value = Json::<serde_json::Value>::from_bytes(&bytes)
            .map_err(|e| Error::JsonRequest {
                message: format!("{} ({})", e, e.status()),
            })?
            .0;

        // Now use `serde_path_to_error::deserialize` to attempt deserialization into `T`
        let deserialized: T =
            serde_path_to_error::deserialize(&value).map_err(|e| Error::JsonRequest {
                message: e.to_string(),
            })?;

        Ok(StructuredJson(deserialized))
    }
}

#[cfg(test)]
mod tests {
    use tracing_test::traced_test;

    use super::*;
    use crate::config_parser::GatewayConfig;

    #[tokio::test]
    #[traced_test]
    async fn test_setup_clickhouse() {
        // Disabled observability
        let gateway_config = GatewayConfig {
            disable_observability: true,
            bind_address: None,
        };

        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));

        let clickhouse_connection_info = setup_clickhouse(config, "").await;
        assert!(matches!(
            clickhouse_connection_info,
            ClickHouseConnectionInfo::Disabled
        ));

        // Bad URL
        let gateway_config = GatewayConfig {
            disable_observability: false,
            bind_address: None,
        };
        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));
        let clickhouse_connection_info = setup_clickhouse(config, "bad_url").await;
        assert!(matches!(
            clickhouse_connection_info,
            ClickHouseConnectionInfo::Disabled
        ));
        assert!(logs_contain("Invalid ClickHouse database URL"));

        // Sensible URL that doesn't point to ClickHouse
        let gateway_config = GatewayConfig {
            disable_observability: false,
            bind_address: None,
        };
        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));
        let clickhouse_connection_info =
            setup_clickhouse(config, "https://tensorzero.com:8123").await;
        assert!(matches!(
            clickhouse_connection_info,
            ClickHouseConnectionInfo::Disabled
        ));
        assert!(logs_contain(
            "Error connecting to ClickHouse: ClickHouse is not healthy"
        ));
    }
}
