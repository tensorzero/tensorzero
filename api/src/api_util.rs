use axum::extract::{rejection::JsonRejection, FromRequest, Json, Request};
use reqwest::Client;
use serde::de::DeserializeOwned;
use std::sync::Arc;
use tracing::instrument;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::{self, Config};
use crate::error::Error;

/// State for the API
#[derive(Clone)]
pub struct AppStateData {
    pub config: Arc<Config>,
    pub http_client: Client,
    pub clickhouse_connection_info: ClickHouseConnectionInfo,
}
pub type AppState = axum::extract::State<AppStateData>;

impl Default for AppStateData {
    fn default() -> Self {
        let config = Arc::new(config_parser::Config::load());
        println!("{:#?}", config); // TODO: temporary
        let clickhouse_url =
            std::env::var("CLICKHOUSE_URL").expect("Missing environment variable CLICKHOUSE_URL");
        Self {
            config,
            http_client: Client::new(),
            clickhouse_connection_info: ClickHouseConnectionInfo::new(&clickhouse_url, false),
        }
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
