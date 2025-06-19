#![deny(clippy::all)]
use std::{path::Path, time::Duration};

use tensorzero::{Client, ClientBuilder, ClientBuilderMode};

#[macro_use]
extern crate napi_derive;

#[napi(js_name = "TensorZeroClient")]
pub struct TensorZeroClient {
    client: Client,
}

#[napi]
impl TensorZeroClient {
    #[napi(factory)]
    pub async fn new(
        config_path: String,
        clickhouse_url: Option<String>,
        timeout: Option<f64>,
    ) -> Result<Self, napi::Error> {
        let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(Path::new(&config_path).to_path_buf()),
            clickhouse_url,
            timeout: timeout.map(Duration::from_secs_f64),
        })
        .build()
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Self { client })
    }
}
