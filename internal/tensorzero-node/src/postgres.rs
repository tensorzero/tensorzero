use secrecy::ExposeSecret;
use tensorzero_auth::constants::{DEFAULT_ORGANIZATION, DEFAULT_WORKSPACE};
use tensorzero_core::{
    config::Config, db::postgres::PostgresConnectionInfo, utils::gateway::setup_postgres,
};

#[napi(js_name = "PostgresClient")]
pub struct PostgresClient {
    connection_info: PostgresConnectionInfo,
}

#[napi]
impl PostgresClient {
    #[napi(factory)]
    pub async fn from_postgres_url(postgres_url: String) -> Result<Self, napi::Error> {
        // Create a minimal config just for postgres connection pool size
        // The default pool size is 10 which should be reasonable
        let config = Config::new_empty()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to setup Postgres: {e}")))?
            .config;

        let connection_info = setup_postgres(&config, Some(postgres_url))
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to setup Postgres: {e}")))?;

        Ok(Self { connection_info })
    }

    #[napi]
    pub async fn create_api_key(&self, description: Option<String>) -> Result<String, napi::Error> {
        let pool = self
            .connection_info
            .get_alpha_pool()
            .ok_or_else(|| napi::Error::from_reason("Postgres connection not available"))?;

        let key = tensorzero_auth::postgres::create_key(
            DEFAULT_ORGANIZATION,
            DEFAULT_WORKSPACE,
            description.as_deref(),
            pool,
        )
        .await
        .map_err(|e| napi::Error::from_reason(format!("Failed to create API key: {e}")))?;

        Ok(key.expose_secret().to_string())
    }

    #[napi]
    pub async fn list_api_keys(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<String, napi::Error> {
        let pool = self
            .connection_info
            .get_alpha_pool()
            .ok_or_else(|| napi::Error::from_reason("Postgres connection not available"))?;

        let keys = tensorzero_auth::postgres::list_key_info(None, limit, offset, pool)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to list API keys: {e}")))?;

        serde_json::to_string(&keys).map_err(|e| {
            napi::Error::from_reason(format!("Failed to serialize API keys list: {e}"))
        })
    }

    #[napi]
    pub async fn disable_api_key(&self, public_id: String) -> Result<String, napi::Error> {
        let pool = self
            .connection_info
            .get_alpha_pool()
            .ok_or_else(|| napi::Error::from_reason("Postgres connection not available"))?;

        let disabled_at = tensorzero_auth::postgres::disable_key(&public_id, pool)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to disable API key: {e}")))?;

        serde_json::to_string(&disabled_at).map_err(|e| {
            napi::Error::from_reason(format!("Failed to serialize disabled_at timestamp: {e}"))
        })
    }
}
