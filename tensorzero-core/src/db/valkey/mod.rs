mod rate_limiting;

use std::time::Duration;

use async_trait::async_trait;
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Client, RedisResult};
use tokio::time::timeout;

use crate::db::HealthCheckable;
use crate::error::{Error, ErrorDetails};

/// Connection info for Valkey (Redis-compatible) rate limiting backend.
///
/// Uses `ConnectionManager` which provides:
/// - Automatic reconnection on connection loss
/// - Connection multiplexing for efficient async operations
/// - No connection pool management needed
#[derive(Clone)]
pub enum ValkeyConnectionInfo {
    Enabled { connection: Box<ConnectionManager> },
    Disabled,
}

impl ValkeyConnectionInfo {
    pub async fn new(valkey_url: &str) -> Result<Self, Error> {
        let client = Client::open(valkey_url).map_err(|e| {
            Error::new(ErrorDetails::ValkeyConnection {
                message: format!("Failed to create Valkey client: {e}"),
            })
        })?;

        let mut connection = ConnectionManager::new(client).await.map_err(|e| {
            Error::new(ErrorDetails::ValkeyConnection {
                message: format!("Failed to connect to Valkey: {e}"),
            })
        })?;

        // When creating the connection, load the function library into Valkey.
        Self::load_function_library(&mut connection).await?;

        Ok(Self::Enabled {
            connection: Box::new(connection),
        })
    }

    pub fn new_disabled() -> Self {
        Self::Disabled
    }

    pub fn get_connection(&self) -> Option<&ConnectionManager> {
        match self {
            Self::Enabled { connection } => Some(connection),
            Self::Disabled => None,
        }
    }

    /// Load the rate limiting function library into Valkey.
    /// This should be called once at startup.
    async fn load_function_library(connection: &mut ConnectionManager) -> Result<(), Error> {
        let lua_code = include_str!("lua/tensorzero_ratelimit.lua");

        // Use FUNCTION LOAD with REPLACE to load/update the library
        let result: RedisResult<()> = redis::cmd("FUNCTION")
            .arg("LOAD")
            .arg("REPLACE")
            .arg(lua_code)
            .query_async(connection)
            .await;
        result.map_err(|e| {
            Error::new(ErrorDetails::ValkeyQuery {
                message: format!("Failed to load function library: {e}"),
            })
        })
    }
}

const HEALTH_CHECK_TIMEOUT_MS: u64 = 1000;

#[async_trait]
impl HealthCheckable for ValkeyConnectionInfo {
    async fn health(&self) -> Result<(), Error> {
        match self {
            Self::Disabled => Ok(()),
            Self::Enabled { connection } => {
                let check = async {
                    let mut conn = connection.clone();
                    let _: String = conn.ping().await.map_err(|e| {
                        Error::new(ErrorDetails::ValkeyConnection {
                            message: format!("Valkey health check failed: {e}"),
                        })
                    })?;
                    Ok(())
                };

                match timeout(Duration::from_millis(HEALTH_CHECK_TIMEOUT_MS), check).await {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(Error::new(ErrorDetails::ValkeyConnection {
                        message: "Valkey health check timed out".to_string(),
                    })),
                }
            }
        }
    }
}
