mod rate_limiting;

use redis::aio::ConnectionManager;
use redis::{Client, RedisResult};

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
