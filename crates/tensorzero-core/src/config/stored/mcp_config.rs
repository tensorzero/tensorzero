use serde::{Deserialize, Serialize};

use crate::config::gateway::McpConfig;

/// Stored version of `McpConfig`.
///
/// Omits `deny_unknown_fields` so that snapshots written by newer versions
/// (with additional fields) remain deserializable by older versions.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredMcpConfig {
    #[serde(default = "default_mcp_enabled")]
    pub enabled: bool,
    #[serde(default = "default_mcp_bind_address")]
    pub bind_address: std::net::SocketAddr,
}

fn default_mcp_enabled() -> bool {
    true
}

fn default_mcp_bind_address() -> std::net::SocketAddr {
    std::net::SocketAddr::from(([0, 0, 0, 0], 3001))
}

impl From<McpConfig> for StoredMcpConfig {
    fn from(config: McpConfig) -> Self {
        let McpConfig {
            enabled,
            bind_address,
        } = config;
        Self {
            enabled,
            bind_address,
        }
    }
}

impl From<StoredMcpConfig> for McpConfig {
    fn from(stored: StoredMcpConfig) -> Self {
        let StoredMcpConfig {
            enabled,
            bind_address,
        } = stored;
        Self {
            enabled,
            bind_address,
        }
    }
}
