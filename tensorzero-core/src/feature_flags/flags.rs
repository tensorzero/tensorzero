//! Feature flag definitions.
//!
//! Add new feature flags here. Each flag is read from an environment variable
//! with the format `TENSORZERO_INTERNAL_FLAG_{NAME}` where NAME is the uppercase version of the flag name.

use super::FlagDefinition;

/// A dummy test flag for unit tests.
#[cfg(test)]
pub static TEST_FLAG: FlagDefinition<bool> = FlagDefinition::new("test_flag", false);

/// Enable writing to Postgres for data in addition to ClickHouse.
pub static ENABLE_POSTGRES_WRITE: FlagDefinition<bool> =
    FlagDefinition::new("enable_postgres_dual_write", false);

/// Enable reading from Postgres for data instead of ClickHouse.
pub static ENABLE_POSTGRES_READ: FlagDefinition<bool> =
    FlagDefinition::new("enable_postgres_read", false);
