# Stored Config Types

`Stored*Config` types represent configuration that is persisted in the database. They must **not** use `#[serde(deny_unknown_fields)]`, because the database may contain fields added by newer versions of the application after a rollback. Rejecting unknown fields would cause deserialization failures when rolling back to an older version.

## Historical snapshot tests

Whenever a config shape changes (fields added, removed, renamed, or restructured), add a test in `mod.rs` that parses the **previous** version of the config TOML and asserts it deserializes into `StoredConfig` and converts to `UninitializedConfig` with the correct values. See the existing `test_historical_*` tests for examples. This ensures that snapshots already persisted in databases remain loadable after the change.

Do not add round-trip tests (e.g. serialize then deserialize and compare). They don't catch real bugs and add noise.
