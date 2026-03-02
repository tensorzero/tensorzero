# Stored Config Types

`Stored*Config` types represent configuration that is persisted in the database. They must **not** use `#[serde(deny_unknown_fields)]`, because the database may contain fields added by newer versions of the application after a rollback. Rejecting unknown fields would cause deserialization failures when rolling back to an older version.
