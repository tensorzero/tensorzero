use serde_json::Value;

use crate::{
    StoredAutopilotConfig, StoredClickHouseConfig, StoredEmbeddingModelConfig,
    StoredEvaluationConfig, StoredFunctionConfig, StoredGatewayConfig, StoredMetricConfig,
    StoredModelConfig, StoredOptimizerConfig, StoredPostgresConfig, StoredProviderTypesConfig,
    StoredRateLimitingConfig, StoredStorageKind, StoredToolConfig, StoredVariantVersionConfig,
};

#[derive(Debug, thiserror::Error)]
pub enum SchemaDispatchError {
    #[error(
        "unsupported schema revision {revision} for `{config_type}`: supported revisions are {supported:?}"
    )]
    UnsupportedRevision {
        config_type: &'static str,
        revision: i32,
        supported: Vec<i32>,
    },
    #[error("failed to deserialize `{config_type}` (schema revision {revision}): {source}")]
    Deserialize {
        config_type: &'static str,
        revision: i32,
        source: serde_json::Error,
    },
}

/// A handler that deserializes a JSON value at one specific schema revision
/// and returns the latest type `T`. Earlier-revision handlers are responsible
/// for performing any `Into<T>` conversions internally.
type RevisionHandler<'a, T> = &'a dyn Fn(Value) -> Result<T, serde_json::Error>;

/// Dispatches a schema-versioned JSON value to the appropriate deserializer.
///
/// Each call site passes a slice of `(revision, handler)` pairs. The handler
/// deserializes the value at that revision and converts it to the latest type
/// `T`. If `schema_revision` matches no entry, returns `UnsupportedRevision`
/// with the list of supported revisions sourced from the slice.
///
/// # When to bump the schema revision
///
/// **Bump only for breaking changes** — changes where an old reader cannot
/// correctly deserialize a row written by a new writer. Examples:
///
/// - Renaming or removing a field
/// - Changing enum tag strings or discriminator values
/// - Changing nesting structure (e.g. moving a field into a sub-object)
/// - Promoting an `Option<T>` to required
///
/// **Do NOT bump for additive changes.** Adding a new `Option<T>` field (even
/// one that takes a default on the read side) is forward- and backward-
/// compatible: old readers ignore the unknown key, and new readers see `None`
/// for rows written before the field existed. The same applies to adding new
/// enum variants — old rows simply never contain them. See
/// `tensorzero-stored-config/AGENTS.md` for the full schema revision policy.
pub fn dispatch_schema<T>(
    config_type: &'static str,
    schema_revision: i32,
    value: Value,
    handlers: &[(i32, RevisionHandler<'_, T>)],
) -> Result<T, SchemaDispatchError> {
    for (revision, handler) in handlers {
        if *revision == schema_revision {
            return handler(value).map_err(|source| SchemaDispatchError::Deserialize {
                config_type,
                revision: schema_revision,
                source,
            });
        }
    }
    Err(SchemaDispatchError::UnsupportedRevision {
        config_type,
        revision: schema_revision,
        supported: handlers.iter().map(|(rev, _)| *rev).collect(),
    })
}

pub fn deserialize_gateway_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredGatewayConfig, SchemaDispatchError> {
    dispatch_schema(
        "gateway_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_clickhouse_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredClickHouseConfig, SchemaDispatchError> {
    dispatch_schema(
        "clickhouse_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_postgres_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredPostgresConfig, SchemaDispatchError> {
    dispatch_schema(
        "postgres_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_storage_kind(
    schema_revision: i32,
    value: Value,
) -> Result<StoredStorageKind, SchemaDispatchError> {
    dispatch_schema(
        "object_storage_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_model_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredModelConfig, SchemaDispatchError> {
    dispatch_schema(
        "models_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_embedding_model_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredEmbeddingModelConfig, SchemaDispatchError> {
    dispatch_schema(
        "embedding_models_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_metric_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredMetricConfig, SchemaDispatchError> {
    dispatch_schema(
        "metrics_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_rate_limiting_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredRateLimitingConfig, SchemaDispatchError> {
    dispatch_schema(
        "rate_limiting_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_autopilot_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredAutopilotConfig, SchemaDispatchError> {
    dispatch_schema(
        "autopilot_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_provider_types_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredProviderTypesConfig, SchemaDispatchError> {
    dispatch_schema(
        "provider_types_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_optimizer_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredOptimizerConfig, SchemaDispatchError> {
    dispatch_schema(
        "optimizers_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_tool_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredToolConfig, SchemaDispatchError> {
    dispatch_schema(
        "tools_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_evaluation_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredEvaluationConfig, SchemaDispatchError> {
    dispatch_schema(
        "evaluations_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_function_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredFunctionConfig, SchemaDispatchError> {
    dispatch_schema(
        "function_versions_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

pub fn deserialize_variant_config(
    schema_revision: i32,
    value: Value,
) -> Result<StoredVariantVersionConfig, SchemaDispatchError> {
    dispatch_schema(
        "variant_versions_config",
        schema_revision,
        value,
        &[(1, &|v| serde_json::from_value(v))],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_revision_returns_error() {
        let value = serde_json::json!({});
        let err = deserialize_gateway_config(999, value).unwrap_err();
        match err {
            SchemaDispatchError::UnsupportedRevision {
                revision: 999,
                supported,
                ..
            } => {
                assert_eq!(supported, vec![1]);
            }
            other => panic!("expected UnsupportedRevision, got: {other}"),
        }
    }

    #[test]
    fn revision_1_deserializes_clickhouse_config() {
        let value = serde_json::json!({"disable_automatic_migrations": true});
        let config = deserialize_clickhouse_config(1, value).unwrap();
        assert_eq!(
            config,
            StoredClickHouseConfig {
                disable_automatic_migrations: Some(true)
            }
        );
    }

    #[test]
    fn revision_1_invalid_json_returns_deserialize_error() {
        let value = serde_json::json!({"disable_automatic_migrations": "not_a_bool"});
        let err = deserialize_clickhouse_config(1, value).unwrap_err();
        assert!(matches!(err, SchemaDispatchError::Deserialize { .. }));
    }
}
