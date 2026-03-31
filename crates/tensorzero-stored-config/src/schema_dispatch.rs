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
        supported: &'static [i32],
    },
    #[error("failed to deserialize `{config_type}` (schema revision {revision}): {source}")]
    Deserialize {
        config_type: &'static str,
        revision: i32,
        source: serde_json::Error,
    },
}

/// Defines a schema-version-dispatched deserializer.
///
/// Each call site specifies which (revision, type) pairs are supported.
/// When a V2 is added for one config type, only that dispatcher changes;
/// all others remain on their own revision set.
///
/// Usage:
/// ```ignore
/// define_dispatcher!(deserialize_foo, "foo_config", {
///     1 => FooConfigV1,
///     2 => FooConfigV2,
/// });
/// ```
///
/// The output type is the *last* listed type (the latest revision).
/// Earlier revisions must implement `Into<LatestType>` so the dispatcher
/// always returns the latest type.
macro_rules! define_dispatcher {
    // Single revision (common case today) -- no Into conversion needed
    ($fn_name:ident, $label:literal, { $rev:literal => $type:ty $(,)? }) => {
        pub fn $fn_name(
            schema_revision: i32,
            value: Value,
        ) -> Result<$type, SchemaDispatchError> {
            match schema_revision {
                $rev => serde_json::from_value(value).map_err(|e| {
                    SchemaDispatchError::Deserialize {
                        config_type: $label,
                        revision: schema_revision,
                        source: e,
                    }
                }),
                _ => Err(SchemaDispatchError::UnsupportedRevision {
                    config_type: $label,
                    revision: schema_revision,
                    supported: &[$rev],
                }),
            }
        }
    };

    // Multiple revisions -- earlier revisions are converted via Into<Latest>
    ($fn_name:ident, $label:literal, { $($rev:literal => $type:ty),+ $(,)? }) => {
        define_dispatcher!(@output_type $fn_name, $label, [] [$($rev => $type),+]);
    };

    // Internal: peel off revision-type pairs, keeping the last as the output type
    (@output_type $fn_name:ident, $label:literal, [$($prev_rev:literal => $prev_type:ty),*] [$rev:literal => $type:ty]) => {
        pub fn $fn_name(
            schema_revision: i32,
            value: Value,
        ) -> Result<$type, SchemaDispatchError> {
            match schema_revision {
                $(
                    $prev_rev => {
                        let v: $prev_type = serde_json::from_value(value).map_err(|e| {
                            SchemaDispatchError::Deserialize {
                                config_type: $label,
                                revision: schema_revision,
                                source: e,
                            }
                        })?;
                        Ok(v.into())
                    }
                )*
                $rev => serde_json::from_value(value).map_err(|e| {
                    SchemaDispatchError::Deserialize {
                        config_type: $label,
                        revision: schema_revision,
                        source: e,
                    }
                }),
                _ => Err(SchemaDispatchError::UnsupportedRevision {
                    config_type: $label,
                    revision: schema_revision,
                    supported: &[$($prev_rev,)* $rev],
                }),
            }
        }
    };

    (@output_type $fn_name:ident, $label:literal, [$($prev_rev:literal => $prev_type:ty),*] [$rev:literal => $type:ty, $($rest_rev:literal => $rest_type:ty),+]) => {
        define_dispatcher!(@output_type $fn_name, $label, [$($prev_rev => $prev_type,)* $rev => $type] [$($rest_rev => $rest_type),+]);
    };
}

define_dispatcher!(deserialize_gateway_config, "gateway_config", {
    1 => StoredGatewayConfig,
});
define_dispatcher!(deserialize_clickhouse_config, "clickhouse_config", {
    1 => StoredClickHouseConfig,
});
define_dispatcher!(deserialize_postgres_config, "postgres_config", {
    1 => StoredPostgresConfig,
});
define_dispatcher!(deserialize_storage_kind, "object_storage_config", {
    1 => StoredStorageKind,
});
define_dispatcher!(deserialize_model_config, "models_config", {
    1 => StoredModelConfig,
});
define_dispatcher!(deserialize_embedding_model_config, "embedding_models_config", {
    1 => StoredEmbeddingModelConfig,
});
define_dispatcher!(deserialize_metric_config, "metrics_config", {
    1 => StoredMetricConfig,
});
define_dispatcher!(deserialize_rate_limiting_config, "rate_limiting_config", {
    1 => StoredRateLimitingConfig,
});
define_dispatcher!(deserialize_autopilot_config, "autopilot_config", {
    1 => StoredAutopilotConfig,
});
define_dispatcher!(deserialize_provider_types_config, "provider_types_config", {
    1 => StoredProviderTypesConfig,
});
define_dispatcher!(deserialize_optimizer_config, "optimizers_config", {
    1 => StoredOptimizerConfig,
});
define_dispatcher!(deserialize_tool_config, "tools_config", {
    1 => StoredToolConfig,
});
define_dispatcher!(deserialize_evaluation_config, "evaluations_config", {
    1 => StoredEvaluationConfig,
});
define_dispatcher!(deserialize_function_config, "function_versions_config", {
    1 => StoredFunctionConfig,
});
define_dispatcher!(deserialize_variant_config, "variant_versions_config", {
    1 => StoredVariantVersionConfig,
});

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
                assert_eq!(supported, &[1]);
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
