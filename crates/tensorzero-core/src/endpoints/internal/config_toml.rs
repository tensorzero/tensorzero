//! Editable TOML config endpoints.
//!
//! These endpoints expose the DB-authoritative stored config as an editable
//! TOML document, validate edits against the config-loading pipeline, and
//! apply them back to the stored-config tables with compare-and-swap.

use std::collections::{BTreeMap, HashMap};

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::Postgres;
use tracing::instrument;

use crate::config::editable::{config_to_toml, toml_to_config};
use crate::config::{Config, UninitializedConfig};
use crate::db::postgres::stored_config_queries::{load_config_from_db, merge_load_config_errors};
use crate::db::postgres::stored_config_writes::{
    WriteStoredConfigParams, write_stored_config_in_tx,
};
use crate::error::{Error, ErrorDetails};
use crate::feature_flags;
use crate::utils::gateway::{
    AppState, PreparedConfigSwap, ResolvedAppStateData, StructuredJson, SwappableAppStateData,
};

const CONFIG_EDITOR_ADVISORY_LOCK_KEY: i64 = 0x434F_4E46_4947_544D;

/// Response containing a TOML-editable config from stored config.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct GetConfigTomlResponse {
    /// Human-readable TOML config with path strings instead of inlined file contents.
    pub toml: String,
    /// File-backed content referenced by the TOML, keyed by the path string in the TOML body.
    pub path_contents: HashMap<String, String>,
    /// The hash identifying this config version.
    pub hash: String,
    /// Compare-and-swap signature for the full editable document.
    pub base_signature: String,
    /// User-defined tags for categorizing or labeling this config.
    pub tags: HashMap<String, String>,
}

impl GetConfigTomlResponse {
    fn from_uninitialized(
        config: UninitializedConfig,
        hash: String,
        tags: HashMap<String, String>,
    ) -> Result<Self, Error> {
        let (toml, path_contents) = config_to_toml(&config)?;
        let base_signature = editable_config_signature(&toml, &path_contents)?;
        Ok(Self {
            toml,
            path_contents,
            hash,
            base_signature,
            tags,
        })
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct ApplyConfigTomlRequest {
    pub base_signature: String,
    pub toml: String,
    #[serde(default)]
    pub path_contents: HashMap<String, String>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct ApplyConfigTomlResponse {
    pub toml: String,
    pub path_contents: HashMap<String, String>,
    pub hash: String,
    pub base_signature: String,
}

fn editable_config_signature(
    toml: &str,
    path_contents: &HashMap<String, String>,
) -> Result<String, Error> {
    let sorted_path_contents = path_contents
        .iter()
        .map(|(path, contents)| (path.clone(), contents.clone()))
        .collect::<BTreeMap<_, _>>();
    let payload = serde_json::to_vec(&json!({
        "toml": toml,
        "path_contents": sorted_path_contents,
    }))
    .map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize editable config signature payload: {e}"),
        })
    })?;
    Ok(blake3::hash(&payload).to_hex().to_string())
}

async fn load_db_authoritative_uninitialized_config(
    app_state: &ResolvedAppStateData,
) -> Result<(UninitializedConfig, String), Error> {
    let pool = app_state
        .postgres_connection_info
        .get_pool_result()
        .map_err(|e| e.log())?;
    let uninitialized = load_config_from_db(pool)
        .await
        .map_err(merge_load_config_errors)?;
    let validated = Config::load_from_uninitialized(uninitialized.clone(), false).await?;
    Ok((uninitialized, validated.hash.to_string()))
}

async fn load_db_authoritative_config_toml(
    app_state: &ResolvedAppStateData,
) -> Result<GetConfigTomlResponse, Error> {
    let (uninitialized, hash) = load_db_authoritative_uninitialized_config(app_state).await?;
    GetConfigTomlResponse::from_uninitialized(uninitialized, hash, HashMap::new())
}

async fn acquire_config_editor_lock(tx: &mut sqlx::Transaction<'_, Postgres>) -> Result<(), Error> {
    sqlx::query("SELECT pg_advisory_xact_lock($1)")
        .bind(CONFIG_EDITOR_ADVISORY_LOCK_KEY)
        .execute(&mut **tx)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to acquire config editor advisory lock: {e}"),
            })
        })?;
    Ok(())
}

/// Handler for `GET /internal/config_toml`
///
/// Returns the latest config in editable TOML form. Only available
/// when the `enable_config_in_database` feature flag is set — the editable-TOML
/// pipeline assumes the stored-config tables are the source of truth, so
/// serving it from a file-backed config would be misleading.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "config.get_latest_toml", skip_all)]
pub async fn get_latest_config_toml_handler(
    State(app_state): AppState,
) -> Result<Json<GetConfigTomlResponse>, Error> {
    if !feature_flags::ENABLE_CONFIG_IN_DATABASE.get() {
        return Err(Error::new(ErrorDetails::NotImplemented {
            message:
                "GET /internal/config_toml requires the `enable_config_in_database` feature flag"
                    .to_string(),
        }));
    }

    Ok(Json(load_db_authoritative_config_toml(&app_state).await?))
}

/// Request body for validating editable config TOML.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct ValidateConfigTomlRequest {
    /// Human-readable TOML config with path strings.
    pub toml: String,
    /// File-backed content referenced by the TOML, keyed by the path string in the TOML body.
    #[serde(default)]
    pub path_contents: HashMap<String, String>,
}

/// Response from validating editable config TOML.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct ValidateConfigTomlResponse {
    pub valid: bool,
}

/// Handler for `POST /internal/config_toml/validate`
///
/// Validates editable TOML by parsing it back into `UninitializedConfig` and
/// running the shared config-loading pipeline without persisting anything.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "config.validate_toml", skip_all)]
pub async fn validate_config_toml_handler(
    StructuredJson(request): StructuredJson<ValidateConfigTomlRequest>,
) -> Result<Json<ValidateConfigTomlResponse>, Error> {
    validate_config_toml_request(&request).await?;
    Ok(Json(ValidateConfigTomlResponse { valid: true }))
}

async fn validate_config_toml_request(request: &ValidateConfigTomlRequest) -> Result<(), Error> {
    let config = toml_to_config(&request.toml, &request.path_contents)?;
    let _validated = Config::load_from_uninitialized(config, false).await?;
    Ok(())
}

/// Handler for `POST /internal/config_toml/apply`
///
/// Applies a full editable TOML document back to the stored-config tables using
/// a compare-and-swap against the current DB-authoritative editable snapshot.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "config.apply_toml", skip_all)]
pub async fn apply_config_toml_handler(
    State(swap_state): State<SwappableAppStateData>,
    StructuredJson(request): StructuredJson<ApplyConfigTomlRequest>,
) -> Result<Json<ApplyConfigTomlResponse>, Error> {
    let app_state = swap_state.load_latest();
    let edited_config = toml_to_config(&request.toml, &request.path_contents)?;
    let (canonical_toml, canonical_path_contents) = config_to_toml(&edited_config)?;
    let canonical_signature = editable_config_signature(&canonical_toml, &canonical_path_contents)?;

    let pool = app_state
        .postgres_connection_info
        .get_pool_result()
        .map_err(|e| e.log())?;
    let mut tx = pool.begin().await.map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to start config TOML apply transaction: {e}"),
        })
    })?;
    acquire_config_editor_lock(&mut tx).await?;

    // The advisory lock above (held on `tx`) serializes concurrent apply
    // handlers. `load_config_from_db` fans its reads out across sibling
    // snapshot transactions on independent pool connections, which cannot
    // observe uncommitted state on `tx` — but that is fine here: at this
    // point `tx` has only acquired the lock, not written anything, so the
    // committed baseline is exactly what we want to CAS against.
    let current_uninitialized = load_config_from_db(pool)
        .await
        .map_err(merge_load_config_errors)?;
    let (current_toml, current_path_contents) = config_to_toml(&current_uninitialized)?;
    let current_signature = editable_config_signature(&current_toml, &current_path_contents)?;

    if current_signature != request.base_signature {
        return Err(Error::new(ErrorDetails::ConfigCompareAndSwapConflict {
            message: "Config changed underneath you, reload the latest snapshot before applying your edits.".to_string(),
        }));
    }

    // `write_stored_config_in_tx` validates `edited_config` internally and
    // returns the resulting `UnwrittenConfig`, so we can hot-swap the live
    // gateway state directly from its output — no second read+validate round
    // trip against the database.
    // `Box::pin` keeps the outer future small (clippy::large_futures).
    let written = Box::pin(write_stored_config_in_tx(
        &mut tx,
        WriteStoredConfigParams {
            config: &edited_config,
            creation_source: "ui-config-editor",
            source_autopilot_session_id: None,
            extra_templates: &request.path_contents,
            path_prefix_to_strip: None,
        },
    ))
    .await?;

    // Write the config snapshot and build new runtime dependencies **before**
    // committing the transaction, so a runtime-dependency failure (e.g. bad
    // connection URL in the new config) can still be rolled back cleanly
    // instead of leaving the DB committed but the in-memory state stale.
    // `Box::pin` keeps the outer future small (clippy::large_futures).
    let db = app_state.get_delegating_database();
    let prepared: PreparedConfigSwap = Box::pin(swap_state.prepare_config_swap(written, &db))
        .await
        .map_err(|e| e.log())?;

    let written_hash = prepared.config().hash.to_string();

    tx.commit().await.map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to commit config TOML apply transaction: {e}"),
        })
    })?;

    // Infallible: all fallible work was done in `prepare_config_swap` above.
    swap_state.swap_config(prepared);

    Ok(Json(ApplyConfigTomlResponse {
        toml: canonical_toml,
        path_contents: canonical_path_contents,
        hash: written_hash,
        base_signature: canonical_signature,
    }))
}

#[cfg(test)]
mod tests {
    use googletest::prelude::*;

    use crate::config::UninitializedToolConfig;
    use crate::config::gateway::UninitializedGatewayConfig;
    use crate::config::path::ResolvedTomlPathData;

    use super::*;

    #[gtest]
    fn config_toml_response_uses_plain_path_strings() {
        let config = UninitializedConfig {
            gateway: Some(UninitializedGatewayConfig {
                bind_address: Some(
                    "127.0.0.1:4000"
                        .parse()
                        .expect("bind address should parse for TOML response test"),
                ),
                ..Default::default()
            }),
            clickhouse: Default::default(),
            postgres: Default::default(),
            rate_limiting: Default::default(),
            object_storage: None,
            models: Some(HashMap::new()),
            embedding_models: Some(HashMap::new()),
            functions: Some(HashMap::new()),
            metrics: Some(HashMap::new()),
            tools: Some(HashMap::from([(
                "weather".to_string(),
                UninitializedToolConfig {
                    description: "Weather tool".to_string(),
                    parameters: ResolvedTomlPathData::new_fake_path(
                        "tools/weather.json".to_string(),
                        "{\"type\":\"object\"}".to_string(),
                    ),
                    name: None,
                    strict: false,
                },
            )])),
            evaluations: Some(HashMap::new()),
            provider_types: Default::default(),
            optimizers: Some(HashMap::new()),
            autopilot: Default::default(),
        };
        let response = GetConfigTomlResponse::from_uninitialized(
            config,
            "test-hash".to_string(),
            HashMap::new(),
        )
        .expect("TOML response generation should succeed");

        expect_that!(
            response.toml.as_str(),
            contains_substring("bind_address = \"127.0.0.1:4000\"")
        );
        expect_that!(
            response.toml.as_str(),
            contains_substring("parameters = \"tools/weather.json\"")
        );
        expect_that!(
            response.path_contents.get("tools/weather.json"),
            some(eq(&"{\"type\":\"object\"}".to_string()))
        );
    }

    #[gtest]
    #[tokio::test]
    async fn validate_config_toml_request_accepts_round_trip_output() {
        let config = UninitializedConfig {
            gateway: Some(UninitializedGatewayConfig {
                bind_address: Some(
                    "127.0.0.1:4000"
                        .parse()
                        .expect("bind address should parse for validation test"),
                ),
                ..Default::default()
            }),
            clickhouse: Default::default(),
            postgres: Default::default(),
            rate_limiting: Default::default(),
            object_storage: None,
            models: Some(HashMap::new()),
            embedding_models: Some(HashMap::new()),
            functions: Some(HashMap::new()),
            metrics: Some(HashMap::new()),
            tools: Some(HashMap::from([(
                "weather".to_string(),
                UninitializedToolConfig {
                    description: "Weather tool".to_string(),
                    parameters: ResolvedTomlPathData::new_fake_path(
                        "tools/weather.json".to_string(),
                        "{\"type\":\"object\"}".to_string(),
                    ),
                    name: None,
                    strict: false,
                },
            )])),
            evaluations: Some(HashMap::new()),
            provider_types: Default::default(),
            optimizers: Some(HashMap::new()),
            autopilot: Default::default(),
        };
        let (toml, path_contents) =
            config_to_toml(&config).expect("editable TOML serialization should succeed");
        let request = ValidateConfigTomlRequest {
            toml,
            path_contents,
        };

        expect_that!(validate_config_toml_request(&request).await, ok(eq(&())));
    }
}
