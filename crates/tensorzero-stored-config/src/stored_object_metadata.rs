//! `ConfigObjectMetadata` — descriptive metadata stored alongside every
//! per-object config row.
//!
//! Excluded from the snapshot canonical hash by design: two rows whose
//! configs hash identically but whose metadata differ are considered the
//! same logical config. Metadata is purely descriptive — provenance,
//! authoring notes, UI tags — and never affects runtime behavior.
//!
//! See `crates/tensorzero-stored-config/src/postgres/migrations/20260428100000_per_object_metadata.sql`
//! for the matching DB columns.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Provenance + descriptive metadata attached to a single per-object
/// config row. Stored as JSONB in the `metadata` column of
/// `tensorzero.<object>_configs`.
///
/// Every field is optional. The default value (`Default::default()`)
/// serializes to `{}`, matching the column default in Postgres so we
/// don't have to special-case "no metadata" anywhere.
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ConfigObjectMetadata {
    /// Free-form notes about why this object exists or why it changed.
    /// Source of truth: the Notes textarea in the UI, or block-level TOML
    /// comments harvested at file-mode parse time.
    pub notes: Option<String>,

    /// Who or what wrote this row. Conventionally one of:
    ///
    /// - `"file-import"`               — `gateway --migrate-config` import
    ///                                    of an on-disk TOML
    /// - `"rest/internal-api"`         — per-object REST endpoints
    ///                                    (`/internal/functions/...`)
    /// - `"ui/<user-id>"`              — UI-driven writes (when auth is on)
    /// - `"autopilot/<session-id>"`    — Autopilot's narrow REST writes
    pub created_by: Option<String>,

    /// UI-assigned tags for grouping and filtering. Indexed via a GIN
    /// index on `function_configs.metadata->'tags'` so filter queries
    /// stay cheap.
    pub tags: Option<Vec<String>>,

    /// Path of the TOML file this object was last loaded from.
    /// File-mode-only; left `None` for UI/REST/Autopilot writes.
    pub source_file: Option<String>,

    /// `mtime` of `source_file` at parse time, RFC 3339. File-mode-only.
    pub created_at_source: Option<String>,
}

impl ConfigObjectMetadata {
    /// Returns `true` when no field is set. Used by callers that want to
    /// avoid writing an explicit `'{}'::jsonb` payload when the column
    /// default already covers them.
    pub fn is_empty(&self) -> bool {
        self.notes.is_none()
            && self.created_by.is_none()
            && self.tags.as_ref().is_none_or(Vec::is_empty)
            && self.source_file.is_none()
            && self.created_at_source.is_none()
    }
}

/// Path-keyed bag of `ConfigObjectMetadata` values for a single
/// `write_stored_config` call.
///
/// Keys are slash-separated object identifiers in the same shape as the
/// canonical-JSON paths used elsewhere:
///
/// - `"functions/foo"` — function `foo`
/// - `"functions/foo/variants/bar"` — variant `bar` of function `foo`
/// - `"models/gpt-4o"` — model `gpt-4o`
/// - `"tools/get_weather"` — tool `get_weather`
/// - `"metrics/correctness"` — metric `correctness`
/// - `"evaluations/myeval"` — evaluation `myeval`
/// - `"embedding_models/foo"` — embedding model `foo`
/// - `"optimizers/myopt"` — optimizer `myopt`
/// - `"gateway"` — gateway singleton
/// - `"clickhouse"` — clickhouse singleton
/// - `"postgres"` — postgres singleton
/// - `"object_storage"` — object_storage singleton
/// - `"rate_limiting"` — rate_limiting singleton
/// - `"autopilot"` — autopilot singleton
/// - `"provider_types"` — provider_types singleton
/// - `"files/<path>"` — stored file at `<path>`
///
/// Path strings rather than a typed enum because:
/// 1. They line up with the canonical JSON paths used by snapshot search.
/// 2. The TOML-comment harvester produces these path strings naturally
///    by walking the AST.
/// 3. Adding a new object kind is one line of caller code, not a new
///    enum variant.
pub type PerObjectMetadata = BTreeMap<String, ConfigObjectMetadata>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_serializes_to_empty_object() {
        let m = ConfigObjectMetadata::default();
        let json = serde_json::to_value(&m).unwrap();
        assert_eq!(json, serde_json::json!({}));
        assert!(m.is_empty());
    }

    #[test]
    fn populated_round_trips_through_json() {
        let m = ConfigObjectMetadata {
            notes: Some("rationale".to_string()),
            created_by: Some("ui/user-1".to_string()),
            tags: Some(vec!["staging".to_string(), "team-a".to_string()]),
            source_file: Some("config/functions.toml".to_string()),
            created_at_source: Some("2026-04-28T12:00:00Z".to_string()),
        };
        let json = serde_json::to_value(&m).unwrap();
        let back: ConfigObjectMetadata = serde_json::from_value(json).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn is_empty_treats_empty_tags_as_empty() {
        let m = ConfigObjectMetadata {
            tags: Some(vec![]),
            ..Default::default()
        };
        assert!(m.is_empty(), "Some(empty Vec) should be considered empty");
    }
}
