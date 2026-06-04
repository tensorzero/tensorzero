//! Shared helpers for persisting stored files to
//! `tensorzero.stored_files`.
//!
//! Both the function-config write path (`function_config_writes`) and the
//! whole-config stored-config write path (`stored_config_writes`) need to
//! insert files with the same semantics:
//!
//! 1. Collect templates into a `BTreeMap` keyed by template key, erroring on
//!    conflicting bodies for the same key.
//! 2. Reuse existing rows that already match `(file_path, content_hash)`
//!    so that idempotent rewrites produce stable version IDs.
//! 3. Batch-insert only the genuinely new rows.
//!
//! The file-specific logic (walking a function config tree, merging
//! filesystem-discovered extra templates, or iterating a tool/evaluation's
//! file references) lives in the callers; this module only owns
//! the collection and SQL dedup.

use std::collections::{BTreeMap, HashMap};

use sqlx::{FromRow, Postgres, QueryBuilder, Transaction};
use uuid::Uuid;

use crate::config::path::ResolvedTomlPathData;
use crate::error::{Error, ErrorDetails};

#[derive(Debug)]
pub(super) struct CollectedFile {
    pub source_body: String,
}

#[derive(Debug, FromRow)]
struct ExistingTemplateRow {
    file_path: String,
    id: Uuid,
}

/// Insert `template` into `templates`, erroring if a different body was
/// already recorded under the same key.
///
pub(super) fn add_file(
    templates: &mut BTreeMap<String, CollectedFile>,
    template: &ResolvedTomlPathData,
) -> Result<(), Error> {
    let file_path = template.get_template_key();
    let source_body = template.data().to_string();
    match templates.get(&file_path) {
        Some(existing) if existing.source_body != source_body => {
            Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "Template key `{file_path}` was provided with conflicting source bodies."
                ),
            }))
        }
        Some(_) => Ok(()),
        None => {
            templates.insert(file_path, CollectedFile { source_body });
            Ok(())
        }
    }
}

/// Persist a collected set of stored files, reusing any existing *active*
/// row that already matches `(file_path, content_hash)` and batch-inserting
/// only the genuinely new ones. After writing, any other active rows for
/// the same `file_path` are tombstoned so that the invariant "at most one
/// active row per `file_path`" (relied on by `load_editor_path_contents`)
/// is preserved. Returns `file_path -> version_id`.
pub(super) async fn write_collected_files(
    tx: &mut Transaction<'_, Postgres>,
    templates: &BTreeMap<String, CollectedFile>,
    creation_source: &str,
    source_autopilot_session_id: Option<Uuid>,
) -> Result<HashMap<String, Uuid>, Error> {
    if templates.is_empty() {
        return Ok(HashMap::new());
    }

    // Compute BLAKE3 content hashes for all templates.
    let hashes: BTreeMap<String, Vec<u8>> = templates
        .iter()
        .map(|(key, t)| {
            (
                key.clone(),
                blake3::hash(t.source_body.as_bytes()).as_bytes().to_vec(),
            )
        })
        .collect();

    // Batch lookup: find existing active template versions with matching
    // `(file_path, content_hash)`. We intentionally filter on
    // `deleted_at IS NULL` so we never reuse a tombstoned row's UUID — doing
    // so would leave zero active rows for the path after the tombstone step
    // below, breaking the editor view.
    let file_paths: Vec<&str> = templates.keys().map(String::as_str).collect();
    let content_hashes: Vec<&[u8]> = file_paths.iter().map(|k| hashes[*k].as_slice()).collect();
    let existing_rows: Vec<ExistingTemplateRow> = sqlx::query_as(
        "SELECT input.file_path, t.id \
         FROM UNNEST($1::text[], $2::bytea[]) AS input(file_path, content_hash) \
         JOIN tensorzero.stored_files t \
           ON t.file_path = input.file_path \
          AND t.content_hash = input.content_hash \
          AND t.deleted_at IS NULL",
    )
    .bind(&file_paths)
    .bind(&content_hashes)
    .fetch_all(&mut **tx)
    .await
    .map_err(|e| postgres_query_error("Failed to look up existing stored file versions", e))?;

    let existing: HashMap<String, Uuid> = existing_rows
        .into_iter()
        .map(|row| (row.file_path, row.id))
        .collect();

    // Partition into reused vs. new templates.
    let mut template_version_ids = HashMap::with_capacity(templates.len());
    let mut new_file_paths = Vec::new();
    for file_path in templates.keys() {
        if let Some(&existing_id) = existing.get(file_path) {
            template_version_ids.insert(file_path.clone(), existing_id);
        } else {
            let id = Uuid::now_v7();
            template_version_ids.insert(file_path.clone(), id);
            new_file_paths.push(file_path.clone());
        }
    }

    // Batch insert only new templates.
    if !new_file_paths.is_empty() {
        let mut qb = QueryBuilder::new(
            "INSERT INTO tensorzero.stored_files \
             (id, file_path, source_body, content_hash, creation_source, source_autopilot_session_id) ",
        );
        qb.push_values(&new_file_paths, |mut b, key| {
            let template = &templates[key];
            let id = template_version_ids[key];
            b.push_bind(id)
                .push_bind(key.clone())
                .push_bind(template.source_body.clone())
                .push_bind(hashes[key].clone())
                .push_bind(creation_source.to_string())
                .push_bind(source_autopilot_session_id);
        });
        qb.build()
            .execute(&mut **tx)
            .await
            .map_err(|e| postgres_query_error("Failed to insert stored file versions", e))?;
    }

    // Tombstone any OTHER active rows for the file paths we just wrote so
    // that at most one active row remains per `file_path`. Config reads go
    // by UUID and do not filter on `deleted_at`, so previously-referenced
    // function/tool/evaluation versions still resolve correctly — the
    // tombstone only hides the old row from the editor's latest-per-path
    // view.
    let kept_ids: Vec<Uuid> = template_version_ids.values().copied().collect();
    sqlx::query(
        "UPDATE tensorzero.stored_files \
         SET deleted_at = NOW() \
         WHERE deleted_at IS NULL \
           AND file_path = ANY($1::text[]) \
           AND id <> ALL($2::uuid[])",
    )
    .bind(&file_paths)
    .bind(&kept_ids)
    .execute(&mut **tx)
    .await
    .map_err(|e| postgres_query_error("Failed to tombstone superseded stored file versions", e))?;

    Ok(template_version_ids)
}

fn postgres_query_error(context: &str, error: impl std::fmt::Display) -> Error {
    Error::new(ErrorDetails::PostgresQuery {
        message: format!("{context}: {error}"),
    })
}
