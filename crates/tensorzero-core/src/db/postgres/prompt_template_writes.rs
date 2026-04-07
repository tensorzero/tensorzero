//! Shared helpers for persisting prompt templates to
//! `tensorzero.prompt_template_configs`.
//!
//! Both the function-config write path (`function_config_writes`) and the
//! whole-config stored-config write path (`stored_config_writes`) need to
//! insert prompt templates with the same semantics:
//!
//! 1. Collect templates into a `BTreeMap` keyed by template key, erroring on
//!    conflicting bodies for the same key.
//! 2. Reuse existing rows that already match `(template_key, content_hash)`
//!    so that idempotent rewrites produce stable version IDs.
//! 3. Batch-insert only the genuinely new rows.
//!
//! The file-specific logic (walking a function config tree, merging
//! filesystem-discovered extra templates, or iterating a tool/evaluation's
//! `prompt_templates_for_db()`) lives in the callers; this module only owns
//! the collection and SQL dedup.

use std::collections::{BTreeMap, HashMap};

use sqlx::{FromRow, Postgres, QueryBuilder, Transaction};
use uuid::Uuid;

use crate::config::path::ResolvedTomlPathData;
use crate::error::{Error, ErrorDetails};

#[derive(Debug)]
pub(super) struct CollectedPromptTemplate {
    pub source_body: String,
}

#[derive(Debug, FromRow)]
struct ExistingTemplateRow {
    template_key: String,
    id: Uuid,
}

/// Insert `template` into `templates`, erroring if a different body was
/// already recorded under the same key.
pub(super) fn add_prompt_template(
    templates: &mut BTreeMap<String, CollectedPromptTemplate>,
    template: &ResolvedTomlPathData,
) -> Result<(), Error> {
    let template_key = template.get_template_key();
    let source_body = template.data().to_string();
    match templates.get(&template_key) {
        Some(existing) if existing.source_body != source_body => {
            Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "Template key `{template_key}` was provided with conflicting source bodies."
                ),
            }))
        }
        Some(_) => Ok(()),
        None => {
            templates.insert(template_key, CollectedPromptTemplate { source_body });
            Ok(())
        }
    }
}

/// Persist a collected set of prompt templates, reusing any existing rows
/// that already match `(template_key, content_hash)` and batch-inserting
/// only the genuinely new ones. Returns `template_key -> version_id`.
pub(super) async fn write_collected_prompt_templates(
    tx: &mut Transaction<'_, Postgres>,
    templates: &BTreeMap<String, CollectedPromptTemplate>,
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

    // Batch lookup: find existing template versions with matching
    // `(template_key, content_hash)`.
    let template_keys: Vec<&str> = templates.keys().map(String::as_str).collect();
    let content_hashes: Vec<&[u8]> = template_keys
        .iter()
        .map(|k| hashes[*k].as_slice())
        .collect();
    let existing_rows: Vec<ExistingTemplateRow> = sqlx::query_as(
        "SELECT input.template_key, t.id \
         FROM UNNEST($1::text[], $2::bytea[]) AS input(template_key, content_hash) \
         JOIN tensorzero.prompt_template_configs t \
           ON t.template_key = input.template_key AND t.content_hash = input.content_hash",
    )
    .bind(&template_keys)
    .bind(&content_hashes)
    .fetch_all(&mut **tx)
    .await
    .map_err(|e| postgres_query_error("Failed to look up existing prompt template versions", e))?;

    let existing: HashMap<String, Uuid> = existing_rows
        .into_iter()
        .map(|row| (row.template_key, row.id))
        .collect();

    // Partition into reused vs. new templates.
    let mut template_version_ids = HashMap::with_capacity(templates.len());
    let mut new_template_keys = Vec::new();
    for template_key in templates.keys() {
        if let Some(&existing_id) = existing.get(template_key) {
            template_version_ids.insert(template_key.clone(), existing_id);
        } else {
            let id = Uuid::now_v7();
            template_version_ids.insert(template_key.clone(), id);
            new_template_keys.push(template_key.clone());
        }
    }

    // Batch insert only new templates.
    if !new_template_keys.is_empty() {
        let mut qb = QueryBuilder::new(
            "INSERT INTO tensorzero.prompt_template_configs \
             (id, template_key, source_body, content_hash, creation_source, source_autopilot_session_id) ",
        );
        qb.push_values(&new_template_keys, |mut b, key| {
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
            .map_err(|e| postgres_query_error("Failed to insert prompt template versions", e))?;
    }

    Ok(template_version_ids)
}

fn postgres_query_error(context: &str, error: impl std::fmt::Display) -> Error {
    Error::new(ErrorDetails::PostgresQuery {
        message: format!("{context}: {error}"),
    })
}
