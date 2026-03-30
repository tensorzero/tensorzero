use std::collections::HashMap;
use std::sync::Arc;

use sqlx::PgPool;
use uuid::Uuid;

use crate::config::namespace::Namespace;
use crate::config::path::ResolvedTomlPathData;
use crate::config::{
    NonStreamingTimeouts, StreamingTimeouts, TimeoutsConfig, UninitializedVariantConfig,
    UninitializedVariantInfo,
};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::extra_body::ExtraBodyConfig;
use crate::inference::types::extra_headers::ExtraHeadersConfig;
use crate::utils::retries::RetryConfig;
use crate::variant::chat_completion::{
    UninitializedChatCompletionConfig, UninitializedChatTemplate, UninitializedChatTemplates,
};
use tensorzero_types::inference_params::{JsonMode, ServiceTier};

// ---- Row types ----

#[derive(Debug, sqlx::FromRow)]
pub struct PromptTemplateVersionRow {
    pub id: Uuid,
    pub template_key: String,
    pub source_body: String,
    pub creation_source: String,
    pub source_autopilot_session_id: Option<Uuid>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct PromptTemplateVersionDependencyRow {
    pub prompt_template_version_id: Uuid,
    pub dependency_prompt_template_version_id: Uuid,
    pub dependency_key: String,
}

#[derive(Debug, sqlx::FromRow)]
pub struct VariantVersionRow {
    pub id: Uuid,
    pub variant_type: String,
    pub weight: Option<f64>,
    pub timeouts_non_streaming_total_ms: Option<i64>,
    pub timeouts_streaming_ttft_ms: Option<i64>,
    pub timeouts_streaming_total_ms: Option<i64>,
    pub namespace: Option<String>,
    pub function_name: Option<String>,
    pub variant_name: Option<String>,
    pub creation_source: String,
    pub source_autopilot_session_id: Option<Uuid>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct ChatCompletionConfigRow {
    pub variant_version_id: Uuid,
    pub model: String,
    pub system_template_prompt_id: Option<Uuid>,
    pub system_template_key: Option<String>,
    pub user_template_prompt_id: Option<Uuid>,
    pub user_template_key: Option<String>,
    pub assistant_template_prompt_id: Option<Uuid>,
    pub assistant_template_key: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<i32>,
    pub stop_sequences: Option<Vec<String>>,
    pub reasoning_effort: Option<String>,
    pub service_tier: Option<String>,
    pub thinking_budget_tokens: Option<i32>,
    pub verbosity: Option<String>,
    pub json_mode: Option<String>,
    pub num_retries: i32,
    pub max_retry_delay_s: f32,
}

#[derive(Debug, sqlx::FromRow)]
pub struct ExtraBodyRow {
    pub variant_version_id: Uuid,
    pub position: i32,
    pub pointer: String,
    pub kind: String,
    pub replacement_value: Option<serde_json::Value>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct ExtraHeaderRow {
    pub variant_version_id: Uuid,
    pub position: i32,
    pub header_name: String,
    pub kind: String,
    pub header_value: Option<String>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct ChatCompletionTemplateRow {
    pub variant_version_id: Uuid,
    pub template_name: String,
    pub prompt_template_version_id: Uuid,
    pub template_key: String,
}

#[derive(Debug, sqlx::FromRow)]
pub struct BestOfNConfigRow {
    pub variant_version_id: Uuid,
    pub evaluator_variant_version_id: Uuid,
}

#[derive(Debug, sqlx::FromRow)]
pub struct BestOfNCandidateRow {
    pub variant_version_id: Uuid,
    pub candidate_name: String,
    pub position: i32,
}

#[derive(Debug, sqlx::FromRow)]
pub struct MixtureOfNConfigRow {
    pub variant_version_id: Uuid,
    pub fuser_variant_version_id: Uuid,
}

#[derive(Debug, sqlx::FromRow)]
pub struct MixtureOfNCandidateRow {
    pub variant_version_id: Uuid,
    pub candidate_name: String,
    pub position: i32,
}

#[derive(Debug, sqlx::FromRow)]
pub struct DiclConfigRow {
    pub variant_version_id: Uuid,
    pub embedding_model: String,
    pub k: i32,
    pub model: String,
    pub system_instructions_prompt_id: Option<Uuid>,
    pub system_instructions_key: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub max_tokens: Option<i32>,
    pub seed: Option<i32>,
    pub reasoning_effort: Option<String>,
    pub thinking_budget_tokens: Option<i32>,
    pub verbosity: Option<String>,
    pub json_mode: Option<String>,
    pub num_retries: i32,
    pub max_retry_delay_s: f32,
    pub max_distance: Option<f32>,
}

// ---- Helper: extract prompt template references from a variant ----

/// Collects all `ResolvedTomlPathData` references from a variant so they can be stored
/// as `prompt_template_versions` rows.
pub(crate) struct PromptTemplateCollector {
    pub(crate) templates: HashMap<String, (String, String)>, // template_key -> (template_key, source_body)
}

impl PromptTemplateCollector {
    pub(crate) fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    pub(crate) fn add(&mut self, path_data: &ResolvedTomlPathData) {
        let key = path_data.get_template_key();
        let body = path_data.data().to_string();
        self.templates.insert(key.clone(), (key, body));
    }

    pub(crate) fn add_option(&mut self, path_data: Option<&ResolvedTomlPathData>) {
        if let Some(pd) = path_data {
            self.add(pd);
        }
    }
}

// ---- Write path ----

/// Inserts a prompt template version row and returns its UUID.
/// If a template with the same key is already tracked, reuses the existing UUID.
pub(crate) async fn insert_prompt_template_version(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    id: Uuid,
    template_key: &str,
    source_body: &str,
    creation_source: &str,
) -> Result<(), Error> {
    sqlx::query(
        r"INSERT INTO tensorzero.prompt_template_versions (id, template_key, source_body, creation_source)
           VALUES ($1, $2, $3, $4)",
    )
    .bind(id)
    .bind(template_key)
    .bind(source_body)
    .bind(creation_source)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Tracks prompt template versions needed for a variant, assigning UUIDs.
/// Returns a map from template_key -> UUID.
async fn write_prompt_templates_for_chat_completion(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    config: &UninitializedChatCompletionConfig,
    creation_source: &str,
) -> Result<HashMap<String, Uuid>, Error> {
    let mut collector = PromptTemplateCollector::new();
    collector.add_option(config.system_template.as_ref());
    collector.add_option(config.user_template.as_ref());
    collector.add_option(config.assistant_template.as_ref());
    for template in config.templates.inner.values() {
        collector.add(&template.path);
    }

    let mut key_to_id: HashMap<String, Uuid> = HashMap::new();
    for (key, (_template_key, source_body)) in &collector.templates {
        let id = Uuid::now_v7();
        insert_prompt_template_version(tx, id, key, source_body, creation_source).await?;
        key_to_id.insert(key.clone(), id);
    }
    Ok(key_to_id)
}

pub(crate) fn prompt_id_for(
    key_to_id: &HashMap<String, Uuid>,
    path_data: Option<&ResolvedTomlPathData>,
) -> Option<Uuid> {
    path_data.and_then(|pd| key_to_id.get(&pd.get_template_key()).copied())
}

pub(crate) fn prompt_key_for(path_data: Option<&ResolvedTomlPathData>) -> Option<String> {
    path_data.map(|pd| pd.get_template_key())
}

/// Writes a chat_completion config row (used for chat_completion, chain_of_thought, evaluator, fuser).
async fn write_chat_completion_config(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    variant_version_id: Uuid,
    config: &UninitializedChatCompletionConfig,
    key_to_id: &HashMap<String, Uuid>,
) -> Result<(), Error> {
    let service_tier_str = config.service_tier.as_ref().map(|st| {
        serde_json::to_value(st)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default()
    });
    let json_mode_str = config.json_mode.as_ref().map(|jm| {
        serde_json::to_value(jm)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default()
    });

    sqlx::query(
        r"INSERT INTO tensorzero.variant_chat_completion_configs (
            variant_version_id, model,
            system_template_prompt_id, system_template_key,
            user_template_prompt_id, user_template_key,
            assistant_template_prompt_id, assistant_template_key,
            temperature, top_p, max_tokens, presence_penalty, frequency_penalty,
            seed, stop_sequences, reasoning_effort, service_tier,
            thinking_budget_tokens, verbosity, json_mode,
            num_retries, max_retry_delay_s
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
            $14, $15, $16, $17, $18, $19, $20, $21, $22
        )",
    )
    .bind(variant_version_id)
    .bind(config.model.as_ref())
    .bind(prompt_id_for(key_to_id, config.system_template.as_ref()))
    .bind(prompt_key_for(config.system_template.as_ref()))
    .bind(prompt_id_for(key_to_id, config.user_template.as_ref()))
    .bind(prompt_key_for(config.user_template.as_ref()))
    .bind(prompt_id_for(key_to_id, config.assistant_template.as_ref()))
    .bind(prompt_key_for(config.assistant_template.as_ref()))
    .bind(config.temperature)
    .bind(config.top_p)
    .bind(config.max_tokens.map(|v| v as i32))
    .bind(config.presence_penalty)
    .bind(config.frequency_penalty)
    .bind(config.seed.map(|v| v as i32))
    .bind(&config.stop_sequences)
    .bind(&config.reasoning_effort)
    .bind(&service_tier_str)
    .bind(config.thinking_budget_tokens)
    .bind(&config.verbosity)
    .bind(&json_mode_str)
    .bind(config.retries.num_retries as i32)
    .bind(config.retries.max_delay_s)
    .execute(&mut **tx)
    .await?;

    // Write extra_body rows
    write_extra_body_rows(
        tx,
        variant_version_id,
        config.extra_body.as_ref(),
        INSERT_CC_EXTRA_BODY,
    )
    .await?;

    // Write extra_headers rows
    write_extra_headers_rows(
        tx,
        variant_version_id,
        config.extra_headers.as_ref(),
        INSERT_CC_EXTRA_HEADERS,
    )
    .await?;

    // Write named templates
    for (template_name, template) in &config.templates.inner {
        let prompt_id = key_to_id
            .get(&template.path.get_template_key())
            .ok_or_else(|| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Missing prompt template ID for template `{template_name}`"),
                })
            })?;
        sqlx::query(
            r"INSERT INTO tensorzero.variant_chat_completion_templates
               (variant_version_id, template_name, prompt_template_version_id, template_key)
               VALUES ($1, $2, $3, $4)",
        )
        .bind(variant_version_id)
        .bind(template_name)
        .bind(prompt_id)
        .bind(template.path.get_template_key())
        .execute(&mut **tx)
        .await?;
    }

    Ok(())
}

/// Writes extra_body replacement rows using the provided SQL insert statement.
pub(crate) async fn write_extra_body_rows(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    variant_version_id: Uuid,
    extra_body: Option<&ExtraBodyConfig>,
    insert_sql: &'static str,
) -> Result<(), Error> {
    use crate::inference::types::extra_body::ExtraBodyReplacementKind;

    let Some(eb) = extra_body else {
        return Ok(());
    };
    for (i, replacement) in eb.data.iter().enumerate() {
        let (kind, value) = match &replacement.kind {
            ExtraBodyReplacementKind::Value(v) => ("value", Some(v.clone())),
            ExtraBodyReplacementKind::Delete => ("delete", None),
        };
        sqlx::query(insert_sql)
            .bind(variant_version_id)
            .bind(i as i32)
            .bind(&replacement.pointer)
            .bind(kind)
            .bind(&value)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

/// Writes extra_headers rows using the provided SQL insert statement.
pub(crate) async fn write_extra_headers_rows(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    variant_version_id: Uuid,
    extra_headers: Option<&ExtraHeadersConfig>,
    insert_sql: &'static str,
) -> Result<(), Error> {
    use crate::inference::types::extra_headers::ExtraHeaderKind;

    let Some(eh) = extra_headers else {
        return Ok(());
    };
    for (i, header) in eh.data.iter().enumerate() {
        let (kind, value) = match &header.kind {
            ExtraHeaderKind::Value(v) => ("value", Some(v.as_str())),
            ExtraHeaderKind::Delete => ("delete", None),
        };
        sqlx::query(insert_sql)
            .bind(variant_version_id)
            .bind(i as i32)
            .bind(&header.name)
            .bind(kind)
            .bind(value)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

const INSERT_CC_EXTRA_BODY: &str = "INSERT INTO tensorzero.variant_chat_completion_extra_body (variant_version_id, position, pointer, kind, replacement_value) VALUES ($1, $2, $3, $4, $5)";
const INSERT_CC_EXTRA_HEADERS: &str = "INSERT INTO tensorzero.variant_chat_completion_extra_headers (variant_version_id, position, header_name, kind, header_value) VALUES ($1, $2, $3, $4, $5)";
const INSERT_DICL_EXTRA_BODY: &str = "INSERT INTO tensorzero.variant_dicl_extra_body (variant_version_id, position, pointer, kind, replacement_value) VALUES ($1, $2, $3, $4, $5)";
const INSERT_DICL_EXTRA_HEADERS: &str = "INSERT INTO tensorzero.variant_dicl_extra_headers (variant_version_id, position, header_name, kind, header_value) VALUES ($1, $2, $3, $4, $5)";

/// Writes a variant version row (the common part).
async fn write_variant_version_row(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    id: Uuid,
    variant_type: &str,
    info: &UninitializedVariantInfo,
    function_name: Option<&str>,
    variant_name: Option<&str>,
    creation_source: &str,
) -> Result<(), Error> {
    let weight = match &info.inner {
        UninitializedVariantConfig::ChatCompletion(c) => c.weight,
        UninitializedVariantConfig::BestOfNSampling(c) => c.weight,
        UninitializedVariantConfig::MixtureOfN(c) => c.weight,
        UninitializedVariantConfig::Dicl(c) => c.weight,
        UninitializedVariantConfig::ChainOfThought(c) => c.inner.weight,
    };

    let (non_streaming_total_ms, streaming_ttft_ms, streaming_total_ms) = match &info.timeouts {
        Some(t) => (
            t.non_streaming.total_ms.map(|v| v as i64),
            t.streaming.ttft_ms.map(|v| v as i64),
            t.streaming.total_ms.map(|v| v as i64),
        ),
        None => (None, None, None),
    };

    sqlx::query(
        r"INSERT INTO tensorzero.variant_versions (
            id, variant_type, weight,
            timeouts_non_streaming_total_ms, timeouts_streaming_ttft_ms, timeouts_streaming_total_ms,
            namespace, function_name, variant_name, creation_source
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
    )
    .bind(id)
    .bind(variant_type)
    .bind(weight)
    .bind(non_streaming_total_ms)
    .bind(streaming_ttft_ms)
    .bind(streaming_total_ms)
    .bind(info.namespace.as_ref().map(|n| n.as_ref()))
    .bind(function_name)
    .bind(variant_name)
    .bind(creation_source)
    .execute(&mut **tx)
    .await?;

    Ok(())
}

/// Writes a variant version to the database with function/variant names, returning the variant_version_id.
pub async fn write_variant_version(
    pool: &PgPool,
    function_name: &str,
    variant_name: &str,
    info: &UninitializedVariantInfo,
    creation_source: &str,
) -> Result<Uuid, Error> {
    let mut tx = pool.begin().await?;
    let variant_version_id = write_variant_version_in_tx(
        &mut tx,
        info,
        Some(function_name),
        Some(variant_name),
        creation_source,
    )
    .await?;
    tx.commit().await?;
    Ok(variant_version_id)
}

/// Writes a variant version within an existing transaction.
pub async fn write_variant_version_in_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    info: &UninitializedVariantInfo,
    function_name: Option<&str>,
    variant_name: Option<&str>,
    creation_source: &str,
) -> Result<Uuid, Error> {
    let variant_version_id = Uuid::now_v7();

    match &info.inner {
        UninitializedVariantConfig::ChatCompletion(config) => {
            write_variant_version_row(
                tx,
                variant_version_id,
                "chat_completion",
                info,
                function_name,
                variant_name,
                creation_source,
            )
            .await?;
            let key_to_id =
                write_prompt_templates_for_chat_completion(tx, config, creation_source).await?;
            write_chat_completion_config(tx, variant_version_id, config, &key_to_id).await?;
        }
        UninitializedVariantConfig::ChainOfThought(config) => {
            write_variant_version_row(
                tx,
                variant_version_id,
                "chain_of_thought",
                info,
                function_name,
                variant_name,
                creation_source,
            )
            .await?;
            let key_to_id =
                write_prompt_templates_for_chat_completion(tx, &config.inner, creation_source)
                    .await?;
            write_chat_completion_config(tx, variant_version_id, &config.inner, &key_to_id).await?;
        }
        UninitializedVariantConfig::BestOfNSampling(config) => {
            write_variant_version_row(
                tx,
                variant_version_id,
                "best_of_n_sampling",
                info,
                function_name,
                variant_name,
                creation_source,
            )
            .await?;

            // Write evaluator as a separate variant_version (no name — internal sub-variant)
            let evaluator_id = Uuid::now_v7();
            let eval_info = UninitializedVariantInfo {
                inner: UninitializedVariantConfig::ChatCompletion(config.evaluator.inner.clone()),
                timeouts: None,
                namespace: None,
            };
            write_variant_version_row(
                tx,
                evaluator_id,
                "chat_completion",
                &eval_info,
                None,
                None,
                creation_source,
            )
            .await?;
            let eval_key_to_id = write_prompt_templates_for_chat_completion(
                tx,
                &config.evaluator.inner,
                creation_source,
            )
            .await?;
            write_chat_completion_config(
                tx,
                evaluator_id,
                &config.evaluator.inner,
                &eval_key_to_id,
            )
            .await?;

            sqlx::query(
                r"INSERT INTO tensorzero.variant_best_of_n_configs (variant_version_id, evaluator_variant_version_id)
                   VALUES ($1, $2)",
            )
            .bind(variant_version_id)
            .bind(evaluator_id)
            .execute(&mut **tx)
            .await?;

            for (i, candidate) in config.candidates.iter().enumerate() {
                sqlx::query(
                    r"INSERT INTO tensorzero.variant_best_of_n_candidates (variant_version_id, candidate_name, position)
                       VALUES ($1, $2, $3)",
                )
                .bind(variant_version_id)
                .bind(candidate)
                .bind(i as i32)
                .execute(&mut **tx)
                .await?;
            }
        }
        UninitializedVariantConfig::MixtureOfN(config) => {
            write_variant_version_row(
                tx,
                variant_version_id,
                "mixture_of_n",
                info,
                function_name,
                variant_name,
                creation_source,
            )
            .await?;

            // Write fuser as a separate variant_version (no name — internal sub-variant)
            let fuser_id = Uuid::now_v7();
            let fuser_info = UninitializedVariantInfo {
                inner: UninitializedVariantConfig::ChatCompletion(config.fuser.inner.clone()),
                timeouts: None,
                namespace: None,
            };
            write_variant_version_row(
                tx,
                fuser_id,
                "chat_completion",
                &fuser_info,
                None,
                None,
                creation_source,
            )
            .await?;
            let fuser_key_to_id = write_prompt_templates_for_chat_completion(
                tx,
                &config.fuser.inner,
                creation_source,
            )
            .await?;
            write_chat_completion_config(tx, fuser_id, &config.fuser.inner, &fuser_key_to_id)
                .await?;

            sqlx::query(
                r"INSERT INTO tensorzero.variant_mixture_of_n_configs (variant_version_id, fuser_variant_version_id)
                   VALUES ($1, $2)",
            )
            .bind(variant_version_id)
            .bind(fuser_id)
            .execute(&mut **tx)
            .await?;

            for (i, candidate) in config.candidates.iter().enumerate() {
                sqlx::query(
                    r"INSERT INTO tensorzero.variant_mixture_of_n_candidates (variant_version_id, candidate_name, position)
                       VALUES ($1, $2, $3)",
                )
                .bind(variant_version_id)
                .bind(candidate)
                .bind(i as i32)
                .execute(&mut **tx)
                .await?;
            }
        }
        UninitializedVariantConfig::Dicl(config) => {
            write_variant_version_row(
                tx,
                variant_version_id,
                "dicl",
                info,
                function_name,
                variant_name,
                creation_source,
            )
            .await?;

            // Write prompt templates for DICL
            let mut key_to_id: HashMap<String, Uuid> = HashMap::new();
            if let Some(ref si) = config.system_instructions {
                let id = Uuid::now_v7();
                insert_prompt_template_version(
                    tx,
                    id,
                    &si.get_template_key(),
                    si.data(),
                    creation_source,
                )
                .await?;
                key_to_id.insert(si.get_template_key(), id);
            }

            let json_mode_str = config.json_mode.as_ref().map(|jm| {
                serde_json::to_value(jm)
                    .ok()
                    .and_then(|v| v.as_str().map(String::from))
                    .unwrap_or_default()
            });

            sqlx::query(
                r"INSERT INTO tensorzero.variant_dicl_configs (
                    variant_version_id, embedding_model, k, model,
                    system_instructions_prompt_id, system_instructions_key,
                    temperature, top_p, stop_sequences, presence_penalty, frequency_penalty,
                    max_tokens, seed, reasoning_effort, thinking_budget_tokens, verbosity,
                    json_mode, num_retries, max_retry_delay_s, max_distance
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
                    $17, $18, $19, $20
                )",
            )
            .bind(variant_version_id)
            .bind(&config.embedding_model)
            .bind(config.k as i32)
            .bind(&config.model)
            .bind(
                config
                    .system_instructions
                    .as_ref()
                    .and_then(|si| key_to_id.get(&si.get_template_key()).copied()),
            )
            .bind(
                config
                    .system_instructions
                    .as_ref()
                    .map(|si| si.get_template_key()),
            )
            .bind(config.temperature)
            .bind(config.top_p)
            .bind(&config.stop_sequences)
            .bind(config.presence_penalty)
            .bind(config.frequency_penalty)
            .bind(config.max_tokens.map(|v| v as i32))
            .bind(config.seed.map(|v| v as i32))
            .bind(&config.reasoning_effort)
            .bind(config.thinking_budget_tokens)
            .bind(&config.verbosity)
            .bind(&json_mode_str)
            .bind(config.retries.num_retries as i32)
            .bind(config.retries.max_delay_s)
            .bind(config.max_distance)
            .execute(&mut **tx)
            .await?;

            // Write extra_body and extra_headers to normalized tables
            write_extra_body_rows(
                tx,
                variant_version_id,
                config.extra_body.as_ref(),
                INSERT_DICL_EXTRA_BODY,
            )
            .await?;
            write_extra_headers_rows(
                tx,
                variant_version_id,
                config.extra_headers.as_ref(),
                INSERT_DICL_EXTRA_HEADERS,
            )
            .await?;
        }
    }

    Ok(variant_version_id)
}

// ---- Read path ----

pub(crate) fn rehydrate_prompt_ref(
    prompt_id: Option<Uuid>,
    template_key: Option<&String>,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<Option<ResolvedTomlPathData>, Error> {
    match (prompt_id, template_key) {
        (Some(pid), Some(key)) => {
            let row = prompt_rows.get(&pid).ok_or_else(|| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Missing prompt template version row for ID `{pid}`"),
                })
            })?;
            Ok(Some(ResolvedTomlPathData::new_fake_path(
                key.clone(),
                row.source_body.clone(),
            )))
        }
        (None, None) => Ok(None),
        _ => Err(Error::new(ErrorDetails::InternalError {
            message: "Inconsistent prompt template reference: one of ID/key is null but not both"
                .to_string(),
        })),
    }
}

fn rehydrate_timeouts(row: &VariantVersionRow) -> Option<TimeoutsConfig> {
    let non_streaming_total_ms = row.timeouts_non_streaming_total_ms.map(|v| v as u64);
    let streaming_ttft_ms = row.timeouts_streaming_ttft_ms.map(|v| v as u64);
    let streaming_total_ms = row.timeouts_streaming_total_ms.map(|v| v as u64);

    if non_streaming_total_ms.is_none()
        && streaming_ttft_ms.is_none()
        && streaming_total_ms.is_none()
    {
        return None;
    }

    Some(TimeoutsConfig {
        non_streaming: NonStreamingTimeouts {
            total_ms: non_streaming_total_ms,
        },
        streaming: StreamingTimeouts {
            ttft_ms: streaming_ttft_ms,
            total_ms: streaming_total_ms,
        },
    })
}

fn rehydrate_chat_completion_config(
    cc_row: &ChatCompletionConfigRow,
    template_rows: &[ChatCompletionTemplateRow],
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
    extra_body_rows: &[ExtraBodyRow],
    extra_header_rows: &[ExtraHeaderRow],
) -> Result<UninitializedChatCompletionConfig, Error> {
    let system_template = rehydrate_prompt_ref(
        cc_row.system_template_prompt_id,
        cc_row.system_template_key.as_ref(),
        prompt_rows,
    )?;
    let user_template = rehydrate_prompt_ref(
        cc_row.user_template_prompt_id,
        cc_row.user_template_key.as_ref(),
        prompt_rows,
    )?;
    let assistant_template = rehydrate_prompt_ref(
        cc_row.assistant_template_prompt_id,
        cc_row.assistant_template_key.as_ref(),
        prompt_rows,
    )?;

    let mut templates_map = HashMap::new();
    for t_row in template_rows {
        let prompt_row = prompt_rows
            .get(&t_row.prompt_template_version_id)
            .ok_or_else(|| {
                Error::new(ErrorDetails::InternalError {
                    message: format!(
                        "Missing prompt template version row for template `{}`",
                        t_row.template_name
                    ),
                })
            })?;
        templates_map.insert(
            t_row.template_name.clone(),
            UninitializedChatTemplate {
                path: ResolvedTomlPathData::new_fake_path(
                    t_row.template_key.clone(),
                    prompt_row.source_body.clone(),
                ),
            },
        );
    }

    let service_tier: Option<ServiceTier> = cc_row
        .service_tier
        .as_deref()
        .map(|s| {
            serde_json::from_value(serde_json::Value::String(s.to_string())).map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Failed to parse service_tier `{s}`: {e}"),
                })
            })
        })
        .transpose()?;

    let json_mode: Option<JsonMode> = cc_row
        .json_mode
        .as_deref()
        .map(|s| {
            serde_json::from_value(serde_json::Value::String(s.to_string())).map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Failed to parse json_mode `{s}`: {e}"),
                })
            })
        })
        .transpose()?;

    let extra_body = rehydrate_extra_body(extra_body_rows)?;
    let extra_headers = rehydrate_extra_headers(extra_header_rows)?;

    Ok(UninitializedChatCompletionConfig {
        weight: None, // weight is stored on variant_versions, not here
        model: Arc::from(cc_row.model.as_str()),
        system_template,
        user_template,
        assistant_template,
        input_wrappers: None, // deprecated, not stored
        templates: UninitializedChatTemplates {
            inner: templates_map,
        },
        temperature: cc_row.temperature,
        top_p: cc_row.top_p,
        max_tokens: cc_row.max_tokens.map(|v| v as u32),
        presence_penalty: cc_row.presence_penalty,
        frequency_penalty: cc_row.frequency_penalty,
        seed: cc_row.seed.map(|v| v as u32),
        stop_sequences: cc_row.stop_sequences.clone(),
        reasoning_effort: cc_row.reasoning_effort.clone(),
        service_tier,
        thinking_budget_tokens: cc_row.thinking_budget_tokens,
        verbosity: cc_row.verbosity.clone(),
        json_mode,
        retries: RetryConfig {
            num_retries: cc_row.num_retries as usize,
            max_delay_s: cc_row.max_retry_delay_s,
        },
        extra_body,
        extra_headers,
    })
}

pub(crate) fn rehydrate_extra_body(
    rows: &[ExtraBodyRow],
) -> Result<Option<ExtraBodyConfig>, Error> {
    use crate::inference::types::extra_body::{ExtraBodyReplacement, ExtraBodyReplacementKind};

    if rows.is_empty() {
        return Ok(None);
    }
    let mut data = Vec::with_capacity(rows.len());
    for row in rows {
        let kind = match row.kind.as_str() {
            "value" => {
                let v = row.replacement_value.clone().ok_or_else(|| {
                    Error::new(ErrorDetails::InternalError {
                        message: "extra_body row with kind='value' missing replacement_value"
                            .to_string(),
                    })
                })?;
                ExtraBodyReplacementKind::Value(v)
            }
            "delete" => ExtraBodyReplacementKind::Delete,
            other => {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: format!("Unknown extra_body kind: `{other}`"),
                }));
            }
        };
        data.push(ExtraBodyReplacement {
            pointer: row.pointer.clone(),
            kind,
        });
    }
    Ok(Some(ExtraBodyConfig { data }))
}

pub(crate) fn rehydrate_extra_headers(
    rows: &[ExtraHeaderRow],
) -> Result<Option<ExtraHeadersConfig>, Error> {
    use crate::inference::types::extra_headers::{ExtraHeader, ExtraHeaderKind};

    if rows.is_empty() {
        return Ok(None);
    }
    let mut data = Vec::with_capacity(rows.len());
    for row in rows {
        let kind = match row.kind.as_str() {
            "value" => {
                let v = row.header_value.clone().ok_or_else(|| {
                    Error::new(ErrorDetails::InternalError {
                        message: "extra_headers row with kind='value' missing header_value"
                            .to_string(),
                    })
                })?;
                ExtraHeaderKind::Value(v)
            }
            "delete" => ExtraHeaderKind::Delete,
            other => {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: format!("Unknown extra_headers kind: `{other}`"),
                }));
            }
        };
        data.push(ExtraHeader {
            name: row.header_name.clone(),
            kind,
        });
    }
    Ok(Some(ExtraHeadersConfig { data }))
}

/// Reads a variant version from the database and rehydrates it to `UninitializedVariantInfo`.
pub async fn read_variant_version(
    pool: &PgPool,
    variant_version_id: Uuid,
) -> Result<UninitializedVariantInfo, Error> {
    // Load the variant_versions row
    let variant_row: VariantVersionRow = sqlx::query_as(
        r"SELECT id, variant_type, weight,
                  timeouts_non_streaming_total_ms, timeouts_streaming_ttft_ms, timeouts_streaming_total_ms,
                  namespace, function_name, variant_name,
                  creation_source, source_autopilot_session_id
           FROM tensorzero.variant_versions
           WHERE id = $1",
    )
    .bind(variant_version_id)
    .fetch_one(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to load variant version `{variant_version_id}`: {e}"),
        })
    })?;

    let timeouts = rehydrate_timeouts(&variant_row);
    let namespace = variant_row
        .namespace
        .clone()
        .map(Namespace::new)
        .transpose()?;

    let inner = match variant_row.variant_type.as_str() {
        "chat_completion" => {
            let (cc_config, _weight) = read_chat_completion_inner(pool, variant_version_id).await?;
            UninitializedVariantConfig::ChatCompletion(UninitializedChatCompletionConfig {
                weight: variant_row.weight,
                ..cc_config
            })
        }
        "chain_of_thought" => {
            let (cc_config, _weight) = read_chat_completion_inner(pool, variant_version_id).await?;
            use crate::variant::chain_of_thought::UninitializedChainOfThoughtConfig;
            UninitializedVariantConfig::ChainOfThought(UninitializedChainOfThoughtConfig {
                inner: UninitializedChatCompletionConfig {
                    weight: variant_row.weight,
                    ..cc_config
                },
            })
        }
        "best_of_n_sampling" => {
            let bon_row: BestOfNConfigRow = sqlx::query_as(
                r"SELECT variant_version_id, evaluator_variant_version_id
                   FROM tensorzero.variant_best_of_n_configs
                   WHERE variant_version_id = $1",
            )
            .bind(variant_version_id)
            .fetch_one(pool)
            .await?;

            let candidate_rows: Vec<BestOfNCandidateRow> = sqlx::query_as(
                r"SELECT variant_version_id, candidate_name, position
                   FROM tensorzero.variant_best_of_n_candidates
                   WHERE variant_version_id = $1
                   ORDER BY position",
            )
            .bind(variant_version_id)
            .fetch_all(pool)
            .await?;

            let (eval_cc, _) =
                read_chat_completion_inner(pool, bon_row.evaluator_variant_version_id).await?;

            use crate::variant::best_of_n_sampling::{
                UninitializedBestOfNEvaluatorConfig, UninitializedBestOfNSamplingConfig,
            };
            #[expect(deprecated)]
            let config = UninitializedBestOfNSamplingConfig {
                weight: variant_row.weight,
                timeout_s: None,
                candidates: candidate_rows
                    .iter()
                    .map(|c| c.candidate_name.clone())
                    .collect(),
                evaluator: UninitializedBestOfNEvaluatorConfig { inner: eval_cc },
            };
            UninitializedVariantConfig::BestOfNSampling(config)
        }
        "mixture_of_n" => {
            let mon_row: MixtureOfNConfigRow = sqlx::query_as(
                r"SELECT variant_version_id, fuser_variant_version_id
                   FROM tensorzero.variant_mixture_of_n_configs
                   WHERE variant_version_id = $1",
            )
            .bind(variant_version_id)
            .fetch_one(pool)
            .await?;

            let candidate_rows: Vec<MixtureOfNCandidateRow> = sqlx::query_as(
                r"SELECT variant_version_id, candidate_name, position
                   FROM tensorzero.variant_mixture_of_n_candidates
                   WHERE variant_version_id = $1
                   ORDER BY position",
            )
            .bind(variant_version_id)
            .fetch_all(pool)
            .await?;

            let (fuser_cc, _) =
                read_chat_completion_inner(pool, mon_row.fuser_variant_version_id).await?;

            use crate::variant::mixture_of_n::{
                UninitializedFuserConfig, UninitializedMixtureOfNConfig,
            };
            #[expect(deprecated)]
            let config = UninitializedMixtureOfNConfig {
                weight: variant_row.weight,
                timeout_s: None,
                candidates: candidate_rows
                    .iter()
                    .map(|c| c.candidate_name.clone())
                    .collect(),
                fuser: UninitializedFuserConfig { inner: fuser_cc },
            };
            UninitializedVariantConfig::MixtureOfN(config)
        }
        "dicl" => read_dicl_inner(pool, variant_version_id, &variant_row).await?,
        other => {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!("Unknown variant type `{other}`"),
            }));
        }
    };

    Ok(UninitializedVariantInfo {
        inner,
        timeouts,
        namespace,
    })
}

/// Reads a chat_completion config + its prompt templates. Returns the config.
async fn read_chat_completion_inner(
    pool: &PgPool,
    variant_version_id: Uuid,
) -> Result<(UninitializedChatCompletionConfig, Option<f64>), Error> {
    let cc_row: ChatCompletionConfigRow = sqlx::query_as(
        r"SELECT variant_version_id, model,
                  system_template_prompt_id, system_template_key,
                  user_template_prompt_id, user_template_key,
                  assistant_template_prompt_id, assistant_template_key,
                  temperature, top_p, max_tokens, presence_penalty, frequency_penalty,
                  seed, stop_sequences, reasoning_effort, service_tier,
                  thinking_budget_tokens, verbosity, json_mode,
                  num_retries, max_retry_delay_s
           FROM tensorzero.variant_chat_completion_configs
           WHERE variant_version_id = $1",
    )
    .bind(variant_version_id)
    .fetch_one(pool)
    .await?;

    let template_rows: Vec<ChatCompletionTemplateRow> = sqlx::query_as(
        r"SELECT variant_version_id, template_name, prompt_template_version_id, template_key
           FROM tensorzero.variant_chat_completion_templates
           WHERE variant_version_id = $1",
    )
    .bind(variant_version_id)
    .fetch_all(pool)
    .await?;

    let extra_body_rows: Vec<ExtraBodyRow> = sqlx::query_as(
        r"SELECT variant_version_id, position, pointer, kind, replacement_value
           FROM tensorzero.variant_chat_completion_extra_body
           WHERE variant_version_id = $1
           ORDER BY position",
    )
    .bind(variant_version_id)
    .fetch_all(pool)
    .await?;

    let extra_header_rows: Vec<ExtraHeaderRow> = sqlx::query_as(
        r"SELECT variant_version_id, position, header_name, kind, header_value
           FROM tensorzero.variant_chat_completion_extra_headers
           WHERE variant_version_id = $1
           ORDER BY position",
    )
    .bind(variant_version_id)
    .fetch_all(pool)
    .await?;

    // Collect all prompt template IDs we need
    let mut prompt_ids: Vec<Uuid> = Vec::new();
    if let Some(id) = cc_row.system_template_prompt_id {
        prompt_ids.push(id);
    }
    if let Some(id) = cc_row.user_template_prompt_id {
        prompt_ids.push(id);
    }
    if let Some(id) = cc_row.assistant_template_prompt_id {
        prompt_ids.push(id);
    }
    for t in &template_rows {
        prompt_ids.push(t.prompt_template_version_id);
    }

    let prompt_rows = load_prompt_template_versions(pool, &prompt_ids).await?;

    let config = rehydrate_chat_completion_config(
        &cc_row,
        &template_rows,
        &prompt_rows,
        &extra_body_rows,
        &extra_header_rows,
    )?;
    Ok((config, None))
}

/// Reads a DICL variant from its sub-table.
async fn read_dicl_inner(
    pool: &PgPool,
    variant_version_id: Uuid,
    variant_row: &VariantVersionRow,
) -> Result<UninitializedVariantConfig, Error> {
    use crate::variant::dicl::UninitializedDiclConfig;

    let dicl_row: DiclConfigRow = sqlx::query_as(
        r"SELECT variant_version_id, embedding_model, k, model,
                  system_instructions_prompt_id, system_instructions_key,
                  temperature, top_p, stop_sequences, presence_penalty, frequency_penalty,
                  max_tokens, seed, reasoning_effort, thinking_budget_tokens, verbosity,
                  json_mode, num_retries, max_retry_delay_s, max_distance
           FROM tensorzero.variant_dicl_configs
           WHERE variant_version_id = $1",
    )
    .bind(variant_version_id)
    .fetch_one(pool)
    .await?;

    let extra_body_rows: Vec<ExtraBodyRow> = sqlx::query_as(
        r"SELECT variant_version_id, position, pointer, kind, replacement_value
           FROM tensorzero.variant_dicl_extra_body
           WHERE variant_version_id = $1
           ORDER BY position",
    )
    .bind(variant_version_id)
    .fetch_all(pool)
    .await?;

    let extra_header_rows: Vec<ExtraHeaderRow> = sqlx::query_as(
        r"SELECT variant_version_id, position, header_name, kind, header_value
           FROM tensorzero.variant_dicl_extra_headers
           WHERE variant_version_id = $1
           ORDER BY position",
    )
    .bind(variant_version_id)
    .fetch_all(pool)
    .await?;

    // Load prompt template for system_instructions
    let mut prompt_ids = Vec::new();
    if let Some(id) = dicl_row.system_instructions_prompt_id {
        prompt_ids.push(id);
    }
    let prompt_rows = load_prompt_template_versions(pool, &prompt_ids).await?;

    let system_instructions = rehydrate_prompt_ref(
        dicl_row.system_instructions_prompt_id,
        dicl_row.system_instructions_key.as_ref(),
        &prompt_rows,
    )?;

    let json_mode: Option<JsonMode> = dicl_row
        .json_mode
        .as_deref()
        .map(|s| {
            serde_json::from_value(serde_json::Value::String(s.to_string())).map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Failed to parse json_mode `{s}`: {e}"),
                })
            })
        })
        .transpose()?;

    let extra_body = rehydrate_extra_body(&extra_body_rows)?;
    let extra_headers = rehydrate_extra_headers(&extra_header_rows)?;

    Ok(UninitializedVariantConfig::Dicl(UninitializedDiclConfig {
        weight: variant_row.weight,
        embedding_model: dicl_row.embedding_model,
        k: dicl_row.k as u32,
        model: dicl_row.model,
        system_instructions,
        temperature: dicl_row.temperature,
        top_p: dicl_row.top_p,
        stop_sequences: dicl_row.stop_sequences,
        presence_penalty: dicl_row.presence_penalty,
        frequency_penalty: dicl_row.frequency_penalty,
        max_tokens: dicl_row.max_tokens.map(|v| v as u32),
        seed: dicl_row.seed.map(|v| v as u32),
        reasoning_effort: dicl_row.reasoning_effort,
        thinking_budget_tokens: dicl_row.thinking_budget_tokens,
        verbosity: dicl_row.verbosity,
        json_mode,
        extra_body,
        retries: RetryConfig {
            num_retries: dicl_row.num_retries as usize,
            max_delay_s: dicl_row.max_retry_delay_s,
        },
        extra_headers,
        max_distance: dicl_row.max_distance,
    }))
}

/// Loads prompt template version rows by their IDs.
pub(crate) async fn load_prompt_template_versions(
    pool: &PgPool,
    ids: &[Uuid],
) -> Result<HashMap<Uuid, PromptTemplateVersionRow>, Error> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }

    let rows: Vec<PromptTemplateVersionRow> = sqlx::query_as(
        r"SELECT id, template_key, source_body, creation_source, source_autopilot_session_id
           FROM tensorzero.prompt_template_versions
           WHERE id = ANY($1)",
    )
    .bind(ids)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(|r| (r.id, r)).collect())
}

// ---- Load latest variants / merge ----

/// Row for the "latest version per (function_name, variant_name)" query.
#[derive(Debug, sqlx::FromRow)]
struct LatestVariantRow {
    id: Uuid,
    function_name: String,
    variant_name: String,
}

/// Loads the latest variant version for each `(function_name, variant_name)` pair.
///
/// Uses UUIDv7 ordering (latest id = latest version) via `DISTINCT ON` to pick the
/// most recent version per name.
pub async fn load_all_latest_variants(
    pool: &PgPool,
) -> Result<HashMap<(String, String), UninitializedVariantInfo>, Error> {
    let rows: Vec<LatestVariantRow> = sqlx::query_as(
        r"SELECT DISTINCT ON (function_name, variant_name) id, function_name, variant_name
          FROM tensorzero.variant_versions
          WHERE function_name IS NOT NULL AND variant_name IS NOT NULL
          ORDER BY function_name, variant_name, id DESC",
    )
    .fetch_all(pool)
    .await?;

    let mut result = HashMap::with_capacity(rows.len());
    for row in rows {
        let info = read_variant_version(pool, row.id).await?;
        result.insert((row.function_name, row.variant_name), info);
    }

    Ok(result)
}

/// Loads DB variants and merges them into a runtime `Config`.
///
/// For each DB variant whose function exists in the config, the variant is loaded
/// through `UninitializedVariantInfo::load()` and inserted (or overrides) the
/// function's variant map. Variants for unknown functions are skipped with a warning.
pub async fn merge_db_variants_into_config(
    config: &mut crate::config::Config,
    pool: &PgPool,
) -> Result<(), Error> {
    use crate::config::ErrorContext;
    use crate::function::FunctionConfig;

    let db_variants = load_all_latest_variants(pool).await?;

    if db_variants.is_empty() {
        return Ok(());
    }

    tracing::info!(
        count = db_variants.len(),
        "Merging DB-sourced variants into config"
    );

    for ((function_name, variant_name), uninit_info) in db_variants {
        let Some(function_arc) = config.functions.get_mut(&function_name) else {
            tracing::warn!(
                function_name,
                variant_name,
                "DB variant references a function not defined in config, skipping"
            );
            continue;
        };

        // Get the function's schemas for variant loading
        let function_config = Arc::get_mut(function_arc).ok_or_else(|| {
            Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Cannot merge DB variant `{variant_name}` into function `{function_name}`: \
                         function config has multiple owners"
                ),
            })
        })?;

        let (schemas, variants) = match function_config {
            FunctionConfig::Chat(c) => (&c.schemas, &mut c.variants),
            FunctionConfig::Json(c) => (&c.schemas, &mut c.variants),
        };

        let error_context = ErrorContext {
            function_name: function_name.clone(),
            variant_name: variant_name.clone(),
        };

        let loaded = uninit_info.load(schemas, &error_context)?;
        variants.insert(variant_name, Arc::new(loaded));
    }

    Ok(())
}

/// Merges database-sourced variants into a file-loaded config.
///
/// For each DB variant:
/// - If the function exists in the config, the variant is inserted (or overrides the
///   file variant with the same name).
/// - If the function doesn't exist in the config, a warning is logged and the variant
///   is skipped.
pub fn merge_db_variants(
    config: &mut crate::config::UninitializedConfig,
    db_variants: HashMap<(String, String), UninitializedVariantInfo>,
) {
    for ((function_name, variant_name), variant_info) in db_variants {
        let Some(function_config) = config.functions.get_mut(&function_name) else {
            tracing::warn!(
                function_name,
                variant_name,
                "DB variant references a function not defined in config, skipping"
            );
            continue;
        };
        let variants = match function_config {
            crate::config::UninitializedFunctionConfig::Chat(c) => &mut c.variants,
            crate::config::UninitializedFunctionConfig::Json(c) => &mut c.variants,
        };
        variants.insert(variant_name, variant_info);
    }
}
