use std::collections::{BTreeMap, HashMap, HashSet};

use sqlx::{FromRow, Postgres, Transaction};
use tensorzero_stored_config::{
    STORED_FUNCTION_CONFIG_SCHEMA_REVISION, STORED_VARIANT_CONFIG_SCHEMA_REVISION,
    StoredAdaptiveExperimentationAlgorithm, StoredAdaptiveExperimentationConfig,
    StoredBestOfNVariantConfig, StoredChatCompletionVariantConfig, StoredChatFunctionConfig,
    StoredDiclVariantConfig, StoredEvaluatorConfig, StoredExactMatchConfig,
    StoredExperimentationConfig, StoredExperimentationConfigWithNamespaces, StoredFileRef,
    StoredFunctionConfig, StoredInputWrappers, StoredJsonFunctionConfig,
    StoredLLMJudgeBestOfNVariantConfig, StoredLLMJudgeChainOfThoughtVariantConfig,
    StoredLLMJudgeChatCompletionVariantConfig, StoredLLMJudgeConfig,
    StoredLLMJudgeDiclVariantConfig, StoredLLMJudgeIncludeConfig, StoredLLMJudgeInputFormat,
    StoredLLMJudgeMixtureOfNVariantConfig, StoredLLMJudgeOptimize, StoredLLMJudgeOutputType,
    StoredLLMJudgeVariantConfig, StoredLLMJudgeVariantInfo, StoredMixtureOfNVariantConfig,
    StoredRegexConfig, StoredRetryConfig, StoredStaticExperimentationConfig, StoredTimeoutsConfig,
    StoredToolChoice, StoredToolUseConfig, StoredVariantConfig, StoredVariantRef,
    StoredVariantVersionConfig,
};
use uuid::Uuid;

use crate::config::{
    UninitializedFunctionConfig, UninitializedSchemas, UninitializedVariantConfig,
    UninitializedVariantInfo, path::ResolvedTomlPathData,
};
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::evaluations::{
    LLMJudgeInputFormat, LLMJudgeOptimize, LLMJudgeOutputType, ToolUseConfig,
    UninitializedEvaluatorConfig, UninitializedLLMJudgeBestOfNVariantConfig,
    UninitializedLLMJudgeChatCompletionVariantConfig, UninitializedLLMJudgeConfig,
    UninitializedLLMJudgeDiclVariantConfig, UninitializedLLMJudgeMixtureOfNVariantConfig,
    UninitializedLLMJudgeVariantConfig, UninitializedLLMJudgeVariantInfo,
};
use crate::experimentation::{
    AdaptiveExperimentationAlgorithm, StaticExperimentationConfig,
    UninitializedExperimentationConfig, UninitializedExperimentationConfigWithNamespaces,
    track_and_stop::UninitializedTrackAndStopExperimentationConfig,
};
use crate::inference::types::extra_body::extra_body_config_to_stored;
use crate::inference::types::extra_headers::extra_headers_config_to_stored;
use crate::utils::retries::RetryConfig;
use crate::variant::chat_completion::{
    UninitializedChatCompletionConfig, UninitializedInputWrappers,
};
use crate::variant::dicl::UninitializedDiclConfig;

use super::PostgresConnectionInfo;
use super::file_writes::{CollectedFile, add_file, write_collected_files};

#[derive(Debug)]
pub struct WriteFunctionConfigParams<'a> {
    pub function_name: &'a str,
    pub config: &'a UninitializedFunctionConfig,
    pub expected_current_version_id: Option<Uuid>,
    pub creation_source: &'a str,
    pub source_autopilot_session_id: Option<Uuid>,
    /// Extra templates discovered from the filesystem (e.g. via `template_filesystem_access`).
    /// All prompts — whether explicitly specified in the config or dynamically included via
    /// MiniJinja `{% include %}` — must be stored in the database for the config to be
    /// self-contained.
    pub extra_templates: &'a HashMap<String, String>,
}

#[derive(Debug)]
pub struct WriteFunctionConfigResult {
    pub function_version_id: Uuid,
    pub file_version_ids: HashMap<String, Uuid>,
    pub variant_version_ids: HashMap<String, Uuid>,
}

#[derive(Debug, FromRow)]
struct ExistingVariantRow {
    name: String,
    id: Uuid,
}

impl PostgresConnectionInfo {
    pub async fn write_function_config(
        &self,
        params: WriteFunctionConfigParams<'_>,
    ) -> Result<WriteFunctionConfigResult, Error> {
        // Callers are responsible for validating the function config before writing.
        // This method is a pure DB writer — validation (including the `tensorzero::`
        // prefix check) should happen at the AppStateData layer by building a full
        // Config from a patched UninitializedConfig.

        let pool = self.get_pool_result().map_err(|e| e.log())?;
        let mut tx = pool
            .begin()
            .await
            .map_err(|e| postgres_query_error("Failed to start function config transaction", e))?;

        let result = write_function_config_in_tx(&mut tx, params).await?;
        tx.commit()
            .await
            .map_err(|e| postgres_query_error("Failed to commit function config transaction", e))?;
        Ok(result)
    }
}

pub(super) async fn write_function_config_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    params: WriteFunctionConfigParams<'_>,
) -> Result<WriteFunctionConfigResult, Error> {
    // Acquire an advisory lock on the function name to serialize concurrent writes,
    // including first writes where there is no existing row to lock via FOR UPDATE.
    acquire_function_advisory_lock(tx, params.function_name).await?;

    // With the advisory lock held, read the latest version for the CAS check.
    let actual_latest_id = fetch_latest_function_version(tx, params.function_name).await?;

    // Compare-and-swap: verify the caller's expected version matches the current row.
    if actual_latest_id != params.expected_current_version_id {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "Function `{}` was updated during your edit; please refresh before retrying.",
                params.function_name
            ),
        }));
    }

    let file_version_ids = write_file_versions(
        tx,
        params.config,
        params.extra_templates,
        params.creation_source,
        params.source_autopilot_session_id,
    )
    .await?;

    let variant_version_ids = write_variant_versions(
        tx,
        params.function_name,
        params.config,
        &file_version_ids,
        params.creation_source,
        params.source_autopilot_session_id,
    )
    .await?;

    let function_version_id = write_function_version(
        tx,
        params.function_name,
        params.config,
        &file_version_ids,
        &variant_version_ids,
        params.creation_source,
        params.source_autopilot_session_id,
    )
    .await?;

    Ok(WriteFunctionConfigResult {
        function_version_id,
        file_version_ids,
        variant_version_ids,
    })
}

/// Acquires a transaction-level exclusive advisory lock keyed on the function name.
///
/// The lock key is derived by taking the first 8 bytes of the BLAKE3 hash of
/// `"tensorzero::function_name::{name}"` interpreted as a little-endian `i64`.
/// This serializes all concurrent writes for the same function (including first
/// writes where no row exists yet, which `SELECT ... FOR UPDATE` cannot handle).
/// The lock is released automatically when the transaction ends.
async fn acquire_function_advisory_lock(
    tx: &mut Transaction<'_, Postgres>,
    function_name: &str,
) -> Result<(), Error> {
    let lock_key = function_advisory_lock_key(function_name);
    let acquired: bool = sqlx::query_scalar("SELECT pg_try_advisory_xact_lock($1)")
        .bind(lock_key)
        .fetch_one(&mut **tx)
        .await
        .map_err(|e| {
            postgres_query_error("Failed to acquire advisory lock for function config", e)
        })?;
    if !acquired {
        return Err(Error::new(ErrorDetails::PostgresQuery {
            message: format!(
                "Failed to lock function `{function_name}` for update; another client is updating this function. Please reload the function config and retry."
            ),
        }));
    }
    Ok(())
}

fn function_advisory_lock_key(function_name: &str) -> i64 {
    let key = format!("tensorzero::function_name::{function_name}");
    let hash = blake3::hash(key.as_bytes());
    // BLAKE3 output is always 32 bytes; read the first 8 as a little-endian i64.
    let bytes: [u8; 8] = hash.as_bytes()[..8].try_into().unwrap_or([0u8; 8]);
    i64::from_le_bytes(bytes)
}

/// Returns the ID of the latest function config version, or `None` on first write.
async fn fetch_latest_function_version(
    tx: &mut Transaction<'_, Postgres>,
    function_name: &str,
) -> Result<Option<Uuid>, Error> {
    sqlx::query_scalar::<_, Uuid>(
        "SELECT id \
         FROM tensorzero.function_configs \
         WHERE name = $1 \
         ORDER BY created_at DESC \
         LIMIT 1",
    )
    .bind(function_name)
    .fetch_optional(&mut **tx)
    .await
    .map_err(|e| postgres_query_error("Failed to fetch latest function config version", e))
}

async fn write_file_versions(
    tx: &mut Transaction<'_, Postgres>,
    config: &UninitializedFunctionConfig,
    extra_templates: &HashMap<String, String>,
    creation_source: &str,
    source_autopilot_session_id: Option<Uuid>,
) -> Result<HashMap<String, Uuid>, Error> {
    let mut templates = collect_files(config)?;

    // Merge extra templates. All prompts (directly or transitively included) must be stored
    // in the database for the config to be self-contained.
    for (key, body) in extra_templates {
        if !templates.contains_key(key) {
            templates.insert(
                key.clone(),
                CollectedFile {
                    source_body: body.clone(),
                },
            );
        }
    }

    write_collected_files(tx, &templates, creation_source, source_autopilot_session_id).await
}

async fn write_variant_versions(
    tx: &mut Transaction<'_, Postgres>,
    function_name: &str,
    config: &UninitializedFunctionConfig,
    file_version_ids: &HashMap<String, Uuid>,
    creation_source: &str,
    source_autopilot_session_id: Option<Uuid>,
) -> Result<HashMap<String, Uuid>, Error> {
    let variants = function_variants(config);
    let mut variant_names: Vec<_> = variants.keys().cloned().collect();
    variant_names.sort();

    let valid_file_ids: HashSet<Uuid> = file_version_ids.values().copied().collect();

    // Convert, validate, and serialize each variant, computing a content hash for dedup.
    struct PreparedVariant {
        variant_type: String,
        config_json: serde_json::Value,
        content_hash: Vec<u8>,
    }
    let mut prepared: BTreeMap<String, PreparedVariant> = BTreeMap::new();
    for variant_name in &variant_names {
        let variant_info =
            variants
                .get(variant_name)
                .ok_or_else(|| Error::new(ErrorDetails::Config {
                    message: format!(
                    "Variant `{variant_name}` was missing while serializing function config. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
                }))?;
        let stored_config = convert_variant_info(variant_info, file_version_ids)?;
        validate_variant_version_config_refs(&stored_config, &valid_file_ids)?;
        let config_json = serde_json::to_value(&stored_config)
            .map_err(|e| serialization_error("Failed to serialize stored variant config", e))?;
        let content_hash = blake3::hash(config_json.to_string().as_bytes())
            .as_bytes()
            .to_vec();
        prepared.insert(
            variant_name.clone(),
            PreparedVariant {
                variant_type: stored_variant_type(variant_info).to_string(),
                config_json,
                content_hash,
            },
        );
    }

    // Batch lookup: find existing variant versions with matching (function_name, name, content_hash)
    let names: Vec<&str> = prepared.keys().map(String::as_str).collect();
    let content_hashes: Vec<&[u8]> = names
        .iter()
        .map(|n| prepared[*n].content_hash.as_slice())
        .collect();
    let existing_rows: Vec<ExistingVariantRow> = sqlx::query_as(
        "SELECT input.name, v.id \
         FROM UNNEST($1::text[], $2::text[], $3::bytea[]) AS input(function_name, name, content_hash) \
         JOIN tensorzero.variant_configs v \
           ON v.function_name = input.function_name AND v.name = input.name AND v.content_hash = input.content_hash",
    )
    .bind(vec![function_name; names.len()])
    .bind(&names)
    .bind(&content_hashes)
    .fetch_all(&mut **tx)
    .await
    .map_err(|e| postgres_query_error("Failed to look up existing variant versions", e))?;

    let existing: HashMap<String, Uuid> = existing_rows
        .into_iter()
        .map(|row| (row.name, row.id))
        .collect();

    // Partition into reused vs. new variants
    let mut variant_version_ids = HashMap::with_capacity(prepared.len());
    let mut new_variant_names = Vec::new();
    for variant_name in prepared.keys() {
        if let Some(&existing_id) = existing.get(variant_name.as_str()) {
            variant_version_ids.insert(variant_name.clone(), existing_id);
        } else {
            let id = Uuid::now_v7();
            variant_version_ids.insert(variant_name.clone(), id);
            new_variant_names.push(variant_name.clone());
        }
    }

    // Batch insert only new variants
    if !new_variant_names.is_empty() {
        let mut qb = sqlx::QueryBuilder::new(
            "INSERT INTO tensorzero.variant_configs \
             (id, function_name, variant_type, name, schema_revision, config, content_hash, creation_source, source_autopilot_session_id) ",
        );
        qb.push_values(&new_variant_names, |mut b, name: &String| {
            let pv = &prepared[name];
            let id = variant_version_ids[name];
            b.push_bind(id)
                .push_bind(function_name.to_string())
                .push_bind(pv.variant_type.clone())
                .push_bind(name.clone())
                .push_bind(STORED_VARIANT_CONFIG_SCHEMA_REVISION)
                .push_bind(pv.config_json.clone())
                .push_bind(pv.content_hash.clone())
                .push_bind(creation_source.to_string())
                .push_bind(source_autopilot_session_id);
        });
        qb.build()
            .execute(&mut **tx)
            .await
            .map_err(|e| postgres_query_error("Failed to insert variant config versions", e))?;
    }

    Ok(variant_version_ids)
}

async fn write_function_version(
    tx: &mut Transaction<'_, Postgres>,
    function_name: &str,
    config: &UninitializedFunctionConfig,
    file_version_ids: &HashMap<String, Uuid>,
    variant_version_ids: &HashMap<String, Uuid>,
    creation_source: &str,
    source_autopilot_session_id: Option<Uuid>,
) -> Result<Uuid, Error> {
    let function_version_id = Uuid::now_v7();
    let stored_config = convert_function_config(config, file_version_ids, variant_version_ids)?;
    let valid_file_ids: HashSet<Uuid> = file_version_ids.values().copied().collect();
    let valid_variant_ids: HashSet<Uuid> = variant_version_ids.values().copied().collect();
    validate_function_config_refs(&stored_config, &valid_variant_ids, &valid_file_ids)?;
    let config_json = serde_json::to_value(&stored_config)
        .map_err(|e| serialization_error("Failed to serialize stored function config", e))?;
    sqlx::query(
        "INSERT INTO tensorzero.function_configs \
         (id, name, function_type, schema_revision, config, creation_source, source_autopilot_session_id) \
         VALUES ($1, $2, $3, $4, $5, $6, $7)",
    )
    .bind(function_version_id)
    .bind(function_name)
    .bind(stored_function_type(config))
    .bind(STORED_FUNCTION_CONFIG_SCHEMA_REVISION)
    .bind(&config_json)
    .bind(creation_source)
    .bind(source_autopilot_session_id)
    .execute(&mut **tx)
    .await
    .map_err(|e| postgres_query_error("Failed to insert function config version", e))?;

    Ok(function_version_id)
}

fn function_variants(
    config: &UninitializedFunctionConfig,
) -> &HashMap<String, UninitializedVariantInfo> {
    match config {
        UninitializedFunctionConfig::Chat(config) => &config.variants,
        UninitializedFunctionConfig::Json(config) => &config.variants,
    }
}

fn collect_files(
    config: &UninitializedFunctionConfig,
) -> Result<BTreeMap<String, CollectedFile>, Error> {
    let mut templates = BTreeMap::new();
    match config {
        UninitializedFunctionConfig::Chat(config) => {
            collect_function_common_files(
                &mut templates,
                config.system_schema.as_ref(),
                config.user_schema.as_ref(),
                config.assistant_schema.as_ref(),
                &config.schemas,
            )?;
            for variant in config.variants.values() {
                collect_variant_files(&mut templates, &variant.inner)?;
            }
            for evaluator in config.evaluators.values() {
                collect_evaluator_files(&mut templates, evaluator)?;
            }
        }
        UninitializedFunctionConfig::Json(config) => {
            collect_function_common_files(
                &mut templates,
                config.system_schema.as_ref(),
                config.user_schema.as_ref(),
                config.assistant_schema.as_ref(),
                &config.schemas,
            )?;
            if let Some(output_schema) = &config.output_schema {
                add_file(&mut templates, output_schema)?;
            }
            for variant in config.variants.values() {
                collect_variant_files(&mut templates, &variant.inner)?;
            }
            for evaluator in config.evaluators.values() {
                collect_evaluator_files(&mut templates, evaluator)?;
            }
        }
    }

    Ok(templates)
}

fn collect_function_common_files(
    templates: &mut BTreeMap<String, CollectedFile>,
    system_schema: Option<&ResolvedTomlPathData>,
    user_schema: Option<&ResolvedTomlPathData>,
    assistant_schema: Option<&ResolvedTomlPathData>,
    schemas: &UninitializedSchemas,
) -> Result<(), Error> {
    if let Some(system_schema) = system_schema {
        add_file(templates, system_schema)?;
    }
    if let Some(user_schema) = user_schema {
        add_file(templates, user_schema)?;
    }
    if let Some(assistant_schema) = assistant_schema {
        add_file(templates, assistant_schema)?;
    }
    for (_, schema) in schemas.iter() {
        add_file(templates, schema)?;
    }
    Ok(())
}

fn collect_variant_files(
    templates: &mut BTreeMap<String, CollectedFile>,
    config: &UninitializedVariantConfig,
) -> Result<(), Error> {
    match config {
        UninitializedVariantConfig::ChatCompletion(config) => {
            collect_chat_completion_files(templates, config)
        }
        UninitializedVariantConfig::BestOfNSampling(config) => {
            collect_chat_completion_files(templates, &config.evaluator.inner)
        }
        UninitializedVariantConfig::MixtureOfN(config) => {
            collect_chat_completion_files(templates, &config.fuser.inner)
        }
        UninitializedVariantConfig::Dicl(config) => {
            if let Some(system_instructions) = &config.system_instructions {
                add_file(templates, system_instructions)?;
            }
            Ok(())
        }
        UninitializedVariantConfig::ChainOfThought(config) => {
            collect_chat_completion_files(templates, &config.inner)
        }
    }
}

fn collect_chat_completion_files(
    templates: &mut BTreeMap<String, CollectedFile>,
    config: &UninitializedChatCompletionConfig,
) -> Result<(), Error> {
    if let Some(system_template) = &config.system_template {
        add_file(templates, system_template)?;
    }
    if let Some(user_template) = &config.user_template {
        add_file(templates, user_template)?;
    }
    if let Some(assistant_template) = &config.assistant_template {
        add_file(templates, assistant_template)?;
    }
    if let Some(UninitializedInputWrappers {
        user,
        assistant,
        system,
    }) = &config.input_wrappers
    {
        if let Some(user) = user {
            add_file(templates, user)?;
        }
        if let Some(assistant) = assistant {
            add_file(templates, assistant)?;
        }
        if let Some(system) = system {
            add_file(templates, system)?;
        }
    }
    for template in config.templates.inner.values() {
        add_file(templates, &template.path)?;
    }
    Ok(())
}

fn collect_evaluator_files(
    templates: &mut BTreeMap<String, CollectedFile>,
    config: &UninitializedEvaluatorConfig,
) -> Result<(), Error> {
    if let UninitializedEvaluatorConfig::LLMJudge(config) = config {
        for variant in config.variants.values() {
            collect_llm_judge_files(templates, &variant.inner)?;
        }
    }
    Ok(())
}

fn collect_llm_judge_files(
    templates: &mut BTreeMap<String, CollectedFile>,
    config: &UninitializedLLMJudgeVariantConfig,
) -> Result<(), Error> {
    match config {
        UninitializedLLMJudgeVariantConfig::ChatCompletion(config) => {
            add_file(templates, &config.system_instructions)
        }
        UninitializedLLMJudgeVariantConfig::BestOfNSampling(config) => {
            add_file(templates, &config.evaluator.system_instructions)
        }
        UninitializedLLMJudgeVariantConfig::MixtureOfNSampling(config) => {
            add_file(templates, &config.fuser.system_instructions)
        }
        UninitializedLLMJudgeVariantConfig::Dicl(config) => {
            if let Some(system_instructions) = &config.system_instructions {
                add_file(templates, system_instructions)?;
            }
            Ok(())
        }
        UninitializedLLMJudgeVariantConfig::ChainOfThought(config) => {
            add_file(templates, &config.inner.system_instructions)
        }
    }
}

fn convert_function_config(
    config: &UninitializedFunctionConfig,
    file_version_ids: &HashMap<String, Uuid>,
    variant_version_ids: &HashMap<String, Uuid>,
) -> Result<StoredFunctionConfig, Error> {
    match config {
        UninitializedFunctionConfig::Chat(config) => {
            Ok(StoredFunctionConfig::Chat(StoredChatFunctionConfig {
                variants: Some(convert_variant_refs(&config.variants, variant_version_ids)?),
                system_schema: config
                    .system_schema
                    .as_ref()
                    .map(|path| file_ref_for(path, file_version_ids))
                    .transpose()?,
                user_schema: config
                    .user_schema
                    .as_ref()
                    .map(|path| file_ref_for(path, file_version_ids))
                    .transpose()?,
                assistant_schema: config
                    .assistant_schema
                    .as_ref()
                    .map(|path| file_ref_for(path, file_version_ids))
                    .transpose()?,
                schemas: Some(convert_named_file_refs(
                    config.schemas.iter(),
                    file_version_ids,
                )?),
                tools: Some(config.tools.clone()),
                tool_choice: Some(StoredToolChoice::from(&config.tool_choice)),
                parallel_tool_calls: config.parallel_tool_calls,
                description: config.description.clone(),
                experimentation: config
                    .experimentation
                    .as_ref()
                    .map(StoredExperimentationConfigWithNamespaces::from),
                evaluators: Some(convert_evaluators(&config.evaluators, file_version_ids)?),
            }))
        }
        UninitializedFunctionConfig::Json(config) => {
            Ok(StoredFunctionConfig::Json(StoredJsonFunctionConfig {
                variants: Some(convert_variant_refs(&config.variants, variant_version_ids)?),
                system_schema: config
                    .system_schema
                    .as_ref()
                    .map(|path| file_ref_for(path, file_version_ids))
                    .transpose()?,
                user_schema: config
                    .user_schema
                    .as_ref()
                    .map(|path| file_ref_for(path, file_version_ids))
                    .transpose()?,
                assistant_schema: config
                    .assistant_schema
                    .as_ref()
                    .map(|path| file_ref_for(path, file_version_ids))
                    .transpose()?,
                schemas: Some(convert_named_file_refs(
                    config.schemas.iter(),
                    file_version_ids,
                )?),
                output_schema: config
                    .output_schema
                    .as_ref()
                    .map(|path| file_ref_for(path, file_version_ids))
                    .transpose()?,
                description: config.description.clone(),
                experimentation: config
                    .experimentation
                    .as_ref()
                    .map(StoredExperimentationConfigWithNamespaces::from),
                evaluators: Some(convert_evaluators(&config.evaluators, file_version_ids)?),
            }))
        }
    }
}

fn convert_variant_refs(
    variants: &HashMap<String, UninitializedVariantInfo>,
    variant_version_ids: &HashMap<String, Uuid>,
) -> Result<BTreeMap<String, StoredVariantRef>, Error> {
    let mut stored_variants = BTreeMap::new();
    for variant_name in variants.keys() {
        let variant_version_id = variant_version_ids.get(variant_name).ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "Missing stored variant version ID for `{variant_name}`. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            })
        })?;
        stored_variants.insert(
            variant_name.clone(),
            StoredVariantRef {
                variant_version_id: *variant_version_id,
            },
        );
    }
    Ok(stored_variants)
}

fn convert_named_file_refs<'a>(
    entries: impl Iterator<Item = (&'a String, &'a ResolvedTomlPathData)>,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<BTreeMap<String, StoredFileRef>, Error> {
    let mut refs = BTreeMap::new();
    for (name, path) in entries {
        refs.insert(name.clone(), file_ref_for(path, file_version_ids)?);
    }
    Ok(refs)
}

fn file_ref_for(
    path: &ResolvedTomlPathData,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<StoredFileRef, Error> {
    let file_path = path.get_template_key();
    let file_version_id = file_version_ids
        .get(&file_path)
        .copied()
        .ok_or_else(|| missing_file_error(&file_path))?;
    Ok(StoredFileRef {
        file_version_id,
        file_path,
    })
}

#[expect(deprecated)]
fn convert_variant_info(
    variant_info: &UninitializedVariantInfo,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<StoredVariantVersionConfig, Error> {
    Ok(StoredVariantVersionConfig {
        variant: match &variant_info.inner {
            UninitializedVariantConfig::ChatCompletion(config) => {
                StoredVariantConfig::ChatCompletion(convert_chat_completion_variant(
                    config,
                    file_version_ids,
                )?)
            }
            UninitializedVariantConfig::BestOfNSampling(config) => {
                StoredVariantConfig::BestOfNSampling(StoredBestOfNVariantConfig {
                    weight: config.weight,
                    timeout_s: config.timeout_s,
                    candidates: Some(config.candidates.clone()),
                    evaluator: convert_chat_completion_variant(
                        &config.evaluator.inner,
                        file_version_ids,
                    )?,
                })
            }
            UninitializedVariantConfig::MixtureOfN(config) => {
                StoredVariantConfig::MixtureOfN(StoredMixtureOfNVariantConfig {
                    weight: config.weight,
                    timeout_s: config.timeout_s,
                    candidates: Some(config.candidates.clone()),
                    fuser: convert_chat_completion_variant(&config.fuser.inner, file_version_ids)?,
                })
            }
            UninitializedVariantConfig::Dicl(config) => {
                StoredVariantConfig::Dicl(convert_dicl_variant(config, file_version_ids)?)
            }
            UninitializedVariantConfig::ChainOfThought(config) => {
                StoredVariantConfig::ChainOfThought(convert_chat_completion_variant(
                    &config.inner,
                    file_version_ids,
                )?)
            }
        },
        timeouts: variant_info
            .timeouts
            .as_ref()
            .map(StoredTimeoutsConfig::from),
        namespace: variant_info
            .namespace
            .as_ref()
            .map(|namespace| namespace.as_str().to_string()),
    })
}

fn convert_chat_completion_variant(
    config: &UninitializedChatCompletionConfig,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<StoredChatCompletionVariantConfig, Error> {
    Ok(StoredChatCompletionVariantConfig {
        weight: config.weight,
        model: config.model.clone(),
        system_template: config
            .system_template
            .as_ref()
            .map(|path| file_ref_for(path, file_version_ids))
            .transpose()?,
        user_template: config
            .user_template
            .as_ref()
            .map(|path| file_ref_for(path, file_version_ids))
            .transpose()?,
        assistant_template: config
            .assistant_template
            .as_ref()
            .map(|path| file_ref_for(path, file_version_ids))
            .transpose()?,
        input_wrappers: config
            .input_wrappers
            .as_ref()
            .map(|wrappers| convert_input_wrappers(wrappers, file_version_ids))
            .transpose()?,
        templates: Some(
            config
                .templates
                .inner
                .iter()
                .map(|(name, template)| {
                    file_ref_for(&template.path, file_version_ids)
                        .map(|template_ref| (name.clone(), template_ref))
                })
                .collect::<Result<BTreeMap<_, _>, Error>>()?,
        ),
        temperature: config.temperature,
        top_p: config.top_p,
        max_tokens: config.max_tokens,
        presence_penalty: config.presence_penalty,
        frequency_penalty: config.frequency_penalty,
        seed: config.seed,
        json_mode: config.json_mode,
        stop_sequences: config.stop_sequences.clone(),
        reasoning_effort: config.reasoning_effort.clone(),
        service_tier: config.service_tier.clone(),
        thinking_budget_tokens: config.thinking_budget_tokens,
        verbosity: config.verbosity.clone(),
        retries: Some(StoredRetryConfig::from(config.retries)),
        extra_body: config.extra_body.as_ref().map(extra_body_config_to_stored),
        extra_headers: config
            .extra_headers
            .as_ref()
            .map(extra_headers_config_to_stored),
    })
}

fn convert_input_wrappers(
    wrappers: &crate::variant::chat_completion::UninitializedInputWrappers,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<StoredInputWrappers, Error> {
    Ok(StoredInputWrappers {
        user: wrappers
            .user
            .as_ref()
            .map(|path| file_ref_for(path, file_version_ids))
            .transpose()?,
        assistant: wrappers
            .assistant
            .as_ref()
            .map(|path| file_ref_for(path, file_version_ids))
            .transpose()?,
        system: wrappers
            .system
            .as_ref()
            .map(|path| file_ref_for(path, file_version_ids))
            .transpose()?,
    })
}

fn convert_dicl_variant(
    config: &UninitializedDiclConfig,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<StoredDiclVariantConfig, Error> {
    Ok(StoredDiclVariantConfig {
        weight: config.weight,
        embedding_model: config.embedding_model.clone(),
        k: config.k,
        model: config.model.clone(),
        system_instructions: config
            .system_instructions
            .as_ref()
            .map(|path| file_ref_for(path, file_version_ids))
            .transpose()?,
        temperature: config.temperature,
        top_p: config.top_p,
        max_tokens: config.max_tokens,
        presence_penalty: config.presence_penalty,
        frequency_penalty: config.frequency_penalty,
        seed: config.seed,
        json_mode: config.json_mode,
        stop_sequences: config.stop_sequences.clone(),
        reasoning_effort: config.reasoning_effort.clone(),
        thinking_budget_tokens: config.thinking_budget_tokens,
        verbosity: config.verbosity.clone(),
        max_distance: config.max_distance,
        retries: Some(StoredRetryConfig::from(config.retries)),
        extra_body: config.extra_body.as_ref().map(extra_body_config_to_stored),
        extra_headers: config
            .extra_headers
            .as_ref()
            .map(extra_headers_config_to_stored),
    })
}

#[expect(deprecated)]
pub(crate) fn convert_evaluators(
    evaluators: &HashMap<String, UninitializedEvaluatorConfig>,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<BTreeMap<String, StoredEvaluatorConfig>, Error> {
    let mut stored = BTreeMap::new();
    for (name, evaluator) in evaluators {
        stored.insert(
            name.clone(),
            match evaluator {
                UninitializedEvaluatorConfig::ExactMatch(config) => {
                    StoredEvaluatorConfig::ExactMatch(StoredExactMatchConfig {
                        cutoff: config.cutoff,
                    })
                }
                UninitializedEvaluatorConfig::LLMJudge(config) => StoredEvaluatorConfig::LLMJudge(
                    convert_llm_judge_config(config, file_version_ids)?,
                ),
                UninitializedEvaluatorConfig::ToolUse(config) => {
                    StoredEvaluatorConfig::ToolUse(StoredToolUseConfig::from(config))
                }
                UninitializedEvaluatorConfig::Regex(config) => {
                    StoredEvaluatorConfig::Regex(StoredRegexConfig {
                        must_match: config.must_match.clone(),
                        must_not_match: config.must_not_match.clone(),
                    })
                }
            },
        );
    }
    Ok(stored)
}

fn convert_llm_judge_config(
    config: &UninitializedLLMJudgeConfig,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<StoredLLMJudgeConfig, Error> {
    let mut variants = BTreeMap::new();
    for (name, variant) in &config.variants {
        variants.insert(
            name.clone(),
            convert_llm_judge_variant_info(variant, file_version_ids)?,
        );
    }

    Ok(StoredLLMJudgeConfig {
        input_format: config
            .input_format
            .as_ref()
            .map(|f| StoredLLMJudgeInputFormat::from(f.clone())),
        variants: Some(variants),
        output_type: config.output_type.into(),
        optimize: config.optimize.into(),
        #[expect(deprecated)]
        cutoff: config.cutoff,
        include: config
            .include
            .as_ref()
            .map(|inc| StoredLLMJudgeIncludeConfig {
                reference_output: inc.reference_output,
            }),
        description: config.description.clone(),
    })
}

fn convert_llm_judge_variant_info(
    config: &UninitializedLLMJudgeVariantInfo,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<StoredLLMJudgeVariantInfo, Error> {
    Ok(StoredLLMJudgeVariantInfo {
        variant: match &config.inner {
            UninitializedLLMJudgeVariantConfig::ChatCompletion(config) => {
                StoredLLMJudgeVariantConfig::ChatCompletion(
                    convert_llm_judge_chat_completion_variant(config, file_version_ids)?,
                )
            }
            UninitializedLLMJudgeVariantConfig::BestOfNSampling(config) => {
                StoredLLMJudgeVariantConfig::BestOfNSampling(convert_llm_judge_best_of_n_variant(
                    config,
                    file_version_ids,
                )?)
            }
            UninitializedLLMJudgeVariantConfig::MixtureOfNSampling(config) => {
                StoredLLMJudgeVariantConfig::MixtureOfNSampling(
                    convert_llm_judge_mixture_of_n_variant(config, file_version_ids)?,
                )
            }
            UninitializedLLMJudgeVariantConfig::Dicl(config) => StoredLLMJudgeVariantConfig::Dicl(
                convert_llm_judge_dicl_variant(config, file_version_ids)?,
            ),
            UninitializedLLMJudgeVariantConfig::ChainOfThought(config) => {
                StoredLLMJudgeVariantConfig::ChainOfThought(
                    StoredLLMJudgeChainOfThoughtVariantConfig {
                        inner: convert_llm_judge_chat_completion_variant(
                            &config.inner,
                            file_version_ids,
                        )?,
                    },
                )
            }
        },
        timeouts: config.timeouts.as_ref().map(StoredTimeoutsConfig::from),
    })
}

fn convert_llm_judge_chat_completion_variant(
    config: &UninitializedLLMJudgeChatCompletionVariantConfig,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<StoredLLMJudgeChatCompletionVariantConfig, Error> {
    Ok(StoredLLMJudgeChatCompletionVariantConfig {
        active: config.active,
        model: config.model.clone(),
        system_instructions: file_ref_for(&config.system_instructions, file_version_ids)?,
        temperature: config.temperature,
        top_p: config.top_p,
        max_tokens: config.max_tokens,
        presence_penalty: config.presence_penalty,
        frequency_penalty: config.frequency_penalty,
        seed: config.seed,
        json_mode: config.json_mode,
        stop_sequences: config.stop_sequences.clone(),
        reasoning_effort: config.reasoning_effort.clone(),
        service_tier: config.service_tier.clone(),
        thinking_budget_tokens: config.thinking_budget_tokens,
        verbosity: config.verbosity.clone(),
        retries: Some(StoredRetryConfig::from(config.retries)),
        extra_body: config.extra_body.as_ref().map(extra_body_config_to_stored),
        extra_headers: config
            .extra_headers
            .as_ref()
            .map(extra_headers_config_to_stored),
    })
}

#[expect(deprecated)]
fn convert_llm_judge_best_of_n_variant(
    config: &UninitializedLLMJudgeBestOfNVariantConfig,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<StoredLLMJudgeBestOfNVariantConfig, Error> {
    Ok(StoredLLMJudgeBestOfNVariantConfig {
        active: config.active,
        timeout_s: config.timeout_s,
        candidates: Some(config.candidates.clone()),
        evaluator: convert_llm_judge_chat_completion_variant(&config.evaluator, file_version_ids)?,
    })
}

#[expect(deprecated)]
fn convert_llm_judge_mixture_of_n_variant(
    config: &UninitializedLLMJudgeMixtureOfNVariantConfig,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<StoredLLMJudgeMixtureOfNVariantConfig, Error> {
    Ok(StoredLLMJudgeMixtureOfNVariantConfig {
        active: config.active,
        timeout_s: config.timeout_s,
        candidates: Some(config.candidates.clone()),
        fuser: convert_llm_judge_chat_completion_variant(&config.fuser, file_version_ids)?,
    })
}

fn convert_llm_judge_dicl_variant(
    config: &UninitializedLLMJudgeDiclVariantConfig,
    file_version_ids: &HashMap<String, Uuid>,
) -> Result<StoredLLMJudgeDiclVariantConfig, Error> {
    Ok(StoredLLMJudgeDiclVariantConfig {
        active: config.active,
        embedding_model: config.embedding_model.clone(),
        k: config.k,
        model: config.model.clone(),
        system_instructions: config
            .system_instructions
            .as_ref()
            .map(|path| file_ref_for(path, file_version_ids))
            .transpose()?,
        temperature: config.temperature,
        top_p: config.top_p,
        presence_penalty: config.presence_penalty,
        frequency_penalty: config.frequency_penalty,
        max_tokens: config.max_tokens,
        seed: config.seed,
        json_mode: config.json_mode,
        stop_sequences: config.stop_sequences.clone(),
        extra_body: config.extra_body.as_ref().map(extra_body_config_to_stored),
        retries: Some(StoredRetryConfig::from(config.retries)),
        extra_headers: config
            .extra_headers
            .as_ref()
            .map(extra_headers_config_to_stored),
    })
}

impl From<&UninitializedExperimentationConfigWithNamespaces>
    for StoredExperimentationConfigWithNamespaces
{
    fn from(config: &UninitializedExperimentationConfigWithNamespaces) -> Self {
        let namespaces = config
            .namespaces
            .iter()
            .map(|(namespace, namespace_config)| {
                (
                    namespace.clone(),
                    StoredExperimentationConfig::from(namespace_config),
                )
            })
            .collect();

        StoredExperimentationConfigWithNamespaces {
            base: StoredExperimentationConfig::from(&config.base),
            namespaces: Some(namespaces),
        }
    }
}

impl From<&UninitializedExperimentationConfig> for StoredExperimentationConfig {
    fn from(config: &UninitializedExperimentationConfig) -> Self {
        match config {
            UninitializedExperimentationConfig::Static(config) => Self::Static(config.into()),
            UninitializedExperimentationConfig::Adaptive(config) => {
                Self::Adaptive(convert_track_and_stop(
                    config.algorithm.as_ref().map(|alg| match alg {
                        AdaptiveExperimentationAlgorithm::TrackAndStop => {
                            StoredAdaptiveExperimentationAlgorithm::TrackAndStop
                        }
                    }),
                    &config.inner,
                ))
            }
            UninitializedExperimentationConfig::Uniform(config) => {
                Self::Static(StoredStaticExperimentationConfig {
                    candidate_variants: config.candidate_variants.as_ref().map(|variants| {
                        variants
                            .iter()
                            .map(|variant| (variant.clone(), 1.0))
                            .collect::<BTreeMap<_, _>>()
                    }),
                    fallback_variants: config.fallback_variants.clone(),
                })
            }
            UninitializedExperimentationConfig::StaticWeights(config) => {
                Self::Static(StoredStaticExperimentationConfig {
                    candidate_variants: Some(
                        config
                            .candidate_variants
                            .iter()
                            .map(|(k, v)| (k.clone(), *v))
                            .collect::<BTreeMap<_, _>>(),
                    ),
                    fallback_variants: Some(config.fallback_variants.clone()),
                })
            }
            UninitializedExperimentationConfig::TrackAndStop(config) => {
                Self::Adaptive(convert_track_and_stop(
                    Some(StoredAdaptiveExperimentationAlgorithm::TrackAndStop),
                    config,
                ))
            }
        }
    }
}

impl From<&StaticExperimentationConfig> for StoredStaticExperimentationConfig {
    fn from(config: &StaticExperimentationConfig) -> Self {
        Self {
            candidate_variants: Some(
                config
                    .candidate_variants
                    .inner()
                    .iter()
                    .map(|(variant, weight)| (variant.clone(), *weight))
                    .collect(),
            ),
            fallback_variants: Some(config.fallback_variants.clone()),
        }
    }
}

fn convert_track_and_stop(
    algorithm: Option<StoredAdaptiveExperimentationAlgorithm>,
    config: &UninitializedTrackAndStopExperimentationConfig,
) -> StoredAdaptiveExperimentationConfig {
    StoredAdaptiveExperimentationConfig {
        algorithm,
        metric: config.metric.clone(),
        candidate_variants: Some(config.candidate_variants.clone()),
        fallback_variants: Some(config.fallback_variants.clone()),
        min_samples_per_variant: Some(config.min_samples_per_variant),
        delta: Some(config.delta),
        epsilon: Some(config.epsilon),
        update_period_s: Some(config.update_period_s),
        min_prob: config.min_prob,
        max_samples_per_variant: config.max_samples_per_variant,
    }
}

impl From<&ToolUseConfig> for StoredToolUseConfig {
    fn from(config: &ToolUseConfig) -> Self {
        match config {
            ToolUseConfig::None => Self::None,
            ToolUseConfig::NoneOf { tools } => Self::NoneOf {
                tools: tools.clone(),
            },
            ToolUseConfig::Any => Self::Any,
            ToolUseConfig::AnyOf { tools } => Self::AnyOf {
                tools: tools.clone(),
            },
            ToolUseConfig::AllOf { tools } => Self::AllOf {
                tools: tools.clone(),
            },
        }
    }
}

impl From<LLMJudgeInputFormat> for StoredLLMJudgeInputFormat {
    fn from(value: LLMJudgeInputFormat) -> Self {
        match value {
            LLMJudgeInputFormat::Serialized => Self::Serialized,
            LLMJudgeInputFormat::Messages => Self::Messages,
        }
    }
}

impl From<LLMJudgeOutputType> for StoredLLMJudgeOutputType {
    fn from(value: LLMJudgeOutputType) -> Self {
        match value {
            LLMJudgeOutputType::Float => Self::Float,
            LLMJudgeOutputType::Boolean => Self::Boolean,
        }
    }
}

impl From<LLMJudgeOptimize> for StoredLLMJudgeOptimize {
    fn from(value: LLMJudgeOptimize) -> Self {
        match value {
            LLMJudgeOptimize::Min => Self::Min,
            LLMJudgeOptimize::Max => Self::Max,
        }
    }
}

impl From<RetryConfig> for StoredRetryConfig {
    fn from(config: RetryConfig) -> Self {
        Self {
            num_retries: config.num_retries as u32,
            max_delay_s: config.max_delay_s,
        }
    }
}

fn validate_variant_version_config_refs(
    config: &StoredVariantVersionConfig,
    valid_file_ids: &HashSet<Uuid>,
) -> Result<(), Error> {
    match &config.variant {
        StoredVariantConfig::ChatCompletion(config)
        | StoredVariantConfig::ChainOfThought(config) => {
            validate_chat_completion_file_refs(config, valid_file_ids)?;
        }
        StoredVariantConfig::BestOfNSampling(config) => {
            validate_chat_completion_file_refs(&config.evaluator, valid_file_ids)?;
        }
        StoredVariantConfig::MixtureOfN(config) => {
            validate_chat_completion_file_refs(&config.fuser, valid_file_ids)?;
        }
        StoredVariantConfig::Dicl(config) => {
            if let Some(system_instructions) = &config.system_instructions {
                validate_file_ref(system_instructions, valid_file_ids)?;
            }
        }
    }
    Ok(())
}

fn validate_chat_completion_file_refs(
    config: &StoredChatCompletionVariantConfig,
    valid_file_ids: &HashSet<Uuid>,
) -> Result<(), Error> {
    if let Some(system_template) = &config.system_template {
        validate_file_ref(system_template, valid_file_ids)?;
    }
    if let Some(user_template) = &config.user_template {
        validate_file_ref(user_template, valid_file_ids)?;
    }
    if let Some(assistant_template) = &config.assistant_template {
        validate_file_ref(assistant_template, valid_file_ids)?;
    }
    if let Some(StoredInputWrappers {
        user,
        assistant,
        system,
    }) = &config.input_wrappers
    {
        if let Some(user) = user {
            validate_file_ref(user, valid_file_ids)?;
        }
        if let Some(assistant) = assistant {
            validate_file_ref(assistant, valid_file_ids)?;
        }
        if let Some(system) = system {
            validate_file_ref(system, valid_file_ids)?;
        }
    }
    if let Some(templates) = &config.templates {
        for prompt_ref in templates.values() {
            validate_file_ref(prompt_ref, valid_file_ids)?;
        }
    }
    Ok(())
}

struct RefValidationContext<'a> {
    valid_variant_ids: &'a HashSet<Uuid>,
    valid_file_ids: &'a HashSet<Uuid>,
}

fn validate_function_config_refs(
    config: &StoredFunctionConfig,
    valid_variant_ids: &HashSet<Uuid>,
    valid_file_ids: &HashSet<Uuid>,
) -> Result<(), Error> {
    let ctx = RefValidationContext {
        valid_variant_ids,
        valid_file_ids,
    };
    match config {
        StoredFunctionConfig::Chat(config) => {
            validate_common_function_refs(
                config.variants.as_ref(),
                config.system_schema.as_ref(),
                config.user_schema.as_ref(),
                config.assistant_schema.as_ref(),
                config.schemas.as_ref(),
                config.evaluators.as_ref(),
                &ctx,
            )?;
        }
        StoredFunctionConfig::Json(config) => {
            validate_common_function_refs(
                config.variants.as_ref(),
                config.system_schema.as_ref(),
                config.user_schema.as_ref(),
                config.assistant_schema.as_ref(),
                config.schemas.as_ref(),
                config.evaluators.as_ref(),
                &ctx,
            )?;
            if let Some(output_schema) = &config.output_schema {
                validate_file_ref(output_schema, ctx.valid_file_ids)?;
            }
        }
    }
    Ok(())
}

fn validate_common_function_refs(
    variants: Option<&BTreeMap<String, StoredVariantRef>>,
    system_schema: Option<&StoredFileRef>,
    user_schema: Option<&StoredFileRef>,
    assistant_schema: Option<&StoredFileRef>,
    schemas: Option<&BTreeMap<String, StoredFileRef>>,
    evaluators: Option<&BTreeMap<String, StoredEvaluatorConfig>>,
    ctx: &RefValidationContext<'_>,
) -> Result<(), Error> {
    if let Some(variants) = variants {
        for variant_ref in variants.values() {
            validate_variant_ref(variant_ref, ctx.valid_variant_ids)?;
        }
    }
    if let Some(system_schema) = system_schema {
        validate_file_ref(system_schema, ctx.valid_file_ids)?;
    }
    if let Some(user_schema) = user_schema {
        validate_file_ref(user_schema, ctx.valid_file_ids)?;
    }
    if let Some(assistant_schema) = assistant_schema {
        validate_file_ref(assistant_schema, ctx.valid_file_ids)?;
    }
    if let Some(schemas) = schemas {
        for prompt_ref in schemas.values() {
            validate_file_ref(prompt_ref, ctx.valid_file_ids)?;
        }
    }
    if let Some(evaluators) = evaluators {
        for evaluator in evaluators.values() {
            validate_evaluator_refs(evaluator, ctx.valid_file_ids)?;
        }
    }
    Ok(())
}

fn validate_evaluator_refs(
    config: &StoredEvaluatorConfig,
    valid_file_ids: &HashSet<Uuid>,
) -> Result<(), Error> {
    if let StoredEvaluatorConfig::LLMJudge(config) = config
        && let Some(variants) = &config.variants
    {
        for variant in variants.values() {
            validate_llm_judge_variant_refs(variant, valid_file_ids)?;
        }
    }
    Ok(())
}

fn validate_llm_judge_variant_refs(
    config: &StoredLLMJudgeVariantInfo,
    valid_file_ids: &HashSet<Uuid>,
) -> Result<(), Error> {
    match &config.variant {
        StoredLLMJudgeVariantConfig::ChatCompletion(config) => {
            validate_file_ref(&config.system_instructions, valid_file_ids)?;
        }
        StoredLLMJudgeVariantConfig::BestOfNSampling(config) => {
            validate_file_ref(&config.evaluator.system_instructions, valid_file_ids)?;
        }
        StoredLLMJudgeVariantConfig::MixtureOfNSampling(config) => {
            validate_file_ref(&config.fuser.system_instructions, valid_file_ids)?;
        }
        StoredLLMJudgeVariantConfig::Dicl(config) => {
            if let Some(system_instructions) = &config.system_instructions {
                validate_file_ref(system_instructions, valid_file_ids)?;
            }
        }
        StoredLLMJudgeVariantConfig::ChainOfThought(config) => {
            validate_file_ref(&config.inner.system_instructions, valid_file_ids)?;
        }
    }
    Ok(())
}

fn validate_variant_ref(
    variant_ref: &StoredVariantRef,
    valid_variant_ids: &HashSet<Uuid>,
) -> Result<(), Error> {
    if valid_variant_ids.contains(&variant_ref.variant_version_id) {
        Ok(())
    } else {
        Err(Error::new(ErrorDetails::Config {
            message: format!(
                "Stored function config references unknown variant version `{}`.",
                variant_ref.variant_version_id
            ),
        }))
    }
}

fn validate_file_ref(
    prompt_ref: &StoredFileRef,
    valid_file_ids: &HashSet<Uuid>,
) -> Result<(), Error> {
    if valid_file_ids.contains(&prompt_ref.file_version_id) {
        Ok(())
    } else {
        Err(Error::new(ErrorDetails::Config {
            message: format!(
                "Stored config references unknown stored file version `{}` for file path `{}`.",
                prompt_ref.file_version_id, prompt_ref.file_path
            ),
        }))
    }
}

fn stored_variant_type(config: &UninitializedVariantInfo) -> &'static str {
    match &config.inner {
        UninitializedVariantConfig::ChatCompletion(_) => "chat_completion",
        UninitializedVariantConfig::BestOfNSampling(_) => "experimental_best_of_n_sampling",
        UninitializedVariantConfig::MixtureOfN(_) => "experimental_mixture_of_n",
        UninitializedVariantConfig::Dicl(_) => "experimental_dynamic_in_context_learning",
        UninitializedVariantConfig::ChainOfThought(_) => "experimental_chain_of_thought",
    }
}

fn stored_function_type(config: &UninitializedFunctionConfig) -> &'static str {
    match config {
        UninitializedFunctionConfig::Chat(_) => "chat",
        UninitializedFunctionConfig::Json(_) => "json",
    }
}

fn postgres_query_error(context: &str, error: impl std::fmt::Display) -> Error {
    Error::new(ErrorDetails::PostgresQuery {
        message: format!("{context}: {error}"),
    })
}

fn serialization_error(context: &str, error: impl std::fmt::Display) -> Error {
    Error::new(ErrorDetails::Serialization {
        message: format!("{context}: {error}"),
    })
}

fn missing_file_error(file_path: &str) -> Error {
    Error::new(ErrorDetails::Config {
        message: format!("Missing stored file version ID for file path `{file_path}`."),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use super::*;

    #[test]
    fn validate_function_config_refs_rejects_unknown_variant() {
        let config = StoredFunctionConfig::Chat(StoredChatFunctionConfig {
            variants: Some(BTreeMap::from([(
                "chat".to_string(),
                StoredVariantRef {
                    variant_version_id: Uuid::now_v7(),
                },
            )])),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            schemas: Some(BTreeMap::new()),
            tools: Some(vec![]),
            tool_choice: Some(StoredToolChoice::Auto),
            parallel_tool_calls: None,
            description: None,
            experimentation: None,
            evaluators: Some(BTreeMap::new()),
        });
        let error = validate_function_config_refs(&config, &HashSet::new(), &HashSet::new())
            .expect_err("unknown variant refs should fail");
        assert!(
            error
                .to_string()
                .contains("Stored function config references unknown variant version")
        );
    }
}
