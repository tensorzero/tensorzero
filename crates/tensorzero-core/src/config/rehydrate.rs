//! Converts stored config types (from `tensorzero-stored-config`) back into
//! `Uninitialized*` types that the existing `load()` pipeline can consume.
//!
//! This module is the core of the "read path" for config-in-database.
//! Each `Stored*` type has a corresponding conversion function that
//! resolves `StoredFileRef`s into `ResolvedTomlPathData` via a preloaded
//! file map.

use std::collections::{BTreeMap, HashMap};

use tensorzero_stored_config::{
    StoredBestOfNVariantConfig, StoredChatCompletionVariantConfig, StoredDiclVariantConfig,
    StoredEvaluationConfig, StoredEvaluatorConfig, StoredFile, StoredFileRef, StoredFunctionConfig,
    StoredInputWrappers, StoredLLMJudgeBestOfNVariantConfig,
    StoredLLMJudgeChatCompletionVariantConfig, StoredLLMJudgeConfig,
    StoredLLMJudgeDiclVariantConfig, StoredLLMJudgeMixtureOfNVariantConfig,
    StoredLLMJudgeVariantConfig, StoredLLMJudgeVariantInfo, StoredMixtureOfNVariantConfig,
    StoredToolConfig, StoredVariantConfig, StoredVariantVersionConfig,
};
use uuid::Uuid;

use crate::config::path::ResolvedTomlPathData;
use crate::config::{
    Namespace, UninitializedFunctionConfig, UninitializedFunctionConfigChat,
    UninitializedFunctionConfigJson, UninitializedSchemas,
};
use crate::error::{Error, ErrorDetails};
use crate::evaluations::{
    UninitializedEvaluationConfig, UninitializedEvaluatorConfig,
    UninitializedInferenceEvaluationConfig, UninitializedLLMJudgeBestOfNVariantConfig,
    UninitializedLLMJudgeChainOfThoughtVariantConfig,
    UninitializedLLMJudgeChatCompletionVariantConfig, UninitializedLLMJudgeConfig,
    UninitializedLLMJudgeDiclVariantConfig, UninitializedLLMJudgeMixtureOfNVariantConfig,
    UninitializedLLMJudgeVariantConfig, UninitializedLLMJudgeVariantInfo,
    UninitializedTypescriptJudgeConfig,
};
use crate::inference::types::extra_body::ExtraBodyConfig;
use crate::inference::types::extra_headers::ExtraHeadersConfig;
use crate::tool::ToolChoice;
use crate::utils::retries::RetryConfig;
use crate::variant::best_of_n_sampling::{
    UninitializedBestOfNEvaluatorConfig, UninitializedBestOfNSamplingConfig,
};
use crate::variant::chain_of_thought::UninitializedChainOfThoughtConfig;
use crate::variant::chat_completion::{
    UninitializedChatCompletionConfig, UninitializedChatTemplate, UninitializedChatTemplates,
    UninitializedInputWrappers,
};
use crate::variant::dicl::UninitializedDiclConfig;
use crate::variant::mixture_of_n::{UninitializedFuserConfig, UninitializedMixtureOfNConfig};

use super::{UninitializedToolConfig, UninitializedVariantConfig, UninitializedVariantInfo};

// ─── File ref resolution ──────────────────────────────────────────────────

/// Preloaded stored files, keyed by version ID.
pub type FileMap = HashMap<Uuid, StoredFile>;

fn resolve_file_ref(
    file_ref: &StoredFileRef,
    files: &FileMap,
) -> Result<ResolvedTomlPathData, Error> {
    let template = files.get(&file_ref.file_version_id).ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: format!(
                "Missing stored file version `{}` (path: `{}`)",
                file_ref.file_version_id, file_ref.file_path
            ),
        })
    })?;
    Ok(ResolvedTomlPathData::new_fake_path(
        file_ref.file_path.clone(),
        template.source_body.clone(),
    ))
}

fn resolve_optional_file_ref(
    file_ref: Option<&StoredFileRef>,
    files: &FileMap,
) -> Result<Option<ResolvedTomlPathData>, Error> {
    file_ref.map(|r| resolve_file_ref(r, files)).transpose()
}

// ─── Prompt-dependent helper conversions ───────────────────────────────────

fn rehydrate_input_wrappers(
    stored: StoredInputWrappers,
    files: &FileMap,
) -> Result<UninitializedInputWrappers, Error> {
    let StoredInputWrappers {
        user,
        assistant,
        system,
    } = stored;
    Ok(UninitializedInputWrappers {
        user: resolve_optional_file_ref(user.as_ref(), files)?,
        assistant: resolve_optional_file_ref(assistant.as_ref(), files)?,
        system: resolve_optional_file_ref(system.as_ref(), files)?,
    })
}

// ─── Variant conversions ───────────────────────────────────────────────────

fn rehydrate_chat_completion(
    stored: StoredChatCompletionVariantConfig,
    files: &FileMap,
) -> Result<UninitializedChatCompletionConfig, Error> {
    let StoredChatCompletionVariantConfig {
        weight,
        model,
        system_template,
        user_template,
        assistant_template,
        input_wrappers,
        templates,
        temperature,
        top_p,
        max_tokens,
        presence_penalty,
        frequency_penalty,
        seed,
        json_mode,
        stop_sequences,
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
        retries,
        extra_body,
        extra_headers,
    } = stored;

    let resolved_templates = match templates {
        Some(map) => {
            let mut resolved = HashMap::new();
            for (key, file_ref) in map {
                let path = resolve_file_ref(&file_ref, files)?;
                resolved.insert(key, UninitializedChatTemplate { path });
            }
            UninitializedChatTemplates { inner: resolved }
        }
        None => UninitializedChatTemplates::default(),
    };

    let resolved_input_wrappers = match input_wrappers {
        Some(input_wrappers) => Some(rehydrate_input_wrappers(input_wrappers, files)?),
        None => None,
    };

    Ok(UninitializedChatCompletionConfig {
        weight,
        model,
        system_template: resolve_optional_file_ref(system_template.as_ref(), files)?,
        user_template: resolve_optional_file_ref(user_template.as_ref(), files)?,
        assistant_template: resolve_optional_file_ref(assistant_template.as_ref(), files)?,
        input_wrappers: resolved_input_wrappers,
        templates: resolved_templates,
        temperature,
        top_p,
        max_tokens,
        presence_penalty,
        frequency_penalty,
        seed,
        json_mode,
        stop_sequences,
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
        retries: retries.map(RetryConfig::from).unwrap_or_default(),
        extra_body: extra_body.map(ExtraBodyConfig::from),
        extra_headers: extra_headers.map(ExtraHeadersConfig::from),
    })
}

/// Rehydrates a stored variant version config into an `UninitializedVariantInfo`.
pub fn rehydrate_variant(
    stored: StoredVariantVersionConfig,
    files: &FileMap,
) -> Result<UninitializedVariantInfo, Error> {
    let StoredVariantVersionConfig {
        variant,
        timeouts,
        namespace,
    } = stored;

    let inner = match variant {
        StoredVariantConfig::ChatCompletion(c) => {
            UninitializedVariantConfig::ChatCompletion(rehydrate_chat_completion(c, files)?)
        }
        StoredVariantConfig::BestOfNSampling(b) => {
            let StoredBestOfNVariantConfig {
                weight,
                timeout_s,
                candidates,
                evaluator,
            } = b;
            #[expect(deprecated)]
            let config = UninitializedBestOfNSamplingConfig {
                weight,
                timeout_s,
                candidates: candidates.unwrap_or_default(),
                evaluator: UninitializedBestOfNEvaluatorConfig {
                    inner: rehydrate_chat_completion(evaluator, files)?,
                },
            };
            UninitializedVariantConfig::BestOfNSampling(config)
        }
        StoredVariantConfig::MixtureOfN(m) => {
            let StoredMixtureOfNVariantConfig {
                weight,
                timeout_s,
                candidates,
                fuser,
            } = m;
            #[expect(deprecated)]
            let config = UninitializedMixtureOfNConfig {
                weight,
                timeout_s,
                candidates: candidates.unwrap_or_default(),
                fuser: UninitializedFuserConfig {
                    inner: rehydrate_chat_completion(fuser, files)?,
                },
            };
            UninitializedVariantConfig::MixtureOfN(config)
        }
        StoredVariantConfig::Dicl(d) => {
            let StoredDiclVariantConfig {
                weight,
                embedding_model,
                k,
                model,
                system_instructions,
                temperature,
                top_p,
                max_tokens,
                presence_penalty,
                frequency_penalty,
                seed,
                json_mode,
                stop_sequences,
                reasoning_effort,
                thinking_budget_tokens,
                verbosity,
                max_distance,
                retries,
                extra_body,
                extra_headers,
            } = d;
            UninitializedVariantConfig::Dicl(UninitializedDiclConfig {
                weight,
                embedding_model,
                k,
                model,
                system_instructions: resolve_optional_file_ref(
                    system_instructions.as_ref(),
                    files,
                )?,
                temperature,
                top_p,
                max_tokens,
                presence_penalty,
                frequency_penalty,
                seed,
                json_mode,
                stop_sequences,
                reasoning_effort,
                thinking_budget_tokens,
                verbosity,
                max_distance,
                retries: retries.map(RetryConfig::from).unwrap_or_default(),
                extra_body: extra_body.map(ExtraBodyConfig::from),
                extra_headers: extra_headers.map(ExtraHeadersConfig::from),
            })
        }
        StoredVariantConfig::ChainOfThought(c) => {
            let config = UninitializedChainOfThoughtConfig {
                inner: rehydrate_chat_completion(c, files)?,
            };
            UninitializedVariantConfig::ChainOfThought(config)
        }
    };

    let ns = namespace.map(Namespace::new).transpose()?;

    Ok(UninitializedVariantInfo {
        inner,
        timeouts: timeouts.map(Into::into),
        namespace: ns,
    })
}

// ─── Evaluator conversions ─────────────────────────────────────────────────

fn rehydrate_llm_judge_chat_completion(
    stored: StoredLLMJudgeChatCompletionVariantConfig,
    files: &FileMap,
) -> Result<UninitializedLLMJudgeChatCompletionVariantConfig, Error> {
    let StoredLLMJudgeChatCompletionVariantConfig {
        active,
        model,
        system_instructions,
        temperature,
        top_p,
        max_tokens,
        presence_penalty,
        frequency_penalty,
        seed,
        json_mode,
        stop_sequences,
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
        retries,
        extra_body,
        extra_headers,
    } = stored;
    Ok(UninitializedLLMJudgeChatCompletionVariantConfig {
        active,
        model,
        system_instructions: resolve_file_ref(&system_instructions, files)?,
        temperature,
        top_p,
        max_tokens,
        presence_penalty,
        frequency_penalty,
        seed,
        json_mode,
        stop_sequences,
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
        retries: retries.map(RetryConfig::from).unwrap_or_default(),
        extra_body: extra_body.map(ExtraBodyConfig::from),
        extra_headers: extra_headers.map(ExtraHeadersConfig::from),
    })
}

fn rehydrate_llm_judge_variant(
    stored: StoredLLMJudgeVariantConfig,
    files: &FileMap,
) -> Result<UninitializedLLMJudgeVariantConfig, Error> {
    match stored {
        StoredLLMJudgeVariantConfig::ChatCompletion(c) => {
            Ok(UninitializedLLMJudgeVariantConfig::ChatCompletion(
                rehydrate_llm_judge_chat_completion(c, files)?,
            ))
        }
        StoredLLMJudgeVariantConfig::BestOfNSampling(b) => {
            let StoredLLMJudgeBestOfNVariantConfig {
                active,
                timeout_s,
                candidates,
                evaluator,
            } = b;
            #[expect(deprecated)]
            Ok(UninitializedLLMJudgeVariantConfig::BestOfNSampling(
                UninitializedLLMJudgeBestOfNVariantConfig {
                    active,
                    timeout_s,
                    candidates: candidates.unwrap_or_default(),
                    evaluator: rehydrate_llm_judge_chat_completion(evaluator, files)?,
                },
            ))
        }
        StoredLLMJudgeVariantConfig::MixtureOfNSampling(m) => {
            let StoredLLMJudgeMixtureOfNVariantConfig {
                active,
                timeout_s,
                candidates,
                fuser,
            } = m;
            #[expect(deprecated)]
            Ok(UninitializedLLMJudgeVariantConfig::MixtureOfNSampling(
                UninitializedLLMJudgeMixtureOfNVariantConfig {
                    active,
                    timeout_s,
                    candidates: candidates.unwrap_or_default(),
                    fuser: rehydrate_llm_judge_chat_completion(fuser, files)?,
                },
            ))
        }
        StoredLLMJudgeVariantConfig::Dicl(d) => {
            let StoredLLMJudgeDiclVariantConfig {
                active,
                embedding_model,
                k,
                model,
                system_instructions,
                temperature,
                top_p,
                presence_penalty,
                frequency_penalty,
                max_tokens,
                seed,
                json_mode,
                stop_sequences,
                extra_body,
                retries,
                extra_headers,
            } = d;
            Ok(UninitializedLLMJudgeVariantConfig::Dicl(
                UninitializedLLMJudgeDiclVariantConfig {
                    active,
                    embedding_model,
                    k,
                    model,
                    system_instructions: resolve_optional_file_ref(
                        system_instructions.as_ref(),
                        files,
                    )?,
                    temperature,
                    top_p,
                    presence_penalty,
                    frequency_penalty,
                    max_tokens,
                    seed,
                    json_mode,
                    stop_sequences,
                    extra_body: extra_body.map(ExtraBodyConfig::from),
                    retries: retries.map(RetryConfig::from).unwrap_or_default(),
                    extra_headers: extra_headers.map(ExtraHeadersConfig::from),
                },
            ))
        }
        StoredLLMJudgeVariantConfig::ChainOfThought(c) => {
            Ok(UninitializedLLMJudgeVariantConfig::ChainOfThought(
                UninitializedLLMJudgeChainOfThoughtVariantConfig {
                    inner: rehydrate_llm_judge_chat_completion(c.inner, files)?,
                },
            ))
        }
    }
}

fn rehydrate_evaluator(
    stored: StoredEvaluatorConfig,
    files: &FileMap,
) -> Result<UninitializedEvaluatorConfig, Error> {
    match stored {
        StoredEvaluatorConfig::ExactMatch(e) => {
            Ok(UninitializedEvaluatorConfig::ExactMatch(e.into()))
        }
        StoredEvaluatorConfig::Regex(r) => Ok(UninitializedEvaluatorConfig::Regex(r.into())),
        StoredEvaluatorConfig::ToolUse(t) => Ok(UninitializedEvaluatorConfig::ToolUse(t.into())),
        StoredEvaluatorConfig::LLMJudge(j) => {
            let StoredLLMJudgeConfig {
                input_format,
                variants,
                output_type,
                optimize,
                cutoff,
                include,
                description,
            } = j;
            let rehydrated_variants = variants
                .unwrap_or_default()
                .into_iter()
                .map(|(name, vi)| {
                    let StoredLLMJudgeVariantInfo { variant, timeouts } = vi;
                    let inner = rehydrate_llm_judge_variant(variant, files)?;
                    Ok((
                        name,
                        UninitializedLLMJudgeVariantInfo {
                            inner,
                            timeouts: timeouts.map(Into::into),
                        },
                    ))
                })
                .collect::<Result<HashMap<_, _>, Error>>()?;
            #[expect(deprecated)]
            let config = UninitializedLLMJudgeConfig {
                input_format: input_format.map(Into::into),
                variants: rehydrated_variants,
                output_type: output_type.into(),
                optimize: optimize.into(),
                cutoff,
                include: include.map(Into::into),
                description,
            };
            Ok(UninitializedEvaluatorConfig::LLMJudge(config))
        }
        StoredEvaluatorConfig::Typescript(t) => Ok(UninitializedEvaluatorConfig::TypescriptJudge(
            UninitializedTypescriptJudgeConfig {
                typescript_file: ResolvedTomlPathData::new_fake_path(
                    "stored::typescript_evaluator".to_string(),
                    t.typescript_code,
                ),
                output_type: t.output_type.into(),
                optimize: t.optimize.into(),
            },
        )),
    }
}

// ─── Function conversions ──────────────────────────────────────────────────

/// Rehydrates a stored function config into an `UninitializedFunctionConfig`.
///
/// `variant_rows` maps variant_version_id -> (variant_name, stored_config).
/// Rehydrates a stored function config. Returns the config alongside any
/// per-variant errors — the function succeeds even if some variants fail.
/// The outer `Err` is reserved for function-level failures (e.g. broken schema
/// file references) that make the entire function unloadable.
pub fn rehydrate_function(
    stored: StoredFunctionConfig,
    variant_rows: &HashMap<Uuid, (String, StoredVariantVersionConfig)>,
    files: &FileMap,
) -> Result<(UninitializedFunctionConfig, Vec<(String, Error)>), Error> {
    match stored {
        StoredFunctionConfig::Chat(chat) => rehydrate_chat_function(chat, variant_rows, files)
            .map(|(config, errs)| (UninitializedFunctionConfig::Chat(config), errs)),
        StoredFunctionConfig::Json(json) => rehydrate_json_function(json, variant_rows, files)
            .map(|(config, errs)| (UninitializedFunctionConfig::Json(config), errs)),
    }
}

/// Resolves variant refs to fully rehydrated variant infos.
///
/// Per-variant failures (missing version, broken file references) are collected
/// and returned alongside the successfully loaded variants rather than failing
/// the entire function. The outer `Result` is reserved for function-level
/// failures that make the function itself unloadable.
fn resolve_variants(
    stored_variants: Option<BTreeMap<String, tensorzero_stored_config::StoredVariantRef>>,
    variant_rows: &HashMap<Uuid, (String, StoredVariantVersionConfig)>,
    files: &FileMap,
) -> (
    HashMap<String, UninitializedVariantInfo>,
    Vec<(String, Error)>,
) {
    let mut result = HashMap::new();
    let mut errors = Vec::new();
    for (name, vref) in stored_variants.unwrap_or_default() {
        let Some((_, stored_variant)) = variant_rows.get(&vref.variant_version_id) else {
            errors.push((
                name.clone(),
                Error::new(ErrorDetails::Config {
                    message: format!(
                        "Missing or broken variant version `{}` for variant `{name}`",
                        vref.variant_version_id
                    ),
                }),
            ));
            continue;
        };
        match rehydrate_variant(stored_variant.clone(), files) {
            Ok(info) => {
                result.insert(name, info);
            }
            Err(error) => {
                errors.push((name, error));
            }
        }
    }
    (result, errors)
}

struct ResolvedSchemas {
    system: Option<ResolvedTomlPathData>,
    user: Option<ResolvedTomlPathData>,
    assistant: Option<ResolvedTomlPathData>,
    schemas: UninitializedSchemas,
}

fn resolve_schemas(
    system_schema: Option<&StoredFileRef>,
    user_schema: Option<&StoredFileRef>,
    assistant_schema: Option<&StoredFileRef>,
    schemas: Option<BTreeMap<String, StoredFileRef>>,
    files: &FileMap,
) -> Result<ResolvedSchemas, Error> {
    let system = resolve_optional_file_ref(system_schema, files)?;
    let user = resolve_optional_file_ref(user_schema, files)?;
    let assistant = resolve_optional_file_ref(assistant_schema, files)?;
    let schemas = match schemas {
        Some(map) => {
            let mut paths = HashMap::new();
            for (key, file_ref) in map {
                paths.insert(key, resolve_file_ref(&file_ref, files)?);
            }
            UninitializedSchemas::from_paths(paths)
        }
        None => UninitializedSchemas::default(),
    };
    Ok(ResolvedSchemas {
        system,
        user,
        assistant,
        schemas,
    })
}

fn resolve_evaluators(
    stored: Option<BTreeMap<String, StoredEvaluatorConfig>>,
    files: &FileMap,
) -> Result<HashMap<String, UninitializedEvaluatorConfig>, Error> {
    stored
        .unwrap_or_default()
        .into_iter()
        .map(|(name, eval)| {
            let rehydrated = rehydrate_evaluator(eval, files)?;
            Ok((name, rehydrated))
        })
        .collect()
}

fn rehydrate_chat_function(
    stored: tensorzero_stored_config::StoredChatFunctionConfig,
    variant_rows: &HashMap<Uuid, (String, StoredVariantVersionConfig)>,
    files: &FileMap,
) -> Result<(UninitializedFunctionConfigChat, Vec<(String, Error)>), Error> {
    let tensorzero_stored_config::StoredChatFunctionConfig {
        variants,
        system_schema,
        user_schema,
        assistant_schema,
        schemas,
        tools,
        tool_choice,
        parallel_tool_calls,
        description,
        experimentation,
        evaluators,
    } = stored;

    let (resolved_variants, variant_errors) = resolve_variants(variants, variant_rows, files);
    let resolved = resolve_schemas(
        system_schema.as_ref(),
        user_schema.as_ref(),
        assistant_schema.as_ref(),
        schemas,
        files,
    )?;

    Ok((
        UninitializedFunctionConfigChat {
            variants: resolved_variants,
            system_schema: resolved.system,
            user_schema: resolved.user,
            assistant_schema: resolved.assistant,
            schemas: resolved.schemas,
            tools: tools.unwrap_or_default(),
            tool_choice: tool_choice.map(ToolChoice::from).unwrap_or_default(),
            parallel_tool_calls,
            description,
            experimentation: experimentation.map(Into::into),
            evaluators: resolve_evaluators(evaluators, files)?,
        },
        variant_errors,
    ))
}

fn rehydrate_json_function(
    stored: tensorzero_stored_config::StoredJsonFunctionConfig,
    variant_rows: &HashMap<Uuid, (String, StoredVariantVersionConfig)>,
    files: &FileMap,
) -> Result<(UninitializedFunctionConfigJson, Vec<(String, Error)>), Error> {
    let tensorzero_stored_config::StoredJsonFunctionConfig {
        variants,
        system_schema,
        user_schema,
        assistant_schema,
        schemas,
        output_schema,
        description,
        experimentation,
        evaluators,
    } = stored;

    let (resolved_variants, variant_errors) = resolve_variants(variants, variant_rows, files);
    let resolved = resolve_schemas(
        system_schema.as_ref(),
        user_schema.as_ref(),
        assistant_schema.as_ref(),
        schemas,
        files,
    )?;

    Ok((
        UninitializedFunctionConfigJson {
            variants: resolved_variants,
            system_schema: resolved.system,
            user_schema: resolved.user,
            assistant_schema: resolved.assistant,
            schemas: resolved.schemas,
            output_schema: resolve_optional_file_ref(output_schema.as_ref(), files)?,
            description,
            experimentation: experimentation.map(Into::into),
            evaluators: resolve_evaluators(evaluators, files)?,
        },
        variant_errors,
    ))
}

// ─── Tool conversions ──────────────────────────────────────────────────────

pub fn rehydrate_tool(
    stored: StoredToolConfig,
    files: &FileMap,
) -> Result<UninitializedToolConfig, Error> {
    let StoredToolConfig {
        description,
        parameters,
        name,
        strict,
    } = stored;
    Ok(UninitializedToolConfig {
        description,
        parameters: resolve_file_ref(&parameters, files)?,
        name,
        strict,
    })
}

// ─── Evaluation conversions ────────────────────────────────────────────────

pub fn rehydrate_evaluation(
    stored: StoredEvaluationConfig,
    files: &FileMap,
) -> Result<UninitializedEvaluationConfig, Error> {
    match stored {
        StoredEvaluationConfig::Inference(i) => {
            let tensorzero_stored_config::StoredInferenceEvaluationConfig {
                evaluators,
                function_name,
                description,
            } = i;
            Ok(UninitializedEvaluationConfig::Inference(
                UninitializedInferenceEvaluationConfig {
                    evaluators: resolve_evaluators(evaluators, files)?,
                    function_name,
                    description,
                },
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use googletest::prelude::*;
    use tensorzero_stored_config::{
        StoredChatFunctionConfig, StoredExactMatchConfig, StoredFile,
        StoredInferenceEvaluationConfig, StoredJsonFunctionConfig, StoredLLMJudgeConfig,
        StoredLLMJudgeOptimize, StoredLLMJudgeOutputType, StoredVariantRef,
    };
    use tensorzero_types::inference_params::JsonMode;

    use super::*;

    fn stored_file(id: Uuid, source_body: &str) -> StoredFile {
        StoredFile {
            id,
            file_path: format!("ignored-{id}"),
            source_body: source_body.to_string(),
            content_hash: Vec::new(),
            creation_source: "test".to_string(),
            source_autopilot_session_id: None,
        }
    }

    fn file_ref(id: Uuid, file_path: &str) -> StoredFileRef {
        StoredFileRef {
            file_version_id: id,
            file_path: file_path.to_string(),
        }
    }

    #[gtest]
    fn resolve_file_ref_uses_ref_key_and_template_body() -> Result<()> {
        let prompt_id = Uuid::now_v7();
        let files = FileMap::from([(prompt_id, stored_file(prompt_id, "system prompt contents"))]);

        let resolved = resolve_file_ref(&file_ref(prompt_id, "templates/system"), &files)?;

        expect_that!(resolved.get_template_key(), eq("templates/system"));
        expect_that!(resolved.data(), eq("system prompt contents"));
        Ok(())
    }

    #[gtest]
    fn rehydrate_variant_chat_completion_resolves_file_refs() -> Result<()> {
        let system_prompt_id = Uuid::now_v7();
        let template_prompt_id = Uuid::now_v7();
        let files = FileMap::from([
            (
                system_prompt_id,
                stored_file(system_prompt_id, "system body"),
            ),
            (
                template_prompt_id,
                stored_file(template_prompt_id, "template body"),
            ),
        ]);
        let stored = StoredVariantVersionConfig {
            variant: StoredVariantConfig::ChatCompletion(StoredChatCompletionVariantConfig {
                weight: Some(0.5),
                model: Arc::<str>::from("gpt-4o-mini"),
                system_template: Some(file_ref(system_prompt_id, "files/system")),
                user_template: None,
                assistant_template: None,
                input_wrappers: None,
                templates: Some(BTreeMap::from([(
                    "custom".to_string(),
                    file_ref(template_prompt_id, "files/custom"),
                )])),
                temperature: Some(0.2),
                top_p: None,
                max_tokens: Some(512),
                presence_penalty: None,
                frequency_penalty: None,
                seed: None,
                json_mode: Some(JsonMode::Off),
                stop_sequences: None,
                reasoning_effort: None,
                service_tier: None,
                thinking_budget_tokens: None,
                verbosity: None,
                retries: None,
                extra_body: None,
                extra_headers: None,
            }),
            timeouts: Some(tensorzero_stored_config::StoredTimeoutsConfig {
                non_streaming: Some(tensorzero_stored_config::StoredNonStreamingTimeouts {
                    total_ms: Some(500),
                }),
                streaming: None,
            }),
            namespace: Some("tenant_a".to_string()),
        };

        let rehydrated = rehydrate_variant(stored, &files)?;

        expect_that!(
            rehydrated.namespace.as_ref().map(Namespace::as_str),
            some(eq("tenant_a"))
        );
        expect_that!(
            rehydrated
                .timeouts
                .as_ref()
                .and_then(|t| t.non_streaming.as_ref())
                .and_then(|ns| ns.total_ms),
            some(eq(500))
        );

        let UninitializedVariantConfig::ChatCompletion(chat) = rehydrated.inner else {
            panic!("expected chat completion variant");
        };
        expect_that!(
            chat.system_template
                .as_ref()
                .map(ResolvedTomlPathData::get_template_key),
            some(eq("files/system"))
        );
        expect_that!(
            chat.system_template
                .as_ref()
                .map(ResolvedTomlPathData::data),
            some(eq("system body"))
        );
        expect_that!(
            chat.templates
                .inner
                .get("custom")
                .map(|template| template.path.get_template_key()),
            some(eq("files/custom"))
        );
        expect_that!(
            chat.templates
                .inner
                .get("custom")
                .map(|template| template.path.data()),
            some(eq("template body"))
        );
        Ok(())
    }

    #[gtest]
    fn rehydrate_function_collects_missing_variant_as_per_variant_error() {
        let stored = StoredFunctionConfig::Chat(StoredChatFunctionConfig {
            variants: Some(BTreeMap::from([(
                "primary".to_string(),
                StoredVariantRef {
                    variant_version_id: Uuid::now_v7(),
                },
            )])),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            schemas: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            description: None,
            experimentation: None,
            evaluators: None,
        });

        let (config, variant_errors) = rehydrate_function(stored, &HashMap::new(), &FileMap::new())
            .expect("function-level rehydration should succeed even with broken variants");

        // The function itself loaded with an empty variants map
        let UninitializedFunctionConfig::Chat(chat) = config else {
            panic!("expected Chat variant");
        };
        expect_that!(chat.variants.is_empty(), eq(true));

        // The missing variant was collected as a per-variant error
        assert_that!(variant_errors.len(), eq(1));
        let (variant_name, error) = &variant_errors[0];
        expect_that!(variant_name, eq("primary"));
        expect_that!(
            error.to_string(),
            contains_substring("Missing or broken variant version")
        );
    }

    #[gtest]
    fn rehydrate_evaluation_inference_preserves_function_name() -> Result<()> {
        let evaluation = StoredEvaluationConfig::Inference(StoredInferenceEvaluationConfig {
            evaluators: None,
            function_name: "answer_question".to_string(),
            description: Some("eval description".to_string()),
        });

        let rehydrated = rehydrate_evaluation(evaluation, &FileMap::new())?;
        let UninitializedEvaluationConfig::Inference(config) = rehydrated;

        expect_that!(config.function_name, eq("answer_question"));
        expect_that!(config.description, some(eq("eval description")));
        Ok(())
    }

    // ─── Prompt-resolving conversions ──────────────────────────────────────

    fn single_file_map(body: &str) -> (Uuid, FileMap) {
        let id = Uuid::now_v7();
        let map = FileMap::from([(id, stored_file(id, body))]);
        (id, map)
    }

    fn minimal_chat_completion(system_prompt_id: Uuid) -> StoredChatCompletionVariantConfig {
        StoredChatCompletionVariantConfig {
            weight: None,
            model: Arc::<str>::from("gpt-4o-mini"),
            system_template: Some(file_ref(system_prompt_id, "files/system")),
            user_template: None,
            assistant_template: None,
            input_wrappers: None,
            templates: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            json_mode: None,
            stop_sequences: None,
            reasoning_effort: None,
            service_tier: None,
            thinking_budget_tokens: None,
            verbosity: None,
            retries: None,
            extra_body: None,
            extra_headers: None,
        }
    }

    #[gtest]
    fn rehydrate_tool_resolves_parameters_file_ref() -> Result<()> {
        let (prompt_id, files) = single_file_map("{\"type\":\"object\"}");
        let rehydrated = rehydrate_tool(
            StoredToolConfig {
                description: "look up weather".to_string(),
                parameters: file_ref(prompt_id, "tools/weather/params"),
                name: Some("weather".to_string()),
                strict: true,
            },
            &files,
        )?;
        expect_that!(rehydrated.description, eq("look up weather"));
        expect_that!(rehydrated.name, some(eq("weather")));
        expect_that!(rehydrated.strict, eq(true));
        expect_that!(
            rehydrated.parameters.get_template_key(),
            eq("tools/weather/params")
        );
        expect_that!(rehydrated.parameters.data(), eq("{\"type\":\"object\"}"));
        Ok(())
    }

    #[gtest]
    fn rehydrate_variant_best_of_n_resolves_evaluator() -> Result<()> {
        let (prompt_id, files) = single_file_map("system body");
        let stored = StoredVariantVersionConfig {
            variant: StoredVariantConfig::BestOfNSampling(StoredBestOfNVariantConfig {
                weight: Some(1.0),
                timeout_s: Some(2.0),
                candidates: Some(vec!["a".to_string(), "b".to_string()]),
                evaluator: minimal_chat_completion(prompt_id),
            }),
            timeouts: None,
            namespace: None,
        };
        let rehydrated = rehydrate_variant(stored, &files)?;
        let UninitializedVariantConfig::BestOfNSampling(cfg) = rehydrated.inner else {
            panic!("expected best-of-n variant");
        };
        expect_that!(cfg.candidates, elements_are![eq("a"), eq("b")]);
        expect_that!(
            cfg.evaluator
                .inner
                .system_template
                .as_ref()
                .map(ResolvedTomlPathData::data),
            some(eq("system body"))
        );
        Ok(())
    }

    #[gtest]
    fn rehydrate_variant_mixture_of_n_resolves_fuser() -> Result<()> {
        let (prompt_id, files) = single_file_map("fuser body");
        let stored = StoredVariantVersionConfig {
            variant: StoredVariantConfig::MixtureOfN(StoredMixtureOfNVariantConfig {
                weight: None,
                timeout_s: None,
                candidates: None,
                fuser: minimal_chat_completion(prompt_id),
            }),
            timeouts: None,
            namespace: None,
        };
        let rehydrated = rehydrate_variant(stored, &files)?;
        let UninitializedVariantConfig::MixtureOfN(cfg) = rehydrated.inner else {
            panic!("expected mixture-of-n variant");
        };
        expect_that!(cfg.candidates, is_empty());
        expect_that!(
            cfg.fuser
                .inner
                .system_template
                .as_ref()
                .map(ResolvedTomlPathData::data),
            some(eq("fuser body"))
        );
        Ok(())
    }

    #[gtest]
    fn rehydrate_variant_chain_of_thought_resolves_inner() -> Result<()> {
        let (prompt_id, files) = single_file_map("cot body");
        let stored = StoredVariantVersionConfig {
            variant: StoredVariantConfig::ChainOfThought(minimal_chat_completion(prompt_id)),
            timeouts: None,
            namespace: None,
        };
        let rehydrated = rehydrate_variant(stored, &files)?;
        let UninitializedVariantConfig::ChainOfThought(cfg) = rehydrated.inner else {
            panic!("expected chain-of-thought variant");
        };
        expect_that!(
            cfg.inner
                .system_template
                .as_ref()
                .map(ResolvedTomlPathData::data),
            some(eq("cot body"))
        );
        Ok(())
    }

    #[gtest]
    fn rehydrate_variant_dicl_resolves_system_instructions() -> Result<()> {
        let (prompt_id, files) = single_file_map("judge me");
        let stored = StoredVariantVersionConfig {
            variant: StoredVariantConfig::Dicl(StoredDiclVariantConfig {
                weight: Some(0.25),
                embedding_model: "dummy-embed".to_string(),
                k: 3,
                model: "dummy-model".to_string(),
                system_instructions: Some(file_ref(prompt_id, "files/judge")),
                temperature: Some(0.1),
                top_p: None,
                max_tokens: None,
                presence_penalty: None,
                frequency_penalty: None,
                seed: None,
                json_mode: None,
                stop_sequences: None,
                reasoning_effort: None,
                thinking_budget_tokens: None,
                verbosity: None,
                max_distance: None,
                retries: None,
                extra_body: None,
                extra_headers: None,
            }),
            timeouts: None,
            namespace: None,
        };
        let rehydrated = rehydrate_variant(stored, &files)?;
        let UninitializedVariantConfig::Dicl(cfg) = rehydrated.inner else {
            panic!("expected dicl variant");
        };
        expect_that!(cfg.k, eq(3));
        expect_that!(cfg.embedding_model, eq("dummy-embed"));
        expect_that!(cfg.model, eq("dummy-model"));
        expect_that!(
            cfg.system_instructions
                .as_ref()
                .map(ResolvedTomlPathData::data),
            some(eq("judge me"))
        );
        Ok(())
    }

    #[gtest]
    fn rehydrate_variant_chat_completion_resolves_input_wrappers() -> Result<()> {
        let user_id = Uuid::now_v7();
        let assistant_id = Uuid::now_v7();
        let system_id = Uuid::now_v7();
        let files = FileMap::from([
            (user_id, stored_file(user_id, "user wrapper")),
            (assistant_id, stored_file(assistant_id, "assistant wrapper")),
            (system_id, stored_file(system_id, "system wrapper")),
        ]);
        let mut chat = minimal_chat_completion(system_id);
        chat.input_wrappers = Some(StoredInputWrappers {
            user: Some(file_ref(user_id, "wrappers/user")),
            assistant: Some(file_ref(assistant_id, "wrappers/assistant")),
            system: Some(file_ref(system_id, "wrappers/system")),
        });
        let stored = StoredVariantVersionConfig {
            variant: StoredVariantConfig::ChatCompletion(chat),
            timeouts: None,
            namespace: None,
        };

        let rehydrated = rehydrate_variant(stored, &files)?;
        let UninitializedVariantConfig::ChatCompletion(chat) = rehydrated.inner else {
            panic!("expected chat completion variant");
        };
        let wrappers = chat
            .input_wrappers
            .as_ref()
            .expect("input wrappers should be set");
        expect_that!(
            wrappers.user.as_ref().map(ResolvedTomlPathData::data),
            some(eq("user wrapper"))
        );
        expect_that!(
            wrappers.assistant.as_ref().map(ResolvedTomlPathData::data),
            some(eq("assistant wrapper"))
        );
        expect_that!(
            wrappers.system.as_ref().map(ResolvedTomlPathData::data),
            some(eq("system wrapper"))
        );
        Ok(())
    }

    #[gtest]
    fn rehydrate_function_json_resolves_schemas_and_variants() -> Result<()> {
        let system_prompt_id = Uuid::now_v7();
        let output_schema_id = Uuid::now_v7();
        let named_schema_id = Uuid::now_v7();
        let files = FileMap::from([
            (system_prompt_id, stored_file(system_prompt_id, "sys body")),
            (
                output_schema_id,
                stored_file(output_schema_id, "{\"type\":\"string\"}"),
            ),
            (
                named_schema_id,
                stored_file(named_schema_id, "{\"type\":\"number\"}"),
            ),
        ]);
        let variant_version_id = Uuid::now_v7();
        let variant_rows = HashMap::from([(
            variant_version_id,
            (
                "primary".to_string(),
                StoredVariantVersionConfig {
                    variant: StoredVariantConfig::ChatCompletion(minimal_chat_completion(
                        system_prompt_id,
                    )),
                    timeouts: None,
                    namespace: None,
                },
            ),
        )]);

        let stored = StoredFunctionConfig::Json(StoredJsonFunctionConfig {
            variants: Some(BTreeMap::from([(
                "primary".to_string(),
                StoredVariantRef { variant_version_id },
            )])),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            schemas: Some(BTreeMap::from([(
                "extra".to_string(),
                file_ref(named_schema_id, "schemas/extra"),
            )])),
            output_schema: Some(file_ref(output_schema_id, "schemas/output")),
            description: Some("a json function".to_string()),
            experimentation: None,
            evaluators: None,
        });

        let (rehydrated, variant_errors) = rehydrate_function(stored, &variant_rows, &files)?;
        assert_that!(variant_errors.is_empty(), eq(true));
        let UninitializedFunctionConfig::Json(json) = rehydrated else {
            panic!("expected json function");
        };
        expect_that!(json.description, some(eq("a json function")));
        expect_that!(
            json.output_schema.as_ref().map(ResolvedTomlPathData::data),
            some(eq("{\"type\":\"string\"}"))
        );
        let schemas: HashMap<_, _> = json
            .schemas
            .iter()
            .map(|(name, path)| (name.clone(), path.data().to_string()))
            .collect();
        expect_that!(
            schemas.get("extra").map(String::as_str),
            some(eq("{\"type\":\"number\"}"))
        );
        expect_that!(json.variants.contains_key("primary"), eq(true));
        Ok(())
    }

    #[gtest]
    fn rehydrate_evaluation_with_llm_judge_evaluator_resolves_variants() -> Result<()> {
        let system_instructions_id = Uuid::now_v7();
        let files = FileMap::from([(
            system_instructions_id,
            stored_file(system_instructions_id, "grade me"),
        )]);

        let judge = StoredLLMJudgeConfig {
            input_format: None,
            variants: Some(BTreeMap::from([(
                "judge-variant".to_string(),
                StoredLLMJudgeVariantInfo {
                    variant: StoredLLMJudgeVariantConfig::ChatCompletion(
                        StoredLLMJudgeChatCompletionVariantConfig {
                            active: Some(true),
                            model: Arc::<str>::from("gpt-4o-mini"),
                            system_instructions: file_ref(system_instructions_id, "files/judge"),
                            temperature: None,
                            top_p: None,
                            max_tokens: None,
                            presence_penalty: None,
                            frequency_penalty: None,
                            seed: None,
                            json_mode: JsonMode::Off,
                            stop_sequences: None,
                            reasoning_effort: None,
                            service_tier: None,
                            thinking_budget_tokens: None,
                            verbosity: None,
                            retries: None,
                            extra_body: None,
                            extra_headers: None,
                        },
                    ),
                    timeouts: None,
                },
            )])),
            output_type: StoredLLMJudgeOutputType::Boolean,
            optimize: StoredLLMJudgeOptimize::Max,
            cutoff: None,
            include: None,
            description: Some("judge".to_string()),
        };

        let evaluation = StoredEvaluationConfig::Inference(StoredInferenceEvaluationConfig {
            evaluators: Some(BTreeMap::from([
                (
                    "exact".to_string(),
                    StoredEvaluatorConfig::ExactMatch(StoredExactMatchConfig { cutoff: None }),
                ),
                ("judge".to_string(), StoredEvaluatorConfig::LLMJudge(judge)),
            ])),
            function_name: "my_fn".to_string(),
            description: None,
        });

        let rehydrated = rehydrate_evaluation(evaluation, &files)?;
        let UninitializedEvaluationConfig::Inference(config) = rehydrated;
        expect_that!(config.function_name, eq("my_fn"));
        let judge_evaluator = config
            .evaluators
            .get("judge")
            .expect("judge evaluator should exist");
        let UninitializedEvaluatorConfig::LLMJudge(judge_cfg) = judge_evaluator else {
            panic!("expected LLMJudge evaluator");
        };
        let variant = judge_cfg
            .variants
            .get("judge-variant")
            .expect("judge variant should exist");
        let UninitializedLLMJudgeVariantConfig::ChatCompletion(chat) = &variant.inner else {
            panic!("expected chat completion judge variant");
        };
        expect_that!(chat.system_instructions.data(), eq("grade me"));
        expect_that!(config.evaluators.contains_key("exact"), eq(true));
        Ok(())
    }
}
