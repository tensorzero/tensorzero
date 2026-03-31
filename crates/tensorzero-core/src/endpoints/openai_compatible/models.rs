use axum::extract::{Path, State};
use axum::Json;
use tracing::instrument;

use crate::config::Config;
use crate::endpoints::openai_compatible::types::models::{OpenAIModel, OpenAIModelsListResponse};
use crate::error::{Error, ErrorDetails};
use crate::model_table::RESERVED_MODEL_PREFIXES;
use crate::utils::gateway::{AppState, AppStateData};

const MODEL_ID_PREFIX: &str = "tensorzero::model_name::";
const EMBEDDING_MODEL_ID_PREFIX: &str = "tensorzero::embedding_model_name::";
const FUNCTION_ID_PREFIX: &str = "tensorzero::function_name::";

fn format_model_id(model_name: &str) -> String {
    format!("{}{}", MODEL_ID_PREFIX, model_name)
}

fn format_embedding_model_id(model_name: &str) -> String {
    format!("{}{}", EMBEDDING_MODEL_ID_PREFIX, model_name)
}

fn format_function_id(function_name: &str) -> String {
    format!("{}{}", FUNCTION_ID_PREFIX, function_name)
}

fn get_all_models(config: &Config) -> Vec<OpenAIModel> {
    let mut models = Vec::new();
    let owned_by = config.gateway.openai_models_owned_by.clone();

    for (model_name, _model_config) in config.models.iter_static_models() {
        let id = format_model_id(model_name);
        models.push(OpenAIModel::new(id, owned_by.clone()));
    }

    for (model_name, _model_config) in config.embedding_models.iter_static_models() {
        let id = format_embedding_model_id(model_name);
        models.push(OpenAIModel::new(id, owned_by.clone()));
    }

    for (function_name, _function_config) in &config.functions {
        if function_name.starts_with("tensorzero::") {
            continue;
        }
        let id = format_function_id(function_name);
        models.push(OpenAIModel::new(id, owned_by.clone()));
    }

    models.sort_by(|a, b| a.id.cmp(&b.id));
    models
}

fn find_model(config: &Config, model_name: &str) -> Option<OpenAIModel> {
    let owned_by = config.gateway.openai_models_owned_by.clone();

    if config.models.table.contains_key(model_name) {
        let id = format_model_id(model_name);
        return Some(OpenAIModel::new(id, owned_by));
    }

    if config.embedding_models.table.contains_key(model_name) {
        let id = format_embedding_model_id(model_name);
        return Some(OpenAIModel::new(id, owned_by));
    }

    if config.functions.contains_key(model_name) && !model_name.starts_with("tensorzero::") {
        let id = format_function_id(model_name);
        return Some(OpenAIModel::new(id, owned_by));
    }

    if let Some(name) = model_name.strip_prefix(MODEL_ID_PREFIX) {
        if config.models.table.contains_key(name) {
            let id = format_model_id(name);
            return Some(OpenAIModel::new(id, owned_by));
        }
    }

    if let Some(name) = model_name.strip_prefix(EMBEDDING_MODEL_ID_PREFIX) {
        if config.embedding_models.table.contains_key(name) {
            let id = format_embedding_model_id(name);
            return Some(OpenAIModel::new(id, owned_by));
        }
    }

    if let Some(name) = model_name.strip_prefix(FUNCTION_ID_PREFIX) {
        if config.functions.contains_key(name) && !name.starts_with("tensorzero::") {
            let id = format_function_id(name);
            return Some(OpenAIModel::new(id, owned_by));
        }
    }

    if let Some(provider_type) = check_shorthand(model_name) {
        let id = format_model_id(model_name);
        return Some(OpenAIModel::new(id, provider_type.to_string()));
    }

    None
}

fn check_shorthand(model_name: &str) -> Option<&'static str> {
    for prefix in RESERVED_MODEL_PREFIXES.iter() {
        if let Some(name) = model_name.strip_prefix(prefix) {
            if !name.is_empty() {
                let provider = &prefix[..prefix.len() - 2];
                return Some(provider);
            }
        }
    }
    None
}

#[instrument(name = "models.list", skip_all)]
pub async fn list_models_handler(
    State(AppStateData { config, .. }): AppState,
) -> Result<Json<OpenAIModelsListResponse>, Error> {
    let models = get_all_models(&config);
    Ok(Json(OpenAIModelsListResponse::new(models)))
}

#[instrument(name = "models.retrieve", skip_all)]
pub async fn retrieve_model_handler(
    State(AppStateData { config, .. }): AppState,
    Path(model_name): Path<String>,
) -> Result<Json<OpenAIModel>, Error> {
    match find_model(&config, &model_name) {
        Some(model) => Ok(Json(model)),
        None => Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: format!("Model '{}' not found", model_name),
        })),
    }
}
