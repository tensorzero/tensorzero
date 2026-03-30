use axum::extract::{Path, State};
use axum::Json;
use tracing::instrument;

use crate::config::Config;
use crate::endpoints::openai_compatible::types::models::{OpenAIModel, OpenAIModelsListResponse};
use crate::error::{Error, ErrorDetails};
use crate::model_table::RESERVED_MODEL_PREFIXES;
use crate::utils::gateway::{AppState, AppStateData};

fn extract_provider_from_model_name(model_name: &str) -> String {
    for prefix in RESERVED_MODEL_PREFIXES.iter() {
        if let Some(provider_and_model) = model_name.strip_prefix(prefix) {
            if !provider_and_model.is_empty() {
                let provider = &prefix[..prefix.len() - 2];
                return provider.to_string();
            }
        }
    }
    "tensorzero".to_string()
}

fn get_all_models(config: &Config) -> Vec<OpenAIModel> {
    let mut models = Vec::new();

    for (model_name, _model_config) in config.models.iter_static_models() {
        let owned_by = extract_provider_from_model_name(model_name);
        models.push(OpenAIModel::new(model_name.to_string(), owned_by));
    }

    for (model_name, _model_config) in config.embedding_models.iter_static_models() {
        let owned_by = extract_provider_from_model_name(model_name);
        models.push(OpenAIModel::new(model_name.to_string(), owned_by));
    }

    models.sort_by(|a, b| a.id.cmp(&b.id));
    models
}

fn find_model(config: &Config, model_name: &str) -> Option<OpenAIModel> {
    if config.models.table.contains_key(model_name)
        || config.embedding_models.table.contains_key(model_name)
    {
        let owned_by = extract_provider_from_model_name(model_name);
        return Some(OpenAIModel::new(model_name.to_string(), owned_by));
    }

    if let Some(provider_type) = check_shorthand(model_name) {
        return Some(OpenAIModel::new(model_name.to_string(), provider_type.to_string()));
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
