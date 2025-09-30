use std::{
    collections::HashMap,
    env,
    fmt::Display,
    fs,
    ops::Deref,
    sync::{Arc, OnceLock},
};

use secrecy::SecretString;
use serde::Serialize;

use crate::{
    config::{provider_types::ProviderTypesConfig, skip_credential_validation},
    error::{Error, ErrorDetails},
    model::{Credential, CredentialLocation, UninitializedProviderConfig},
};
use lazy_static::lazy_static;
use strum::VariantNames;

// Reserve prefixes for all supported providers, regardless of whether or not a particular `BaseModelTable`
// currently supports them.
lazy_static! {
    pub static ref RESERVED_MODEL_PREFIXES: Vec<String> = {
        let mut prefixes: Vec<String> = UninitializedProviderConfig::VARIANTS
            .iter()
            .map(|&v| format!("{v}::"))
            .collect();
        prefixes.push("tensorzero::".to_string());
        prefixes
    };
}

pub enum ProviderType {
    Anthropic,
    Azure,
    Deepseek,
    Fireworks,
    AWSBedrock,
    GCPVertexAnthropic,
    GCPVertexGemini,
    GoogleAIStudioGemini,
    Groq,
    Hyperbolic,
    Mistral,
    OpenAI,
    OpenRouter,
    SGLang,
    TGI,
    Together,
    VLLM,
    XAI,
}

impl Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderType::Anthropic => write!(f, "Anthropic"),
            ProviderType::Azure => write!(f, "Azure"),
            ProviderType::Deepseek => write!(f, "Deepseek"),
            ProviderType::Fireworks => write!(f, "Fireworks"),
            ProviderType::AWSBedrock => write!(f, "AWSBedrock"),
            ProviderType::GCPVertexAnthropic => write!(f, "GCPVertexAnthropic"),
            ProviderType::GCPVertexGemini => write!(f, "GCPVertexGemini"),
            ProviderType::GoogleAIStudioGemini => write!(f, "GoogleAIStudioGemini"),
            ProviderType::Groq => write!(f, "Groq"),
            ProviderType::Hyperbolic => write!(f, "Hyperbolic"),
            ProviderType::Mistral => write!(f, "Mistral"),
            ProviderType::OpenAI => write!(f, "OpenAI"),
            ProviderType::OpenRouter => write!(f, "OpenRouter"),
            ProviderType::SGLang => write!(f, "SGLang"),
            ProviderType::TGI => write!(f, "TGI"),
            ProviderType::Together => write!(f, "Together"),
            ProviderType::VLLM => write!(f, "VLLM"),
            ProviderType::XAI => write!(f, "XAI"),
        }
    }
}

#[derive(Serialize, Debug, ts_rs::TS)]
#[ts(export)]
// TODO: investigate why derive(TS) doesn't work if we add bounds to BaseModelTable itself
// #[serde(bound(deserialize = "T: ShorthandModelConfig + Deserialize<'de>"))]
// #[serde(try_from = "HashMap<Arc<str>, T>")]
pub struct BaseModelTable<T> {
    table: HashMap<Arc<str>, T>,
    #[serde(skip)]
    #[ts(skip)]
    #[expect(dead_code, reason = "Will be used in future implementation")]
    pub default_credentials: ProviderTypeDefaultCredentials,
}

impl<T: ShorthandModelConfig> Default for BaseModelTable<T> {
    fn default() -> Self {
        BaseModelTable {
            table: HashMap::new(),
            default_credentials: ProviderTypeDefaultCredentials::default(),
        }
    }
}

pub trait ShorthandModelConfig: Sized {
    const SHORTHAND_MODEL_PREFIXES: &[&str];
    /// Used in error messages (e.g. 'Model' or 'Embedding model')
    const MODEL_TYPE: &str;
    async fn from_shorthand(
        provider_type: &str,
        model_name: &str,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self, Error>;
    fn validate(&self, key: &str) -> Result<(), Error>;
}

// impl<T: ShorthandModelConfig> TryFrom<HashMap<Arc<str>, T>> for BaseModelTable<T> {
//     type Error = String;

//     fn try_from(map: HashMap<Arc<str>, T>) -> Result<Self, Self::Error> {
//         for key in map.keys() {
//             if RESERVED_MODEL_PREFIXES
//                 .iter()
//                 .any(|name| key.starts_with(name))
//             {
//                 return Err(format!(
//                     "{} name '{}' contains a reserved prefix",
//                     T::MODEL_TYPE,
//                     key
//                 ));
//             }
//         }
//         Ok(BaseModelTable(map))
//     }
// }

impl<T: ShorthandModelConfig> std::ops::Deref for BaseModelTable<T> {
    type Target = HashMap<Arc<str>, T>;

    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

/// This is `Cow` without the `T: Clone` bound.
/// Useful when we want a `Cow`, but don't want to (or can't) implement `Clone`
#[derive(Debug)]
pub enum CowNoClone<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<T> Deref for CowNoClone<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        match self {
            CowNoClone::Borrowed(t) => t,
            CowNoClone::Owned(t) => t,
        }
    }
}

pub struct Shorthand<'a> {
    pub provider_type: &'a str,
    pub model_name: &'a str,
}

fn check_shorthand<'a>(prefixes: &[&'a str], key: &'a str) -> Option<Shorthand<'a>> {
    for prefix in prefixes {
        if let Some(model_name) = key.strip_prefix(prefix) {
            // Remove the last two characters of the prefix to get the provider type
            let provider_type = &prefix[..prefix.len() - 2];
            return Some(Shorthand {
                provider_type,
                model_name,
            });
        }
    }
    None
}

impl<T: ShorthandModelConfig> BaseModelTable<T> {
    pub fn new(
        models: HashMap<Arc<str>, T>,
        provider_type_default_credentials: ProviderTypeDefaultCredentials,
    ) -> Result<Self, String> {
        for key in models.keys() {
            if RESERVED_MODEL_PREFIXES
                .iter()
                .any(|name| key.starts_with(name))
            {
                return Err(format!(
                    "{} name '{}' contains a reserved prefix",
                    T::MODEL_TYPE,
                    key
                ));
            }
        }

        Ok(Self {
            table: models,
            default_credentials: provider_type_default_credentials,
        })
    }

    pub async fn get(&self, key: &str) -> Result<Option<CowNoClone<'_, T>>, Error> {
        if let Some(model_config) = self.table.get(key) {
            return Ok(Some(CowNoClone::Borrowed(model_config)));
        }
        if let Some(shorthand) = check_shorthand(T::SHORTHAND_MODEL_PREFIXES, key) {
            return Ok(Some(CowNoClone::Owned(
                T::from_shorthand(
                    shorthand.provider_type,
                    shorthand.model_name,
                    &self.default_credentials,
                )
                .await?,
            )));
        }
        Ok(None)
    }
    /// Check that a model name is valid
    /// This is either true because it's in the table, or because it's a valid shorthand name
    pub fn validate(&self, key: &str) -> Result<(), Error> {
        // Try direct lookup (if it's blacklisted, it's not in the table)
        // If it's shorthand and already in the table, it's valid
        if let Some(model_config) = self.table.get(key) {
            model_config.validate(key)?;
            return Ok(());
        }

        if check_shorthand(T::SHORTHAND_MODEL_PREFIXES, key).is_some() {
            return Ok(());
        }

        Err(ErrorDetails::Config {
            message: format!("Model name '{key}' not found in model table"),
        }
        .into())
    }

    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn static_model_len(&self) -> usize {
        self.table.len()
    }

    pub fn iter_static_models(&self) -> impl Iterator<Item = (&Arc<str>, &T)> {
        self.table.iter()
    }
}

pub struct LazyCredential {
    cell: OnceLock<Result<Credential, Error>>,
    loader: Box<dyn Fn() -> Result<Credential, Error> + Send + Sync>,
}

impl LazyCredential {
    pub fn new<F>(loader: F) -> Self
    where
        F: Fn() -> Result<Credential, Error> + Send + Sync + 'static,
    {
        Self {
            cell: OnceLock::new(),
            loader: Box::new(loader),
        }
    }

    pub fn get(&self) -> Result<&Credential, &Error> {
        self.cell.get_or_init(|| (self.loader)()).as_ref()
    }

    pub fn get_cloned(&self) -> Result<Credential, Error>
    where
        Error: Clone,
    {
        self.get().map(|t| t.clone()).map_err(|e| e.clone())
    }
}
/*

use tokio::sync::OnceCell as AsyncOnceCell;
pub struct LazyAsyncCredential<T> {
    cell: AsyncOnceCell<Result<T, Error>>,
    loader: fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, Error>> + Send>>,
}

impl<T> LazyAsyncCredential<T> {
    pub fn new(
        loader: fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, Error>> + Send>>,
    ) -> Self {
        Self {
            cell: AsyncOnceCell::new(),
            loader,
        }
    }

    pub async fn get(&self) -> Result<&T, &Error> {
        self.cell
            .get_or_init(|| async { (self.loader)().await })
            .await
            .as_ref()
    }

    pub async fn get_cloned(&self) -> Result<T, Error>
    where
        T: Clone,
        Error: Clone,
    {
        self.get().await.map(|t| t.clone()).map_err(|e| e.clone())
    }
}*/

pub struct ProviderTypeDefaultCredentials {
    pub anthropic: LazyCredential,
    // TODO: do this lazy
    // aws_bedrock:
    // aws_sagemaker:
    pub azure: LazyCredential,
    pub deepseek: LazyCredential,
    pub fireworks: LazyCredential,
    // needs to be lazy / shared
    // gcp_vertex_anthropic: GCPVertexCredentials,
    // gcp_vertex_gemini: GCPVertexCredentials,
    pub google_ai_studio_gemini: LazyCredential,
    pub groq: LazyCredential,
    pub hyperbolic: LazyCredential,
    pub mistral: LazyCredential,
    pub openai: LazyCredential,
    pub openrouter: LazyCredential,
    pub sglang: LazyCredential,
    pub tgi: LazyCredential,
    pub together: LazyCredential,
    pub vllm: LazyCredential,
    pub xai: LazyCredential,
}

impl ProviderTypeDefaultCredentials {
    pub fn new(provider_types_config: &ProviderTypesConfig) -> Self {
        let anthropic_location = provider_types_config
            .anthropic
            .defaults
            .credential_location
            .clone();
        let azure_location = provider_types_config
            .azure
            .defaults
            .credential_location
            .clone();
        let deepseek_location = provider_types_config
            .deepseek
            .defaults
            .credential_location
            .clone();
        let fireworks_location = provider_types_config
            .fireworks
            .defaults
            .credential_location
            .clone();
        let google_ai_studio_gemini_location = provider_types_config
            .google_ai_studio_gemini
            .defaults
            .credential_location
            .clone();
        let groq_location = provider_types_config
            .groq
            .defaults
            .credential_location
            .clone();
        let hyperbolic_location = provider_types_config
            .hyperbolic
            .defaults
            .credential_location
            .clone();
        let mistral_location = provider_types_config
            .mistral
            .defaults
            .credential_location
            .clone();
        let openai_location = provider_types_config
            .openai
            .defaults
            .credential_location
            .clone();
        let openrouter_location = provider_types_config
            .openrouter
            .defaults
            .credential_location
            .clone();
        let sglang_location = provider_types_config
            .sglang
            .defaults
            .credential_location
            .clone();
        let tgi_location = provider_types_config
            .tgi
            .defaults
            .credential_location
            .clone();
        let together_location = provider_types_config
            .together
            .defaults
            .credential_location
            .clone();
        let vllm_location = provider_types_config
            .vllm
            .defaults
            .credential_location
            .clone();
        let xai_location = provider_types_config
            .xai
            .defaults
            .credential_location
            .clone();

        ProviderTypeDefaultCredentials {
            anthropic: LazyCredential::new(move || {
                load_credential(&anthropic_location, ProviderType::Anthropic)
            }),
            azure: LazyCredential::new(move || {
                load_credential(&azure_location, ProviderType::Azure)
            }),
            deepseek: LazyCredential::new(move || {
                load_credential(&deepseek_location, ProviderType::Deepseek)
            }),
            fireworks: LazyCredential::new(move || {
                load_credential(&fireworks_location, ProviderType::Fireworks)
            }),
            google_ai_studio_gemini: LazyCredential::new(move || {
                load_credential(
                    &google_ai_studio_gemini_location,
                    ProviderType::GoogleAIStudioGemini,
                )
            }),
            groq: LazyCredential::new(move || load_credential(&groq_location, ProviderType::Groq)),
            hyperbolic: LazyCredential::new(move || {
                load_credential(&hyperbolic_location, ProviderType::Hyperbolic)
            }),
            mistral: LazyCredential::new(move || {
                load_credential(&mistral_location, ProviderType::Mistral)
            }),
            openai: LazyCredential::new(move || {
                load_credential(&openai_location, ProviderType::OpenAI)
            }),
            openrouter: LazyCredential::new(move || {
                load_credential(&openrouter_location, ProviderType::OpenRouter)
            }),
            sglang: LazyCredential::new(move || {
                load_credential(&sglang_location, ProviderType::SGLang)
            }),
            tgi: LazyCredential::new(move || load_credential(&tgi_location, ProviderType::TGI)),
            together: LazyCredential::new(move || {
                load_credential(&together_location, ProviderType::Together)
            }),
            vllm: LazyCredential::new(move || load_credential(&vllm_location, ProviderType::VLLM)),
            xai: LazyCredential::new(move || load_credential(&xai_location, ProviderType::XAI)),
        }
    }

    pub fn get_defaulted_credential(
        &self,
        api_key_location: Option<&CredentialLocation>,
        provider_type: ProviderType,
    ) -> Result<Credential, Error> {
        if let Some(api_key_location) = api_key_location {
            return load_credential(api_key_location, provider_type);
        }
        match provider_type {
            ProviderType::Anthropic => self.anthropic.get_cloned(),
            _ => todo!(),
        }
    }
}

impl Default for ProviderTypeDefaultCredentials {
    fn default() -> Self {
        todo!()
        // Self::new(&ProviderTypesConfig::default())
    }
}

impl std::fmt::Debug for ProviderTypeDefaultCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderTypeDefaultCredentials")
            .finish_non_exhaustive()
    }
}

fn load_credential(
    location: &CredentialLocation,
    provider_type: ProviderType,
) -> Result<Credential, Error> {
    match location {
        CredentialLocation::Env(key_name) => match env::var(&key_name) {
            Ok(value) => Ok(Credential::Static(SecretString::from(value))),
            Err(_) => {
                if skip_credential_validation() {
                    #[cfg(any(test, feature = "e2e_tests"))]
                    {
                        tracing::warn!(
                                "You are missing the credentials required for a model provider of type {provider_type} (environment variable `{key_name}` is unset), so the associated tests will likely fail.",
                            );
                    }
                    Ok(Credential::Missing)
                } else {
                    Err(Error::new(ErrorDetails::ApiKeyMissing {
                        provider_name: provider_type.to_string(),
                        message: format!("Environment variable `{key_name}` is missing"),
                    }))
                }
            }
        },
        CredentialLocation::PathFromEnv(env_key) => {
            // First get the path from environment variable
            let path = match env::var(&env_key) {
                Ok(path) => path,
                Err(_) => {
                    if skip_credential_validation() {
                        #[cfg(any(test, feature = "e2e_tests"))]
                        {
                            tracing::warn!(
                                "Environment variable {} is required for a model provider of type {} but is missing, so the associated tests will likely fail.",
                                env_key, provider_type

                            );
                        }
                        return Ok(Credential::Missing);
                    } else {
                        return Err(Error::new(ErrorDetails::ApiKeyMissing {
                            provider_name: provider_type.to_string(),
                            message: format!(
                                "Environment variable `{env_key}` for credentials path is missing"
                            ),
                        }));
                    }
                }
            };
            // Then read the file contents
            match fs::read_to_string(path) {
                Ok(contents) => Ok(Credential::FileContents(SecretString::from(contents))),
                Err(e) => {
                    if skip_credential_validation() {
                        #[cfg(any(test, feature = "e2e_tests"))]
                        {
                            tracing::warn!(
                                "Failed to read credentials file for a model provider of type {}, so the associated tests will likely fail: {}",
                                provider_type, e
                            );
                        }
                        Ok(Credential::Missing)
                    } else {
                        Err(Error::new(ErrorDetails::ApiKeyMissing {
                            provider_name: provider_type.to_string(),
                            message: format!("Failed to read credentials file - {e}"),
                        }))
                    }
                }
            }
        }
        CredentialLocation::Path(path) => match fs::read_to_string(path) {
            Ok(contents) => Ok(Credential::FileContents(SecretString::from(contents))),
            Err(e) => {
                if skip_credential_validation() {
                    #[cfg(any(test, feature = "e2e_tests"))]
                    {
                        tracing::warn!(
                                "Failed to read credentials file for a model provider of type {}, so the associated tests will likely fail: {}",
                            provider_type, e
                        );
                    }
                    Ok(Credential::Missing)
                } else {
                    Err(Error::new(ErrorDetails::ApiKeyMissing {
                        provider_name: provider_type.to_string(),
                        message: format!("Failed to read credentials file - {e}"),
                    }))
                }
            }
        },
        CredentialLocation::Dynamic(key_name) => Ok(Credential::Dynamic(key_name.clone())),
        CredentialLocation::Sdk => Ok(Credential::Sdk),
        CredentialLocation::None => Ok(Credential::None),
    }
}
