use std::{
    collections::HashMap,
    env,
    fmt::Display,
    fs,
    ops::Deref,
    sync::{Arc, OnceLock},
};

use crate::{
    config::{provider_types::ProviderTypesConfig, skip_credential_validation},
    error::{Error, ErrorDetails},
    model::{
        Credential, CredentialLocation, CredentialLocationWithFallback, UninitializedProviderConfig,
    },
    providers::{
        anthropic::AnthropicCredentials,
        azure::AzureCredentials,
        deepseek::DeepSeekCredentials,
        fireworks::FireworksCredentials,
        gcp_vertex_anthropic::make_gcp_sdk_credentials,
        gcp_vertex_gemini::{build_gcp_non_sdk_credentials, GCPVertexCredentials},
        google_ai_studio_gemini::GoogleAIStudioCredentials,
        groq::GroqCredentials,
        hyperbolic::HyperbolicCredentials,
        mistral::MistralCredentials,
        openai::OpenAICredentials,
        openrouter::OpenRouterCredentials,
        sglang::SGLangCredentials,
        tgi::TGICredentials,
        together::TogetherCredentials,
        vllm::VLLMCredentials,
        xai::XAICredentials,
    },
};
use lazy_static::lazy_static;
use secrecy::SecretString;
use serde::Serialize;
use strum::VariantNames;
use tokio::sync::OnceCell;

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

pub trait ProviderKind {
    type Credential: Clone;
    fn get_provider_type(&self) -> ProviderType;
    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error>;
    async fn get_defaulted_credential(
        &self,
        api_key_location: Option<&CredentialLocationWithFallback>,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error>
    where
        Self::Credential: TryFrom<Credential, Error = Error>,
    {
        let provider_type = self.get_provider_type();
        if let Some(api_key_location) = api_key_location {
            return load_credential_with_fallback(api_key_location, provider_type)?.try_into();
        }

        Ok(self
            .get_credential_field(default_credentials)
            .await?
            .clone())
    }
}

#[derive(Debug, Clone, Copy)]
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
    pub table: HashMap<Arc<str>, T>,
    #[serde(skip)]
    #[ts(skip)]
    pub default_credentials: Arc<ProviderTypeDefaultCredentials>,
    global_outbound_http_timeout: chrono::Duration,
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
    fn validate(
        &self,
        key: &str,
        global_outbound_http_timeout: &chrono::Duration,
    ) -> Result<(), Error>;
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

impl<T: ShorthandModelConfig> Default for BaseModelTable<T> {
    fn default() -> Self {
        Self {
            table: HashMap::new(),
            default_credentials: Arc::new(ProviderTypeDefaultCredentials::default()),
            global_outbound_http_timeout: chrono::Duration::seconds(120),
        }
    }
}

impl<T: ShorthandModelConfig> BaseModelTable<T> {
    pub fn new(
        models: HashMap<Arc<str>, T>,
        provider_type_default_credentials: Arc<ProviderTypeDefaultCredentials>,
        global_outbound_http_timeout: chrono::Duration,
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
            global_outbound_http_timeout,
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
            model_config.validate(key, &self.global_outbound_http_timeout)?;
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

pub struct LazyCredential<T: Clone> {
    cell: OnceLock<Result<T, Error>>,
    loader: Box<dyn Fn() -> Result<T, Error> + Send + Sync>,
}

impl<T: Clone> LazyCredential<T> {
    pub fn new<F>(loader: F) -> Self
    where
        F: Fn() -> Result<T, Error> + Send + Sync + 'static,
    {
        Self {
            cell: OnceLock::new(),
            loader: Box::new(loader),
        }
    }

    pub fn get(&self) -> Result<&T, &Error> {
        self.cell.get_or_init(|| (self.loader)()).as_ref()
    }

    pub fn get_cloned(&self) -> Result<T, Error>
    where
        Error: Clone,
    {
        self.get().cloned().map_err(std::clone::Clone::clone)
    }
}

type AsyncCredentialLoader<T> = Box<
    dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, Error>> + Send>>
        + Send
        + Sync,
>;

pub struct LazyAsyncCredential<T: Clone> {
    cell: OnceCell<Result<T, Error>>,
    loader: AsyncCredentialLoader<T>,
}

impl<T: Clone> LazyAsyncCredential<T> {
    pub fn new<F, Fut>(loader: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<T, Error>> + Send + 'static,
    {
        Self {
            cell: OnceCell::new(),
            loader: Box::new(move || Box::pin(loader())),
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
        self.get().await.cloned().map_err(std::clone::Clone::clone)
    }
}

pub struct ProviderTypeDefaultCredentials {
    anthropic: LazyCredential<AnthropicCredentials>,
    // Note: we currently do not support shorthand for either AWS Bedrock or AWS Sagemaker
    // aws_bedrock:
    // aws_sagemaker:
    azure: LazyCredential<AzureCredentials>,
    deepseek: LazyCredential<DeepSeekCredentials>,
    fireworks: LazyCredential<FireworksCredentials>,
    gcp_vertex_anthropic: LazyAsyncCredential<GCPVertexCredentials>,
    gcp_vertex_gemini: LazyAsyncCredential<GCPVertexCredentials>,
    google_ai_studio_gemini: LazyCredential<GoogleAIStudioCredentials>,
    groq: LazyCredential<GroqCredentials>,
    hyperbolic: LazyCredential<HyperbolicCredentials>,
    mistral: LazyCredential<MistralCredentials>,
    openai: LazyCredential<OpenAICredentials>,
    openrouter: LazyCredential<OpenRouterCredentials>,
    sglang: LazyCredential<SGLangCredentials>,
    tgi: LazyCredential<TGICredentials>,
    together: LazyCredential<TogetherCredentials>,
    vllm: LazyCredential<VLLMCredentials>,
    xai: LazyCredential<XAICredentials>,
}

impl ProviderTypeDefaultCredentials {
    pub fn new(provider_types_config: &ProviderTypesConfig) -> Self {
        let anthropic_location = provider_types_config
            .anthropic
            .defaults
            .api_key_location
            .clone();
        let azure_location = provider_types_config
            .azure
            .defaults
            .api_key_location
            .clone();
        let deepseek_location = provider_types_config
            .deepseek
            .defaults
            .api_key_location
            .clone();
        let fireworks_location = provider_types_config
            .fireworks
            .defaults
            .api_key_location
            .clone();
        let google_ai_studio_gemini_location = provider_types_config
            .google_ai_studio_gemini
            .defaults
            .api_key_location
            .clone();
        let gcp_vertex_anthropic_location = provider_types_config
            .gcp_vertex_anthropic
            .defaults
            .credential_location
            .clone();
        let gcp_vertex_gemini_location = provider_types_config
            .gcp_vertex_gemini
            .defaults
            .credential_location
            .clone();
        let groq_location = provider_types_config.groq.defaults.api_key_location.clone();
        let hyperbolic_location = provider_types_config
            .hyperbolic
            .defaults
            .api_key_location
            .clone();
        let mistral_location = provider_types_config
            .mistral
            .defaults
            .api_key_location
            .clone();
        let openai_location = provider_types_config
            .openai
            .defaults
            .api_key_location
            .clone();
        let openrouter_location = provider_types_config
            .openrouter
            .defaults
            .api_key_location
            .clone();
        let sglang_location = provider_types_config
            .sglang
            .defaults
            .api_key_location
            .clone();
        let tgi_location = provider_types_config.tgi.defaults.api_key_location.clone();
        let together_location = provider_types_config
            .together
            .defaults
            .api_key_location
            .clone();
        let vllm_location = provider_types_config.vllm.defaults.api_key_location.clone();
        let xai_location = provider_types_config.xai.defaults.api_key_location.clone();

        ProviderTypeDefaultCredentials {
            anthropic: LazyCredential::new(move || {
                load_credential_with_fallback(&anthropic_location, ProviderType::Anthropic)?
                    .try_into()
            }),
            azure: LazyCredential::new(move || {
                load_credential_with_fallback(&azure_location, ProviderType::Azure)?.try_into()
            }),
            deepseek: LazyCredential::new(move || {
                load_credential_with_fallback(&deepseek_location, ProviderType::Deepseek)?
                    .try_into()
            }),
            fireworks: LazyCredential::new(move || {
                load_credential_with_fallback(&fireworks_location, ProviderType::Fireworks)?
                    .try_into()
            }),
            google_ai_studio_gemini: LazyCredential::new(move || {
                load_credential_with_fallback(
                    &google_ai_studio_gemini_location,
                    ProviderType::GoogleAIStudioGemini,
                )?
                .try_into()
            }),
            gcp_vertex_anthropic: LazyAsyncCredential::new(move || {
                let location = gcp_vertex_anthropic_location.clone();
                async move {
                    make_gcp_credentials_with_fallback(ProviderType::GCPVertexAnthropic, &location)
                        .await
                }
            }),
            gcp_vertex_gemini: LazyAsyncCredential::new(move || {
                let location = gcp_vertex_gemini_location.clone();
                async move {
                    make_gcp_credentials_with_fallback(ProviderType::GCPVertexGemini, &location)
                        .await
                }
            }),

            groq: LazyCredential::new(move || {
                load_credential_with_fallback(&groq_location, ProviderType::Groq)?.try_into()
            }),
            hyperbolic: LazyCredential::new(move || {
                load_credential_with_fallback(&hyperbolic_location, ProviderType::Hyperbolic)?
                    .try_into()
            }),
            mistral: LazyCredential::new(move || {
                load_credential_with_fallback(&mistral_location, ProviderType::Mistral)?.try_into()
            }),
            openai: LazyCredential::new(move || {
                load_credential_with_fallback(&openai_location, ProviderType::OpenAI)?.try_into()
            }),
            openrouter: LazyCredential::new(move || {
                load_credential_with_fallback(&openrouter_location, ProviderType::OpenRouter)?
                    .try_into()
            }),
            sglang: LazyCredential::new(move || {
                load_credential_with_fallback(&sglang_location, ProviderType::SGLang)?.try_into()
            }),
            tgi: LazyCredential::new(move || {
                load_credential_with_fallback(&tgi_location, ProviderType::TGI)?.try_into()
            }),
            together: LazyCredential::new(move || {
                load_credential_with_fallback(&together_location, ProviderType::Together)?
                    .try_into()
            }),
            vllm: LazyCredential::new(move || {
                load_credential_with_fallback(&vllm_location, ProviderType::VLLM)?.try_into()
            }),
            xai: LazyCredential::new(move || {
                load_credential_with_fallback(&xai_location, ProviderType::XAI)?.try_into()
            }),
        }
    }
}

async fn make_gcp_credentials_with_fallback(
    provider_type: ProviderType,
    location: &CredentialLocationWithFallback,
) -> Result<GCPVertexCredentials, Error> {
    // Build default credential
    let default_cred = match location.default_location() {
        CredentialLocation::Sdk => make_gcp_sdk_credentials(provider_type).await?,
        loc => build_gcp_non_sdk_credentials(load_credential(loc, provider_type)?, &provider_type)?,
    };

    // If fallback location is specified, construct a WithFallback credential
    if let Some(fallback_location) = location.fallback_location() {
        let fallback_cred = match fallback_location {
            CredentialLocation::Sdk => make_gcp_sdk_credentials(provider_type).await?,
            fallback_loc => build_gcp_non_sdk_credentials(
                load_credential(fallback_loc, provider_type)?,
                &provider_type,
            )?,
        };
        Ok(GCPVertexCredentials::WithFallback {
            default: Box::new(default_cred),
            fallback: Box::new(fallback_cred),
        })
    } else {
        Ok(default_cred)
    }
}

impl Default for ProviderTypeDefaultCredentials {
    fn default() -> Self {
        Self::new(&ProviderTypesConfig::default())
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
        CredentialLocation::Env(key_name) => match env::var(key_name) {
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
            let path = match env::var(env_key) {
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

/// Load credential with fallback support
/// Constructs a WithFallback credential that will be resolved at inference time
fn load_credential_with_fallback(
    location_with_fallback: &crate::model::CredentialLocationWithFallback,
    provider_type: ProviderType,
) -> Result<Credential, Error> {
    let default_credential =
        load_credential(location_with_fallback.default_location(), provider_type)?;

    // If fallback location is specified, construct a WithFallback credential
    if let Some(fallback_location) = location_with_fallback.fallback_location() {
        let fallback_credential = load_credential(fallback_location, provider_type)?;
        Ok(Credential::WithFallback {
            default: Box::new(default_credential),
            fallback: Box::new(fallback_credential),
        })
    } else {
        Ok(default_credential)
    }
}

pub struct AnthropicKind;

impl ProviderKind for AnthropicKind {
    type Credential = AnthropicCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.anthropic.get_cloned()
    }
}

pub struct OpenAIKind;

impl ProviderKind for OpenAIKind {
    type Credential = OpenAICredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.openai.get_cloned()
    }
}

pub struct AzureKind;

impl ProviderKind for AzureKind {
    type Credential = AzureCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Azure
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.azure.get_cloned()
    }
}

pub struct DeepSeekKind;

impl ProviderKind for DeepSeekKind {
    type Credential = DeepSeekCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Deepseek
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.deepseek.get_cloned()
    }
}

pub struct FireworksKind;

impl ProviderKind for FireworksKind {
    type Credential = FireworksCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Fireworks
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.fireworks.get_cloned()
    }
}

pub struct GCPVertexAnthropicKind;

impl ProviderKind for GCPVertexAnthropicKind {
    type Credential = GCPVertexCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::GCPVertexAnthropic
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.gcp_vertex_anthropic.get_cloned().await
    }
}

impl GCPVertexAnthropicKind {
    pub async fn get_defaulted_credential(
        &self,
        api_key_location: Option<&CredentialLocationWithFallback>,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<GCPVertexCredentials, Error> {
        if let Some(api_key_location) = api_key_location {
            return make_gcp_credentials_with_fallback(
                ProviderType::GCPVertexAnthropic,
                api_key_location,
            )
            .await;
        }

        Ok(self
            .get_credential_field(default_credentials)
            .await?
            .clone())
    }
}

pub struct GCPVertexGeminiKind;

impl ProviderKind for GCPVertexGeminiKind {
    type Credential = GCPVertexCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::GCPVertexGemini
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.gcp_vertex_gemini.get_cloned().await
    }
}

impl GCPVertexGeminiKind {
    pub async fn get_defaulted_credential(
        &self,
        api_key_location: Option<&CredentialLocationWithFallback>,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<GCPVertexCredentials, Error> {
        if let Some(api_key_location) = api_key_location {
            return make_gcp_credentials_with_fallback(
                ProviderType::GCPVertexGemini,
                api_key_location,
            )
            .await;
        }

        Ok(self
            .get_credential_field(default_credentials)
            .await?
            .clone())
    }
}

pub struct GoogleAIStudioGeminiKind;

impl ProviderKind for GoogleAIStudioGeminiKind {
    type Credential = GoogleAIStudioCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::GoogleAIStudioGemini
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.google_ai_studio_gemini.get_cloned()
    }
}

pub struct GroqKind;

impl ProviderKind for GroqKind {
    type Credential = GroqCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Groq
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.groq.get_cloned()
    }
}

pub struct HyperbolicKind;

impl ProviderKind for HyperbolicKind {
    type Credential = HyperbolicCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Hyperbolic
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.hyperbolic.get_cloned()
    }
}

pub struct MistralKind;

impl ProviderKind for MistralKind {
    type Credential = MistralCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Mistral
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.mistral.get_cloned()
    }
}

pub struct OpenRouterKind;

impl ProviderKind for OpenRouterKind {
    type Credential = OpenRouterCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::OpenRouter
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.openrouter.get_cloned()
    }
}

pub struct SGLangKind;

impl ProviderKind for SGLangKind {
    type Credential = SGLangCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::SGLang
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.sglang.get_cloned()
    }
}

pub struct TGIKind;

impl ProviderKind for TGIKind {
    type Credential = TGICredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::TGI
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.tgi.get_cloned()
    }
}

pub struct TogetherKind;

impl ProviderKind for TogetherKind {
    type Credential = TogetherCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Together
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.together.get_cloned()
    }
}

pub struct VLLMKind;

impl ProviderKind for VLLMKind {
    type Credential = VLLMCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::VLLM
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.vllm.get_cloned()
    }
}

pub struct XAIKind;

impl ProviderKind for XAIKind {
    type Credential = XAICredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::XAI
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.xai.get_cloned()
    }
}
