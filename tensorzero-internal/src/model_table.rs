use std::{collections::HashMap, ops::Deref, sync::Arc};

use serde::Deserialize;

use crate::{
    error::{Error, ErrorDetails},
    model::ProviderConfigHelper,
};
use lazy_static::lazy_static;
use strum::VariantNames;

// Reserve prefixes for all supported providers, regardless of whether or not a particular `BaseModelTable`
// currently supports them.
lazy_static! {
    pub static ref RESERVED_MODEL_PREFIXES: Vec<String> = {
        let mut prefixes: Vec<String> = ProviderConfigHelper::VARIANTS
            .iter()
            .map(|&v| format!("{}::", v))
            .collect();
        prefixes.push("tensorzero::".to_string());
        prefixes
    };
}

#[derive(Debug, Deserialize)]
#[serde(try_from = "HashMap<Arc<str>, T>")]
pub struct BaseModelTable<T: ShorthandModelConfig>(HashMap<Arc<str>, T>);

impl<T: ShorthandModelConfig> Default for BaseModelTable<T> {
    fn default() -> Self {
        BaseModelTable(HashMap::new())
    }
}

pub trait ShorthandModelConfig: Sized {
    const SHORTHAND_MODEL_PREFIXES: &[&str];
    /// Used in error messages (e.g. 'Model' or 'Embedding model')
    const MODEL_TYPE: &str;
    fn from_shorthand(provider_type: &str, model_name: &str) -> Result<Self, Error>;
    fn validate(&self, key: &str) -> Result<(), Error>;
}

impl<T: ShorthandModelConfig> TryFrom<HashMap<Arc<str>, T>> for BaseModelTable<T> {
    type Error = String;

    fn try_from(map: HashMap<Arc<str>, T>) -> Result<Self, Self::Error> {
        for key in map.keys() {
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
        Ok(BaseModelTable(map))
    }
}

impl<T: ShorthandModelConfig> std::ops::Deref for BaseModelTable<T> {
    type Target = HashMap<Arc<str>, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
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
    pub fn get(&self, key: &str) -> Result<Option<CowNoClone<'_, T>>, Error> {
        if let Some(model_config) = self.0.get(key) {
            return Ok(Some(CowNoClone::Borrowed(model_config)));
        }
        if let Some(shorthand) = check_shorthand(T::SHORTHAND_MODEL_PREFIXES, key) {
            return Ok(Some(CowNoClone::Owned(T::from_shorthand(
                shorthand.provider_type,
                shorthand.model_name,
            )?)));
        }
        Ok(None)
    }
    /// Check that a model name is valid
    /// This is either true because it's in the table, or because it's a valid shorthand name
    pub fn validate(&self, key: &str) -> Result<(), Error> {
        // Try direct lookup (if it's blacklisted, it's not in the table)
        // If it's shorthand and already in the table, it's valid
        if let Some(model_config) = self.0.get(key) {
            model_config.validate(key)?;
            return Ok(());
        }

        if check_shorthand(T::SHORTHAND_MODEL_PREFIXES, key).is_some() {
            return Ok(());
        }

        Err(ErrorDetails::Config {
            message: format!("Model name '{}' not found in model table", key),
        }
        .into())
    }

    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn static_model_len(&self) -> usize {
        self.0.len()
    }

    pub fn iter_static_models(&self) -> impl Iterator<Item = (&Arc<str>, &T)> {
        self.0.iter()
    }
}
