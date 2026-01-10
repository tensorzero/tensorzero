mod middleware;

pub use middleware::feature_flags_middleware;

use std::collections::HashMap;
use std::future::Future;

tokio::task_local! {
    static FEATURE_FLAGS: FeatureFlags;
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeatureFlagValue {
    Bool(bool),
}

impl FeatureFlagValue {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            FeatureFlagValue::Bool(b) => Some(*b),
        }
    }
}

pub struct FlagDefinition {
    pub name: &'static str,
    pub default: FeatureFlagValue,
}

pub mod flags {
    use super::FeatureFlagValue;
    pub use super::FlagDefinition;

    pub const USE_POSTGRES: FlagDefinition = FlagDefinition {
        name: "use_postgres",
        default: FeatureFlagValue::Bool(false),
    };

    pub const ALL_FLAGS: &[&FlagDefinition] = &[&USE_POSTGRES];
}

#[derive(Debug, Clone)]
pub struct FeatureFlags {
    values: HashMap<&'static str, FeatureFlagValue>,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        let mut values = HashMap::new();
        for flag in flags::ALL_FLAGS {
            values.insert(flag.name, flag.default.clone());
        }
        Self { values }
    }
}

impl FeatureFlags {
    pub fn with_overrides(overrides: HashMap<&'static str, FeatureFlagValue>) -> Self {
        let mut flags = Self::default();
        for (name, value) in overrides {
            flags.values.insert(name, value);
        }
        flags
    }

    pub fn get_bool(&self, flag: &FlagDefinition) -> bool {
        self.values
            .get(flag.name)
            .and_then(|v| v.as_bool())
            .unwrap_or_else(|| flag.default.as_bool().unwrap_or(false))
    }

    pub fn is_enabled(&self, flag: &FlagDefinition) -> bool {
        self.get_bool(flag)
    }
}

pub fn get_feature_flags() -> FeatureFlags {
    FEATURE_FLAGS.try_with(|f| f.clone()).unwrap_or_default()
}

pub async fn with_feature_flags<F, R>(flags: FeatureFlags, f: F) -> R
where
    F: Future<Output = R>,
{
    FEATURE_FLAGS.scope(flags, f).await
}
