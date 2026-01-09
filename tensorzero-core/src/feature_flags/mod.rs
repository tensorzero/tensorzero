//! Feature flags for TensorZero.
//!
//! Feature flags are read from environment variables at startup and cached.
//! Each flag has a typed definition with a default value.

use std::sync::OnceLock;

/// Trait for types that can be used as feature flag values.
pub trait FlagValue: Clone + Send + Sync + 'static {
    /// Parse a value from an environment variable string.
    /// Returns `None` if the string is not a valid representation.
    fn parse_from_env(s: &str) -> Option<Self>;

    /// Convert the value to JSON for introspection APIs.
    fn to_json(&self) -> serde_json::Value;
}

/// Boolean flag definition.
impl FlagValue for bool {
    fn parse_from_env(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::Value::Bool(*self)
    }
}

/// Trait for introspecting feature flags without knowing their concrete type.
pub trait Flag: Send + Sync {
    /// The name of the flag.
    fn name(&self) -> &'static str;

    /// The environment variable name for this flag.
    fn env_var_name(&self) -> String;

    /// The default value as JSON.
    fn default_value_json(&self) -> serde_json::Value;

    /// The current value as JSON.
    fn current_value_json(&self) -> serde_json::Value;
}

/// Definition of a feature flag with a specific value type.
pub struct FlagDefinition<T: FlagValue> {
    /// The name of the flag (used as env var suffix: TENSORZERO_FLAG_{NAME}).
    pub name: &'static str,
    /// The default value if the environment variable is not set.
    pub default: T,
    /// Cached value read from environment.
    value: OnceLock<T>,
}

impl<T: FlagValue> FlagDefinition<T> {
    /// Creates a new flag definition.
    pub const fn new(name: &'static str, default: T) -> Self {
        Self {
            name,
            default,
            value: OnceLock::new(),
        }
    }

    /// Returns the environment variable name for this flag.
    pub fn env_var_name(&self) -> String {
        format!(
            "TENSORZERO_FLAG_{}",
            self.name.to_uppercase().replace('-', "_")
        )
    }

    /// Gets the value of this flag, reading from environment on first access.
    fn get_value_from_env(&self) -> T {
        self.value
            .get_or_init(|| {
                let env_var = self.env_var_name();
                match std::env::var(&env_var) {
                    Ok(val) => match T::parse_from_env(&val) {
                        Some(parsed) => parsed,
                        None => {
                            tracing::warn!(
                                "Invalid value '{}' for feature flag env var {}, using default",
                                val,
                                env_var
                            );
                            self.default.clone()
                        }
                    },
                    Err(_) => self.default.clone(),
                }
            })
            .clone()
    }

    /// Gets the current value of this flag.
    ///
    /// In tests, this checks thread-local overrides first (set via [`set`]).
    /// Otherwise, it reads from the environment variable or uses the default.
    ///
    /// This returns a clone of the value. We currently only support boolean flags, so this
    /// performs a cheap copy.
    ///
    /// TODO(shuyangli): When we need to support complex flag types, figure out how to
    /// handle this efficiently.
    pub fn get(&self) -> T {
        // Check test override first
        #[cfg(test)]
        {
            if let Some(value) = test_flags::get_override::<T>(self.name) {
                return value;
            }
        }

        self.get_value_from_env()
    }

    /// Temporarily overrides this flag's value for the current thread.
    ///
    /// Returns a guard that restores the flag to its non-overridden state when dropped.
    /// This is useful for testing code that depends on feature flags.
    ///
    /// # Panics
    ///
    /// Panics if this flag is already overridden in the current thread.
    /// Each flag can only have one active override at a time.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tensorzero_core::feature_flags::flags;
    ///
    /// #[test]
    /// fn test_with_flag_enabled() {
    ///     let _guard = flags::MY_FLAG.set(true);
    ///     assert!(flags::MY_FLAG.get());
    ///     // Guard dropped here, flag restored to non-overridden state
    /// }
    /// ```
    #[cfg(test)]
    #[must_use = "the guard must be held for the override to remain active"]
    pub fn set(&'static self, value: T) -> FlagOverrideGuard {
        test_flags::set_override(self.name, value);
        FlagOverrideGuard {
            flag_name: self.name,
        }
    }
}

impl<T: FlagValue> Flag for FlagDefinition<T> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn env_var_name(&self) -> String {
        self.env_var_name()
    }

    fn default_value_json(&self) -> serde_json::Value {
        self.default.to_json()
    }

    fn current_value_json(&self) -> serde_json::Value {
        self.get().to_json()
    }
}

/// Guard that removes a flag's override when dropped.
///
/// Created by [`FlagDefinition::set`].
#[cfg(test)]
#[must_use = "the guard must be held for the override to remain active"]
pub struct FlagOverrideGuard {
    flag_name: &'static str,
}

#[cfg(test)]
impl Drop for FlagOverrideGuard {
    fn drop(&mut self) {
        test_flags::clear_override(self.flag_name);
    }
}

/// Test-only utilities for flag overrides.
#[cfg(test)]
mod test_flags {
    use std::any::Any;
    use std::cell::RefCell;
    use std::collections::HashMap;

    use super::FlagValue;

    thread_local! {
        /// Thread-local storage for flag overrides during tests.
        /// Values are stored as `Box<dyn Any>` to support different types.
        static OVERRIDES: RefCell<HashMap<&'static str, Box<dyn Any>>> = RefCell::new(HashMap::new());
    }

    /// Gets the override value for a flag, if one is set.
    pub fn get_override<T: FlagValue>(name: &'static str) -> Option<T> {
        OVERRIDES.with(|overrides| {
            overrides
                .borrow()
                .get(name)
                .and_then(|v| v.downcast_ref::<T>())
                .cloned()
        })
    }

    /// Sets an override for a flag.
    ///
    /// # Panics
    ///
    /// Panics if the flag is already overridden.
    pub fn set_override<T: FlagValue>(name: &'static str, value: T) {
        OVERRIDES.with(|overrides| {
            let mut map = overrides.borrow_mut();
            assert!(
                !map.contains_key(name),
                "Feature flag '{name}' is already overridden in this thread. \
                 Each flag can only have one active override at a time."
            );
            map.insert(name, Box::new(value));
        });
    }

    /// Clears the override for a flag.
    pub fn clear_override(name: &'static str) {
        OVERRIDES.with(|overrides| {
            overrides.borrow_mut().remove(name);
        });
    }
}

pub mod flags;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_var_name() {
        let flag: FlagDefinition<bool> = FlagDefinition::new("use_postgres", false);
        assert_eq!(
            flag.env_var_name(),
            "TENSORZERO_FLAG_USE_POSTGRES",
            "Environment variable name should be uppercase with prefix"
        );
    }

    #[test]
    fn test_env_var_name_with_hyphens() {
        let flag: FlagDefinition<bool> = FlagDefinition::new("my-feature-flag", false);
        assert_eq!(
            flag.env_var_name(),
            "TENSORZERO_FLAG_MY_FEATURE_FLAG",
            "Hyphens should be converted to underscores"
        );
    }

    #[test]
    fn test_default_value() {
        // Create a fresh flag that won't have env var set
        let flag: FlagDefinition<bool> = FlagDefinition::new("test_flag_default", false);
        assert!(
            !flag.get(),
            "Flag should return default value when env var not set"
        );
    }

    #[test]
    fn test_override_flag() {
        // Use the actual TEST_FLAG from flags module
        assert!(
            !flags::TEST_FLAG.get(),
            "TEST_FLAG should be disabled by default"
        );

        {
            let _guard = flags::TEST_FLAG.set(true);
            assert!(
                flags::TEST_FLAG.get(),
                "TEST_FLAG should be enabled after set(true)"
            );
        }

        // After guard dropped, should restore to default
        assert!(
            !flags::TEST_FLAG.get(),
            "TEST_FLAG should be disabled after guard dropped"
        );
    }

    #[test]
    #[should_panic(expected = "already overridden")]
    fn test_double_override_panics() {
        let _guard1 = flags::TEST_FLAG.set(true);
        let _guard2 = flags::TEST_FLAG.set(false); // Should panic
    }
}
