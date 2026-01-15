//! Feature flags for TensorZero.
//!
//! Feature flags are read from environment variables at startup and cached.
//! Each flag has a typed definition with a default value.
//!
//! In tests, use [`FlagDefinition::override_for_test`] to override flag values.
//! Test builds initialize the flag value on first access to support this.

use std::fmt::Debug;
use std::sync::OnceLock;

use crate::error::{Error, ErrorDetails};

/// Trait for types that can be used as feature flag values.
pub trait FlagValue: Clone + Send + Sync + PartialEq + Debug + 'static {
    /// Parse a value from an environment variable string.
    /// Returns `None` if the string is not a valid representation.
    fn parse_from_env(s: &str) -> Option<Self>;

    /// Convert the value to a string for environment variable storage.
    fn to_env_string(&self) -> String;
}

impl FlagValue for bool {
    fn parse_from_env(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        }
    }

    fn to_env_string(&self) -> String {
        if *self { "true" } else { "false" }.to_string()
    }
}

/// Trait for introspecting feature flags without knowing their concrete type.
/// All flags must be initialized before use in production.
pub trait Flag: Send + Sync {
    /// The name of the flag.
    fn name(&self) -> &'static str;

    /// The environment variable name for this flag.
    fn env_var_name(&self) -> String;

    /// Initializes the flag value to default value or environment variable override.
    /// Returns an error if an environment variable is set but its value is invalid.
    fn init(&self) -> Result<(), Error>;
}

/// Definition of a feature flag with a specific value type.
pub struct FlagDefinition<T: FlagValue> {
    /// The name of the flag (used as env var suffix: TENSORZERO_INTERNAL_FLAG_{NAME}).
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
            "TENSORZERO_INTERNAL_FLAG_{}",
            self.name.to_uppercase().replace('-', "_")
        )
    }

    /// Reads the flag value from the environment variable.
    fn read_from_env(&self) -> Result<T, Error> {
        let env_var = self.env_var_name();
        match std::env::var(&env_var) {
            Ok(val) => match T::parse_from_env(&val) {
                Some(parsed) => Ok(parsed),
                None => Err(Error::new(ErrorDetails::AppState {
                    message: format!(
                        "Invalid value '{}' for feature flag {} (env var: {})",
                        val, self.name, env_var
                    ),
                })),
            },
            Err(_) => Ok(self.default.clone()),
        }
    }

    /// Gets the value of this flag.
    ///
    /// # Panics
    ///
    /// Panics if the flag is not initialized.
    #[cfg(not(test))]
    #[expect(clippy::expect_used)]
    pub fn get(&self) -> T {
        self.value
            .get()
            .expect("All feature flags must be initialized and validated at startup")
            .clone()
    }

    /// Gets the value of this flag.
    /// In tests, this initializes the flag value on first access.
    ///
    /// # Panics
    ///
    /// Panics if the flag value from the environment cannot be parsed into the correct type.
    #[cfg(test)]
    pub fn get(&self) -> T {
        self.value
            .get_or_init(|| {
                self.read_from_env()
                    .expect("Failed to parse flag value in tests")
            })
            .clone()
    }

    /// Overrides this flag's value for the current test process.
    ///
    /// nextest runs each test in its own process, so overrides are scoped to the running test.
    ///
    /// IMPORTANT: this must be set before the first flag access, because flag values are
    /// cached upon first read.
    #[cfg(test)]
    pub fn override_for_test(&self, value: T) {
        let env_var = self.env_var_name();
        tensorzero_unsafe_helpers::set_env_var_tests_only(&env_var, value.to_env_string());
    }
}

impl<T: FlagValue> Flag for FlagDefinition<T> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn env_var_name(&self) -> String {
        self.env_var_name()
    }

    fn init(&self) -> Result<(), Error> {
        let flag_value = self.read_from_env()?;
        match self.value.set(flag_value.clone()) {
            Ok(()) => Ok(()),
            Err(e) => {
                if e == flag_value {
                    Ok(())
                } else {
                    // To prevent subtle errors from initializing flags multiple times with different values
                    // (so at runtime it's hard to reason about the flag value), we error out.
                    Err(Error::new(ErrorDetails::AppState {
                        message: format!(
                            "Attempted to reinitialize flag {} with different value {:?} (current value: {:?})",
                            self.name,
                            flag_value,
                            e.clone()
                        ),
                    }))
                }
            }
        }
    }
}

mod flags;

pub use flags::*;

/// Initializes all feature flags.
///
/// This should be called on application startup.
pub fn init_flags() -> Result<(), Error> {
    for flag in ALL_FLAGS {
        flag.init()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_var_name() {
        let flag: FlagDefinition<bool> = FlagDefinition::new("use_postgres", false);
        assert_eq!(
            flag.env_var_name(),
            "TENSORZERO_INTERNAL_FLAG_USE_POSTGRES",
            "Environment variable name should be uppercase with prefix"
        );
    }

    #[test]
    fn test_env_var_name_with_hyphens() {
        let flag: FlagDefinition<bool> = FlagDefinition::new("my-feature-flag", false);
        assert_eq!(
            flag.env_var_name(),
            "TENSORZERO_INTERNAL_FLAG_MY_FEATURE_FLAG",
            "Hyphens should be converted to underscores"
        );
    }

    #[test]
    fn test_default_value() {
        let flag: FlagDefinition<bool> = FlagDefinition::new("test_flag_default", false);
        assert!(
            !flag.get(),
            "Flag should return default value when env var not set"
        );
    }

    #[test]
    fn test_override_for_test() {
        flags::TEST_FLAG.override_for_test(true);
        assert!(
            flags::TEST_FLAG.get(),
            "TEST_FLAG should be enabled after override_for_test(true)"
        );
    }

    #[test]
    #[should_panic(expected = "Failed to parse flag value")]
    fn test_invalid_env_value_panics_in_test() {
        let env_var = flags::TEST_FLAG.env_var_name();
        tensorzero_unsafe_helpers::set_env_var_tests_only(&env_var, "not-a-bool");
        let _value = flags::TEST_FLAG.get();
    }

    #[test]
    fn test_all_flags_includes_test_flag() {
        // To test the define_flags macro is correct, we check that ALL_FLAGS includes TEST_FLAG.
        assert!(
            ALL_FLAGS.iter().any(|f| f.name() == TEST_FLAG.name()),
            "ALL_FLAGS should include TEST_FLAG"
        );
    }
}
