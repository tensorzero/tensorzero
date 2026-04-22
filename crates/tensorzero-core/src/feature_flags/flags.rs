//! Feature flag definitions.
//!
//! Add new feature flags here. Each flag is read from an environment variable
//! with the format `TENSORZERO_INTERNAL_FLAG_{NAME}` where NAME is the uppercase version of the flag name.

use super::{Flag, FlagDefinition};

macro_rules! define_flags {
    (
        $(
            $(#[$meta:meta])*
            $vis:vis $ident:ident : $ty:ty = ($name:expr, $default:expr);
        )+
    ) => {
        $(
            $(#[$meta])*
            $vis static $ident: FlagDefinition<$ty> = FlagDefinition::new($name, $default);
        )+

        pub static ALL_FLAGS: &[&'static dyn Flag] = &[
            $(
                & $ident,
            )+
        ];
    };
}

define_flags! {
    /// A dummy test flag for unit tests.
    pub TEST_FLAG: bool = ("test_flag", false);

    /// Enables reading configuration from the database instead of (only) from the config file.
    pub ENABLE_CONFIG_IN_DATABASE: bool = ("enable_config_in_database", false);
}
