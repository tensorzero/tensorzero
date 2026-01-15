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

    /// Enable writing to Postgres for data in addition to ClickHouse.
    pub ENABLE_POSTGRES_WRITE: bool = ("enable_postgres_write", false);

    /// Enable reading from Postgres for data instead of ClickHouse.
    pub ENABLE_POSTGRES_READ: bool = ("enable_postgres_read", false);
}
