use std::ffi::OsStr;

fn check_nextest() {
    if std::env::var("NEXTEST").is_err() {
        panic!(
            "The `tensorzero-unsafe-helpers` crate should only be used in tests (specifically, when `cargo nextest` is running)."
        )
    }
}

/// A helper function to set an environment variable.
/// This function is safe, even though the underlying `std::env::set_var` function in unsafe.
/// It's the responsibility of test code to only set 'reasonable' environment variables
/// (e.g. don't modify an environment variable that another thread is in the middle of reading)
pub fn set_env_var_tests_only<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) {
    check_nextest();
    // SAFETY: We are only calling this function in tests
    // It's the responsibility of the test to only try to modify a 'reasonable' environment variable.
    // We unfortunately cannot use the [env](https://crates.io/crates/env) crate, since there are multiple
    // threads active during tests, even when using nextest: https://github.com/rust-lang/rust/issues/104053
    //
    // As a sanity check, the 'check_nextest' function will panic if we're not running under 'cargo nextest',
    // to try to prevent using this crate outside of tests.
    unsafe {
        std::env::set_var(key, value);
    }
}

/// A helper function to remove an environment variable.
/// See `set_env_var_tests_only` for more details.
pub fn remove_env_var_tests_only<K: AsRef<OsStr>>(key: K) {
    check_nextest();
    // SAFETY: See 'set_env_var_tests_only' for more details
    unsafe {
        std::env::remove_var(key);
    }
}
