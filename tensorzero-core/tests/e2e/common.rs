use reqwest::Url;
use tensorzero_core::feature_flags;

lazy_static::lazy_static! {
    static ref GATEWAY_URL: String = std::env::var("TENSORZERO_GATEWAY_URL")
        .unwrap_or_else(|_| "http://localhost:3000".to_string());
}

/// Returns true if we're testing against Postgres.
///
/// This is not really perfect because we rely on the tests running in the same context as when we
/// launch the gateway container, but it's true for our CI setup and is good enough for today.
#[expect(dead_code)]
pub fn is_postgres_test() -> bool {
    feature_flags::ENABLE_POSTGRES_READ.get() || feature_flags::ENABLE_POSTGRES_WRITE.get()
}

/// Skips the current test if running against Postgres.
/// Use this for tests that don't have Postgres implementations yet.
///
/// TODO(#5691): Remove this once we have Postgres implementations for all tests.
#[macro_export]
macro_rules! skip_for_postgres {
    () => {
        if $crate::common::is_postgres_test() {
            eprintln!("Skipping test: Postgres implementation not yet available");
            return;
        }
    };
}

pub fn get_gateway_endpoint(endpoint: &str) -> Url {
    let base_url: Url = GATEWAY_URL
        .parse()
        .expect("Invalid gateway URL (check environment variable TENSORZERO_GATEWAY_URL)");

    base_url.join(endpoint).unwrap()
}
