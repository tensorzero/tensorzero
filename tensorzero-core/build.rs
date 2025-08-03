#![expect(clippy::expect_used)]

fn main() {
    // If this env var is set, emit a 'cargo:rerun-if-env-changed', which will disable
    // all of Cargo's default build.rs change detection logic. This is useful for local development,
    // where we don't want to force unnecessary rebuilds of the build.rs file (e.g. when changing
    // config files that happen to be inside the 'tensorzero-core' directory)
    if std::env::var("TENSORZERO_SKIP_BUILD_INFO").is_ok() {
        println!("cargo:rerun-if-env-changed=TENSORZERO_SKIP_BUILD_INFO");
    }
    built::write_built_file().expect("Failed to acquire build-time information");
}
