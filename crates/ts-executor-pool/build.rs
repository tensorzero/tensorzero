#![expect(clippy::unwrap_used, clippy::print_stderr, clippy::panic)]

use std::path::Path;

const TS_VERSION: &str = "5.7.3";
const TS_BLAKE3: &str = "f4c7abd9363389427c082edf3056f17ac724a8f046d7266d44d456fede5692a9";

const SES_VERSION: &str = "1.14.0";
const SES_BLAKE3: &str = "8620364f19ff2f35689ae8600ecbdb248aff04b33171bf8821b83eff650c98d4";

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // -----------------------------------------------------------------------
    // TypeScript checker snapshot (unchanged)
    // -----------------------------------------------------------------------

    // Include the version in the filename so bumping TS_VERSION invalidates the cache.
    let ts_path = format!("{out_dir}/typescript-{TS_VERSION}.js");
    let checker_js = include_str!("src/js/ts_checker.js");
    // Include hashes of both JS inputs in the snapshot filename so it is
    // regenerated whenever either changes.
    let checker_hash = &blake3::hash(checker_js.as_bytes()).to_hex()[..16];
    let snapshot_path = format!("{out_dir}/TS_CHECKER_SNAPSHOT_{TS_BLAKE3}_{checker_hash}.bin");

    // Step 1: Download typescript.js (cached by version)
    if !Path::new(&ts_path).exists() {
        let url = format!("https://unpkg.com/typescript@{TS_VERSION}/lib/typescript.js");
        eprintln!("Downloading typescript.js v{TS_VERSION} from {url}...");
        let mut body = ureq::get(&url)
            .call()
            .unwrap_or_else(|e| panic!("Failed to download typescript.js from {url}: {e}"))
            .into_body();
        let bytes = body
            .read_to_vec()
            .unwrap_or_else(|e| panic!("Failed to read response body from {url}: {e}"));

        let actual_hash = blake3::hash(&bytes).to_hex().to_string();
        assert_eq!(
            actual_hash, TS_BLAKE3,
            "BLAKE3 mismatch for typescript.js v{TS_VERSION}: expected {TS_BLAKE3}, got {actual_hash}"
        );

        std::fs::write(&ts_path, bytes).unwrap();
        eprintln!(
            "Downloaded typescript.js ({} bytes, blake3: {TS_BLAKE3})",
            std::fs::metadata(&ts_path).unwrap().len()
        );
    }

    // Step 2: Create V8 snapshot with TS compiler + checker helper preloaded
    if !Path::new(&snapshot_path).exists() {
        eprintln!("Creating TypeScript checker V8 snapshot...");
        let ts_js = std::fs::read_to_string(&ts_path).unwrap();

        let mut runtime =
            deno_core::JsRuntimeForSnapshot::new(deno_core::RuntimeOptions::default());
        runtime
            .execute_script("<typescript.js>", ts_js)
            .unwrap_or_else(|e| panic!("Failed to evaluate typescript.js: {e}"));
        runtime
            .execute_script("<ts_checker.js>", checker_js.to_string())
            .unwrap_or_else(|e| panic!("Failed to evaluate ts_checker.js: {e}"));

        let snapshot = runtime.snapshot();
        std::fs::write(&snapshot_path, &*snapshot).unwrap();
        eprintln!("Snapshot created ({} bytes)", snapshot.len());
    }

    // Tell downstream code where to find the snapshot.
    println!("cargo:rustc-env=TS_CHECKER_SNAPSHOT_PATH={snapshot_path}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/js/ts_checker.js");

    // -----------------------------------------------------------------------
    // SES (Secure ECMAScript) — downloaded and embedded at compile time
    // -----------------------------------------------------------------------

    // Step 3: Download ses.umd.min.js (cached by version)
    let ses_path = format!("{out_dir}/ses-{SES_VERSION}.umd.min.js");
    if !Path::new(&ses_path).exists() {
        let url = format!("https://unpkg.com/ses@{SES_VERSION}/dist/ses.umd.min.js");
        eprintln!("Downloading ses.umd.min.js v{SES_VERSION} from {url}...");
        let mut body = ureq::get(&url)
            .call()
            .unwrap_or_else(|e| panic!("Failed to download ses.umd.min.js from {url}: {e}"))
            .into_body();
        let bytes = body
            .read_to_vec()
            .unwrap_or_else(|e| panic!("Failed to read response body from {url}: {e}"));

        let actual_hash = blake3::hash(&bytes).to_hex().to_string();
        assert_eq!(
            actual_hash, SES_BLAKE3,
            "BLAKE3 mismatch for ses.umd.min.js v{SES_VERSION}: expected {SES_BLAKE3}, got {actual_hash}"
        );

        std::fs::write(&ses_path, bytes).unwrap();
        eprintln!(
            "Downloaded ses.umd.min.js ({} bytes, blake3: {SES_BLAKE3})",
            std::fs::metadata(&ses_path).unwrap().len()
        );
    }

    // Expose the path so runtime.rs can include_str! it at compile time.
    println!("cargo:rustc-env=SES_JS_PATH={ses_path}");
    println!("cargo:rerun-if-changed=src/js/ses_init.js");
}
