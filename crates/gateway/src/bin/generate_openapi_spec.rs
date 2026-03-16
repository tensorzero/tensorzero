use std::error::Error as StdError;
use std::path::PathBuf;
use std::thread;

use clap::Parser;

#[path = "../routes/mod.rs"]
mod routes;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "openapi/tensorzero-gateway-external.json")]
    external_output: PathBuf,

    #[arg(long, default_value = "openapi/tensorzero-gateway-internal.json")]
    internal_output: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let _ = routes::build_api_routes;
    let _ = routes::build_internal_openapi_spec as fn() -> utoipa::openapi::OpenApi;

    let (external_openapi, internal_openapi) = thread::Builder::new()
        .name("generate-openapi-specs".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(async {
                let metrics_handle = tensorzero_core::observability::setup_metrics(None)?;
                Ok::<_, Box<dyn StdError + Send + Sync>>((
                    routes::build_external_openapi_spec(metrics_handle),
                    routes::build_internal_openapi_spec(),
                ))
            })
        })?
        .join()
        .map_err(|_| "failed to generate OpenAPI spec on worker thread")?
        .map_err(|error| -> Box<dyn StdError> { error })?;

    if let Some(parent) = args.external_output.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    if let Some(parent) = args.internal_output.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let external_spec = serde_json::to_string_pretty(&external_openapi)?;
    tokio::fs::write(&args.external_output, external_spec).await?;

    let internal_spec = serde_json::to_string_pretty(&internal_openapi)?;
    tokio::fs::write(&args.internal_output, internal_spec).await?;
    Ok(())
}
