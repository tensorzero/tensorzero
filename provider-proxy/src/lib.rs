//! An HTTP/HTTPS proxy that caches non-error responses to disk.
//! Heavily based on https://github.com/hatoo/http-mitm-proxy (MIT-licensed),
//! with the openssl dependency and `default_client` removed.
#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

mod mitm_server;
mod streaming_body_collector;
mod tls;

use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::{fs::OpenOptions, future::Future};

use anyhow::Context as _;
use bytes::{Bytes, BytesMut};
use clap::{ArgAction, Parser};
use http::HeaderValue;
use http_body_util::{combinators::BoxBody, BodyExt, Full};
use hyper::service::service_fn;
use mitm_server::MitmProxy;
use moka::sync::Cache;
use sha2::{Digest, Sha256};
use streaming_body_collector::StreamingBodyCollector;
use tokio::sync::oneshot;
use tracing::level_filters::LevelFilter;

const CACHE_HEADER_NAME: &str = "x-tensorzero-provider-proxy-cache";

fn make_root_cert() -> rcgen::CertifiedKey {
    let mut param = rcgen::CertificateParams::default();

    param.distinguished_name = rcgen::DistinguishedName::new();
    param.distinguished_name.push(
        rcgen::DnType::CommonName,
        rcgen::DnValue::Utf8String("<HTTP-MITM-PROXY CA>".to_string()),
    );
    param.key_usages = vec![
        rcgen::KeyUsagePurpose::KeyCertSign,
        rcgen::KeyUsagePurpose::CrlSign,
    ];
    param.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);

    let key_pair = rcgen::KeyPair::generate().unwrap();
    let cert = param.self_signed(&key_pair).unwrap();

    rcgen::CertifiedKey { cert, key_pair }
}

fn hash_value(request: &serde_json::Value) -> Result<String, anyhow::Error> {
    let mut hasher = Sha256::new();
    hasher.update(
        serde_json::to_string(&request).with_context(|| "Failed to stringify request json")?,
    );
    Ok(hex::encode(hasher.finalize()))
}

fn save_cache_body(
    path: PathBuf,
    parts: http::response::Parts,
    body: BytesMut,
) -> Result<(), anyhow::Error> {
    let path_str = path.to_string_lossy().into_owned();
    tracing::info!(path = path_str, "Finished processing request");
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            let body_str = String::from_utf8(body.to_vec())
                .with_context(|| format!("Failed to convert body to string for path {path_str}"))?;
            let mut reconstructed = hyper::Response::from_parts(parts, body_str);
            reconstructed.extensions_mut().clear();
            let json_response =
                http_serde_ext::response::serialize(&reconstructed, serde_json::value::Serializer)
                    .with_context(|| format!("Failed to serialize response for path {path_str}"))?;
            let json_str = serde_json::to_string(&json_response)
                .with_context(|| format!("Failed to stringify response for path {path_str}"))?;
            file.write_all(json_str.as_bytes())
                .with_context(|| format!("Failed to write to file for path {path_str}"))?;
            file.write_all(b"\n").with_context(|| {
                format!("Failed to write EOL newline to file for path {path_str}")
            })?;
            tracing::info!(path = path_str, "Wrote response to cache");
            Ok(())
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                // Log the error but otherwise continue on, as it's the client's fault
                tracing::error!(
                    path = path_str,
                    "Cache file already exists - two duplicate requests were likely made in parallel"
                );
                Ok(())
            } else {
                Err(e).with_context(|| format!("Failed to open cache file for path {path_str}"))
            }
        }
    }
}

const HEADER_TRUE: HeaderValue = HeaderValue::from_static("true");
const HEADER_FALSE: HeaderValue = HeaderValue::from_static("false");

async fn check_cache<
    E: std::fmt::Debug + 'static,
    T: Future<Output = Result<hyper::Response<BoxBody<Bytes, E>>, anyhow::Error>>,
    F: FnOnce() -> T,
>(
    args: &Args,
    request: &hyper::Request<Bytes>,
    missing: F,
) -> Result<hyper::Response<BoxBody<Bytes, E>>, anyhow::Error> {
    let mut request = request.clone();
    request.extensions_mut().clear();
    let mut sanitized_header = false;
    if args.sanitize_bearer_auth {
        if let Some(auth_header) = request.headers().get("Authorization") {
            if auth_header.to_str().unwrap().starts_with("Bearer ") {
                request.headers_mut().insert(
                    "Authorization",
                    HeaderValue::from_static("Bearer TENSORZERO_PROVIDER_PROXY_TOKEN"),
                );
                sanitized_header = true;
            }
        }
    }
    let json_request = http_serde_ext::request::serialize(&request, serde_json::value::Serializer)
        .with_context(|| "Failed to serialize request")?;
    let hash = hash_value(&json_request)?;
    let filename = format!(
        "{}-{}",
        request.uri().host().expect("Missing request host"),
        hash
    );

    let path = args.cache_path.join(filename);
    let path_str = path.to_string_lossy().into_owned();
    let (mut resp, cache_hit) = if path.exists() {
        tracing::info!(sanitized_header, "Cache hit: {}", path_str);
        let path_str_clone = path_str.clone();
        let resp = tokio::task::spawn_blocking(move || {
            let file = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read cache file {path_str}"))?;
            let response: serde_json::Value = serde_json::from_str(&file).with_context(|| {
                format!("Failed to deserialize response to JSON from {path_str}")
            })?;
            let response: hyper::Response<Bytes> = http_serde_ext::response::deserialize(response)
                .with_context(|| format!("Failed to deserialize HTTP response from {path_str}"))?;
            Ok::<_, anyhow::Error>(
                response.map(|b| BoxBody::new(Full::new(b).map_err(|e| match e {}))),
            )
        })
        .await
        .with_context(|| format!("Failed to await tokio spawn_blocking for {path_str_clone}"))??;
        (resp, HEADER_TRUE)
    } else {
        tracing::info!("Cache miss: {}", path_str);
        let response = match missing().await {
            Ok(response) => response,
            Err(e) => {
                tracing::error!(
                    e = e.as_ref() as &dyn std::error::Error,
                    "Failed to forward request"
                );
                let body = Full::new(Bytes::from(format!("Failed to forward request: {e:?}")));
                http::Response::builder()
                    .status(http::StatusCode::BAD_GATEWAY)
                    .body(BoxBody::new(body.map_err(|e| match e {})))
                    .with_context(|| "Failed to build response")?
            }
        };
        if response.status().is_success() {
            let (parts, body) = response.into_parts();
            let mut hyper_response = hyper::Response::from_parts(parts.clone(), body);
            // We need to clear the extensions in order to be able to serialize the response
            hyper_response.extensions_mut().clear();

            let write = args.write;

            // Start streaming the response to the client, running the provided callback once the whole body has been received
            // This lets us forward streaming responses without needing to wait for the entire response, while
            // still caching the entire response to disk.
            // Note that we make a `StreamingBodyCollector` even when `write` is false, so that
            // the HTTP behavior is consistent regardless of whether `write` is enabled.
            let body_collector = hyper_response.map(|b| {
                StreamingBodyCollector::new(
                    b,
                    Box::new(move |body| {
                        if write {
                            tokio::task::spawn_blocking(move || {
                                if let Err(e) = save_cache_body(path, parts, body) {
                                    tracing::error!(
                                        err = e.as_ref() as &dyn std::error::Error,
                                        "Failed to save cache body"
                                    );
                                }
                            });
                        }
                    }),
                )
            });

            (body_collector.map(|b| BoxBody::new(b)), HEADER_FALSE)
        } else {
            tracing::warn!("Skipping caching of non-success response: {:?}", response);
            (response, HEADER_FALSE)
        }
    };
    // Insert this header at the very end, to ensure that we never store this
    // header in the cache.
    resp.headers_mut().insert(CACHE_HEADER_NAME, cache_hit);
    Ok(resp)
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Path to the cache directory
    #[arg(long, default_value = "request_cache")]
    pub cache_path: PathBuf,
    /// Port to listen on
    #[arg(long, default_value = "3003")]
    pub port: u16,
    /// If `true`, replaces `Authorization: Bearer <token>` with `Authorization: Bearer TENSORZERO_PROVIDER_PROXY_TOKEN`
    /// when constructing a cache key.
    #[arg(long, default_value = "true")]
    pub sanitize_bearer_auth: bool,
    /// Whether to write to the cache when a cache miss occurs.
    /// If false, the proxy will still read existing entries from the cache, but not write new ones.
    #[arg(long, action = ArgAction::Set, default_value_t = true, num_args = 1)]
    pub write: bool,
}

pub async fn run_server(args: Args, server_started: oneshot::Sender<SocketAddr>) {
    use tracing_subscriber::EnvFilter;

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .try_init()
        .unwrap();

    let args = Arc::new(args);

    std::fs::create_dir_all(&args.cache_path).expect("Failed to create cache directory");

    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls ring provider");

    let root_cert = make_root_cert();

    let proxy = MitmProxy::new(
        // This is the root cert that will be used to sign the fake certificates
        Some(root_cert),
        Some(Cache::new(128)),
    );

    let client = reqwest::Client::new();
    let args_clone = args.clone();
    let (server_addr, server) = proxy
        .bind(
            ("127.0.0.1", args.port),
            service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                let client = client.clone();
                let args = args_clone.clone();
                async move {
                    let (parts, body) = req.into_parts();
                    let body_bytes = body
                        .collect()
                        .await
                        .with_context(|| "Failed to collect body")?
                        .to_bytes();
                    let bytes_request = hyper::Request::from_parts(parts, body_bytes);
                    let response = check_cache(&args, &bytes_request, || async {
                        let request: reqwest::Request =
                            bytes_request.clone().try_into().with_context(|| {
                                "Failed to convert Request from `hyper` to `reqwest`"
                            })?;
                        Ok(http::Response::from(client.execute(request).await?).map(BoxBody::new))
                    })
                    .await?;

                    Ok::<_, anyhow::Error>(response)
                }
            }),
        )
        .await
        .unwrap();

    tracing::info!(?args, "HTTP Proxy is listening on http://{server_addr}");
    server_started
        .send(server_addr)
        .expect("Failed to send server started signal");
    server.await;
}
