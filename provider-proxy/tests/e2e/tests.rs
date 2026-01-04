#![expect(clippy::unwrap_used)]

use std::{
    future::{Future, IntoFuture},
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Instant,
};

use axum::{
    Router,
    response::{Sse, sse::Event},
    routing::post,
};
use futures_util::StreamExt;
use provider_proxy::{Args, CacheMode, run_server};
use rand::Rng;
use reqwest_eventsource::RequestBuilderExt;
use serde_json::Value;
use tokio::{sync::oneshot, task::JoinHandle};

async fn start_target_server(
    shutdown_signal: impl Future<Output = ()> + Send + 'static,
) -> (SocketAddr, JoinHandle<Result<(), std::io::Error>>) {
    let app = Router::new()
        .route(
            "/timestamp-good",
            post(|| async {
                let mut res = http::Response::new(format!(
                    "Hello at {:?} {}",
                    Instant::now(),
                    rand::rng().random::<u32>()
                ));
                res.headers_mut().insert(
                    "x-my-custom-header",
                    http::HeaderValue::from_str(&format!("{}", rand::rng().random::<u32>()))
                        .unwrap(),
                );
                res
            }),
        )
        .route(
            "/timestamp-bad",
            post(|| async {
                let mut resp = http::Response::new(format!(
                    "Bad request at {:?} {}",
                    Instant::now(),
                    rand::rng().random::<u32>()
                ));
                *resp.status_mut() = http::StatusCode::BAD_REQUEST;
                resp
            }),
        )
        .route(
            "/slow",
            post(|| async {
                Sse::new(async_stream::stream! {
                    yield Ok::<_, String>(Event::default().data("Hello"));
                    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                    yield Ok(Event::default().data("World"));
                    yield Ok(Event::default().data("[DONE]"));
                })
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
    #[expect(clippy::disallowed_methods)]
    let handle = tokio::spawn(
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal)
            .into_future(),
    );
    (addr, handle)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_provider_proxy() {
    let (server_started_tx, server_started_rx) = oneshot::channel();

    let temp_dir = tempfile::tempdir().unwrap();
    // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
    #[expect(clippy::disallowed_methods)]
    let _proxy_handle = tokio::spawn(run_server(
        Args {
            cache_path: temp_dir.path().to_path_buf(),
            port: 0,
            sanitize_traceparent: true,
            sanitize_bearer_auth: true,
            sanitize_aws_sigv4: true,
            sanitize_model_headers: true,
            remove_user_agent_non_amazon: false,
            health_port: 0,
            mode: CacheMode::ReadWrite,
        },
        server_started_tx,
    ));

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let shutdown_fut = async {
        shutdown_rx.await.unwrap();
    };

    let (target_server_addr, target_server_handle) = start_target_server(shutdown_fut).await;

    let proxy_addr = server_started_rx.await.unwrap();

    let client = reqwest::Client::builder()
        .proxy(reqwest::Proxy::all(format!("http://{proxy_addr}")).unwrap())
        .danger_accept_invalid_certs(true)
        .build()
        .unwrap();

    let first_local_response = client
        .post(format!("http://{target_server_addr}/timestamp-good"))
        .send()
        .await
        .unwrap();
    assert_eq!(first_local_response.status(), 200);
    let mut first_local_headers = first_local_response.headers().clone();
    assert!(first_local_headers.contains_key("x-my-custom-header"));
    // Remove this header so that we can compare the remaining headers between requests
    let cached = first_local_headers
        .remove("x-tensorzero-provider-proxy-cache")
        .unwrap();
    assert_eq!(cached, "false");
    let first_local_response_body = first_local_response.text().await.unwrap();

    // Wait for a file to show up on disk
    loop {
        let temp_path = temp_dir.path().to_path_buf();
        let found_file = tokio::task::spawn_blocking(move || {
            let files = std::fs::read_dir(temp_path).unwrap();
            for file in files {
                if file.unwrap().path().to_string_lossy().contains("127.0.0.1") {
                    return true;
                }
            }
            false
        })
        .await
        .unwrap();
        if found_file {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    let second_local_response = client
        .post(format!("http://{target_server_addr}/timestamp-good"))
        .send()
        .await
        .unwrap();
    assert_eq!(second_local_response.status(), 200);
    let mut second_local_headers = second_local_response.headers().clone();
    let cached = second_local_headers
        .remove("x-tensorzero-provider-proxy-cache")
        .unwrap();
    assert_eq!(cached, "true");
    assert_eq!(first_local_headers, second_local_headers);
    let second_local_response_body = second_local_response.text().await.unwrap();

    tracing::info!("first_local_response_body: {}", first_local_response_body);
    tracing::info!("second_local_response_body: {}", second_local_response_body);

    // The response should be cached, so we should get back the same value
    assert_eq!(first_local_response_body, second_local_response_body);

    // An error response should not be cached
    let first_bad_response = client
        .post(format!("http://{target_server_addr}/timestamp-bad"))
        .send()
        .await
        .unwrap();
    assert_eq!(first_bad_response.status(), 400);
    let first_bad_response_body = first_bad_response.text().await.unwrap();

    // Sleep for a little bit, to try to catch a bug where we incorrectly write the bad response to disk
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let second_bad_response = client
        .post(format!("http://{target_server_addr}/timestamp-bad"))
        .send()
        .await
        .unwrap();
    assert_eq!(second_bad_response.status(), 400);
    let second_bad_response_body = second_bad_response.text().await.unwrap();

    assert_ne!(first_bad_response_body, second_bad_response_body);

    shutdown_tx.send(()).unwrap();
    target_server_handle.await.unwrap().unwrap();

    // When the target server is down, we should get an error
    let bad_gateway_response = client
        .post(format!("http://{target_server_addr}/timestamp-bad"))
        .send()
        .await
        .unwrap();
    assert_eq!(bad_gateway_response.status(), 502);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_read_old_write_new() {
    let (server_started_tx, server_started_rx) = oneshot::channel();

    let temp_dir = tempfile::tempdir().unwrap();
    // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
    #[expect(clippy::disallowed_methods)]
    let _proxy_handle = tokio::spawn(run_server(
        Args {
            cache_path: temp_dir.path().to_path_buf(),
            port: 0,
            sanitize_traceparent: true,
            sanitize_bearer_auth: true,
            sanitize_aws_sigv4: true,
            health_port: 0,
            sanitize_model_headers: true,
            remove_user_agent_non_amazon: false,
            mode: CacheMode::ReadOldWriteNew,
        },
        server_started_tx,
    ));

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let shutdown_fut = async {
        shutdown_rx.await.unwrap();
    };

    let (target_server_addr, target_server_handle) = start_target_server(shutdown_fut).await;

    let proxy_addr = server_started_rx.await.unwrap();

    let client = reqwest::Client::builder()
        .proxy(reqwest::Proxy::all(format!("http://{proxy_addr}")).unwrap())
        .danger_accept_invalid_certs(true)
        .build()
        .unwrap();

    let first_local_response = client
        .post(format!("http://{target_server_addr}/timestamp-good"))
        .send()
        .await
        .unwrap();
    assert_eq!(first_local_response.status(), 200);
    let mut first_local_headers = first_local_response.headers().clone();
    assert!(first_local_headers.contains_key("x-my-custom-header"));
    // Remove this header so that we can compare the remaining headers between requests
    let cached = first_local_headers
        .remove("x-tensorzero-provider-proxy-cache")
        .unwrap();
    assert_eq!(cached, "false");

    let file_path = Arc::new(Mutex::new(None));
    let file_mtime = Arc::new(Mutex::new(None));

    // Wait for a file to show up on disk
    loop {
        let temp_path = temp_dir.path().to_path_buf();

        let file_path_clone = file_path.clone();
        let file_mtime_clone = file_mtime.clone();

        let found_file = tokio::task::spawn_blocking(move || {
            let files = std::fs::read_dir(temp_path).unwrap();
            for file in files {
                let file = file.unwrap();
                if file.path().to_string_lossy().contains("127.0.0.1") {
                    file_path_clone.lock().unwrap().replace(file.path());
                    file_mtime_clone
                        .lock()
                        .unwrap()
                        .replace(file.path().metadata().unwrap().modified().unwrap());
                    return true;
                }
            }
            false
        })
        .await
        .unwrap();
        if found_file {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    // The cache file was written after the proxy started, so we should get a cache miss
    let second_local_response = client
        .post(format!("http://{target_server_addr}/timestamp-good"))
        .send()
        .await
        .unwrap();
    assert_eq!(second_local_response.status(), 200);
    let mut second_local_headers = second_local_response.headers().clone();
    let cached = second_local_headers
        .remove("x-tensorzero-provider-proxy-cache")
        .unwrap();
    assert_eq!(cached, "false");
    let second_local_response_body = second_local_response.text().await.unwrap();

    shutdown_tx.send(()).unwrap();
    target_server_handle.await.unwrap().unwrap();

    let file_path = file_path.lock().unwrap().as_ref().unwrap().clone();
    let file_mtime = *file_mtime.lock().unwrap().as_ref().unwrap();

    // Wait for the file to be modified on disk
    loop {
        let new_mtime = file_path.metadata().unwrap().modified().unwrap();
        if new_mtime > file_mtime {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    // Start a new proxy server with the same settings
    let (server_started_tx, server_started_rx) = oneshot::channel();

    // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
    #[expect(clippy::disallowed_methods)]
    let _proxy_handle = tokio::spawn(run_server(
        Args {
            cache_path: temp_dir.path().to_path_buf(),
            port: 0,
            sanitize_traceparent: true,
            sanitize_bearer_auth: true,
            sanitize_aws_sigv4: true,
            health_port: 0,
            sanitize_model_headers: true,
            remove_user_agent_non_amazon: false,
            mode: CacheMode::ReadOldWriteNew,
        },
        server_started_tx,
    ));

    let proxy_addr = server_started_rx.await.unwrap();

    let client = reqwest::Client::builder()
        .proxy(reqwest::Proxy::all(format!("http://{proxy_addr}")).unwrap())
        .danger_accept_invalid_certs(true)
        .build()
        .unwrap();

    let third_local_response = client
        .post(format!("http://{target_server_addr}/timestamp-good"))
        .send()
        .await
        .unwrap();

    // We should now get a cache hit, because the cache file was written before the proxy started
    assert_eq!(third_local_response.status(), 200);
    let mut third_local_headers = third_local_response.headers().clone();
    let cached = third_local_headers
        .remove("x-tensorzero-provider-proxy-cache")
        .unwrap();
    assert_eq!(cached, "true");
    let third_local_response_body = third_local_response.text().await.unwrap();
    // We should use the second response body (from the original run of provider-proxy), which
    // should have overwritten the first response body on disk.
    assert_eq!(second_local_response_body, third_local_response_body);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dropped_stream_body() {
    let (server_started_tx, server_started_rx) = oneshot::channel();

    let temp_dir = tempfile::tempdir().unwrap();
    // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
    #[expect(clippy::disallowed_methods)]
    let _proxy_handle = tokio::spawn(run_server(
        Args {
            cache_path: temp_dir.path().to_path_buf(),
            port: 0,
            sanitize_traceparent: true,
            sanitize_bearer_auth: true,
            sanitize_aws_sigv4: true,
            health_port: 0,
            sanitize_model_headers: true,
            remove_user_agent_non_amazon: false,
            mode: CacheMode::ReadOldWriteNew,
        },
        server_started_tx,
    ));

    let (_shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let shutdown_fut = async {
        shutdown_rx.await.unwrap();
    };

    let proxy_addr = server_started_rx.await.unwrap();

    let (target_server_addr, _target_server_handle) = start_target_server(shutdown_fut).await;

    {
        let good_client = reqwest::Client::builder()
            .proxy(reqwest::Proxy::all(format!("http://{proxy_addr}")).unwrap())
            .danger_accept_invalid_certs(true)
            .build()
            .unwrap();

        let mut good_stream = good_client
            .post(format!("http://{target_server_addr}/slow"))
            .eventsource()
            .unwrap();
        // Read the entire stream, so that we're sure that provider-proxy will write the file to disk
        while let Some(event) = good_stream.next().await {
            match event {
                Err(reqwest_eventsource::Error::StreamEnded) => break,
                Err(e) => panic!("Unexpected error: {e:?}"),
                Ok(_) => continue,
            }
        }
    }

    // Check that the file is on disk
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let mut files = std::fs::read_dir(temp_dir.path())
        .unwrap()
        .collect::<Vec<_>>();
    assert!(files.len() == 1);
    let file = files.pop().unwrap().unwrap();
    let file_content = std::fs::read_to_string(file.path()).unwrap();
    let file_json = serde_json::from_str::<Value>(&file_content).unwrap();
    assert_eq!(
        file_json["body"],
        "data: Hello\n\ndata: World\n\ndata: [DONE]\n\n"
    );

    // Now, make the same request, but drop the stream (due to a timeout) before it's done

    let client = reqwest::Client::builder()
        .proxy(reqwest::Proxy::all(format!("http://{proxy_addr}")).unwrap())
        .danger_accept_invalid_certs(true)
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();

    let mut first_stream = client
        .post(format!("http://{target_server_addr}/slow"))
        .eventsource()
        .unwrap();

    let first_event = first_stream.next().await.unwrap().unwrap();
    assert_eq!(first_event, reqwest_eventsource::Event::Open);

    let second_event = first_stream.next().await.unwrap().unwrap();
    let reqwest_eventsource::Event::Message(second_event) = second_event else {
        panic!("Unexpected event: {second_event:?}");
    };
    assert_eq!(second_event.data, "Hello");
    // We should get a timeout
    let err = first_stream.next().await.unwrap().unwrap_err();
    assert!(
        matches!(&err, reqwest_eventsource::Error::Transport(e) if e.is_timeout()),
        "Unexpected error: {err:?}"
    );

    drop(first_stream);
    // Nothing should be on disk
    // The previous cache file should have been *deleted* at the start of the second request,
    // and no new file should have been written (because the stream was dropped before it was done)
    let files = std::fs::read_dir(temp_dir.path()).unwrap();
    assert!(files.count() == 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stream_body() {
    let (server_started_tx, server_started_rx) = oneshot::channel();

    let temp_dir = tempfile::tempdir().unwrap();
    // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
    #[expect(clippy::disallowed_methods)]
    let _proxy_handle = tokio::spawn(run_server(
        Args {
            cache_path: temp_dir.path().to_path_buf(),
            port: 0,
            sanitize_traceparent: true,
            sanitize_bearer_auth: true,
            sanitize_aws_sigv4: true,
            health_port: 0,
            sanitize_model_headers: true,
            remove_user_agent_non_amazon: false,
            mode: CacheMode::ReadOldWriteNew,
        },
        server_started_tx,
    ));

    let (_shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let shutdown_fut = async {
        shutdown_rx.await.unwrap();
    };

    let (target_server_addr, _target_server_handle) = start_target_server(shutdown_fut).await;

    let proxy_addr = server_started_rx.await.unwrap();

    let client = reqwest::Client::builder()
        .proxy(reqwest::Proxy::all(format!("http://{proxy_addr}")).unwrap())
        .danger_accept_invalid_certs(true)
        .build()
        .unwrap();

    let mut second_stream = client
        .post(format!("http://{target_server_addr}/slow"))
        .eventsource()
        .unwrap();

    while let Some(event) = second_stream.next().await {
        let event = event.unwrap();
        if let reqwest_eventsource::Event::Message(event) = event
            && event.data == "[DONE]"
        {
            break;
        }
    }

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // We should see a file on disk
    let files = std::fs::read_dir(temp_dir.path()).unwrap();
    let files = files.collect::<Vec<_>>();
    assert!(files.len() == 1);
    let file = files[0].as_ref().unwrap();
    let file_content = std::fs::read_to_string(file.path()).unwrap();
    let file_json = serde_json::from_str::<Value>(&file_content).unwrap();
    assert_eq!(
        file_json["body"],
        "data: Hello\n\ndata: World\n\ndata: [DONE]\n\n"
    );
}
