use std::time::Instant;

use provider_proxy::{run_server, Args};
use rand::Rng;
use tokio::sync::oneshot;
use warp::Filter;

#[tokio::test]
async fn test_provider_proxy() {
    let (server_started_tx, server_started_rx) = oneshot::channel();

    let temp_dir = tempfile::tempdir().unwrap();
    let _proxy_handle = tokio::spawn(run_server(
        Args {
            cache_path: temp_dir.path().to_path_buf(),
            port: 0,
        },
        server_started_tx,
    ));

    let target_server = warp::path("timestamp-good")
        .map(|| {
            format!(
                "Hello at {:?} {}",
                Instant::now(),
                rand::rng().random::<u32>()
            )
        })
        .or(warp::path("timestamp-bad").map(|| {
            let mut resp = warp::http::Response::new(format!(
                "Bad request at {:?} {}",
                Instant::now(),
                rand::rng().random::<u32>()
            ));
            *resp.status_mut() = warp::http::StatusCode::BAD_REQUEST;
            resp
        }));

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let shutdown_fut = async {
        shutdown_rx.await.unwrap();
    };

    let (target_server_addr, target_server_fut) =
        warp::serve(target_server).bind_with_graceful_shutdown(([127, 0, 0, 1], 0), shutdown_fut);
    let target_server_handle = tokio::spawn(target_server_fut);

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
    target_server_handle.await.unwrap();

    // When the target server is down, we should get an error
    let bad_gateway_response = client
        .post(format!("http://{target_server_addr}/timestamp-bad"))
        .send()
        .await
        .unwrap();
    assert_eq!(bad_gateway_response.status(), 502);
}
