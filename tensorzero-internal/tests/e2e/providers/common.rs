#![allow(clippy::print_stdout)]
use std::{collections::HashMap, net::SocketAddr};

#[cfg(feature = "e2e_tests")]
use aws_config::Region;
#[cfg(feature = "e2e_tests")]
use aws_sdk_bedrockruntime::error::SdkError;
#[cfg(feature = "e2e_tests")]
use aws_sdk_s3::operation::get_object::GetObjectError;
use axum::{routing::get, Router};
use base64::prelude::*;
use futures::StreamExt;
use object_store::path::Path;
#[cfg(feature = "e2e_tests")]
use rand::Rng;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use std::future::IntoFuture;
use tensorzero::{
    CacheParamsOptions, ClientInferenceParams, InferenceOutput, InferenceResponse, Input,
    InputMessage, InputMessageContent,
};
use tensorzero_internal::inference::types::TextKind;
use tensorzero_internal::{
    cache::CacheEnabledMode,
    inference::types::{
        resolved_input::ImageWithPath,
        storage::{StorageKind, StoragePath},
        Base64Image, ContentBlock, ContentBlockChatOutput, Image, ImageKind, RequestMessage, Role,
        Text,
    },
    tool::{ToolCall, ToolResult},
};
use url::Url;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use crate::common::{
    get_clickhouse, select_chat_inference_clickhouse, select_inference_tags_clickhouse,
    select_json_inference_clickhouse, select_model_inference_clickhouse,
};

#[derive(Clone, Debug)]
pub struct E2ETestProvider {
    pub variant_name: String,
    pub model_name: String,
    pub model_provider_name: String,
    pub credentials: HashMap<String, String>,
}

/// Enforce that every provider implements a common set of tests.
///
/// To achieve that, each provider should call the `generate_provider_tests!` macro along with a
/// function that returns a `E2ETestProviders` struct.
///
/// If some test doesn't apply to a particular provider (e.g. provider doesn't support tool use),
/// then the provider should return an empty vector for the corresponding test.
pub struct E2ETestProviders {
    pub simple_inference: Vec<E2ETestProvider>,
    #[cfg_attr(not(feature = "e2e_tests"), allow(dead_code))]
    pub extra_body_inference: Vec<E2ETestProvider>,
    #[cfg_attr(not(feature = "e2e_tests"), allow(dead_code))]
    pub reasoning_inference: Vec<E2ETestProvider>,
    pub inference_params_inference: Vec<E2ETestProvider>,
    pub tool_use_inference: Vec<E2ETestProvider>,
    pub tool_multi_turn_inference: Vec<E2ETestProvider>,
    pub dynamic_tool_use_inference: Vec<E2ETestProvider>,
    pub parallel_tool_use_inference: Vec<E2ETestProvider>,
    pub json_mode_inference: Vec<E2ETestProvider>,
    #[cfg_attr(not(feature = "e2e_tests"), allow(dead_code))]
    pub image_inference: Vec<E2ETestProvider>,
    #[cfg(feature = "e2e_tests")]
    pub shorthand_inference: Vec<E2ETestProvider>,
    #[cfg(feature = "batch_tests")]
    pub supports_batch_inference: bool,
}

#[cfg(feature = "e2e_tests")]
pub async fn make_http_gateway() -> tensorzero::Client {
    tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::HTTPGateway {
        url: get_gateway_endpoint("/"),
    })
    .build()
    .await
    .unwrap()
}

#[allow(dead_code)]
pub async fn make_embedded_gateway() -> tensorzero::Client {
    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("tests/e2e/tensorzero.toml");
    tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: Some(config_path),
        clickhouse_url: Some(crate::common::CLICKHOUSE_URL.clone()),
    })
    .build()
    .await
    .unwrap()
}

#[allow(dead_code)]
pub async fn make_embedded_gateway_no_config() -> tensorzero::Client {
    tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: None,
        clickhouse_url: Some(crate::common::CLICKHOUSE_URL.clone()),
    })
    .build()
    .await
    .unwrap()
}

#[allow(dead_code)]
pub async fn make_embedded_gateway_with_config(config: &str) -> tensorzero::Client {
    let tmp_config = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();
    tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(crate::common::CLICKHOUSE_URL.clone()),
    })
    .build()
    .await
    .unwrap()
}

// We use a multi-threaded runtime so that the embedded gateway can use 'block_on'.
// For consistency, we also use a multi-threaded runtime for the http gateway test.
#[cfg(feature = "e2e_tests")]
#[macro_export]
macro_rules! make_gateway_test_functions {
    ($prefix:ident) => {
        paste::paste! {
            #[cfg(feature = "e2e_tests")]
            #[tokio::test(flavor = "multi_thread")]
            async fn [<$prefix _embedded_gateway>]() {
                $prefix ($crate::providers::common::make_embedded_gateway().await).await;
            }

            #[cfg(feature = "e2e_tests")]
            #[tokio::test(flavor = "multi_thread")]
            async fn [<$prefix _http_gateway>]() {
                $prefix ($crate::providers::common::make_http_gateway().await).await;
            }
        }
    };
}

#[cfg(feature = "e2e_tests")]
#[macro_export]
macro_rules! generate_provider_tests {
    ($func:ident) => {
        use $crate::providers::common::test_dynamic_tool_use_inference_request_with_provider;
        use $crate::providers::common::test_dynamic_tool_use_streaming_inference_request_with_provider;
        use $crate::providers::common::test_inference_params_inference_request_with_provider;
        use $crate::providers::common::test_inference_params_streaming_inference_request_with_provider;
        use $crate::providers::common::test_json_mode_inference_request_with_provider;
        use $crate::providers::common::test_json_mode_streaming_inference_request_with_provider;
        use $crate::providers::common::test_image_inference_with_provider_filesystem;
        use $crate::providers::common::test_image_inference_with_provider_amazon_s3;
        use $crate::providers::common::test_dynamic_json_mode_inference_request_with_provider;
        use $crate::providers::common::test_parallel_tool_use_inference_request_with_provider;
        use $crate::providers::common::test_parallel_tool_use_streaming_inference_request_with_provider;
        use $crate::providers::common::test_simple_inference_request_with_provider;
        use $crate::providers::common::test_simple_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_multi_turn_inference_request_with_provider;
        use $crate::providers::common::test_tool_multi_turn_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_allowed_tools_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_allowed_tools_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_auto_unused_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_auto_unused_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_auto_used_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_auto_used_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_none_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_none_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_required_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_required_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_specific_inference_request_with_provider;
        use $crate::providers::common::test_image_url_inference_with_provider_filesystem;
        use $crate::providers::common::test_tool_use_tool_choice_specific_streaming_inference_request_with_provider;
        use $crate::providers::common::test_extra_body_with_provider;
        use $crate::providers::reasoning::test_reasoning_inference_request_with_provider;
        use $crate::providers::reasoning::test_streaming_reasoning_inference_request_with_provider;
        use $crate::providers::reasoning::test_reasoning_inference_request_with_provider_json_mode;
        use $crate::providers::reasoning::test_streaming_reasoning_inference_request_with_provider_json_mode;

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_simple_inference_request() {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_simple_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_reasoning_inference_request() {
            let providers = $func().await.reasoning_inference;
            for provider in providers {
                test_reasoning_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_streaming_reasoning_inference_request() {
            let providers = $func().await.reasoning_inference;
            for provider in providers {
                test_streaming_reasoning_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_shorthand_inference_request() {
            let providers = $func().await.shorthand_inference;
            for provider in providers {
                test_simple_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_simple_streaming_inference_request() {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_simple_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_inference_params_inference_request() {
            let providers = $func().await.inference_params_inference;
            for provider in providers {
                test_inference_params_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_inference_params_streaming_inference_request() {
            let providers = $func().await.inference_params_inference;
            for provider in providers {
                test_inference_params_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_use_tool_choice_auto_used_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_auto_used_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_use_tool_choice_auto_used_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_auto_used_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_use_tool_choice_auto_unused_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_auto_unused_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_use_tool_choice_auto_unused_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_auto_unused_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_use_tool_choice_required_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_required_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_use_tool_choice_required_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_required_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_use_tool_choice_none_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_none_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_use_tool_choice_none_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_none_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_use_tool_choice_specific_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_specific_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_use_tool_choice_specific_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_specific_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_use_allowed_tools_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_allowed_tools_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_use_allowed_tools_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_allowed_tools_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_multi_turn_inference_request() {
            let providers = $func().await.tool_multi_turn_inference;
            for provider in providers {
                test_tool_multi_turn_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_tool_multi_turn_streaming_inference_request() {
            let providers = $func().await.tool_multi_turn_inference;
            for provider in providers {
                test_tool_multi_turn_streaming_inference_request_with_provider(provider).await;
            }
        }

        async fn test_dynamic_tool_use_inference_request(client: tensorzero::Client) {
            let providers = $func().await.dynamic_tool_use_inference;
            for provider in providers {
                test_dynamic_tool_use_inference_request_with_provider(provider, &client).await;
            }
        }
        $crate::make_gateway_test_functions!(test_dynamic_tool_use_inference_request);


        async fn test_dynamic_tool_use_streaming_inference_request(client: tensorzero::Client) {
            let providers = $func().await.dynamic_tool_use_inference;
            for provider in providers {
                test_dynamic_tool_use_streaming_inference_request_with_provider(provider, &client).await;
            }
        }
        $crate::make_gateway_test_functions!(test_dynamic_tool_use_streaming_inference_request);

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_parallel_tool_use_inference_request() {
            let providers = $func().await.parallel_tool_use_inference;
            for provider in providers {
                test_parallel_tool_use_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_parallel_tool_use_streaming_inference_request() {
            let providers = $func().await.parallel_tool_use_inference;
            for provider in providers {
                test_parallel_tool_use_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_json_mode_inference_request() {
            let providers = $func().await.json_mode_inference;
            for provider in providers {
                test_json_mode_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_reasoning_inference_request_json_mode() {
            let providers = $func().await.reasoning_inference;
            for provider in providers {
                test_reasoning_inference_request_with_provider_json_mode(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_streaming_reasoning_inference_request_json_mode() {
            let providers = $func().await.reasoning_inference;
            for provider in providers {
                test_streaming_reasoning_inference_request_with_provider_json_mode(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_dynamic_json_mode_inference_request() {
            let providers = $func().await.json_mode_inference;
            for provider in providers {
                test_dynamic_json_mode_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_json_mode_streaming_inference_request() {
            let providers = $func().await.json_mode_inference;
            for provider in providers {
                test_json_mode_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_image_inference_store_filesystem() {
            let providers = $func().await.image_inference;
            for provider in providers {
                test_image_inference_with_provider_filesystem(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_image_url_inference_store_filesystem() {
            let providers = $func().await.image_inference;
            for provider in providers {
                test_image_url_inference_with_provider_filesystem(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_image_inference_store_amazon_s3() {
            let providers = $func().await.image_inference;
            for provider in providers {
                test_image_inference_with_provider_amazon_s3(provider).await;
            }
        }

        #[cfg(feature = "e2e_tests")]
        #[tokio::test]
        async fn test_extra_body() {
            let providers = $func().await.extra_body_inference;
            for provider in providers {
                test_extra_body_with_provider(provider).await;
            }
        }
    };
}

#[cfg_attr(not(feature = "e2e_tests"), allow(dead_code))]
pub static FERRIS_PNG: &[u8] = include_bytes!("./ferris.png");

#[cfg_attr(not(feature = "e2e_tests"), allow(dead_code))]
pub async fn test_image_url_inference_with_provider_filesystem(provider: E2ETestProvider) {
    let temp_dir = tempfile::tempdir().unwrap();
    println!("Temporary image dir: {}", temp_dir.path().to_string_lossy());
    test_url_image_inference_with_provider_and_store(
        provider,
        StorageKind::Filesystem {
            path: temp_dir.path().to_string_lossy().to_string(),
        },
        &format!(
            r#"
        [object_storage]
        type = "filesystem"
        path = "{}"
        [functions]
        "#,
            temp_dir.path().to_string_lossy()
        ),
    )
    .await;

    // Check that image was stored in filesystem
    let result = std::fs::read(temp_dir.path().join(
        "observability/images/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
    ))
    .unwrap();
    assert_eq!(result, FERRIS_PNG);
}

#[cfg_attr(not(feature = "e2e_tests"), allow(dead_code))]
pub async fn test_image_inference_with_provider_filesystem(provider: E2ETestProvider) {
    let temp_dir = tempfile::tempdir().unwrap();
    println!("Temporary image dir: {}", temp_dir.path().to_string_lossy());
    test_base64_image_inference_with_provider_and_store(
        provider,
        &StorageKind::Filesystem {
            path: temp_dir.path().to_string_lossy().to_string(),
        },
        &format!(
            r#"
        [object_storage]
        type = "filesystem"
        path = "{}"
        [functions]
        "#,
            temp_dir.path().to_string_lossy()
        ),
        "",
    )
    .await;

    // Check that image was stored in filesystem
    let result = std::fs::read(temp_dir.path().join(
        "observability/images/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
    ))
    .unwrap();
    assert_eq!(result, FERRIS_PNG);
}

#[cfg(feature = "e2e_tests")]
pub async fn test_image_inference_with_provider_amazon_s3(provider: E2ETestProvider) {
    let test_bucket = "tensorzero-e2e-test-images";
    let test_bucket_region = "us-east-1";
    let config = aws_config::load_from_env()
        .await
        .to_builder()
        .region(Region::new(test_bucket_region))
        .build();

    let client = aws_sdk_s3::Client::new(&config);

    use rand::distributions::Alphanumeric;
    use rand::distributions::DistString;

    let mut prefix = Alphanumeric.sample_string(&mut rand::thread_rng(), 6);
    prefix += "-";

    test_image_inference_with_provider_s3_compatible(
        provider,
        &StorageKind::S3Compatible {
            bucket_name: Some(test_bucket.to_string()),
            region: Some("us-east-1".to_string()),
            prefix: prefix.clone(),
            endpoint: None,
        },
        &client,
        &format!(
            r#"
    [object_storage]
    type = "s3_compatible"
    region = "us-east-1"
    bucket_name = "{test_bucket}"
    prefix = "{prefix}"

    [functions]
    "#
        ),
        test_bucket,
        &prefix,
    )
    .await;
}

#[cfg(feature = "e2e_tests")]
pub async fn test_image_inference_with_provider_s3_compatible(
    provider: E2ETestProvider,
    storage_kind: &StorageKind,
    client: &aws_sdk_s3::Client,
    toml: &str,
    bucket_name: &str,
    prefix: &str,
) {
    let expected_key =
        format!("{prefix}observability/images/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png");

    // Check that object is deleted
    let err = client
        .get_object()
        .bucket(bucket_name)
        .key(&expected_key)
        .send()
        .await
        .expect_err("Image should not exist in s3 after deletion");

    if let SdkError::ServiceError(err) = err {
        let err = err.err();
        assert!(
            matches!(err, GetObjectError::NoSuchKey(_)),
            "Unexpected service error: {err:?}"
        );
    } else {
        panic!("Expected ServiceError: {err:?}");
    }

    test_base64_image_inference_with_provider_and_store(provider, storage_kind, toml, prefix).await;

    let result = client
        .get_object()
        .bucket(bucket_name)
        .key(&expected_key)
        .send()
        .await
        .expect("Failed to get image from S3-compatible store");

    assert_eq!(result.body.collect().await.unwrap().to_vec(), FERRIS_PNG);

    client
        .delete_object()
        .key(&expected_key)
        .bucket(bucket_name)
        .send()
        .await
        .unwrap();
}

async fn make_temp_image_server() -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let addr = SocketAddr::from(([0, 0, 0, 0], 0));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {addr}: {e}"));
    let real_addr = listener.local_addr().unwrap();

    let app = Router::new().route("/ferris.png", get(|| async { FERRIS_PNG.to_vec() }));

    let (send, recv) = tokio::sync::oneshot::channel::<()>();
    let shutdown_fut = async move {
        let _ = recv.await;
    };

    tokio::spawn(
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_fut)
            .into_future(),
    );

    (real_addr, send)
}

#[cfg_attr(not(feature = "e2e_tests"), allow(dead_code))]
pub async fn test_url_image_inference_with_provider_and_store(
    provider: E2ETestProvider,
    kind: StorageKind,
    config_toml: &str,
) {
    let episode_id = Uuid::now_v7();

    // The '_shutdown_sender' will wake up the receiver on drop
    let (server_addr, _shutdown_sender) = make_temp_image_server().await;
    let image_url = Url::parse(&format!("http://{}/ferris.png", server_addr)).unwrap();

    let client = make_embedded_gateway_with_config(config_toml).await;

    for should_be_cached in [false, true] {
        let response = client
            .inference(ClientInferenceParams {
                model_name: Some(provider.model_name.clone()),
                episode_id: Some(episode_id),
                input: Input {
                    system: None,
                    messages: vec![InputMessage {
                        role: Role::User,
                        content: vec![
                            InputMessageContent::Text(TextKind::Text {
                                text: "Describe the contents of the image".to_string(),
                            }),
                            InputMessageContent::Image(Image::Url {
                                url: image_url.clone(),
                            }),
                        ],
                    }],
                },
                cache_options: CacheParamsOptions {
                    enabled: CacheEnabledMode::On,
                    max_age_s: Some(10),
                },
                ..Default::default()
            })
            .await
            .unwrap();

        let InferenceOutput::NonStreaming(response) = response else {
            panic!("Expected non-streaming inference response");
        };

        check_url_image_response(
            response,
            Some(episode_id),
            &provider,
            should_be_cached,
            &kind,
            &image_url,
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

#[cfg_attr(not(feature = "e2e_tests"), allow(dead_code))]
pub async fn test_base64_image_inference_with_provider_and_store(
    provider: E2ETestProvider,
    kind: &StorageKind,
    config_toml: &str,
    prefix: &str,
) {
    let episode_id = Uuid::now_v7();

    let image_data = BASE64_STANDARD.encode(FERRIS_PNG);

    let client = make_embedded_gateway_with_config(config_toml).await;

    for should_be_cached in [false, true] {
        let response = client
            .inference(ClientInferenceParams {
                model_name: Some(provider.model_name.clone()),
                episode_id: Some(episode_id),
                input: Input {
                    system: None,
                    messages: vec![InputMessage {
                        role: Role::User,
                        content: vec![
                            InputMessageContent::Text(TextKind::Text {
                                text: "Describe the contents of the image".to_string(),
                            }),
                            InputMessageContent::Image(Image::Base64 {
                                mime_type: ImageKind::Png,
                                data: image_data.clone(),
                            }),
                        ],
                    }],
                },
                cache_options: CacheParamsOptions {
                    enabled: CacheEnabledMode::On,
                    max_age_s: Some(10),
                },
                ..Default::default()
            })
            .await
            .unwrap();

        let InferenceOutput::NonStreaming(response) = response else {
            panic!("Expected non-streaming inference response");
        };

        check_base64_image_response(
            response,
            Some(episode_id),
            &provider,
            should_be_cached,
            kind,
            prefix,
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

#[cfg_attr(not(feature = "e2e_tests"), allow(dead_code))]
pub async fn test_extra_body_with_provider(provider: E2ETestProvider) {
    test_extra_body_with_provider_and_stream(&provider, false).await;
    test_extra_body_with_provider_and_stream(&provider, true).await;
}

#[cfg_attr(not(feature = "e2e_tests"), allow(dead_code))]
pub async fn test_extra_body_with_provider_and_stream(provider: &E2ETestProvider, stream: bool) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "params": {
            "chat_completion": {
                "temperature": 9000
            }
        },
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "stream": stream,
        "tags": {"foo": "bar"},
    });

    let inference_id = if stream {
        let mut event_source = Client::new()
            .post(get_gateway_endpoint("/inference"))
            .json(&payload)
            .eventsource()
            .unwrap();

        let mut chunks = vec![];
        let mut found_done_chunk = false;
        while let Some(event) = event_source.next().await {
            let event = event.unwrap();
            match event {
                Event::Open => continue,
                Event::Message(message) => {
                    if message.data == "[DONE]" {
                        found_done_chunk = true;
                        break;
                    }
                    chunks.push(message.data);
                }
            }
        }
        assert!(found_done_chunk);

        let response_json = serde_json::from_str::<Value>(&chunks[0]).unwrap();
        let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
        Uuid::parse_str(inference_id).unwrap()
    } else {
        let response = Client::new()
            .post(get_gateway_endpoint("/inference"))
            .json(&payload)
            .send()
            .await
            .unwrap();

        // Check that the API response is ok
        assert_eq!(response.status(), StatusCode::OK);
        let response_json = response.json::<Value>().await.unwrap();

        println!("API response: {response_json:#?}");

        let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
        Uuid::parse_str(inference_id).unwrap()
    };

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;

    // Check the ModelInference Table. We don't check the ChatInference table, since we only care about the contents
    // of the raw request sent to the model provider.
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    let raw_request_val: serde_json::Value = serde_json::from_str::<Value>(raw_request).unwrap();
    let temp = if provider.variant_name.contains("aws-bedrock") {
        raw_request_val
            .get("inferenceConfig")
            .unwrap()
            .get("temperature")
    } else if provider
        .variant_name
        .contains("google-ai-studio-gemini-flash-8b")
        || provider.variant_name.contains("gcp-vertex-gemini-flash")
    {
        raw_request_val
            .get("generationConfig")
            .unwrap()
            .get("temperature")
    } else {
        raw_request_val.get("temperature")
    };
    assert_eq!(
        temp.expect("Missing temperature")
            .as_f64()
            .expect("Temperature is not a number"),
        0.123
    );
}

#[cfg(feature = "e2e_tests")]
pub async fn test_simple_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "stream": false,
        "tags": {"foo": "bar"},
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    check_simple_inference_response(response_json, Some(episode_id), &provider, false, false).await;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "stream": false,
        "tags": {"foo": "bar"},
        "cache_options": {"enabled": "on", "lookback_s": 10}
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    check_simple_inference_response(response_json, Some(episode_id), &provider, false, true).await;
}

#[cfg_attr(not(feature = "e2e_tests"), allow(dead_code))]
pub async fn check_base64_image_response(
    response: InferenceResponse,
    episode_id: Option<Uuid>,
    provider: &E2ETestProvider,
    should_be_cached: bool,
    kind: &StorageKind,
    prefix: &str,
) {
    let inference_id = response.inference_id();

    let episode_id_response = response.episode_id();
    if let Some(episode_id) = episode_id {
        assert_eq!(episode_id_response, episode_id);
    }

    let InferenceResponse::Chat(response) = response else {
        panic!("Expected chat inference response");
    };

    let content = response.content;
    assert_eq!(content.len(), 1);
    let content_block = content.first().unwrap();
    let ContentBlockChatOutput::Text(text) = content_block else {
        panic!("Expected text content block: {content_block:?}");
    };
    let content = &text.text;
    assert!(
        content.to_lowercase().contains("cartoon") || content.to_lowercase().contains("crab"),
        "Content should contain 'cartoon' or 'crab': {content}"
    );

    let usage = response.usage;
    let input_tokens = usage.input_tokens;
    let output_tokens = usage.output_tokens;
    if should_be_cached {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "tensorzero::default");

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    if let Some(episode_id) = episode_id {
        assert_eq!(retrieved_episode_id, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();

    let kind_json = serde_json::to_value(kind).unwrap();

    let correct_input = json!({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "value": "Describe the contents of the image"},
                    {
                        "type": "image",
                        "image": {
                            "url": null,
                            "mime_type": "image/png",
                        },
                        "storage_path": {
                            "kind": kind_json,
                            "path": format!("{prefix}observability/images/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png"),
                        },
                    }
                ]
            }
        ]
    });
    assert_eq!(input, correct_input);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(
        input_messages,
        vec![
            RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::Text(Text {
                    text: "Describe the contents of the image".to_string(),
                }), ContentBlock::Image(ImageWithPath {
                    image: Base64Image {
                        url: None,
                        data: None,
                        mime_type: ImageKind::Png,
                    },
                    storage_path: StoragePath {
                        kind: kind.clone(),
                        path: Path::parse(format!("{prefix}observability/images/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png")).unwrap(),
                    }
                })]
            },
        ]
    );

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(
        raw_request.contains("<TENSORZERO_IMAGE_0>"),
        "Unexpected raw_request: {raw_request}"
    );
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );
    assert_eq!(
        result.get("cached").unwrap().as_bool().unwrap(),
        should_be_cached
    );
}

#[cfg_attr(not(feature = "e2e_tests"), allow(dead_code))]
pub async fn check_url_image_response(
    response: InferenceResponse,
    episode_id: Option<Uuid>,
    provider: &E2ETestProvider,
    should_be_cached: bool,
    kind: &StorageKind,
    image_url: &Url,
) {
    let inference_id = response.inference_id();

    let episode_id_response = response.episode_id();
    if let Some(episode_id) = episode_id {
        assert_eq!(episode_id_response, episode_id);
    }

    let InferenceResponse::Chat(response) = response else {
        panic!("Expected chat inference response");
    };

    let content = response.content;
    assert_eq!(content.len(), 1);
    let content_block = content.first().unwrap();
    let ContentBlockChatOutput::Text(text) = content_block else {
        panic!("Expected text content block: {content_block:?}");
    };
    let content = &text.text;
    assert!(
        content.to_lowercase().contains("cartoon") || content.to_lowercase().contains("crab"),
        "Content should contain 'cartoon' or 'crab': {content}"
    );

    let usage = response.usage;
    let input_tokens = usage.input_tokens;
    let output_tokens = usage.output_tokens;
    if should_be_cached {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "tensorzero::default");

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    if let Some(episode_id) = episode_id {
        assert_eq!(retrieved_episode_id, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();

    let kind_json = serde_json::to_value(kind).unwrap();

    let correct_input = json!({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "value": "Describe the contents of the image"},
                    {
                        "type": "image",
                        "image": {
                            "url": image_url.to_string(),
                            "mime_type": "image/png",
                        },
                        "storage_path": {
                            "kind": kind_json,
                            "path": "observability/images/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png"
                        },
                    }
                ]
            }
        ]
    });
    assert_eq!(input, correct_input);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(
        input_messages,
        vec![
            RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::Text(Text {
                    text: "Describe the contents of the image".to_string(),
                }), ContentBlock::Image(ImageWithPath {
                    image: Base64Image {
                        url: Some(image_url.clone()),
                        data: None,
                        mime_type: ImageKind::Png,
                    },
                    storage_path: StoragePath {
                        kind: kind.clone(),
                        path: Path::parse("observability/images/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png").unwrap(),
                    }
                })]
            },
        ]
    );

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(
        raw_request.contains("<TENSORZERO_IMAGE_0>"),
        "Unexpected raw_request: {raw_request}"
    );
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );
    assert_eq!(
        result.get("cached").unwrap().as_bool().unwrap(),
        should_be_cached
    );
}

pub async fn check_simple_inference_response(
    response_json: Value,
    episode_id: Option<Uuid>,
    provider: &E2ETestProvider,
    is_batch: bool,
    should_be_cached: bool,
) {
    let hardcoded_function_name = "basic_test";
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    if let Some(episode_id) = episode_id {
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content.len(), 1);
    let content_block = content.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert!(content.to_lowercase().contains("tokyo"));

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    if should_be_cached {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, hardcoded_function_name);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    if let Some(episode_id) = episode_id {
        assert_eq!(retrieved_episode_id, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the name of the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);

    if !is_batch {
        let tags = result.get("tags").unwrap().as_object().unwrap();
        assert_eq!(tags.get("foo").unwrap().as_str().unwrap(), "bar");
    }

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    assert!(tool_params.is_empty());

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    let max_tokens = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        max_tokens
    );

    if !is_batch {
        let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
        assert!(processing_time_ms > 0);
    }

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();
    if should_be_cached {
        assert!(input_tokens.is_null());
        assert!(output_tokens.is_null());
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }
    if !is_batch && !should_be_cached {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }
    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the name of the capital city of Japan?"
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);

    if !is_batch {
        // Check the InferenceTag Table
        let result = select_inference_tags_clickhouse(
            &clickhouse,
            hardcoded_function_name,
            "foo",
            "bar",
            inference_id,
        )
        .await
        .unwrap();
        let id = result.get("inference_id").unwrap().as_str().unwrap();
        let id = Uuid::parse_str(id).unwrap();
        assert_eq!(id, inference_id);
    }
    assert_eq!(
        result.get("cached").unwrap().as_bool().unwrap(),
        should_be_cached
    );
}

#[cfg(feature = "e2e_tests")]
pub async fn test_simple_streaming_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();
    let tag_value = Uuid::now_v7().to_string();
    // Generate random u32
    let seed = rand::thread_rng().gen_range(0..u32::MAX);

    let original_content = test_simple_streaming_inference_request_with_provider_cache(
        &provider, episode_id, seed, &tag_value, false,
    )
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let cached_content = test_simple_streaming_inference_request_with_provider_cache(
        &provider, episode_id, seed, &tag_value, true,
    )
    .await;
    assert_eq!(original_content, cached_content);
}

#[cfg(feature = "e2e_tests")]
pub async fn test_simple_streaming_inference_request_with_provider_cache(
    provider: &E2ETestProvider,
    episode_id: Uuid,
    seed: u32,
    tag_value: &str,
    check_cache: bool,
) -> String {
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": format!("Dr. Mehta #{seed}")},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "stream": true,
        "tags": {"key": tag_value},
        "cache_options": {"enabled": "on", "lookback_s": 10}
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id: Option<Uuid> = None;
    let mut full_content = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for chunk in chunks.clone() {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            Some(inference_id) => {
                assert_eq!(inference_id, chunk_inference_id);
            }
            None => {
                inference_id = Some(chunk_inference_id);
            }
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        let content_blocks = chunk_json.get("content").unwrap().as_array().unwrap();
        if !content_blocks.is_empty() {
            let content_block = content_blocks.first().unwrap();
            let content = content_block.get("text").unwrap().as_str().unwrap();
            full_content.push_str(content);
        }

        if let Some(usage) = chunk_json.get("usage") {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    let inference_id = inference_id.unwrap();
    assert!(full_content.to_lowercase().contains("tokyo"));

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") || check_cache {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": format!("Dr. Mehta #{seed}")},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the name of the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<Value> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let content_block = output.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, full_content);

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    assert!(tool_params.is_empty());

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    let expected_max_tokens = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        expected_max_tokens
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags.get("key").unwrap().as_str().unwrap(), tag_value);

    // Check ClickHouse - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();

    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") || check_cache {
        assert!(input_tokens.is_null());
        assert!(output_tokens.is_null());
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    if check_cache {
        assert_eq!(response_time_ms, 0);
    } else {
        assert!(response_time_ms > 0);
    }

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    if check_cache {
        assert_eq!(ttft_ms, 0);
    } else {
        assert!(ttft_ms >= 1);
    }
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        format!("You are a helpful and friendly assistant named Dr. Mehta #{seed}")
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the name of the capital city of Japan?"
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    // Check the InferenceTag Table
    let result =
        select_inference_tags_clickhouse(&clickhouse, "basic_test", "key", tag_value, inference_id)
            .await
            .unwrap();
    let id = result.get("inference_id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);
    full_content
}

#[cfg(feature = "e2e_tests")]
pub async fn test_inference_params_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": [{"type": "raw_text", "value": "What is the name of the capital city of Japan?"}],
                }
            ]},
        "params": {
            "chat_completion": {
                "temperature": 0.9,
                "seed": 1337,
                "max_tokens": 120,
                "top_p": 0.9,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
            }
        },
        "stream": false,
        "credentials": provider.credentials,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    let response_status = response.status();
    assert_eq!(response_status, StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    check_inference_params_response(response_json, &provider, Some(episode_id), false).await;
}

// This function is also used by batch tests. If you adjust the prompt checked by this function
// ("What is the name of the capital city of Japan?"), make sure to update the batch tests to start batch
// jobs with the correct prompt.
pub async fn check_inference_params_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    is_batch: bool,
) {
    let hardcoded_function_name = "basic_test";
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content.len(), 1);
    let content_block = content.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert!(content.to_lowercase().contains("tokyo"));

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, hardcoded_function_name);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    if let Some(episode_id) = episode_id {
        let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
        let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
        assert_eq!(retrieved_episode_id, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "raw_text", "value": "What is the name of the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    assert!(tool_params.is_empty());

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    let temperature = inference_params
        .get("temperature")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(temperature, 0.9);
    let seed = inference_params.get("seed").unwrap().as_u64().unwrap();
    assert_eq!(seed, 1337);
    let max_tokens = inference_params
        .get("max_tokens")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(max_tokens, 120);
    let top_p = inference_params.get("top_p").unwrap().as_f64().unwrap();
    assert_eq!(top_p, 0.9);
    let presence_penalty = inference_params
        .get("presence_penalty")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(presence_penalty, 0.1);
    let frequency_penalty = inference_params
        .get("frequency_penalty")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(frequency_penalty, 0.2);

    if !is_batch {
        let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
        assert!(processing_time_ms > 0);
    } else {
        assert!(result.get("processing_time_ms").unwrap().is_null());
    }

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    } else {
        assert!(result.get("response_time_ms").unwrap().is_null());
        assert!(result.get("ttft_ms").unwrap().is_null());
    }
    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the name of the capital city of Japan?"
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
}

#[cfg(feature = "e2e_tests")]
pub async fn test_inference_params_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "params": {
            "chat_completion": {
                "temperature": 0.9,
                "seed": 1337,
                "max_tokens": 120,
                "top_p": 0.9,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
            }
        },
        "stream": true,
        "credentials": provider.credentials,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id: Option<Uuid> = None;
    let mut full_content = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for chunk in chunks.clone() {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            Some(inference_id) => {
                assert_eq!(inference_id, chunk_inference_id);
            }
            None => {
                inference_id = Some(chunk_inference_id);
            }
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        let content_blocks = chunk_json.get("content").unwrap().as_array().unwrap();
        if !content_blocks.is_empty() {
            let content_block = content_blocks.first().unwrap();
            let content = content_block.get("text").unwrap().as_str().unwrap();
            full_content.push_str(content);
        }

        if let Some(usage) = chunk_json.get("usage") {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    let inference_id = inference_id.unwrap();
    assert!(full_content.to_lowercase().contains("tokyo"));

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the name of the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<Value> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let content_block = output.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, full_content);

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    assert!(tool_params.is_empty());

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    let temperature = inference_params
        .get("temperature")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(temperature, 0.9);
    let seed = inference_params.get("seed").unwrap().as_u64().unwrap();
    assert_eq!(seed, 1337);
    let max_tokens = inference_params
        .get("max_tokens")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(max_tokens, 120);
    let top_p = inference_params.get("top_p").unwrap().as_f64().unwrap();
    assert_eq!(top_p, 0.9);
    let presence_penalty = inference_params
        .get("presence_penalty")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(presence_penalty, 0.1);
    let frequency_penalty = inference_params
        .get("frequency_penalty")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(frequency_penalty, 0.2);

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check ClickHouse - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();

    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert!(input_tokens.is_null());
        assert!(output_tokens.is_null());
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms >= 1);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the name of the capital city of Japan?"
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
}

#[cfg(feature = "e2e_tests")]
pub async fn test_tool_use_tool_choice_auto_used_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
        "stream": false,
        "variant_name": provider.variant_name,
        "tags": {"test_type": "auto_used"}
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    check_tool_use_tool_choice_auto_used_inference_response(
        response_json,
        &provider,
        Some(episode_id),
        false,
    )
    .await;
}

pub async fn check_tool_use_tool_choice_auto_used_inference_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    is_batch: bool,
) {
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty()); // could be > 1 if the model returns text as well
    let content_block = content
        .iter()
        .find(|block| block["type"] == "tool_call")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 2);
    let location = raw_arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");
    let units = raw_arguments.get("units").unwrap().as_str().unwrap();
    assert!(units == "celsius");

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();
    assert!(arguments.len() == 2);
    let location = arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");
    let units = arguments.get("units").unwrap().as_str().unwrap();
    assert!(units == "celsius");

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    if let Some(episode_id) = episode_id {
        let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
        assert_eq!(episode_id_result, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(raw_request.to_lowercase().contains("celsius"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(raw_response.contains("get_temperature"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            let arguments =
                serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
            let expected_arguments = json!({
                "location": "tokyo",
                "units": "celsius"
            });
            assert_eq!(arguments, expected_arguments);
        }
        _ => panic!("Unreachable"),
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_tool_use_tool_choice_auto_used_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // OpenAI O1 doesn't support streaming responses
    if provider.model_provider_name == "openai" && provider.model_name.starts_with("o1") {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut tool_id: Option<String> = None;
    let mut arguments = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    assert_eq!(
                        block.get("raw_name").unwrap().as_str().unwrap(),
                        "get_temperature"
                    );

                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    match &tool_id {
                        None => tool_id = Some(block_tool_id.to_string()),
                        Some(tool_id) => assert_eq!(tool_id, block_tool_id),
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();
                    arguments.push_str(chunk_arguments);
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else if provider.variant_name.contains("together") {
        // Do nothing: Together is flaky. Sometimes it returns non-zero usage, sometimes it returns zero usage...
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();
    let tool_id = tool_id.unwrap();
    assert!(serde_json::from_str::<Value>(&arguments).is_ok());

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(content_block.get("id").unwrap().as_str().unwrap(), tool_id);
    assert_eq!(
        content_block.get("name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block.get("arguments").unwrap().as_object().unwrap(),
        &serde_json::from_str::<serde_json::Map<String, Value>>(&arguments).unwrap()
    );
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        arguments
    );

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_temperature"));
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert!(input_tokens.is_null());
        assert!(output_tokens.is_null());
    } else if provider.variant_name.contains("together") {
        // Do nothing: Together is flaky. Sometimes it returns non-zero usage, sometimes it returns zero usage...
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms >= 1);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            let arguments =
                serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
            let expected_arguments = json!({
                "location": "tokyo",
                "units": "celsius"
            });
            assert_eq!(arguments, expected_arguments);
        }
        _ => panic!("Unreachable"),
    }
}

/// This test is similar to `test_tool_use_tool_choice_auto_used_inference_request_with_provider`, but it steers the model to not use the tool.
/// This ensures that ToolChoice::Auto is working as expected.
#[cfg(feature = "e2e_tests")]
pub async fn test_tool_use_tool_choice_auto_unused_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is your name?"
                }
            ]},
        "stream": false,
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    check_tool_use_tool_choice_auto_unused_inference_response(
        response_json,
        &provider,
        Some(episode_id),
        false,
    )
    .await;
}

pub async fn check_tool_use_tool_choice_auto_unused_inference_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    is_batch: bool,
) {
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.iter().any(|block| block["type"] == "tool_call"));
    let content_block = content
        .iter()
        .find(|block| block["type"] == "text")
        .unwrap();
    let content_block_text = content_block.get("text").unwrap().as_str().unwrap();
    assert!(content_block_text.to_lowercase().contains("mehta"));

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    if let Some(episode_id) = episode_id {
        let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
        assert_eq!(episode_id_result, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is your name?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );
    if !is_batch {
        let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
        assert!(processing_time_ms > 0);
    } else {
        assert!(result.get("processing_time_ms").unwrap().is_null());
    }

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("what is your name"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    } else {
        assert!(result.get("response_time_ms").unwrap().is_null());
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("mehta"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    } else {
        assert!(result.get("response_time_ms").unwrap().is_null());
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is your name?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let first = output.first().unwrap();
    match first {
        ContentBlock::Text(_text) => {}
        _ => {
            panic!("Expected a text block, got {:?}", first);
        }
    }
}

/// This test is similar to `test_tool_use_tool_choice_auto_used_streaming_inference_request_with_provider`, but it steers the model to not use the tool.
/// This ensures that ToolChoice::Auto is working as expected.
#[cfg(feature = "e2e_tests")]
pub async fn test_tool_use_tool_choice_auto_unused_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // OpenAI O1 doesn't support streaming responses
    if provider.model_provider_name == "openai" && provider.model_name.starts_with("o1") {
        return;
    }
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is your name?"
                }
            ]},
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut full_text = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    panic!("Tool call found in streaming inference response");
                }
                "text" => {
                    full_text.push_str(block.get("text").unwrap().as_str().unwrap());
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();

    assert!(full_text.to_lowercase().contains("mehta"));

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is your name?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();

    assert!(!output_clickhouse
        .iter()
        .any(|block| block["type"] == "tool_call"));

    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    assert!(content_block
        .get("text")
        .unwrap()
        .as_str()
        .unwrap()
        .to_lowercase()
        .contains("mehta"));

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("what is your name"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert!(input_tokens.is_null());
        assert!(output_tokens.is_null());
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms >= 1);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is your name?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let first = output.first().unwrap();
    match first {
        ContentBlock::Text(_text) => {}
        _ => {
            panic!("Expected a text block, got {:?}", first);
        }
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_tool_use_tool_choice_required_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Azure and Together don't support `tool_choice: "required"`
    if provider.model_provider_name == "azure" || provider.model_provider_name == "together" {
        return;
    }

    // GCP Vertex doesn't support `tool_choice: "required"` for Gemini 1.5 Flash
    if provider.model_provider_name.contains("gcp_vertex")
        && provider.model_name == "gemini-1.5-flash-001"
    {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is your name?"
                }
            ]},
        "tool_choice": "required",
        "stream": false,
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    check_tool_use_tool_choice_required_inference_response(
        response_json,
        &provider,
        Some(episode_id),
        false,
    )
    .await;
}

pub async fn check_tool_use_tool_choice_required_inference_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    is_batch: bool,
) {
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty()); // could be > 1 if the model returns text as well
    let content_block = content
        .iter()
        .find(|block| block["type"] == "tool_call")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 1 || raw_arguments.len() == 2);
    assert!(raw_arguments.get("location").unwrap().as_str().is_some());
    if raw_arguments.len() == 2 {
        let units = raw_arguments.get("units").unwrap().as_str().unwrap();
        assert!(units == "celsius" || units == "fahrenheit");
    }

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();
    assert!(arguments.len() == 1 || arguments.len() == 2);
    assert!(arguments.get("location").unwrap().as_str().is_some());
    if arguments.len() == 2 {
        let units = arguments.get("units").unwrap().as_str().unwrap();
        assert!(units == "celsius" || units == "fahrenheit");
    }

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    if let Some(episode_id) = episode_id {
        let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
        assert_eq!(episode_id_result, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is your name?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "required");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("what is your name"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_temperature"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    } else {
        assert!(result.get("response_time_ms").unwrap().is_null());
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is your name?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_tool_use_tool_choice_required_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Azure and Together don't support `tool_choice: "required"`
    if provider.model_provider_name == "azure" || provider.model_provider_name == "together" {
        return;
    }

    // GCP Vertex doesn't support `tool_choice: "required"` for Gemini 1.5 Flash
    if provider.model_provider_name.contains("gcp_vertex")
        && provider.model_name == "gemini-1.5-flash-001"
    {
        return;
    }

    // OpenAI O1 doesn't support streaming responses
    if provider.model_provider_name == "openai" && provider.model_name.starts_with("o1") {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is your name?"
                }
            ]},
        "tool_choice": "required",
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut tool_id: Option<String> = None;
    let mut arguments = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    assert_eq!(
                        block.get("raw_name").unwrap().as_str().unwrap(),
                        "get_temperature"
                    );

                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    match &tool_id {
                        None => tool_id = Some(block_tool_id.to_string()),
                        Some(tool_id) => assert_eq!(tool_id, block_tool_id),
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();
                    arguments.push_str(chunk_arguments);
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();
    let tool_id = tool_id.unwrap();
    assert!(serde_json::from_str::<Value>(&arguments).is_ok());

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is your name?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(content_block.get("id").unwrap().as_str().unwrap(), tool_id);
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        arguments
    );
    assert_eq!(
        content_block.get("name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block.get("arguments").unwrap().as_object().unwrap(),
        &serde_json::from_str::<serde_json::Map<String, Value>>(&arguments).unwrap()
    );

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "required");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("what is your name"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_temperature"));
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms >= 1);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is your name?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_tool_use_tool_choice_none_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // NOTE: The xAI API occasionally returns mangled output most of the time when this test runs.
    // The bug has been reported to the xAI team.
    //
    // https://gist.github.com/virajmehta/2911580b09713fc58aabfeb9ad62cf3b
    // We have disabled this test for that provider for now.
    if provider.model_provider_name == "xai" {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
        "tool_choice": "none",
        "stream": false,
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    check_tool_use_tool_choice_none_inference_response(
        response_json,
        &provider,
        Some(episode_id),
        false,
    )
    .await;
}

pub async fn check_tool_use_tool_choice_none_inference_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    is_batch: bool,
) {
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.iter().any(|block| block["type"] == "tool_call"));
    let content_block = content
        .iter()
        .find(|block| block["type"] == "text")
        .unwrap();
    assert!(content_block.get("text").unwrap().as_str().is_some());

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    if let Some(episode_id) = episode_id {
        let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
        assert_eq!(episode_id_result, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "none");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(raw_request.to_lowercase().contains("celsius"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    assert!(result.get("raw_response").unwrap().as_str().is_some());

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let first = output.first().unwrap();
    match first {
        ContentBlock::Text(_text) => {}
        _ => {
            panic!("Expected a text block, got {:?}", first);
        }
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_tool_use_tool_choice_none_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // OpenAI O1 doesn't support streaming responses
    if provider.model_provider_name == "openai" && provider.model_name.starts_with("o1") {
        return;
    }

    // NOTE: the xAI API now returns mangled output most of the time when this test runs.
    // The bug has been reported to the xAI team.
    //
    // https://gist.github.com/virajmehta/2911580b09713fc58aabfeb9ad62cf3b
    // We have disabled this test for that provider for now.
    if provider.model_provider_name == "xai" {
        return;
    }
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
        "tool_choice": "none",
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut full_text = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    panic!("Tool call found in streaming inference response");
                }
                "text" => {
                    full_text.push_str(block.get("text").unwrap().as_str().unwrap());
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();

    assert!(!output_clickhouse
        .iter()
        .any(|block| block["type"] == "tool_call"));

    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    assert!(content_block.get("text").unwrap().as_str().is_some());

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "none");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);

    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_temperature")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request
        .to_lowercase()
        .contains("what is the weather like in tokyo (in celsius)"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert!(input_tokens.is_null());
        assert!(output_tokens.is_null());
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms >= 1);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let first = output.first().unwrap();
    match first {
        ContentBlock::Text(_text) => {}
        _ => {
            panic!("Expected a text block, got {:?}", first);
        }
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_tool_use_tool_choice_specific_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // GCP Vertex AI, Mistral, and Together don't support ToolChoice::Specific.
    // (Together AI claims to support it, but we can't get it to behave strictly.)
    // In those cases, we use ToolChoice::Any with a single tool under the hood.
    // Even then, they seem to hallucinate a new tool.
    if provider.model_provider_name.contains("gcp_vertex")
        || provider.model_provider_name == "mistral"
        || provider.model_provider_name == "together"
    {
        return;
    }

    // OpenAI O1 doesn't support streaming responses
    if provider.model_provider_name == "openai" && provider.model_name.starts_with("o1") {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
        "tool_choice": {"specific": "self_destruct"},
        "additional_tools": [
            {
                "name": "self_destruct",
                "description": "Do not call this function under any circumstances.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fast": {
                            "type": "boolean",
                            "description": "Whether to use a fast method to self-destruct."
                        },
                    },
                    "required": ["fast"],
                    "additionalProperties": false
                },
            }
        ],
        "stream": false,
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    check_tool_use_tool_choice_specific_inference_response(
        response_json,
        &provider,
        Some(episode_id),
        false,
    )
    .await;
}

pub async fn check_tool_use_tool_choice_specific_inference_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    is_batch: bool,
) {
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty()); // could be > 1 if the model returns text as well
    let content_block = content
        .iter()
        .find(|block| block["type"] == "tool_call")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "self_destruct");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "self_destruct");

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 1);
    assert!(raw_arguments.get("fast").unwrap().as_bool().is_some());

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();
    assert!(arguments.len() == 1);
    assert!(arguments.get("fast").unwrap().as_bool().is_some());

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    if let Some(episode_id) = episode_id {
        let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
        assert_eq!(episode_id_result, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(
        tool_params["tool_choice"],
        json!({"specific": "self_destruct"})
    );
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 2);
    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_temperature")
        .unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "self_destruct")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Do not call this function under any circumstances."
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters
        .get("required")
        .unwrap()
        .as_array()
        .unwrap()
        .contains(&json!("fast")));
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    println!("Properties: {properties:#?}");
    assert!(properties.get("fast").is_some());

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("self_destruct"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("self_destruct"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "self_destruct");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_tool_use_tool_choice_specific_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // GCP Vertex AI, Mistral, and Together don't support ToolChoice::Specific.
    // (Together AI claims to support it, but we can't get it to behave strictly.)
    // In those cases, we use ToolChoice::Any with a single tool under the hood.
    // Even then, they seem to hallucinate a new tool.
    if provider.model_provider_name.contains("gcp_vertex")
        || provider.model_provider_name == "mistral"
        || provider.model_provider_name == "together"
    {
        return;
    }

    // OpenAI O1 doesn't support streaming responses
    if provider.model_provider_name == "openai" && provider.model_name.starts_with("o1") {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
        "additional_tools": [
            {
                "name": "self_destruct",
                "description": "Do not call this function under any circumstances.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fast": {
                            "type": "boolean",
                            "description": "Whether to use a fast method to self-destruct."
                        },
                    },
                    "required": ["fast"],
                    "additionalProperties": false
                },
            }
        ],
        "tool_choice": {"specific": "self_destruct"},
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut tool_id: Option<String> = None;
    let mut arguments = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    assert_eq!(
                        block.get("raw_name").unwrap().as_str().unwrap(),
                        "self_destruct"
                    );

                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    match &tool_id {
                        None => tool_id = Some(block_tool_id.to_string()),
                        Some(tool_id) => assert_eq!(
                            tool_id, block_tool_id,
                            "Provider returned multiple tool calls"
                        ),
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();
                    arguments.push_str(chunk_arguments);
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();
    let tool_id = tool_id.unwrap();
    assert!(
        serde_json::from_str::<Value>(&arguments).is_ok(),
        "Arguments: {arguments}"
    );

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(content_block.get("id").unwrap().as_str().unwrap(), tool_id);
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "self_destruct"
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        arguments
    );
    assert_eq!(
        content_block.get("name").unwrap().as_str().unwrap(),
        "self_destruct"
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        arguments
    );

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(
        tool_params["tool_choice"],
        json!({"specific": "self_destruct"})
    );
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 2);
    let tool = tools_available
        .iter()
        .find(|t| t["name"] == "get_temperature")
        .unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    let tool = tools_available
        .iter()
        .find(|t| t["name"] == "self_destruct")
        .unwrap();

    assert_eq!(
        tool["description"],
        "Do not call this function under any circumstances."
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters
        .get("required")
        .unwrap()
        .as_array()
        .unwrap()
        .contains(&json!("fast")));
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("fast"));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("self_destruct"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("self_destruct"));
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert!(input_tokens.is_null());
        assert!(output_tokens.is_null());
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms >= 1);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "self_destruct");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_tool_use_allowed_tools_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // GCP Vertex doesn't support `tool_choice: "required"` for Gemini 1.5 Flash,
    // and it won't call `get_humidity` on auto.
    if provider.model_provider_name.contains("gcp_vertex")
        && provider.model_name == "gemini-1.5-flash-001"
    {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."
                }
            ]},
        "tool_choice": "required",
        "allowed_tools": ["get_humidity"],
        "stream": false,
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    check_tool_use_tool_choice_allowed_tools_inference_response(
        response_json,
        &provider,
        Some(episode_id),
        false,
    )
    .await;
}

pub async fn check_tool_use_tool_choice_allowed_tools_inference_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    is_batch: bool,
) {
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty()); // could be > 1 if the model returns text as well
    let content_block = content
        .iter()
        .find(|block| block["type"] == "tool_call")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_humidity");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_humidity");

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 1);
    assert!(raw_arguments.get("location").unwrap().as_str().is_some());

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();
    assert!(arguments.len() == 1);
    assert!(arguments.get("location").unwrap().as_str().is_some());

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "basic_test");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    if let Some(episode_id) = episode_id {
        let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
        assert_eq!(episode_id_result, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "required");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);

    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_humidity")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Get the current humidity in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters
        .get("required")
        .unwrap()
        .as_array()
        .unwrap()
        .contains(&json!("location")));
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    println!("Properties: {properties:#?}");
    assert!(properties.get("location").is_some());

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_humidity"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_humidity"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_humidity");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_tool_use_allowed_tools_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // GCP Vertex doesn't support `tool_choice: "required"` for Gemini 1.5 Flash,
    // and it won't call `get_humidity` on auto.
    if provider.model_provider_name.contains("gcp_vertex")
        && provider.model_name == "gemini-1.5-flash-001"
    {
        return;
    }

    // OpenAI O1 doesn't support streaming responses
    if provider.model_provider_name == "openai" && provider.model_name.starts_with("o1") {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."
                }
            ]},
        "tool_choice": "required",
        "allowed_tools": ["get_humidity"],
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut tool_id: Option<String> = None;
    let mut arguments = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    assert_eq!(
                        block.get("raw_name").unwrap().as_str().unwrap(),
                        "get_humidity"
                    );

                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    match &tool_id {
                        None => tool_id = Some(block_tool_id.to_string()),
                        Some(tool_id) => assert_eq!(tool_id, block_tool_id),
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();
                    arguments.push_str(chunk_arguments);
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else if provider.variant_name.contains("together") {
        // Do nothing: Together is flaky. Sometimes it returns non-zero usage, sometimes it returns zero usage...
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();
    let tool_id = tool_id.unwrap();
    assert!(serde_json::from_str::<Value>(&arguments).is_ok());

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "basic_test");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(content_block.get("id").unwrap().as_str().unwrap(), tool_id);
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "get_humidity"
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        arguments
    );
    assert_eq!(
        content_block.get("name").unwrap().as_str().unwrap(),
        "get_humidity"
    );
    assert_eq!(
        content_block.get("arguments").unwrap().as_object().unwrap(),
        &serde_json::from_str::<serde_json::Map<String, Value>>(&arguments).unwrap()
    );

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "required");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_humidity");
    assert_eq!(
        tool["description"],
        "Get the current humidity in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the humidity for (e.g. \"New York\")"
    );

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_humidity"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_humidity"));
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert!(input_tokens.is_null());
        assert!(output_tokens.is_null());
    } else if provider.variant_name.contains("together") {
        // Do nothing: Together is flaky. Sometimes it returns non-zero usage, sometimes it returns zero usage...
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms >= 1);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_humidity");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_tool_multi_turn_inference_request_with_provider(provider: E2ETestProvider) {
    // NOTE: The xAI API returns an error for multi-turn tool use when the assistant message only has tool_calls but no content.
    // The xAI team has acknowledged the issue and is working on a fix.
    // We skip this test for xAI until the fix is deployed.
    // https://gist.github.com/GabrielBianconi/47a4247cfd8b6689e7228f654806272d
    if provider.model_provider_name == "xai" {
        return;
    }

    // OpenAI O1 doesn't support streaming responses
    if provider.model_provider_name == "openai" && provider.model_name.starts_with("o1") {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
       "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "123456789",
                            "name": "get_temperature",
                            "arguments": "{\"location\": \"Tokyo\", \"units\": \"celsius\"}"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "id": "123456789",
                            "name": "get_temperature",
                            "result": "70"
                        }
                    ]
                }
            ]},
        "variant_name": provider.variant_name,
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    check_tool_use_multi_turn_inference_response(response_json, &provider, Some(episode_id), false)
        .await;
}

pub async fn check_tool_use_multi_turn_inference_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    is_batch: bool,
) {
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content.len(), 1);
    let content_block = content.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert!(content.to_lowercase().contains("tokyo"));

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    if let Some(episode_id) = episode_id {
        let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
        let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
        assert_eq!(retrieved_episode_id, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "id": "123456789", "name": "get_temperature", "arguments": "{\"location\": \"Tokyo\", \"units\": \"celsius\"}"}]
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "id": "123456789", "name": "get_temperature", "result": "70"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    let max_tokens = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        max_tokens,
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![
        RequestMessage {
            role: Role::User,
            content: vec![
                "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        },
        RequestMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "123456789".to_string(),
                name: "get_temperature".to_string(),
                arguments: "{\"location\": \"Tokyo\", \"units\": \"celsius\"}".to_string(),
            })],
        },
        RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                id: "123456789".to_string(),
                name: "get_temperature".to_string(),
                result: "70".to_string(),
            })],
        },
    ];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let first = output.first().unwrap();
    match first {
        ContentBlock::Text(text) => {
            assert!(text.text.to_lowercase().contains("tokyo"));
        }
        _ => {
            panic!("Expected a text block, got {:?}", first);
        }
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_tool_multi_turn_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // NOTE: The xAI API returns an error for multi-turn tool use when the assistant message only has tool_calls but no content.
    // The xAI team has acknowledged the issue and is working on a fix.
    // We skip this test for xAI until the fix is deployed.
    // https://gist.github.com/GabrielBianconi/47a4247cfd8b6689e7228f654806272d
    if provider.model_provider_name == "xai" {
        return;
    }

    // OpenAI O1 doesn't support streaming responses
    if provider.model_provider_name == "openai" && provider.model_name.starts_with("o1") {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
       "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "123456789",
                            "name": "get_temperature",
                            "arguments": "{\"location\": \"Tokyo\"}"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "id": "123456789",
                            "name": "get_temperature",
                            "result": "30"
                        }
                    ]
                }
            ]},
        "variant_name": provider.variant_name,
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id: Option<Uuid> = None;
    let mut full_content = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for chunk in chunks.clone() {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            Some(inference_id) => {
                assert_eq!(inference_id, chunk_inference_id);
            }
            None => {
                inference_id = Some(chunk_inference_id);
            }
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        let content_blocks = chunk_json.get("content").unwrap().as_array().unwrap();
        if !content_blocks.is_empty() {
            let content_block = content_blocks.first().unwrap();
            let content = content_block.get("text").unwrap().as_str().unwrap();
            full_content.push_str(content);
        }

        if let Some(usage) = chunk_json.get("usage") {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    let inference_id = inference_id.unwrap();
    assert!(full_content.to_lowercase().contains("tokyo"));

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "123456789",
                        "name": "get_temperature",
                        "arguments": "{\"location\": \"Tokyo\"}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "id": "123456789",
                        "name": "get_temperature",
                        "result": "30"
                    }
                ]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<Value> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let content_block = output.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, full_content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check ClickHouse - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();

    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert!(input_tokens.is_null());
        assert!(output_tokens.is_null());
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms >= 1);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![
        RequestMessage {
            role: Role::User,
            content: vec![
                "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        },
        RequestMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "123456789".to_string(),
                name: "get_temperature".to_string(),
                arguments: "{\"location\": \"Tokyo\"}".to_string(),
            })],
        },
        RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                id: "123456789".to_string(),
                name: "get_temperature".to_string(),
                result: "30".to_string(),
            })],
        },
    ];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let first = output.first().unwrap();
    match first {
        ContentBlock::Text(text) => {
            assert!(text.text.to_lowercase().contains("tokyo"));
        }
        _ => {
            panic!("Expected a text block, got {:?}", first);
        }
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_dynamic_tool_use_inference_request_with_provider(
    provider: E2ETestProvider,
    client: &tensorzero::Client,
) {
    let episode_id = Uuid::now_v7();

    let response = client.inference(tensorzero::ClientInferenceParams {
        function_name: Some("basic_test".to_string()),
        model_name: None,
        variant_name: Some(provider.variant_name.clone()),
        episode_id: Some(episode_id),
        input: tensorzero::Input {
            system: Some(json!({"assistant_name": "Dr. Mehta"})),
            messages: vec![tensorzero::InputMessage {
                role: Role::User,
                content: vec![
                    tensorzero::InputMessageContent::Text(
                        TextKind::Text {
                            text: "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function.".to_string()
                        }
                    )
                ],
            }],
        },
        stream: Some(false),
        dynamic_tool_params: tensorzero::DynamicToolParams {
            additional_tools: Some(vec![tensorzero::Tool {
                name: "get_temperature".to_string(),
                description: "Get the current temperature in a given location".to_string(),
                parameters: json!({
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for (e.g. \"New York\")"
                        },
                        "units": {
                            "type": "string",
                            "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",
                            "enum": ["fahrenheit", "celsius"]
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }),
                strict: false,
            }]),
            ..Default::default()
        },
        ..Default::default()
    }).await.unwrap();

    match response {
        tensorzero::InferenceOutput::NonStreaming(response) => {
            let response_json = serde_json::to_value(&response).unwrap();

            println!("API response: {response_json:#?}");

            check_dynamic_tool_use_inference_response(
                response_json,
                &provider,
                Some(episode_id),
                false,
            )
            .await;
        }
        tensorzero::InferenceOutput::Streaming(_) => {
            panic!("Unexpected streaming response");
        }
    }
}

pub async fn check_dynamic_tool_use_inference_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    is_batch: bool,
) {
    let hardcoded_function_name = "basic_test";
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty()); // could be > 1 if the model returns text as well
    let content_block = content
        .iter()
        .find(|block| block["type"] == "tool_call")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 2);
    let location = raw_arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");
    let units = raw_arguments.get("units").unwrap().as_str().unwrap();
    assert!(units == "celsius");

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();
    assert!(arguments.len() == 2);
    let location = arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");
    let units = arguments.get("units").unwrap().as_str().unwrap();
    assert!(units == "celsius");

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, hardcoded_function_name);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    if let Some(episode_id) = episode_id {
        let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
        assert_eq!(episode_id_result, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(raw_response.contains("get_temperature"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_dynamic_tool_use_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
    client: &tensorzero::Client,
) {
    // OpenAI O1 doesn't support streaming responses
    if provider.model_provider_name == "openai" && provider.model_name.starts_with("o1") {
        return;
    }

    let episode_id = Uuid::now_v7();

    let input_function_name = "basic_test";

    let stream = client.inference(tensorzero::ClientInferenceParams {
        function_name: Some(input_function_name.to_string()),
        model_name: None,
        variant_name: Some(provider.variant_name.clone()),
        episode_id: Some(episode_id),
        input: tensorzero::Input {
            system: Some(json!({"assistant_name": "Dr. Mehta"})),
            messages: vec![tensorzero::InputMessage {
                role: Role::User,
                content: vec![tensorzero::InputMessageContent::Text(TextKind::Text { text: "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function.".to_string() })],
            }],
        },
        stream: Some(true),
        dynamic_tool_params: tensorzero::DynamicToolParams {
            additional_tools: Some(vec![tensorzero::Tool {
                name: "get_temperature".to_string(),
                description: "Get the current temperature in a given location".to_string(),
                parameters: json!({
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for (e.g. \"New York\")"
                        },
                        "units": {
                            "type": "string",
                            "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",
                            "enum": ["fahrenheit", "celsius"]
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }),
                strict: false,
            }]),
            ..Default::default()
        },
        ..Default::default()
    }).await.unwrap();

    let tensorzero::InferenceOutput::Streaming(mut stream) = stream else {
        panic!("Expected streaming response");
    };

    let mut chunks = vec![];
    while let Some(event) = stream.next().await {
        let chunk = event.unwrap();
        chunks.push(serde_json::to_value(&chunk).unwrap());
    }

    let mut inference_id = None;
    let mut tool_id: Option<String> = None;
    let mut arguments = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk_json in chunks {
        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    assert_eq!(
                        block.get("raw_name").unwrap().as_str().unwrap(),
                        "get_temperature"
                    );

                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    match &tool_id {
                        None => tool_id = Some(block_tool_id.to_string()),
                        Some(tool_id) => assert_eq!(tool_id, block_tool_id),
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();
                    arguments.push_str(chunk_arguments);
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else if provider.variant_name.contains("together") {
        // Do nothing: Together is flaky. Sometimes it returns non-zero usage, sometimes it returns zero usage...
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();
    let tool_id = tool_id.unwrap();
    assert!(serde_json::from_str::<Value>(&arguments).is_ok());

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, input_function_name);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(content_block.get("id").unwrap().as_str().unwrap(), tool_id);
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block.get("name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        arguments
    );
    assert_eq!(
        content_block.get("arguments").unwrap().as_object().unwrap(),
        &serde_json::from_str::<serde_json::Map<String, Value>>(&arguments).unwrap()
    );

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_temperature"));
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert!(input_tokens.is_null());
        assert!(output_tokens.is_null());
    } else if provider.variant_name.contains("together") {
        // Do nothing: Together is flaky. Sometimes it returns non-zero usage, sometimes it returns zero usage...
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms >= 1);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_parallel_tool_use_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper_parallel",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
                }
            ]},
        "parallel_tool_calls": true,
        "stream": false,
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    check_parallel_tool_use_inference_response(response_json, &provider, Some(episode_id), false)
        .await;
}

pub async fn check_parallel_tool_use_inference_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    is_batch: bool,
) {
    let hardcoded_function_name = "weather_helper_parallel";
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();

    // Validate the `get_temperature` tool call
    let content_block = content
        .iter()
        .find(|block| block["name"] == "get_temperature")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");
    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 2);
    let location = raw_arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");
    let units = raw_arguments.get("units").unwrap().as_str().unwrap();
    assert!(units == "celsius");

    let arguments = content_block.get("arguments").unwrap().as_object().unwrap();
    assert!(arguments.len() == 2);
    let location = arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");
    let units = arguments.get("units").unwrap().as_str().unwrap();
    assert!(units == "celsius");

    // Validate the `get_humidity` tool call
    let content_block = content
        .iter()
        .find(|block| block["name"] == "get_humidity")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_humidity");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_humidity");

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 1);
    let location = raw_arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();
    assert!(arguments.len() == 1);
    let location = arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, hardcoded_function_name);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    if let Some(episode_id) = episode_id {
        let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
        assert_eq!(episode_id_result, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "value": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
                    }]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], true);

    let tools_available = tool_params["tools_available"].as_array().unwrap();

    // Validate the `get_temperature` tool
    assert_eq!(tools_available.len(), 2);
    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_temperature")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Validate the `get_humidity` tool
    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_humidity")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Get the current humidity in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the humidity for (e.g. \"New York\")"
    );

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_temperature"));
    assert!(raw_response.to_lowercase().contains("tokyo"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with two tool calls. Use BOTH the \"get_temperature\" and \"get_humidity\" tools.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit with 50% humidity.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 2);
    let mut tool_call_names = vec![];
    for block in output {
        match block {
            ContentBlock::ToolCall(tool_call) => {
                tool_call_names.push(tool_call.name);
                serde_json::from_str::<Value>(&tool_call.arguments).unwrap();
            }
            _ => {
                panic!("Expected a tool call, got {:?}", block);
            }
        }
    }
    assert!(tool_call_names.contains(&"get_temperature".to_string()));
    assert!(tool_call_names.contains(&"get_humidity".to_string()));
}

#[cfg(feature = "e2e_tests")]
pub async fn test_parallel_tool_use_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper_parallel",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
                }
            ]},
        "parallel_tool_calls": true,
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut get_temperature_tool_id: Option<String> = None;
    let mut get_temperature_arguments = String::new();
    let mut get_humidity_tool_id: Option<String> = None;
    let mut get_humidity_arguments = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    let tool_name = block.get("raw_name").unwrap().as_str().unwrap();
                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();

                    match tool_name {
                        "get_temperature" => {
                            match &get_temperature_tool_id {
                                None => get_temperature_tool_id = Some(block_tool_id.to_string()),
                                Some(tool_id) => assert_eq!(tool_id, block_tool_id),
                            };
                            get_temperature_arguments.push_str(chunk_arguments);
                        }
                        "get_humidity" => {
                            match &get_humidity_tool_id {
                                None => get_humidity_tool_id = Some(block_tool_id.to_string()),
                                Some(tool_id) => assert_eq!(tool_id, block_tool_id),
                            };
                            get_humidity_arguments.push_str(chunk_arguments);
                        }
                        _ => {
                            panic!("Unexpected tool name: {}", tool_name);
                        }
                    }
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else if provider.variant_name.contains("together") {
        // Do nothing: Together is flaky. Sometimes it returns non-zero usage, sometimes it returns zero usage...
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();
    let get_temperature_tool_id = get_temperature_tool_id.unwrap();
    let get_humidity_tool_id = get_humidity_tool_id.unwrap();
    assert!(serde_json::from_str::<Value>(&get_temperature_arguments).is_ok());
    assert!(serde_json::from_str::<Value>(&get_humidity_arguments).is_ok());

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse - Inference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper_parallel");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well

    // Validate the `get_temperature` tool call
    let content_block = output_clickhouse
        .iter()
        .find(|block| block["name"] == "get_temperature")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(
        content_block.get("id").unwrap().as_str().unwrap(),
        get_temperature_tool_id
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        get_temperature_arguments
    );
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block.get("arguments").unwrap().as_object().unwrap(),
        &serde_json::from_str::<serde_json::Map<String, Value>>(&get_temperature_arguments)
            .unwrap()
    );

    // Validate the `get_humidity` tool call
    let content_block = output_clickhouse
        .iter()
        .find(|block| block["name"] == "get_humidity")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(
        content_block.get("id").unwrap().as_str().unwrap(),
        get_humidity_tool_id
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        get_humidity_arguments
    );
    assert_eq!(
        content_block.get("arguments").unwrap().as_object().unwrap(),
        &serde_json::from_str::<serde_json::Map<String, Value>>(&get_humidity_arguments).unwrap()
    );
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "get_humidity"
    );

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], true);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 2);

    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_temperature")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_humidity")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Get the current humidity in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the humidity for (e.g. \"New York\")"
    );

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("get_humidity"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(raw_request.to_lowercase().contains("celsius"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_temperature"));
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert!(input_tokens.is_null());
        assert!(output_tokens.is_null());
    } else if provider.variant_name.contains("together") {
        // Do nothing: Together is flaky. Sometimes it returns non-zero usage, sometimes it returns zero usage...
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms >= 1);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with two tool calls. Use BOTH the \"get_temperature\" and \"get_humidity\" tools.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit with 50% humidity.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 2);
    let mut tool_call_names = vec![];
    for block in output {
        match block {
            ContentBlock::ToolCall(tool_call) => {
                tool_call_names.push(tool_call.name);
                serde_json::from_str::<Value>(&tool_call.arguments).unwrap();
            }
            _ => {
                panic!("Expected a tool call, got {:?}", block);
            }
        }
    }
    assert!(tool_call_names.contains(&"get_temperature".to_string()));
    assert!(tool_call_names.contains(&"get_humidity".to_string()));
}

#[cfg(feature = "e2e_tests")]
pub async fn test_json_mode_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": {"country": "Japan"}
                }
            ]},
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    check_json_mode_inference_response(response_json, &provider, Some(episode_id), false).await;
}

pub async fn check_json_mode_inference_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    is_batch: bool,
) {
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let output = response_json.get("output").unwrap().as_object().unwrap();
    let parsed_output = output.get("parsed").unwrap().as_object().unwrap();
    assert!(parsed_output
        .get("answer")
        .unwrap()
        .as_str()
        .unwrap()
        .to_lowercase()
        .contains("tokyo"));
    let raw_output = output.get("raw").unwrap().as_str().unwrap();
    let raw_output: Value = serde_json::from_str(raw_output).unwrap();
    assert_eq!(&raw_output, output.get("parsed").unwrap());

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is ok - JsonInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - JsonInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "json_success");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    if let Some(episode_id) = episode_id {
        let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
        let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
        assert_eq!(retrieved_episode_id, episode_id);
    }

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": {"country": "Japan"}}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output_clickhouse = result.get("output").unwrap().as_str().unwrap();
    let output_clickhouse: Value = serde_json::from_str(output_clickhouse).unwrap();
    let output_clickhouse = output_clickhouse.as_object().unwrap();
    assert_eq!(output_clickhouse, output);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    let max_tokens = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        max_tokens
    );

    if !is_batch {
        let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
        assert!(processing_time_ms > 0);
    }
    let retrieved_output_schema = result.get("output_schema").unwrap().as_str().unwrap();
    let retrieved_output_schema: Value = serde_json::from_str(retrieved_output_schema).unwrap();
    let expected_output_schema = json!({
        "type": "object",
        "properties": {
          "answer": {
            "type": "string"
          }
        },
        "required": ["answer"],
        "additionalProperties": false
      }
    );
    assert_eq!(retrieved_output_schema, expected_output_schema);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the name of the capital city of Japan?"
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        ContentBlock::Text(text) => {
            let parsed: Value = serde_json::from_str(&text.text).unwrap();
            let answer = parsed.get("answer").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        ContentBlock::ToolCall(tool_call) => {
            // Handles implicit tool calls
            assert_eq!(tool_call.name, "respond");
            let arguments: Value = serde_json::from_str(&tool_call.arguments).unwrap();
            let answer = arguments.get("answer").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        _ => {
            panic!("Expected a text block, got {:?}", output[0]);
        }
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_dynamic_json_mode_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();
    let output_schema = json!({
      "type": "object",
      "properties": {
        "response": {
          "type": "string"
        }
      },
      "required": ["response"],
      "additionalProperties": false
    });
    let serialized_output_schema = serde_json::to_string(&output_schema).unwrap();

    let payload = json!({
        "function_name": "dynamic_json",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta", "schema": serialized_output_schema},
               "messages": [
                {
                    "role": "user",
                    "content": {"country": "Japan"}
                }
            ]},
        "stream": false,
        "output_schema": output_schema.clone(),
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    check_dynamic_json_mode_inference_response(
        response_json,
        &provider,
        Some(episode_id),
        Some(output_schema),
        false,
    )
    .await;
}

pub async fn check_dynamic_json_mode_inference_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    output_schema: Option<Value>,
    is_batch: bool,
) {
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let output = response_json.get("output").unwrap().as_object().unwrap();
    let parsed_output = output.get("parsed").unwrap().as_object().unwrap();
    assert!(parsed_output
        .get("response")
        .unwrap()
        .as_str()
        .unwrap()
        .to_lowercase()
        .contains("tokyo"));
    let raw_output = output.get("raw").unwrap().as_str().unwrap();
    let raw_output: Value = serde_json::from_str(raw_output).unwrap();
    assert_eq!(&raw_output, output.get("parsed").unwrap());

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check if ClickHouse is ok - JsonInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - JsonInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "dynamic_json");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    if let Some(episode_id) = episode_id {
        let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
        let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
        assert_eq!(retrieved_episode_id, episode_id);
    }

    if let Some(output_schema) = &output_schema {
        let serialized_output_schema = serde_json::to_string(&output_schema).unwrap();
        let input: Value =
            serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
        let correct_input = json!({
            "system": {"assistant_name": "Dr. Mehta", "schema": serialized_output_schema},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": {"country": "Japan"}}]
                }
            ]
        });
        assert_eq!(input, correct_input);
    }

    let output_clickhouse = result.get("output").unwrap().as_str().unwrap();
    let output_clickhouse: Value = serde_json::from_str(output_clickhouse).unwrap();
    let output_clickhouse = output_clickhouse.as_object().unwrap();
    assert_eq!(output_clickhouse, output);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    let max_tokens = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        max_tokens
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    if let Some(output_schema) = &output_schema {
        let retrieved_output_schema = result.get("output_schema").unwrap().as_str().unwrap();
        let retrieved_output_schema: Value = serde_json::from_str(retrieved_output_schema).unwrap();
        assert_eq!(retrieved_output_schema, *output_schema);
    }

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if !is_batch {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nDo not include any other text than the JSON object.  Do not include \"```json\" or \"```\" or anything else.\n\nPlease answer the questions in a JSON with the following schema:\n\n{\"type\":\"object\",\"properties\":{\"response\":{\"type\":\"string\"}},\"required\":[\"response\"],\"additionalProperties\":false}"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the name of the capital city of Japan?"
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        ContentBlock::Text(text) => {
            let parsed: Value = serde_json::from_str(&text.text).unwrap();
            let answer = parsed.get("response").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        ContentBlock::ToolCall(tool_call) => {
            // Handles implicit tool calls
            assert_eq!(tool_call.name, "respond");
            let arguments: Value = serde_json::from_str(&tool_call.arguments).unwrap();
            let answer = arguments.get("response").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        _ => {
            panic!("Expected a text block, got {:?}", output[0]);
        }
    }
}

#[cfg(feature = "e2e_tests")]
pub async fn test_json_mode_streaming_inference_request_with_provider(provider: E2ETestProvider) {
    if provider.variant_name.contains("tgi") {
        // TGI does not support streaming in JSON mode (because it doesn't support streaming tools)
        return;
    }
    // OpenAI O1 doesn't support streaming responses
    if provider.model_provider_name == "openai" && provider.model_name.starts_with("o1") {
        return;
    }
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": {"country": "Japan"}}]
                }
            ]},
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id: Option<Uuid> = None;
    let mut full_content = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for chunk in chunks.clone() {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            Some(inference_id) => {
                assert_eq!(inference_id, chunk_inference_id);
            }
            None => {
                inference_id = Some(chunk_inference_id);
            }
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);
        if let Some(raw) = chunk_json.get("raw").and_then(|raw| raw.as_str()) {
            if !raw.is_empty() {
                full_content.push_str(raw);
            }
        }

        if let Some(usage) = chunk_json.get("usage") {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    let inference_id = inference_id.unwrap();
    assert!(full_content.to_lowercase().contains("tokyo"));

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Check ClickHouse - JsonInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - JsonInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "json_success");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": {"country": "Japan"}}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Value = serde_json::from_str(output).unwrap();
    let output = output.as_object().unwrap();
    assert_eq!(output.keys().len(), 2);
    let clickhouse_parsed = output.get("parsed").unwrap().as_object().unwrap();
    let clickhouse_raw = output.get("parsed").unwrap().as_object().unwrap();
    assert_eq!(clickhouse_parsed, clickhouse_raw);
    let full_content_parsed: Value = serde_json::from_str(&full_content).unwrap();
    let full_content_parsed = full_content_parsed.as_object().unwrap();
    assert_eq!(clickhouse_parsed, full_content_parsed);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    let retrieved_output_schema = result.get("output_schema").unwrap().as_str().unwrap();
    let retrieved_output_schema: Value = serde_json::from_str(retrieved_output_schema).unwrap();
    let expected_output_schema = json!({
        "type": "object",
        "properties": {
          "answer": {
            "type": "string"
          }
        },
        "required": ["answer"],
        "additionalProperties": false
    });
    assert_eq!(retrieved_output_schema, expected_output_schema);

    // Check ClickHouse - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(raw_request.to_lowercase().contains("mehta"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();

    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert!(input_tokens.is_null());
        assert!(output_tokens.is_null());
    } else {
        assert!(input_tokens.as_u64().unwrap() > 0);
        assert!(output_tokens.as_u64().unwrap() > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms >= 1);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the name of the capital city of Japan?"
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        ContentBlock::Text(text) => {
            let parsed: Value = serde_json::from_str(&text.text).unwrap();
            let answer = parsed.get("answer").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        ContentBlock::ToolCall(tool_call) => {
            // Handles implicit tool calls
            assert_eq!(tool_call.name, "respond");
            let arguments: Value = serde_json::from_str(&tool_call.arguments).unwrap();
            let answer = arguments.get("answer").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        _ => {
            panic!("Expected a text block, got {:?}", output[0]);
        }
    }
}
