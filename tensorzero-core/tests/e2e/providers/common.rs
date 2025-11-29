#![expect(clippy::print_stdout)]
use std::io::Cursor;
use std::time::Duration;
use std::{collections::HashMap, net::SocketAddr};

use secrecy::SecretString;
use tensorzero::ClientExt;

use object_store::{aws::AmazonS3Builder, ObjectStore};
use std::sync::Arc;
use tensorzero_core::config::provider_types::{
    AnthropicDefaults, AzureDefaults, DeepSeekDefaults, FireworksDefaults, GCPDefaults,
    GoogleAIStudioGeminiDefaults, GroqDefaults, HyperbolicDefaults, MistralDefaults,
    OpenAIDefaults, OpenRouterDefaults, SGLangDefaults, TGIDefaults, TogetherDefaults,
    VLLMDefaults, XAIDefaults,
};
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_core::tool::Tool;

use axum::body::Body;
use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::{routing::get, Router};
use base64::prelude::*;
use futures::StreamExt;
use image::{ImageFormat, ImageReader};
use object_store::path::Path;

use rand::Rng;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use std::future::IntoFuture;
use tensorzero::{
    CacheParamsOptions, ClientInferenceParams, ClientInput, ClientInputMessage,
    ClientInputMessageContent, ClientSecretString, InferenceOutput, InferenceResponse,
};
use tensorzero_core::endpoints::inference::ChatCompletionInferenceParams;
use tensorzero_core::endpoints::object_storage::{get_object_handler, ObjectResponse, PathParams};
use tensorzero_core::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;

use tensorzero_core::inference::types::file::{Base64File, Detail, ObjectStoragePointer, UrlFile};
use tensorzero_core::inference::types::stored_input::StoredFile;
use tensorzero_core::inference::types::{Arguments, FinishReason, System, TextKind, Thought};
use tensorzero_core::utils::gateway::AppStateData;
use tensorzero_core::utils::testing::reset_capture_logs;
use tensorzero_core::{
    cache::CacheEnabledMode,
    inference::types::{
        storage::{StorageKind, StoragePath},
        ContentBlockChatOutput, File, Role, StoredContentBlock, StoredRequestMessage, Text,
    },
    tool::{ToolCall, ToolResult},
};
use url::Url;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
// use tensorzero_core::config::provider_types::*;
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_inference_tags_clickhouse,
    select_json_inference_clickhouse, select_model_inference_clickhouse,
};
use tensorzero_core::model::CredentialLocation;

use super::helpers::{get_modal_extra_headers, get_test_model_extra_headers};

#[derive(Clone, Debug)]
pub struct E2ETestProvider {
    pub variant_name: String,
    pub model_name: String,
    pub model_provider_name: String,

    pub credentials: HashMap<String, String>,

    pub supports_batch_inference: bool,
}

impl E2ETestProvider {
    /// Returns true if the provider is vLLM or SGLang (which typically require Modal headers)
    pub fn is_modal_provider(&self) -> bool {
        self.model_provider_name == "vllm" || self.model_provider_name == "sglang"
    }
}

#[derive(Clone, Debug)]
pub struct ModelTestProvider {
    pub provider_type: String,
    // These are keys and values that we would put into the model block for that provider
    pub model_info: HashMap<String, String>,
    pub use_modal_headers: bool,
}

#[derive(Clone, Debug)]
pub struct EmbeddingTestProvider {
    pub model_name: String,
    pub dimensions: usize,
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

    pub bad_auth_extra_headers: Vec<E2ETestProvider>,
    pub extra_body_inference: Vec<E2ETestProvider>,

    pub reasoning_inference: Vec<E2ETestProvider>,

    pub inference_params_dynamic_credentials: Vec<E2ETestProvider>,

    pub provider_type_default_credentials: Vec<E2ETestProvider>,
    pub provider_type_default_credentials_shorthand: Vec<E2ETestProvider>,
    pub credential_fallbacks: Vec<ModelTestProvider>,

    pub inference_params_inference: Vec<E2ETestProvider>,
    pub tool_use_inference: Vec<E2ETestProvider>,
    pub tool_multi_turn_inference: Vec<E2ETestProvider>,
    pub dynamic_tool_use_inference: Vec<E2ETestProvider>,
    pub parallel_tool_use_inference: Vec<E2ETestProvider>,
    pub json_mode_inference: Vec<E2ETestProvider>,
    pub json_mode_off_inference: Vec<E2ETestProvider>,

    pub image_inference: Vec<E2ETestProvider>,
    pub pdf_inference: Vec<E2ETestProvider>,
    pub input_audio: Vec<E2ETestProvider>,

    pub shorthand_inference: Vec<E2ETestProvider>,
    pub embeddings: Vec<EmbeddingTestProvider>,
}

#[macro_export]
macro_rules! generate_provider_tests {
    ($func:ident) => {
        use $crate::providers::common::test_dynamic_tool_use_inference_request_with_provider;
        use $crate::providers::common::test_dynamic_tool_use_streaming_inference_request_with_provider;
        use $crate::providers::common::test_inference_params_inference_request_with_provider;
        use $crate::providers::common::test_inference_params_streaming_inference_request_with_provider;
        use $crate::providers::common::test_provider_type_default_credentials_with_provider;
        use $crate::providers::common::test_provider_type_default_credentials_shorthand_with_provider;
        use $crate::providers::common::test_json_mode_inference_request_with_provider;
        use $crate::providers::common::test_json_mode_streaming_inference_request_with_provider;
        use $crate::providers::common::test_bad_auth_extra_headers_with_provider;
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
        use $crate::providers::common::test_inference_extra_body_with_provider;
        use $crate::providers::reasoning::test_reasoning_inference_request_simple_with_provider;
        use $crate::providers::reasoning::test_streaming_reasoning_inference_request_simple_with_provider;
        use $crate::providers::reasoning::test_reasoning_inference_request_with_provider_json_mode;
        use $crate::providers::reasoning::test_streaming_reasoning_inference_request_with_provider_json_mode;
        use $crate::providers::common::test_short_inference_request_with_provider;
        use $crate::providers::common::test_multi_turn_parallel_tool_use_inference_request_with_provider;
        use $crate::providers::common::test_multi_turn_parallel_tool_use_streaming_inference_request_with_provider;
        use $crate::providers::common::test_streaming_invalid_request_with_provider;
        use $crate::providers::common::test_json_mode_off_inference_request_with_provider;
        use $crate::providers::common::test_multiple_text_blocks_in_message_with_provider;
        use $crate::providers::common::test_streaming_include_original_response_with_provider;
        use $crate::providers::common::test_pdf_inference_with_provider_filesystem;
        use $crate::providers::common::test_stop_sequences_inference_request_with_provider;
        use $crate::providers::common::test_warn_ignored_thought_block_with_provider;
        use $crate::providers::embeddings::test_basic_embedding_with_provider;
        use $crate::providers::embeddings::test_bulk_embedding_with_provider;
        use $crate::providers::embeddings::test_embedding_with_dimensions_with_provider;
        use $crate::providers::common::test_provider_type_fallback_credentials_with_provider;
        use $crate::providers::embeddings::test_embedding_with_encoding_format_with_provider;
        use $crate::providers::embeddings::test_embedding_with_user_parameter_with_provider;
        use $crate::providers::embeddings::test_embedding_invalid_model_error_with_provider;
        use $crate::providers::embeddings::test_embedding_large_bulk_with_provider;
        use $crate::providers::embeddings::test_embedding_consistency_with_provider;
        use $crate::providers::embeddings::test_embedding_cache_with_provider;
        use $crate::providers::embeddings::test_embedding_cache_options_with_provider;
        use $crate::providers::embeddings::test_embedding_dryrun_with_provider;
        use $crate::providers::embeddings::test_single_token_array_with_provider;
        use $crate::providers::embeddings::test_batch_token_arrays_semantic_similarity_with_provider;

        #[tokio::test]
        async fn test_simple_inference_request() {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_simple_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn test_warn_ignored_thought_block() {
            let logs_contain = tensorzero_core::utils::testing::capture_logs();
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_warn_ignored_thought_block_with_provider(provider, &logs_contain).await;
            }
        }

        #[tokio::test]
        async fn test_reasoning_inference_request_simple() {
            let providers = $func().await.reasoning_inference;
            for provider in providers {
                test_reasoning_inference_request_simple_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_streaming_reasoning_inference_request_simple() {
            let providers = $func().await.reasoning_inference;
            for provider in providers {
                test_streaming_reasoning_inference_request_simple_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_bad_auth_extra_headers() {
            let providers = $func().await.bad_auth_extra_headers;
            for provider in providers {
                test_bad_auth_extra_headers_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_shorthand_inference_request() {
            let providers = $func().await.shorthand_inference;
            for provider in providers {
                test_simple_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_streaming_include_original_response() {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_streaming_include_original_response_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_simple_streaming_inference_request() {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_simple_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_streaming_invalid_request() {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_streaming_invalid_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_inference_params_inference_request() {
            let providers = $func().await.inference_params_dynamic_credentials;
            for provider in providers {
                test_inference_params_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_inference_params_streaming_inference_request() {
            let providers = $func().await.inference_params_dynamic_credentials;
            for provider in providers {
                test_inference_params_streaming_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_provider_type_default_credentials() {
            let providers = $func().await.provider_type_default_credentials;
            for provider in providers {
                test_provider_type_default_credentials_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_provider_type_default_credentials_shorthand() {
            let providers = $func().await.provider_type_default_credentials_shorthand;
            for provider in providers {
                test_provider_type_default_credentials_shorthand_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_provider_type_fallback_credentials() {
            let logs_contain = tensorzero_core::utils::testing::capture_logs();
            // We just need a longhand model
            let all_providers = $func().await;
            let providers = all_providers.credential_fallbacks;
            let supports_dynamic_credentials = !all_providers.provider_type_default_credentials.is_empty();
            for provider in providers {
                test_provider_type_fallback_credentials_with_provider(provider, supports_dynamic_credentials, &logs_contain).await;
            }
        }


        #[tokio::test]
        async fn test_tool_use_tool_choice_auto_used_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_auto_used_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_tool_use_tool_choice_auto_used_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_auto_used_streaming_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_tool_use_tool_choice_auto_unused_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_auto_unused_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_tool_use_tool_choice_auto_unused_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_auto_unused_streaming_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_tool_use_tool_choice_required_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_required_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_tool_use_tool_choice_required_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_required_streaming_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_tool_use_tool_choice_none_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_none_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_tool_use_tool_choice_none_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_none_streaming_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_tool_use_tool_choice_specific_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_specific_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_tool_use_tool_choice_specific_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_specific_streaming_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_tool_use_allowed_tools_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_allowed_tools_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_tool_use_allowed_tools_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_allowed_tools_streaming_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_tool_multi_turn_inference_request() {
            let providers = $func().await.tool_multi_turn_inference;
            for provider in providers {
                test_tool_multi_turn_inference_request_with_provider(provider).await;
            }
        }


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
        tensorzero::make_gateway_test_functions!(test_dynamic_tool_use_inference_request);

        async fn test_stop_sequences_inference_request(client: tensorzero::Client) {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_stop_sequences_inference_request_with_provider(provider, &client).await;
            }
        }
        tensorzero::make_gateway_test_functions!(test_stop_sequences_inference_request);

        async fn test_dynamic_tool_use_streaming_inference_request(client: tensorzero::Client) {
            let providers = $func().await.dynamic_tool_use_inference;
            for provider in providers {
                test_dynamic_tool_use_streaming_inference_request_with_provider(provider, &client).await;
            }
        }
        tensorzero::make_gateway_test_functions!(test_dynamic_tool_use_streaming_inference_request);


        #[tokio::test]
        async fn test_parallel_tool_use_inference_request() {
            let providers = $func().await.parallel_tool_use_inference;
            for provider in providers {
                test_parallel_tool_use_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_parallel_tool_use_streaming_inference_request() {
            let providers = $func().await.parallel_tool_use_inference;
            for provider in providers {
                test_parallel_tool_use_streaming_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_json_mode_inference_request() {
            let providers = $func().await.json_mode_inference;
            for provider in providers {
                test_json_mode_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_reasoning_inference_request_json_mode() {
            let providers = $func().await.reasoning_inference;
            for provider in providers {
                test_reasoning_inference_request_with_provider_json_mode(provider).await;
            }
        }


        #[tokio::test]
        async fn test_streaming_reasoning_inference_request_json_mode() {
            let providers = $func().await.reasoning_inference;
            for provider in providers {
                test_streaming_reasoning_inference_request_with_provider_json_mode(provider).await;
            }
        }


        #[tokio::test]
        async fn test_dynamic_json_mode_inference_request() {
            let providers = $func().await.json_mode_inference;
            for provider in providers {
                test_dynamic_json_mode_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_json_mode_streaming_inference_request() {
            let providers = $func().await.json_mode_inference;
            for provider in providers {
                test_json_mode_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn test_pdf_inference_store_filesystem() {
            let providers = $func().await.pdf_inference;
            for provider in providers {
                test_pdf_inference_with_provider_filesystem(provider).await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn test_audio_inference_store_filesystem() {
            let providers = $func().await.input_audio;
            for provider in providers {
                $crate::providers::commonv2::input_audio::test_audio_inference_with_provider_filesystem(provider).await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn test_image_inference_store_filesystem() {
            let providers = $func().await.image_inference;
            for provider in providers {
                test_image_inference_with_provider_filesystem(provider).await;
            }
        }


        #[tokio::test(flavor = "multi_thread")]
        async fn test_image_url_inference_store_filesystem() {
            let providers = $func().await.image_inference;
            for provider in providers {
                test_image_url_inference_with_provider_filesystem(provider).await;
            }
        }


        #[tokio::test(flavor = "multi_thread")]
        async fn test_image_inference_store_amazon_s3() {
            let providers = $func().await.image_inference;
            for provider in providers {
                test_image_inference_with_provider_amazon_s3(provider).await;
            }
        }


        #[tokio::test]
        async fn test_extra_body() {
            let providers = $func().await.extra_body_inference;
            for provider in providers {
                test_extra_body_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_inference_extra_body() {
            let providers = $func().await.extra_body_inference;
            for provider in providers {
                test_inference_extra_body_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_short_inference_request() {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_short_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_multi_turn_parallel_tool_use_inference_request() {
            let providers = $func().await.parallel_tool_use_inference;
            for provider in providers {
                test_multi_turn_parallel_tool_use_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_multi_turn_parallel_tool_use_streaming_inference_request() {
            let providers = $func().await.parallel_tool_use_inference;
            for provider in providers {
                test_multi_turn_parallel_tool_use_streaming_inference_request_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_json_mode_off_inference_request() {
            let providers = $func().await.json_mode_off_inference;
            for provider in providers {
                test_json_mode_off_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_multiple_text_blocks_in_message() {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_multiple_text_blocks_in_message_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_basic_embedding() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_basic_embedding_with_provider(provider).await;
            }
        }


        #[tokio::test]
        async fn test_bulk_embedding() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_bulk_embedding_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_embedding_with_dimensions() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_embedding_with_dimensions_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_embedding_with_encoding_format() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_embedding_with_encoding_format_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_embedding_with_user_parameter() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_embedding_with_user_parameter_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_embedding_invalid_model_error() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_embedding_invalid_model_error_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_embedding_large_bulk() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_embedding_large_bulk_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_embedding_consistency() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_embedding_consistency_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_embedding_cache() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_embedding_cache_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_embedding_cache_options() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_embedding_cache_options_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_embedding_dryrun() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_embedding_dryrun_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_single_token_array() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_single_token_array_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_batch_token_arrays_semantic_similarity() {
            let providers = $func().await.embeddings;
            for provider in providers {
                test_batch_token_arrays_semantic_similarity_with_provider(provider).await;
            }
        }

    };
}

pub const PDF_FUNCTION_CONFIG: &str = r#"
[functions.pdf_test]
type = "chat"

[functions.pdf_test.variants.openai]
type = "chat_completion"
model = "openai::gpt-4o-mini-2024-07-18"

[functions.pdf_test.variants.openai-responses]
type = "chat_completion"
model = "responses-gpt-4o-mini-2024-07-18"

[functions.pdf_test.variants.gcp_vertex_gemini]
type = "chat_completion"
model = "gcp_vertex_gemini::projects/tensorzero-public/locations/us-central1/publishers/google/models/gemini-2.0-flash-lite"

[functions.pdf_test.variants.gcp_vertex_anthropic]
type = "chat_completion"
model = "gcp_vertex_anthropic::projects/tensorzero-public/locations/global/publishers/anthropic/models/claude-sonnet-4-5@20250929"

[functions.pdf_test.variants.google_ai_studio]
type = "chat_completion"
model = "google_ai_studio_gemini::gemini-2.0-flash-lite"

[functions.pdf_test.variants.anthropic]
type = "chat_completion"
model = "anthropic::claude-sonnet-4-5-20250929"

[functions.pdf_test.variants.gcp-vertex-sonnet]
type = "chat_completion"
model = "claude-sonnet-4-5-gcp-vertex"

[functions.pdf_test.variants.aws-bedrock]
type = "chat_completion"
model = "claude-3-haiku-20240307-aws-bedrock"

[models.claude-sonnet-4-5-gcp-vertex]
routing = ["gcp_vertex_anthropic"]

[models.claude-sonnet-4-5-gcp-vertex.providers.gcp_vertex_anthropic]
type = "gcp_vertex_anthropic"
model_id = "claude-sonnet-4-5@20250929"
location = "us-east5"
project_id = "tensorzero-public"

[models."responses-gpt-4o-mini-2024-07-18"]
routing = ["openai"]

[models."responses-gpt-4o-mini-2024-07-18".providers.openai]
type = "openai"
model_name = "gpt-4o-mini-2024-07-18"
api_type = "responses"

[models.claude-3-haiku-20240307-aws-bedrock]
routing = ["aws_bedrock"]

[models.claude-3-haiku-20240307-aws-bedrock.providers.aws_bedrock]
type = "aws_bedrock"
model_id = "us.anthropic.claude-3-haiku-20240307-v1:0"
region = "us-east-1"
"#;

pub static FERRIS_PNG: &[u8] = include_bytes!("./ferris.png");
pub static DEEPSEEK_PAPER_PDF: &[u8] = include_bytes!("./deepseek_paper.pdf");

pub const IMAGE_FUNCTION_CONFIG: &str = r#"
[functions.image_test]
type = "chat"

[functions.image_test.variants.openai]
type = "chat_completion"
model = "openai::gpt-4o-mini-2024-07-18"

[functions.image_test.variants.openai-responses]
type = "chat_completion"
model = "responses-gpt-4o-mini-2024-07-18"

[functions.image_test.variants.anthropic]
type = "chat_completion"
model = "anthropic::claude-3-haiku-20240307"

[functions.image_test.variants.google_ai_studio]
type = "chat_completion"
model = "google_ai_studio_gemini::gemini-2.0-flash-lite"

[functions.image_test.variants.gcp_vertex]
type = "chat_completion"
model = "gcp-gemini-2.5-pro"

[models."gcp-gemini-2.5-pro"]
routing = ["gcp_vertex_gemini"]

[models."gcp-gemini-2.5-pro".providers.gcp_vertex_gemini]
type = "gcp_vertex_gemini"
model_id = "gemini-2.5-pro"
location = "global"
project_id = "tensorzero-public"

[models."responses-gpt-4o-mini-2024-07-18"]
routing = ["openai"]

[models."responses-gpt-4o-mini-2024-07-18".providers.openai]
type = "openai"
model_name = "gpt-4o-mini-2024-07-18"
api_type = "responses"

[functions.image_test.variants.gcp-vertex-haiku]
type = "chat_completion"
model = "claude-3-haiku-20240307-gcp-vertex"

[models.claude-3-haiku-20240307-gcp-vertex]
routing = ["gcp_vertex_anthropic"]

[models.claude-3-haiku-20240307-gcp-vertex.providers.gcp_vertex_anthropic]
type = "gcp_vertex_anthropic"
model_id = "claude-3-haiku@20240307"
location = "us-central1"
project_id = "tensorzero-public"

[functions.image_test.variants.aws-bedrock]
type = "chat_completion"
model = "claude-3-haiku-20240307-aws-bedrock"

[models.claude-3-haiku-20240307-aws-bedrock]
routing = ["aws_bedrock"]

[models.claude-3-haiku-20240307-aws-bedrock.providers.aws_bedrock]
type = "aws_bedrock"
model_id = "us.anthropic.claude-3-haiku-20240307-v1:0"
region = "us-east-1"
"#;

/// Helper function to get the default credential location for a provider type.
/// This calls the Default implementation directly, ensuring we test the actual defaults.
fn get_default_credential_location(provider_type: &str) -> CredentialLocation {
    match provider_type {
        "anthropic" => AnthropicDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "openai" => OpenAIDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "azure" => AzureDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "deepseek" => DeepSeekDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "fireworks" => FireworksDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "gcp_vertex_anthropic" => GCPDefaults::default()
            .credential_location
            .default_location()
            .clone(),
        "gcp_vertex_gemini" => GCPDefaults::default()
            .credential_location
            .default_location()
            .clone(),
        "google_ai_studio_gemini" => GoogleAIStudioGeminiDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "groq" => GroqDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "hyperbolic" => HyperbolicDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "mistral" => MistralDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "openrouter" => OpenRouterDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "sglang" => SGLangDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "tgi" => TGIDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "together" => TogetherDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "vllm" => VLLMDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        "xai" => XAIDefaults::default()
            .api_key_location
            .default_location()
            .clone(),
        _ => panic!("Unknown provider type: {provider_type}"),
    }
}

fn uses_credential_location(provider_type: &str) -> bool {
    matches!(provider_type, "gcp_vertex_gemini" | "gcp_vertex_anthropic")
}

/// Test that provider type default credentials work correctly.
/// This test:
/// 1. Gets the default credential location from the provider's Default impl
/// 2. Removes the credential from its default location
/// 3. Sets it at a custom location
/// 4. Configures the gateway to use the custom location via provider_types config
/// 5. Verifies inference works
pub async fn test_provider_type_default_credentials_with_provider(provider: E2ETestProvider) {
    // Get the default credential location for this provider
    let default_location = get_default_credential_location(&provider.model_provider_name);

    // Extract the env var name from the credential location
    let (original_env_var, is_path_env) = match &default_location {
        CredentialLocation::Env(var_name) => (var_name.clone(), false),
        CredentialLocation::PathFromEnv(var_name) => (var_name.clone(), true),
        _ => {
            println!(
                "Skipping test for {} - unsupported credential location type",
                provider.model_provider_name
            );
            return;
        }
    };

    // Save the original credential value if it exists
    let original_value = std::env::var(&original_env_var).unwrap();

    // Get the credential value from the provider (it should be set for the test to run)
    let credential_value = original_value.clone();

    // Remove the default env var
    std::env::remove_var(&original_env_var);

    // Set up a custom env var with a test-specific name
    let custom_env_var = format!(
        "TENSORZERO_TEST_{}_KEY",
        provider.model_provider_name.to_uppercase()
    );
    std::env::set_var(&custom_env_var, &credential_value);

    // Create the credential location config based on the type
    let default_credential_location_key = if uses_credential_location(&provider.model_provider_name)
    {
        "credential_location"
    } else {
        "api_key_location"
    };
    let credential_location_config = if is_path_env {
        format!(r#"{default_credential_location_key}= "path_from_env::{custom_env_var}""#)
    } else {
        format!(r#"{default_credential_location_key}= "env::{custom_env_var}""#)
    };

    // Create a config with the custom credential location
    let config = format!(
        r#"
[provider_types.{}]
defaults.{}

[models."test-model"]
routing = ["test-provider"]

[models."test-model".providers.test-provider]
type = "{}"
model_name = "{}"

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "test-model"
"#,
        provider.model_provider_name,
        credential_location_config,
        provider.model_provider_name,
        provider.model_name
    );

    println!(
        "Testing provider type default credentials for {}",
        provider.model_provider_name
    );
    println!("Config:\n{config}");

    // Create an embedded gateway with this config
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(&config).await;

    // Make a simple inference request to verify it works
    let episode_id = Uuid::now_v7();
    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Say hello".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    // Assert the inference succeeded
    assert!(
        result.is_ok(),
        "Inference failed for {}: {:?}",
        provider.model_provider_name,
        result.err()
    );
}

/// Test that provider type default credentials work correctly with shorthand model syntax.
/// This test is similar to test_provider_type_default_credentials_with_provider but uses
/// shorthand model syntax (e.g., "openai::gpt-4o-mini") instead of explicit provider configs.
pub async fn test_provider_type_default_credentials_shorthand_with_provider(
    provider: E2ETestProvider,
) {
    // Get the default credential location for this provider
    let default_location = get_default_credential_location(&provider.model_provider_name);

    // Extract the env var name from the credential location
    let (original_env_var, is_path_env) = match &default_location {
        CredentialLocation::Env(var_name) => (var_name.clone(), false),
        CredentialLocation::PathFromEnv(var_name) => (var_name.clone(), true),
        _ => {
            println!(
                "Skipping test for {} - unsupported credential location type",
                provider.model_provider_name
            );
            return;
        }
    };

    // Save the original credential value if it exists
    let original_value = std::env::var(&original_env_var).unwrap();

    // Get the credential value from the provider (it should be set for the test to run)
    let credential_value = original_value.clone();

    // Remove the default env var
    std::env::remove_var(&original_env_var);

    // Set up a custom env var with a test-specific name
    let custom_env_var = format!(
        "TENSORZERO_TEST_{}_KEY",
        provider.model_provider_name.to_uppercase()
    );
    std::env::set_var(&custom_env_var, &credential_value);

    // Create the credential location config based on the type
    let default_credential_location_key = if uses_credential_location(&provider.model_provider_name)
    {
        "credential_location"
    } else {
        "api_key_location"
    };
    let credential_location_config = if is_path_env {
        format!(r#"{default_credential_location_key}= "path_from_env::{custom_env_var}""#)
    } else {
        format!(r#"{default_credential_location_key}= "env::{custom_env_var}""#)
    };

    // Create a config with the custom credential location and shorthand model syntax
    let config = format!(
        r"
[provider_types.{}]
defaults.{}

",
        provider.model_provider_name, credential_location_config
    );

    println!(
        "Testing provider type default credentials (shorthand) for {}",
        provider.model_provider_name
    );
    println!("Config:\n{config}");

    // Create an embedded gateway with this config
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(&config).await;

    // Make a simple inference request to verify it works
    let episode_id = Uuid::now_v7();
    let result = client
        .inference(ClientInferenceParams {
            function_name: None,
            model_name: Some(provider.model_name),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Say hello".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            ..Default::default()
        })
        .await;

    // Assert the inference succeeded
    assert!(
        result.is_ok(),
        "Inference failed for {} (shorthand): {:?}",
        provider.model_provider_name,
        result.err()
    );
}

/// Test that fallback credentials work correctly.
/// This test:
/// 1. Gets the default credential location from the provider's Default impl
/// 2. Gets the credential value from the env var or path
/// 3. Sets up a provider with a dynamic credential location that falls back to the default location
/// 4. Infers with the dynamic credential
/// 5. Infers with the default credential
/// 6. Asserts that the logs contain exactly one message about falling back
pub async fn test_provider_type_fallback_credentials_with_provider(
    provider: ModelTestProvider,
    supports_dynamic_credentials_test: bool,
    logs_contain: &impl Fn(&str) -> bool,
) {
    reset_capture_logs();
    // Get the default credential location for this provider
    let default_location = get_default_credential_location(&provider.provider_type);

    // Create the credential location config based on the type
    let credential_location_key = if uses_credential_location(&provider.provider_type) {
        "credential_location"
    } else {
        "api_key_location"
    };
    let default_credential_location_str = serde_json::to_string(&default_location).unwrap();
    let credential_location_config = format!(
        r#"{credential_location_key}= {{default = "dynamic::test_credential", fallback = {default_credential_location_str}}}"#
    );

    // Extract the env var name from the credential location
    let original_env_var = match &default_location {
        CredentialLocation::Env(var_name) => var_name.clone(),
        CredentialLocation::PathFromEnv(var_name) => var_name.clone(),
        _ => {
            println!(
                "Skipping test for {} - unsupported credential location type",
                provider.provider_type
            );
            return;
        }
    };

    // Create a config with the custom credential location
    // Build the model_info config lines
    let model_info_config = provider
        .model_info
        .iter()
        .map(|(key, value)| format!("{key} = \"{value}\""))
        .collect::<Vec<_>>()
        .join("\n");

    let config = format!(
        r#"
[models."test-model"]
routing = ["test-provider"]

[models."test-model".providers.test-provider]
{}
type = "{}"
{}

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "test-model"
"#,
        credential_location_config, provider.provider_type, model_info_config
    );

    println!(
        "Testing provider type fallback credentials for {}",
        provider.provider_type
    );
    println!("Config:\n{config}");

    // Create an embedded gateway with this config
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(&config).await;

    // Save the original credential value if it exists
    let original_value = std::env::var(&original_env_var).unwrap();
    let extra_headers = if provider.use_modal_headers {
        get_test_model_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders {
            extra_headers: vec![],
        }
    };

    if supports_dynamic_credentials_test {
        // Make a simple inference request with primary credentials to verify it works
        let episode_id = Uuid::now_v7();
        let result = client
            .inference(ClientInferenceParams {
                function_name: Some("basic_test".to_string()),
                variant_name: Some("default".to_string()),
                episode_id: Some(episode_id),
                input: ClientInput {
                    system: None,
                    messages: vec![ClientInputMessage {
                        role: Role::User,
                        content: vec![ClientInputMessageContent::Text(TextKind::Text {
                            text: "Say hello".to_string(),
                        })],
                    }],
                },
                stream: Some(false),
                credentials: HashMap::from([(
                    "test_credential".to_string(),
                    ClientSecretString(SecretString::new(original_value.clone().into())),
                )]),
                extra_headers: extra_headers.clone(),
                // pass dynamic credentials here
                ..Default::default()
            })
            .await;

        // Assert the inference succeeded
        assert!(
            result.is_ok(),
            "Inference failed for {}: {:?}",
            provider.provider_type,
            result.err()
        );

        assert!(!logs_contain("attempting fallback"));
    }

    // Make a simple inference request without primary credentials to verify it works
    // with a fallback
    let episode_id = Uuid::now_v7();
    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("default".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Say hello".to_string(),
                    })],
                }],
            },
            stream: Some(false),
            credentials: HashMap::from([(
                "test_credentials".to_string(),
                ClientSecretString(SecretString::new(original_value.into())),
            )]),
            extra_headers,
            // pass dynamic credentials here
            ..Default::default()
        })
        .await;

    // Assert the inference succeeded
    assert!(
        result.is_ok(),
        "Inference failed for {}: {:?}",
        provider.provider_type,
        result.err()
    );

    assert!(logs_contain("Using fallback credential"));
    assert!(
        !logs_contain("ERROR"),
        "We should not log an error when credential fallback occurs"
    );
}

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
        gateway.fetch_and_encode_input_files_before_inference = true
        [object_storage]
        type = "filesystem"
        path = "{}"

        {IMAGE_FUNCTION_CONFIG}
        "#,
            temp_dir.path().to_string_lossy()
        ),
    )
    .await;

    // Check that image was stored in filesystem
    let result = std::fs::read(temp_dir.path().join(
        "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
    ))
    .unwrap();
    assert_eq!(result, FERRIS_PNG);
}

async fn check_object_fetch(data: AppStateData, storage_path: &StoragePath, expected_data: &[u8]) {
    check_object_fetch_via_embedded(data.clone(), storage_path, expected_data).await;
    check_object_fetch_via_gateway(storage_path, expected_data).await;
}

async fn check_object_fetch_via_embedded(
    data: AppStateData,
    storage_path: &StoragePath,
    expected_data: &[u8],
) {
    let res = get_object_handler(
        State(data),
        Query(PathParams {
            storage_path: serde_json::to_string(storage_path).unwrap(),
        }),
    )
    .await
    .unwrap();
    assert_eq!(
        res.0,
        ObjectResponse {
            data: BASE64_STANDARD.encode(expected_data),
            reused_object_store: true,
        }
    );
}

async fn check_object_fetch_via_gateway(storage_path: &StoragePath, expected_data: &[u8]) {
    // Try using the running HTTP gateway (which is *not* configured with an object store)
    // to fetch the `StoragePath`
    let client = TensorzeroHttpClient::new_testing().unwrap();
    let res = client
        .get(get_gateway_endpoint(&format!(
            "/internal/object_storage?storage_path={}",
            serde_json::to_string(storage_path).unwrap()
        )))
        .send()
        .await
        .unwrap();

    let response_json = res.json::<Value>().await.unwrap();
    assert_eq!(
        response_json,
        serde_json::json!({
            "data": BASE64_STANDARD.encode(expected_data),
            "reused_object_store": false,
        })
    );
}

/// We already test all of our object store providers with image inputs,
/// so there's no need to re-test them with PDF inputs.
/// All of our PDF-capable providers are tested against the filesystem object store.
pub async fn test_pdf_inference_with_provider_filesystem(provider: E2ETestProvider) {
    let temp_dir = tempfile::tempdir().unwrap();
    println!("Temporary pdf dir: {}", temp_dir.path().to_string_lossy());
    let (client, storage_path) = test_base64_pdf_inference_with_provider_and_store(
        provider,
        &StorageKind::Filesystem {
            path: temp_dir.path().to_string_lossy().to_string(),
        },
        &format!(
            r#"
        [object_storage]
        type = "filesystem"
        path = "{}"

        {PDF_FUNCTION_CONFIG}
        "#,
            temp_dir.path().to_string_lossy()
        ),
        "",
    )
    .await;

    // Check that PDF was stored in filesystem
    let result = std::fs::read(temp_dir.path().join(
        "observability/files/3e127d9a726f6be0fd81d73ccea97d96ec99419f59650e01d49183cd3be999ef.pdf",
    ))
    .unwrap();
    // Don't use assert_eq! because we don't want to print the entire PDF if the check fails
    assert!(
        result == DEEPSEEK_PAPER_PDF,
        "PDF in object store does not match expect pdf"
    );
    check_object_fetch(
        client.get_app_state_data().unwrap().clone(),
        &storage_path,
        DEEPSEEK_PAPER_PDF,
    )
    .await;
}

pub async fn test_image_inference_with_provider_filesystem(provider: E2ETestProvider) {
    let temp_dir = tempfile::tempdir().unwrap();
    println!("Temporary image dir: {}", temp_dir.path().to_string_lossy());
    let (client, storage_path) = test_base64_image_inference_with_provider_and_store(
        provider,
        &StorageKind::Filesystem {
            path: temp_dir.path().to_string_lossy().to_string(),
        },
        &format!(
            r#"
        [object_storage]
        type = "filesystem"
        path = "{}"

        {IMAGE_FUNCTION_CONFIG}
        "#,
            temp_dir.path().to_string_lossy()
        ),
        "",
    )
    .await;

    // Check that image was stored in filesystem
    let result = std::fs::read(temp_dir.path().join(
        "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
    ))
    .unwrap();
    assert_eq!(result, FERRIS_PNG);
    check_object_fetch(
        client.get_app_state_data().unwrap().clone(),
        &storage_path,
        FERRIS_PNG,
    )
    .await;
}

async fn create_s3_object_store(
    bucket_name: Option<String>,
    region: Option<String>,
    endpoint: Option<String>,
    allow_http: Option<bool>,
) -> Result<Arc<dyn ObjectStore>, Box<dyn std::error::Error>> {
    let mut builder = AmazonS3Builder::from_env();

    if let Some(bucket) = bucket_name {
        builder = builder.with_bucket_name(bucket);
    }

    if let Some(region) = region {
        builder = builder.with_region(region);
    }

    if let Some(endpoint) = endpoint {
        builder = builder.with_endpoint(endpoint);
    }

    if let Some(allow_http) = allow_http {
        builder = builder.with_allow_http(allow_http);
    }

    Ok(Arc::new(builder.build()?))
}

pub async fn test_image_inference_with_provider_amazon_s3(provider: E2ETestProvider) {
    let test_bucket = "tensorzero-e2e-test-images";
    let test_bucket_region = "us-east-1";
    let client = create_s3_object_store(
        Some(test_bucket.to_string()),
        Some(test_bucket_region.to_string()),
        None,
        None,
    )
    .await
    .unwrap();

    use rand::distr::Alphanumeric;
    use rand::distr::SampleString;

    let mut prefix = Alphanumeric.sample_string(&mut rand::rng(), 6);
    prefix += "-";

    let (tensorzero_client, expected_key, storage_path) =
        test_image_inference_with_provider_s3_compatible(
            provider,
            &StorageKind::S3Compatible {
                bucket_name: Some(test_bucket.to_string()),
                region: Some("us-east-1".to_string()),
                prefix: prefix.clone(),
                endpoint: None,
                allow_http: None,
            },
            &client,
            &format!(
                r#"
    [object_storage]
    type = "s3_compatible"
    region = "us-east-1"
    bucket_name = "{test_bucket}"
    prefix = "{prefix}"

    {IMAGE_FUNCTION_CONFIG}
    "#
            ),
            &prefix,
        )
        .await;

    check_object_fetch(
        tensorzero_client.get_app_state_data().unwrap().clone(),
        &storage_path,
        FERRIS_PNG,
    )
    .await;

    client
        .delete(&object_store::path::Path::parse(&expected_key).unwrap())
        .await
        .unwrap();
}

pub async fn test_image_inference_with_provider_s3_compatible(
    provider: E2ETestProvider,
    storage_kind: &StorageKind,
    client: &Arc<dyn ObjectStore>,
    toml: &str,
    prefix: &str,
) -> (tensorzero::Client, String, StoragePath) {
    let expected_key = format!(
        "{prefix}observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png"
    );

    // Check that object is deleted
    let path = object_store::path::Path::parse(&expected_key).unwrap();
    let result = client.get(&path).await;
    assert!(
        result.is_err(),
        "Image should not exist in object store after deletion"
    );

    match result {
        Err(object_store::Error::NotFound { .. }) => {
            // Expected - object should not exist
        }
        Err(e) => {
            panic!("Unexpected error: {e}");
        }
        Ok(_) => {
            panic!("Object should not exist after deletion");
        }
    }

    let (tensorzero_client, storage_path) =
        test_base64_image_inference_with_provider_and_store(provider, storage_kind, toml, prefix)
            .await;

    let path = object_store::path::Path::parse(&expected_key).unwrap();
    let result = client
        .get(&path)
        .await
        .expect("Failed to get image from S3-compatible store");

    let bytes = result.bytes().await.unwrap();
    assert_eq!(bytes.to_vec(), FERRIS_PNG);

    (tensorzero_client, expected_key, storage_path)
}

async fn make_temp_image_server() -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {addr}: {e}"));
    let real_addr = listener.local_addr().unwrap();

    async fn get_ferris_png() -> impl IntoResponse {
        Response::builder()
            .header(http::header::CONTENT_TYPE, "image/png")
            .body(Body::from(FERRIS_PNG.to_vec()))
            .unwrap()
    }

    let app = Router::new().route("/ferris.png", get(get_ferris_png));

    let (send, recv) = tokio::sync::oneshot::channel::<()>();
    let shutdown_fut = async move {
        let _ = recv.await;
    };

    // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_fut)
            .into_future(),
    );

    (real_addr, send)
}

pub async fn test_url_image_inference_with_provider_and_store(
    provider: E2ETestProvider,
    kind: StorageKind,
    config_toml: &str,
) {
    let episode_id = Uuid::now_v7();

    // The '_shutdown_sender' will wake up the receiver on drop
    let (server_addr, _shutdown_sender) = make_temp_image_server().await;
    let image_url = Url::parse(&format!("http://{server_addr}/ferris.png")).unwrap();

    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(config_toml).await;

    for should_be_cached in [false, true] {
        let response = client
            .inference(ClientInferenceParams {
                model_name: Some(provider.model_name.clone()),
                episode_id: Some(episode_id),
                input: ClientInput {
                    system: None,
                    messages: vec![ClientInputMessage {
                        role: Role::User,
                        content: vec![
                            ClientInputMessageContent::Text(TextKind::Text {
                                text: "Describe the contents of the image".to_string(),
                            }),
                            ClientInputMessageContent::File(File::Url(UrlFile {
                                url: image_url.clone(),
                                mime_type: None,
                                detail: Some(Detail::Low),
                                filename: None,
                            })),
                        ],
                    }],
                },
                cache_options: CacheParamsOptions {
                    enabled: CacheEnabledMode::On,
                    max_age_s: Some(10),
                },
                extra_headers: if provider.model_provider_name == "vllm"
                    || provider.model_provider_name == "sglang"
                {
                    get_modal_extra_headers()
                } else {
                    UnfilteredInferenceExtraHeaders::default()
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

pub async fn test_base64_pdf_inference_with_provider_and_store(
    provider: E2ETestProvider,
    kind: &StorageKind,
    config_toml: &str,
    prefix: &str,
) -> (tensorzero::Client, StoragePath) {
    let episode_id = Uuid::now_v7();

    let pdf_data = BASE64_STANDARD.encode(DEEPSEEK_PAPER_PDF);

    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(config_toml).await;
    let mut storage_path = None;

    for should_be_cached in [false, true] {
        let response = client
            .inference(ClientInferenceParams {
                function_name: Some("pdf_test".to_string()),
                variant_name: Some(provider.variant_name.clone()),
                episode_id: Some(episode_id),
                input: ClientInput {
                    system: None,
                    messages: vec![ClientInputMessage {
                        role: Role::User,
                        content: vec![
                            ClientInputMessageContent::Text(TextKind::Text {
                                text: "Describe the contents of the PDF".to_string(),
                            }),
                            ClientInputMessageContent::File(File::Base64(
                                Base64File::new(
                                    None,
                                    mime::APPLICATION_PDF,
                                    pdf_data.clone(),
                                    None,
                                    None,
                                )
                                .expect("test data should be valid"),
                            )),
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

        let latest_storage_path = check_base64_pdf_response(
            response,
            Some(episode_id),
            &provider,
            should_be_cached,
            kind,
            prefix,
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        storage_path = Some(latest_storage_path);
    }
    (client, storage_path.unwrap())
}

pub async fn test_base64_image_inference_with_provider_and_store(
    provider: E2ETestProvider,
    kind: &StorageKind,
    config_toml: &str,
    prefix: &str,
) -> (tensorzero::Client, StoragePath) {
    let episode_id = Uuid::now_v7();

    let image_data = BASE64_STANDARD.encode(FERRIS_PNG);

    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(config_toml).await;
    let mut storage_path = None;

    let mut params = ClientInferenceParams {
        function_name: Some("image_test".to_string()),
        variant_name: Some(provider.variant_name.clone()),
        episode_id: Some(episode_id),
        input: ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![
                    ClientInputMessageContent::Text(TextKind::Text {
                        text: "Describe the contents of the image".to_string(),
                    }),
                    ClientInputMessageContent::File(File::Base64(
                        Base64File::new(
                            None,
                            mime::IMAGE_PNG,
                            image_data.clone(),
                            Some(Detail::Low),
                            None,
                        )
                        .expect("test data should be valid"),
                    )),
                ],
            }],
        },
        cache_options: CacheParamsOptions {
            enabled: CacheEnabledMode::On,
            max_age_s: Some(10),
        },
        ..Default::default()
    };

    for should_be_cached in [false, true] {
        let response = client.inference(params.clone()).await.unwrap();

        let InferenceOutput::NonStreaming(response) = response else {
            panic!("Expected non-streaming inference response");
        };

        let latest_storage_path = check_base64_image_response(
            response,
            Some(episode_id),
            &provider,
            should_be_cached,
            kind,
            prefix,
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        storage_path = Some(latest_storage_path);
    }

    let mut image_png = ImageReader::new(Cursor::new(FERRIS_PNG))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();

    // Get 32 random bytes, and write then to the image. This should force a cache miss
    let mut rng = rand::rng();
    let random_bytes: Vec<u8> = (0..32)
        .map(|_| rng.sample(rand::distr::StandardUniform))
        .collect();
    image_png
        .as_mut_rgba8()
        .unwrap()
        .as_flat_samples_mut()
        .samples[0..(random_bytes.len())]
        .copy_from_slice(&random_bytes);

    let mut updated_image = Cursor::new(Vec::new());
    image_png
        .write_to(&mut updated_image, ImageFormat::Png)
        .unwrap();

    let updated_base64 = BASE64_STANDARD.encode(updated_image.into_inner());

    params.input.messages[0].content[1] = ClientInputMessageContent::File(File::Base64(
        Base64File::new(None, mime::IMAGE_PNG, updated_base64, None, None)
            .expect("test data should be valid"),
    ));

    let response = client.inference(params.clone()).await.unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    let inference_id = response.inference_id();

    let clickhouse = get_clickhouse().await;
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    assert_eq!(result["cached"], false);
    // Should be a cache miss since the image data was changed
    (client, storage_path.unwrap())
}

pub async fn test_extra_body_with_provider(provider: E2ETestProvider) {
    test_extra_body_with_provider_and_stream(&provider, false).await;
    test_extra_body_with_provider_and_stream(&provider, true).await;
}

pub async fn test_extra_body_with_provider_and_stream(provider: &E2ETestProvider, stream: bool) {
    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "extra_headers": extra_headers.extra_headers,
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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

pub async fn test_inference_extra_body_with_provider(provider: E2ETestProvider) {
    test_inference_extra_body_with_provider_and_stream(&provider, false).await;
    test_inference_extra_body_with_provider_and_stream(&provider, true).await;
}

pub async fn test_inference_extra_body_with_provider_and_stream(
    provider: &E2ETestProvider,
    stream: bool,
) {
    let episode_id = Uuid::now_v7();
    println!("Provider name: {}", provider.model_provider_name);

    let extra_body = if provider.model_provider_name == "aws_bedrock" {
        json!([
            {
                "variant_name": provider.variant_name,
                "pointer": "/inferenceConfig/temperature",
                "value": 0.5
            },
            {
                "model_name": provider.model_name,
                "provider_name": provider.model_provider_name,
                "pointer": "/inferenceConfig/top_p",
                "value": 0.8
            }
        ])
    } else if provider.model_provider_name == "google_ai_studio_gemini"
        || provider.model_provider_name == "gcp_vertex_gemini"
    {
        json!([
            {
                "variant_name": provider.variant_name,
                "pointer": "/generationConfig/temperature",
                "value": 0.5
            },
            {
                "model_name": provider.model_name,
                "provider_name": provider.model_provider_name,
                "pointer": "/generationConfig/top_p",
                "value": 0.8
            }
        ])
    } else {
        json!([
            {
                "variant_name": provider.variant_name,
                "pointer": "/temperature",
                "value": 0.5
            },
            {
                "model_name": provider.model_name,
                "provider_name": provider.model_provider_name,
                "pointer": "/top_p",
                "value": 0.8
            }
        ])
    };
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "extra_body": extra_body,
        "stream": stream,
        "tags": {"foo": "bar"},
        "extra_headers": extra_headers.extra_headers,
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
        let status = response.status();

        let response_text = response.text().await.unwrap();
        println!("API response text: {response_text}");

        assert_eq!(status, StatusCode::OK);
        let response_json = serde_json::from_str::<Value>(&response_text).unwrap();

        println!("API response: {response_json:#?}");

        let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
        Uuid::parse_str(inference_id).unwrap()
    };

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let chat_result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {chat_result:#?}");

    let id = chat_result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let clickhouse_extra_body = chat_result.get("extra_body").unwrap().as_str().unwrap();
    let clickhouse_extra_body: serde_json::Value =
        serde_json::from_str(clickhouse_extra_body).unwrap();
    // We store the *original* inference-level extra_body in clickhouse, without any filtering
    // This allows us to later re-run the inference with a different variant.
    assert_eq!(extra_body, clickhouse_extra_body);

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
        0.5
    );

    let top_p = if provider.model_provider_name == "aws_bedrock" {
        raw_request_val.get("inferenceConfig").unwrap().get("top_p")
    } else if provider.model_provider_name == "google_ai_studio_gemini"
        || provider.model_provider_name == "gcp_vertex_gemini"
    {
        raw_request_val
            .get("generationConfig")
            .unwrap()
            .get("top_p")
    } else {
        raw_request_val.get("top_p")
    };
    assert_eq!(top_p.unwrap().as_f64().expect("Top P is not a number"), 0.8);
}

pub async fn test_bad_auth_extra_headers_with_provider(provider: E2ETestProvider) {
    test_bad_auth_extra_headers_with_provider_and_stream(&provider, false).await;
    test_bad_auth_extra_headers_with_provider_and_stream(&provider, true).await;
}

pub async fn test_bad_auth_extra_headers_with_provider_and_stream(
    provider: &E2ETestProvider,
    stream: bool,
) {
    // Inject randomness to prevent this from being cached, since provider-proxy will ignore the (invalid) auth header
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": format!("If you see this, something has gone wrong - the request should have failed: {}", Uuid::now_v7())
                }
            ]},
        "stream": stream,
        "extra_headers": extra_headers.extra_headers,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let res = response.json::<Value>().await.unwrap();
    if stream {
        assert!(
            res["error"]
                .as_str()
                .unwrap()
                .contains(format!("from {} server", provider.model_provider_name).as_str()),
            "Missing provider type in error: {res}"
        );
    }
    // The status codes/messages from providers are inconsistent,
    // so we manually check for auth-related strings (where possible)
    match provider.model_provider_name.as_str() {
        "openai" => assert!(
            res["error"]
                .as_str()
                .unwrap()
                .contains("You didn't provide an API key")
                || res["error"].as_str().unwrap().contains("400 Bad Request"),
            "Unexpected error: {res}"
        ),
        "deepseek" => {
            assert!(
                res["error"]
                    .as_str()
                    .unwrap()
                    .contains("Authentication Fails"),
                "Unexpected error: {res}"
            );
        }
        "google_ai_studio_gemini" => {
            // We produce an error by setting a bad 'Content-Length', so just
            // check that an error occurs
            assert!(!res["error"].as_str().unwrap().is_empty());
        }
        "aws_bedrock" => {
            assert!(
                res["error"].as_str().unwrap().contains("Bad Request")
                    || res["error"].as_str().unwrap().contains("ConnectorError"),
                "Unexpected error: {res}"
            );
        }
        "aws_sagemaker" => {
            assert!(
                res["error"]
                    .as_str()
                    .unwrap()
                    .contains("The security token included in the request is invalid"),
                "Unexpected error: {res}"
            );
        }
        "anthropic" => {
            assert!(
                res["error"].as_str().unwrap().contains("invalid x-api-key"),
                "Unexpected error: {res}"
            );
        }
        "azure" => {
            assert!(
                res["error"].as_str().unwrap().contains("Access denied"),
                "Unexpected error: {res}"
            );
        }
        "fireworks" => {
            assert!(
                res["error"]
                    .as_str()
                    .unwrap()
                    .to_lowercase()
                    .contains("unauthorized"),
                "Unexpected error: {res}"
            );
        }
        "gcp_vertex_anthropic" => {
            // We produce an error by setting a bad 'Content-Length', so just
            // check that an error occurs
            assert!(!res["error"].as_str().unwrap().is_empty());
        }
        "groq" => {
            assert!(
                res["error"].as_str().unwrap().contains("400 Bad Request")
                    || res["error"].as_str().unwrap().contains("Invalid API Key"),
                "Unexpected error: {res}"
            );
        }
        "hyperbolic" => {
            assert!(
                res["error"]
                    .as_str()
                    .unwrap()
                    .contains("Could not validate credentials")
                    || res["error"].as_str().unwrap().contains("401 Unauthorized"),
                "Unexpected error: {res}"
            );
        }
        "mistral" => {
            assert!(
                res["error"].as_str().unwrap().contains("Bearer token"),
                "Unexpected error: {res}"
            );
        }
        "openrouter" => {
            assert!(
                res["error"].as_str().unwrap().contains("400 Bad Request")
                    || res["error"].as_str().unwrap().contains("Invalid API Key")
                    || res["error"]
                        .as_str()
                        .unwrap()
                        .contains("No auth credentials found")
                    || res["error"]
                        .as_str()
                        .unwrap()
                        .to_lowercase()
                        .contains("no cookie auth"),
                "Unexpected error: {res}"
            );
        }
        "sglang" | "tgi" => {
            assert!(
                res["error"]
                    .as_str()
                    .is_some_and(|e| e.contains("401 Authorization")),
                "Unexpected error: {res}"
            );
        }
        "together" => {
            assert!(
                res["error"].as_str().unwrap().contains("Invalid API key"),
                "Unexpected error: {res}"
            );
        }
        "vllm" => {
            // vLLM returns different errors if you mess with the request headers,
            // so we just check that an error occurs
            assert!(res["error"].as_str().is_some(), "Unexpected error: {res}");
        }
        "xai" => {
            assert!(
                res["error"].as_str().unwrap().contains("Incorrect"),
                "Unexpected error: {res}"
            );
        }
        "gcp_vertex_gemini" => {
            // We produce an error by setting a bad 'Content-Length', so just
            // check that an error occurs
            assert!(!res["error"].as_str().unwrap().is_empty());
        }
        _ => {
            panic!("Got error: {res}");
        }
    }

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
}
pub async fn test_warn_ignored_thought_block_with_provider(
    provider: E2ETestProvider,
    logs_contain: &impl Fn(&str) -> bool,
) {
    reset_capture_logs();
    // Bedrock rejects input thoughts for these models
    if provider.model_name == "claude-3-haiku-20240307-aws-bedrock"
        || provider.model_name == "deepseek-r1-aws-bedrock"
    {
        return;
    }

    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let res = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some(provider.variant_name.clone()),
            input: ClientInput {
                system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                    "assistant_name".to_string(),
                    "Dr. Mehta".into(),
                )])))),
                messages: vec![
                    ClientInputMessage {
                        role: Role::Assistant,
                        content: vec![ClientInputMessageContent::Thought(Thought {
                            text: Some("My TensorZero thought".to_string()),
                            signature: Some("My new TensorZero signature".to_string()),
                            summary: None,
                            provider_type: None,
                        })],
                    },
                    ClientInputMessage {
                        role: Role::User,
                        content: vec![ClientInputMessageContent::Text(TextKind::Text {
                            text: "What is the name of the capital city of Japan?".to_string(),
                        })],
                    },
                ],
            },
            extra_headers: if provider.model_provider_name == "vllm"
                || provider.model_provider_name == "sglang"
            {
                get_modal_extra_headers()
            } else {
                UnfilteredInferenceExtraHeaders::default()
            },
            ..Default::default()
        })
        .await;

    if provider.model_provider_name.as_str() == "anthropic"
        || provider.model_provider_name.as_str() == "gcp_vertex_anthropic"
    {
        // Anthropic rejects requests with invalid thought signatures
        let err = res.unwrap_err();
        assert!(err.to_string().contains("signature"));
    } else if provider.variant_name.as_str() == "openai-responses" {
        // OpenAI Responses rejects requests with invalid thought signatures
        let err = res.unwrap_err();
        assert!(err.to_string().contains("signature"));
    } else {
        let _ = res.unwrap();
    }

    if ["anthropic", "aws-bedrock", "gcp_vertex_anthropic"]
        .contains(&provider.model_provider_name.as_str())
        || ["openai-responses"].contains(&provider.variant_name.as_str())
    {
        assert!(
            !logs_contain("TensorZero doesn't support input thought blocks for the"),
            "Should not have warned about dropping thought blocks"
        );
    } else {
        assert!(
            logs_contain("TensorZero doesn't support input thought blocks for the"),
            "Missing expected warning for model_provider {} variant {}",
            provider.model_provider_name,
            provider.variant_name
        );
    }
}

pub async fn test_simple_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "extra_headers": extra_headers.extra_headers,
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
        "cache_options": {"enabled": "on", "lookback_s": 10},
        "extra_headers": extra_headers.extra_headers,
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

pub async fn check_base64_pdf_response(
    response: InferenceResponse,
    episode_id: Option<Uuid>,
    provider: &E2ETestProvider,
    should_be_cached: bool,
    kind: &StorageKind,
    prefix: &str,
) -> StoragePath {
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
        content.to_lowercase().contains("deepseek"),
        "Content should contain 'deepseek': {content}"
    );

    let usage = response.usage;
    let input_tokens = usage.input_tokens;
    let output_tokens = usage.output_tokens;
    if should_be_cached {
        assert_eq!(input_tokens, Some(0));
        assert_eq!(output_tokens, Some(0));
    } else {
        assert!(input_tokens.unwrap() > 0);
        assert!(output_tokens.unwrap() > 0);
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
    assert_eq!(function_name, "pdf_test");

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
                    {"type": "text", "text": "Describe the contents of the PDF"},
                    {
                        "type": "file",
                        "source_url": null,
                        "mime_type": "application/pdf",
                        "storage_path": {
                            "kind": kind_json,
                            "path": format!("{prefix}observability/files/3e127d9a726f6be0fd81d73ccea97d96ec99419f59650e01d49183cd3be999ef.pdf"),
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

    let expected_storage_path = StoragePath {
        kind: kind.clone(),
        path: Path::parse(format!("{prefix}observability/files/3e127d9a726f6be0fd81d73ccea97d96ec99419f59650e01d49183cd3be999ef.pdf")).unwrap(),
    };

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(
        input_messages,
        vec![StoredRequestMessage {
            role: Role::User,
            content: vec![
                StoredContentBlock::Text(Text {
                    text: "Describe the contents of the PDF".to_string(),
                }),
                StoredContentBlock::File(Box::new(StoredFile(ObjectStoragePointer {
                    source_url: None,
                    mime_type: mime::APPLICATION_PDF,
                    storage_path: expected_storage_path.clone(),
                    detail: None,
                    filename: None,
                },)))
            ]
        },]
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
        raw_request.contains("<TENSORZERO_FILE_0>"),
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
    expected_storage_path
}

pub async fn check_base64_image_response(
    response: InferenceResponse,
    episode_id: Option<Uuid>,
    provider: &E2ETestProvider,
    should_be_cached: bool,
    kind: &StorageKind,
    prefix: &str,
) -> StoragePath {
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
        assert_eq!(input_tokens, Some(0));
        assert_eq!(output_tokens, Some(0));
    } else {
        assert!(input_tokens.unwrap() > 0);
        assert!(output_tokens.unwrap() > 0);
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
    assert_eq!(function_name, "image_test");

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
                    {"type": "text", "text": "Describe the contents of the image"},
                    {
                        "type": "file",
                        "source_url": null,
                        "mime_type": "image/png",
                        "storage_path": {
                            "kind": kind_json,
                            "path": format!("{prefix}observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png"),
                        },
                        "detail": "low"
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

    let expected_storage_path = StoragePath {
        kind: kind.clone(),
        path: Path::parse(format!("{prefix}observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png")).unwrap(),
    };

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(
        input_messages,
        vec![StoredRequestMessage {
            role: Role::User,
            content: vec![
                StoredContentBlock::Text(Text {
                    text: "Describe the contents of the image".to_string(),
                }),
                StoredContentBlock::File(Box::new(StoredFile(ObjectStoragePointer {
                    source_url: None,
                    mime_type: mime::IMAGE_PNG,
                    storage_path: expected_storage_path.clone(),
                    detail: Some(Detail::Low),
                    filename: None,
                },)))
            ]
        },]
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
        raw_request.contains("<TENSORZERO_FILE_0>"),
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
    expected_storage_path
}

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
        assert_eq!(input_tokens, Some(0));
        assert_eq!(output_tokens, Some(0));
    } else {
        assert!(input_tokens.unwrap() > 0);
        assert!(output_tokens.unwrap() > 0);
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
                    {"type": "text", "text": "Describe the contents of the image"},
                    {
                        "type": "file",
                        "source_url": image_url.to_string(),
                        "mime_type": "image/png",
                        "storage_path": {
                            "kind": kind_json,
                            "path": "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png"
                        },
                        "detail": "low"
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(
        input_messages,
        vec![
            StoredRequestMessage {
                role: Role::User,
                content: vec![StoredContentBlock::Text(Text {
                    text: "Describe the contents of the image".to_string(),
                }), StoredContentBlock::File(Box::new(StoredFile(
                    ObjectStoragePointer {
                        source_url: Some(image_url.clone()),
                        mime_type: mime::IMAGE_PNG,
                        storage_path: StoragePath {
                            kind: kind.clone(),
                            path: Path::parse("observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png").unwrap(),
                        },
                        detail: Some(Detail::Low),
                        filename: None,
                    },
                )))]
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
        raw_request.contains("<TENSORZERO_FILE_0>"),
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

    let mut content = response_json
        .get("content")
        .unwrap()
        .as_array()
        .unwrap()
        .clone();
    // Some providers always produce thought blocks - we don't care about them in this test
    content.retain(|c| c.get("type").unwrap().as_str().unwrap() != "thought");
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
    if provider.variant_name != "openai-responses" {
        let finish_reason = response_json
            .get("finish_reason")
            .unwrap()
            .as_str()
            .unwrap();
        // Some providers return "stop" and others return "length"
        assert!(finish_reason == "stop" || finish_reason == "length");
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                "content": [{"type": "text", "text": "What is the name of the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let mut content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    // Some providers always produce thought blocks - we don't care about them in this test
    content_blocks.retain(|c| c.get("type").unwrap().as_str().unwrap() != "thought");
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);

    let tags = result.get("tags").unwrap().as_object().unwrap();
    if !is_batch {
        assert_eq!(tags.get("foo").unwrap().as_str().unwrap(), "bar");
    }
    // Since the variant was pinned, the variant_pinned tag should be present
    assert_eq!(
        tags.get("tensorzero::variant_pinned")
            .unwrap()
            .as_str()
            .unwrap(),
        provider.variant_name
    );

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    assert!(tool_params.is_empty());

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());

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
    assert!(input_tokens.as_u64().unwrap() > 0);
    assert!(output_tokens.as_u64().unwrap() > 0);
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is the name of the capital city of Japan?".to_string(),
        })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let mut output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    // Some providers always produce thought blocks - we don't care about them in this test
    output.retain(|c| !matches!(c, StoredContentBlock::Thought(_)));
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

pub async fn check_simple_image_inference_response(
    response_json: Value,
    episode_id: Option<Uuid>,
    provider: &E2ETestProvider,
    is_batch: bool,
    should_be_cached: bool,
) {
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
    assert!(content.to_lowercase().contains("crab"));

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
    let finish_reason = response_json
        .get("finish_reason")
        .unwrap()
        .as_str()
        .unwrap();
    // Some providers return "stop" and others return "length"
    assert!(finish_reason == "stop" || finish_reason == "length");

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
    assert_eq!(function_name, "basic_test");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    if let Some(episode_id) = episode_id {
        assert_eq!(retrieved_episode_id, episode_id);
    }
    let tags = result.get("tags").unwrap();
    // All callers set this tag so this tests that tags are propagated to the ultimate sink of the inference data
    tags.get("test_type").unwrap();

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What kind of animal is in this image?"},
                    {
                        "type": "file",
                        "source_url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
                        "mime_type": "image/png",
                        "storage_path": {
                            "kind": {"type": "disabled"},
                            "path": "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png"
                        }
                    }
                ]
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
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());

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
    assert!(raw_request.to_lowercase().contains("animal"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("crab"));
    assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();
    assert!(input_tokens.as_u64().unwrap() > 0);
    assert!(output_tokens.as_u64().unwrap() > 0);
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
    let output = result.get("output").unwrap().as_str().unwrap();
    assert!(
        output.to_lowercase().contains("crab"),
        "Unexpected output: {output}",
    );
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);

    if !is_batch {
        // Check the InferenceTag Table
        let result =
            select_inference_tags_clickhouse(&clickhouse, "basic_test", "foo", "bar", inference_id)
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

pub async fn test_streaming_invalid_request_with_provider(provider: E2ETestProvider) {
    // A top_p of -100 and temperature of -100 should produce errors on all providers
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "params": {
            "chat_completion": {
                "temperature": -100,
                "top_p": -100,
            }
        },
        "input":
            {
               "system": {"assistant_name": format!("Dr. Mehta")},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "stream": true,
        "extra_body": [
            {
                "variant_name": "aws-sagemaker-openai",
                "pointer": "/messages/0/content",
                "value": 123,
            },
        ],
        "extra_headers": extra_headers.extra_headers,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    while let Some(event) = event_source.next().await {
        if let Ok(reqwest_eventsource::Event::Open) = event {
            continue;
        }
        let err = event.unwrap_err();
        let reqwest_eventsource::Error::InvalidStatusCode(code, resp) = err else {
            panic!("Unexpected error: {err:?}")
        };
        assert_eq!(code, StatusCode::INTERNAL_SERVER_ERROR);
        let resp: Value = resp.json().await.unwrap();
        let err_msg = resp.get("error").unwrap().as_str().unwrap();
        assert!(
            err_msg.contains("top_p")
                || err_msg.contains("topP")
                || err_msg.contains("temperature"),
            "Unexpected error message: {resp}"
        );
    }
}

pub async fn test_simple_streaming_inference_request_with_provider(provider: E2ETestProvider) {
    // We use a serverless Sagemaker endpoint, which doesn't support streaming
    if provider.variant_name == "aws-sagemaker-tgi" {
        return;
    }
    let episode_id = Uuid::now_v7();
    let tag_value = Uuid::now_v7().to_string();
    // Generate random u32
    let seed = rand::rng().random_range(0..u32::MAX);

    let original_content = test_simple_streaming_inference_request_with_provider_cache(
        &provider, episode_id, seed, &tag_value, false, false,
    )
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let cached_content = test_simple_streaming_inference_request_with_provider_cache(
        &provider, episode_id, seed, &tag_value, true, false,
    )
    .await;
    assert_eq!(original_content, cached_content);
}

pub async fn test_streaming_include_original_response_with_provider(provider: E2ETestProvider) {
    // We use a serverless Sagemaker endpoint, which doesn't support streaming
    if provider.variant_name == "aws-sagemaker-tgi" {
        return;
    }
    let episode_id = Uuid::now_v7();
    let tag_value = Uuid::now_v7().to_string();
    // Generate random u32
    let seed = rand::rng().random_range(0..u32::MAX);

    let original_content = test_simple_streaming_inference_request_with_provider_cache(
        &provider, episode_id, seed, &tag_value, false, true,
    )
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let cached_content = test_simple_streaming_inference_request_with_provider_cache(
        &provider, episode_id, seed, &tag_value, true, true,
    )
    .await;
    assert_eq!(original_content, cached_content);
}

pub async fn test_simple_streaming_inference_request_with_provider_cache(
    provider: &E2ETestProvider,
    episode_id: Uuid,
    seed: u32,
    tag_value: &str,
    check_cache: bool,
    include_original_response: bool,
) -> String {
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "include_original_response": include_original_response,
        "tags": {"key": tag_value},
        "cache_options": {"enabled": "on", "lookback_s": 10},
        "extra_headers": extra_headers.extra_headers,
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
    let mut finish_reason: Option<String> = None;
    for (i, chunk) in chunks.clone().iter().enumerate() {
        let chunk_json: Value = serde_json::from_str(chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        // The `original_chunk` field should only be set if we enable the `include_original_response` flag
        if include_original_response {
            // The last chunk might be a usage chunk generated by TensorZero
            // (if the original stream didn't report usage), so don't check it
            if i != chunks.len() - 1 {
                assert!(chunk_json.get("original_chunk").is_some());
            }
        } else {
            assert_eq!(chunk_json.get("original_chunk"), None);
        }

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
            for block in content_blocks {
                if block["type"] == "text" {
                    let content = block.get("text").unwrap().as_str().unwrap();
                    full_content.push_str(content);
                }
            }
        }

        // When we get a cache hit, the usage should be explicitly set to 0
        if check_cache {
            let usage = chunk_json.get("usage").unwrap();
            assert_eq!(usage.get("input_tokens").unwrap().as_u64().unwrap(), 0);
            assert_eq!(usage.get("output_tokens").unwrap().as_u64().unwrap(), 0);
        }

        if let Some(usage) = chunk_json.get("usage") {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }

        if let Some(chunk_finish_reason) = chunk_json.get("finish_reason") {
            assert!(finish_reason.is_none());
            finish_reason = Some(chunk_finish_reason.as_str().unwrap().to_string());
        }
    }

    let inference_id = inference_id.unwrap();
    assert!(full_content.to_lowercase().contains("tokyo"));

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if (provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry"))
        || check_cache
    {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    assert!(finish_reason.is_some());

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                "content": [{"type": "text", "text": "What is the name of the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output = result.get("output").unwrap().as_str().unwrap();
    let mut output: Vec<Value> = serde_json::from_str(output).unwrap();
    // Some providers always produce thought blocks - we don't care about them in this test
    output.retain(|c| c.get("type").unwrap().as_str().unwrap() != "thought");
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
    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    let tags = result.get("tags").unwrap().as_object().unwrap();
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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is the name of the capital city of Japan?".to_string(),
        })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let mut output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    // Some providers always produce thought blocks - we don't care about them in this test
    output.retain(|c| !matches!(c, StoredContentBlock::Thought(_)));
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

pub async fn test_inference_params_inference_request_with_provider(provider: E2ETestProvider) {
    // Gemini 2.5 Pro gives us 'Penalty is not enabled for models/gemini-2.5-pro'
    if provider.model_name.contains("gemini-2.5-pro") {
        return;
    }
    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "extra_headers": extra_headers.extra_headers,
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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

    if is_batch {
        assert!(result.get("processing_time_ms").unwrap().is_null());
    } else {
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

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if is_batch {
        assert!(result.get("response_time_ms").unwrap().is_null());
        assert!(result.get("ttft_ms").unwrap().is_null());
    } else {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is the name of the capital city of Japan?".to_string(),
        })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
}

pub async fn test_inference_params_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Gemini 2.5 Pro gives us 'Penalty is not enabled for models/gemini-2.5-pro'
    if provider.model_name.contains("gemini-2.5-pro") {
        return;
    }
    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "extra_headers": extra_headers.extra_headers,
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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                "content": [{"type": "text", "text": "What is the name of the capital city of Japan?"}]
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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is the name of the capital city of Japan?".to_string(),
        })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
}

pub async fn test_tool_use_tool_choice_auto_used_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "tags": {"test_type": "auto_used"},
        "extra_headers": extra_headers.extra_headers,
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

// Helper function to verify new Migration 0041 tool call storage columns
fn verify_tool_call_storage_columns(
    result: &Value,
    expected_tool_choice: &str,
    expected_parallel_tool_calls: Option<bool>,
    expected_allowed_tools_choice: &str,
    expected_static_tool_names: &[&str],
    expected_dynamic_tool_count: usize,
    expected_provider_tool_count: usize,
) {
    // Verify allowed_tools column
    let allowed_tools_str = result.get("allowed_tools").unwrap().as_str().unwrap();
    let allowed_tools: Value = serde_json::from_str(allowed_tools_str).unwrap();
    assert_eq!(
        allowed_tools["choice"], expected_allowed_tools_choice,
        "allowed_tools.choice mismatch"
    );

    let actual_tools = allowed_tools["tools"].as_array().unwrap();
    assert_eq!(
        actual_tools.len(),
        expected_static_tool_names.len(),
        "allowed_tools.tools length mismatch"
    );
    for expected_tool in expected_static_tool_names {
        assert!(
            actual_tools.contains(&json!(expected_tool)),
            "allowed_tools.tools missing tool: {expected_tool}"
        );
    }

    // Verify dynamic_tools column
    let dynamic_tools_unparsed = result.get("dynamic_tools").unwrap().as_array().unwrap();
    let dynamic_tools: Result<Vec<Tool>, _> = dynamic_tools_unparsed
        .iter()
        .map(|x| serde_json::from_str::<Tool>(x.as_str().unwrap()))
        .collect();
    let dynamic_tools = dynamic_tools.unwrap();
    assert_eq!(
        dynamic_tools.len(),
        expected_dynamic_tool_count,
        "dynamic_tools length mismatch"
    );

    // Verify dynamic_provider_tools column
    let dynamic_provider_tools_unparsed = result
        .get("dynamic_provider_tools")
        .unwrap()
        .as_array()
        .unwrap();
    let dynamic_provider_tools: Result<Vec<Tool>, _> = dynamic_provider_tools_unparsed
        .iter()
        .map(|x| serde_json::from_str::<Tool>(x.as_str().unwrap()))
        .collect();
    let dynamic_provider_tools = dynamic_provider_tools.unwrap();
    assert_eq!(
        dynamic_provider_tools.len(),
        expected_provider_tool_count,
        "dynamic_provider_tools length mismatch"
    );

    // Verify tool_choice column
    let tool_choice = result.get("tool_choice").unwrap().as_str().unwrap();
    assert_eq!(tool_choice, expected_tool_choice, "tool_choice mismatch");

    // Verify parallel_tool_calls column
    let parallel_tool_calls = result.get("parallel_tool_calls");
    match expected_parallel_tool_calls {
        Some(expected) => {
            let actual = parallel_tool_calls
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            assert_eq!(actual, expected, "parallel_tool_calls mismatch");
        }
        None => {
            // Should be null or not present
            assert!(
                parallel_tool_calls.is_none() || parallel_tool_calls.unwrap().is_null(),
                "parallel_tool_calls should be null"
            );
        }
    }
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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

    // Verify new Migration 0041 columns (decomposed tool call storage format)
    verify_tool_call_storage_columns(
        &result,
        "auto",               // expected_tool_choice
        None,                 // expected_parallel_tool_calls (null/none)
        "function_default",   // expected_allowed_tools_choice (tools from function config)
        &["get_temperature"], // expected_static_tool_names
        0,                    // expected_dynamic_tool_count
        0,                    // expected_provider_tool_count
    );

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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![
            "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, StoredContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        StoredContentBlock::ToolCall(tool_call) => {
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

pub async fn test_tool_use_tool_choice_auto_used_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Together doesn't correctly produce streaming tool call chunks (it produces text chunks with the raw tool call).
    if provider.model_provider_name == "together" {
        return;
    }

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

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
        "extra_headers": extra_headers
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
    let mut tool_name = None;

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

        let blocks = chunk_json.get("content").unwrap().as_array().unwrap();
        for block in blocks {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    if let Some(block_raw_name) = block.get("raw_name") {
                        match tool_name {
                            Some(_) => {
                                assert!(
                                    block_raw_name.as_str().unwrap().is_empty(),
                                    "Raw name already seen, got {block:#?}"
                                );
                            }
                            None => {
                                tool_name = Some(block_raw_name.as_str().unwrap().to_string());
                            }
                        }
                    }

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
                "thought" => {
                    // Gemini models can return thoughts - ignore them
                }
                _ => {
                    panic!("Unexpected block type: {block_type}");
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    assert!(tool_name.is_some());
    assert_eq!(tool_name.as_ref().unwrap(), "get_temperature");

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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
                                            // Ignore other content blocks
    let content_block = output_clickhouse
        .iter()
        .find(|b| b["type"] == "tool_call")
        .unwrap();
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![
            "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, StoredContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        StoredContentBlock::ToolCall(tool_call) => {
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
pub async fn test_tool_use_tool_choice_auto_unused_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "extra_headers": extra_headers.extra_headers,
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
    let content = response_json.get("content").unwrap().as_array().unwrap();

    if let Some(episode_id) = episode_id {
        let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
        let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
        assert_eq!(episode_id_response, episode_id);
    }

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What is your name?"}]
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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
    if is_batch {
        assert!(result.get("processing_time_ms").unwrap().is_null());
    } else {
        let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
        assert!(processing_time_ms > 0);
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
    if is_batch {
        assert!(result.get("response_time_ms").unwrap().is_null());
        assert!(result.get("ttft_ms").unwrap().is_null());
    } else {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("mehta"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    if is_batch {
        assert!(result.get("response_time_ms").unwrap().is_null());
        assert!(result.get("ttft_ms").unwrap().is_null());
    } else {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is your name?".to_string(),
        })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let mut output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    // We don't care about thoughts in this test
    output.retain(|block| !matches!(block, StoredContentBlock::Thought(_)));

    let first = output.first().unwrap();
    match first {
        StoredContentBlock::Text(_text) => {}
        _ => {
            panic!("Expected a text block, got {first:?}");
        }
    }

    assert!(!content.iter().any(|block| block["type"] == "tool_call"));
    let content_block = content
        .iter()
        .find(|block| block["type"] == "text")
        .unwrap();
    let content_block_text = content_block.get("text").unwrap().as_str().unwrap();
    assert!(content_block_text.to_lowercase().contains("mehta"));
}

/// This test is similar to `test_tool_use_tool_choice_auto_used_streaming_inference_request_with_provider`, but it steers the model to not use the tool.
/// This ensures that ToolChoice::Auto is working as expected.
pub async fn test_tool_use_tool_choice_auto_unused_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Together doesn't correctly produce streaming tool call chunks (it produces text chunks with the raw tool call).
    if provider.model_provider_name == "together" {
        return;
    }

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "extra_headers": extra_headers.extra_headers,
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
                "thought" => {
                    // Gemini models can return thoughts - ignore them
                }
                _ => {
                    panic!("Unexpected block type: {block_type}");
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What is your name?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let mut output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();

    // We don't care about thoughts in this test
    output_clickhouse.retain(|block| block["type"] != "thought");

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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is your name?".to_string(),
        })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let mut output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    // We don't care about thoughts in this test
    output.retain(|block| !matches!(block, StoredContentBlock::Thought(_)));

    let first = output.first().unwrap();
    match first {
        StoredContentBlock::Text(_text) => {}
        _ => {
            panic!("Expected a text block, got {first:?}");
        }
    }
}

pub async fn test_tool_use_tool_choice_required_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Azure, Together, and SGLang don't support `tool_choice: "required"`
    // Groq says they support it, but it doesn't return the required tool as expected
    if provider.model_provider_name == "azure"
        || provider.model_provider_name == "together"
        || provider.model_provider_name == "sglang"
        || provider.model_provider_name == "groq"
    {
        return;
    }

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "extra_headers": extra_headers.extra_headers,
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
    // OpenAI occasionally emits a tool call with an empty object for `arguments`
    assert!(raw_arguments.len() <= 2);
    if let Some(location) = raw_arguments.get("location") {
        assert!(location.as_str().is_some());
    }
    if raw_arguments.len() == 2 {
        let units = raw_arguments.get("units").unwrap().as_str().unwrap();
        assert!(units == "celsius" || units == "fahrenheit");
    }

    if let Some(arguments) = content_block["arguments"].as_object() {
        // OpenAI occasionally emits a tool call with an empty object for `arguments`
        assert!(arguments.len() <= 2);
        if let Some(location) = arguments.get("location") {
            assert!(location.as_str().is_some());
        }
        if arguments.len() == 2 {
            let units = arguments.get("units").unwrap().as_str().unwrap();
            assert!(units == "celsius" || units == "fahrenheit");
        }
    }

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What is your name?"}]
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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
    if is_batch {
        assert!(result.get("response_time_ms").unwrap().is_null());
        assert!(result.get("ttft_ms").unwrap().is_null());
    } else {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is your name?".to_string(),
        })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, StoredContentBlock::ToolCall(_)))
        .collect();

    // Assert at least one tool call
    assert!(
        !tool_call_blocks.is_empty(),
        "Expected at least one tool call"
    );

    for tool_call_block in tool_call_blocks {
        match tool_call_block {
            StoredContentBlock::ToolCall(tool_call) => {
                assert_eq!(tool_call.name, "get_temperature");
                serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
            }
            _ => panic!("Unreachable"),
        }
    }
}

pub async fn test_tool_use_tool_choice_required_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Azure, Together, and SGLang don't support `tool_choice: "required"`
    // Groq says they support it, but it doesn't return the required tool as expected
    // Fireworks returns trash.
    if provider.model_provider_name == "azure"
        || provider.model_provider_name == "together"
        || provider.model_provider_name == "sglang"
        || provider.model_provider_name == "groq"
        || provider.model_provider_name == "fireworks"
    {
        return;
    }

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "extra_headers": extra_headers.extra_headers,
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
    let mut tool_name = None;

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
                    if let Some(block_raw_name) = block.get("raw_name") {
                        match tool_name {
                            Some(_) => {
                                assert!(
                                    block_raw_name.as_str().unwrap().is_empty(),
                                    "Raw name already seen, got {block:#?}"
                                );
                            }
                            None => {
                                tool_name = Some(block_raw_name.as_str().unwrap().to_string());
                            }
                        }
                    }

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
                "thought" => {
                    // Gemini models can return thoughts - ignore them
                }
                _ => {
                    panic!("Unexpected block type: {block_type}");
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    assert!(tool_name.is_some());
    assert_eq!(tool_name.as_ref().unwrap(), "get_temperature");

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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What is your name?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
                                            // Ignore other content blocks
    let content_block = output_clickhouse
        .iter()
        .find(|b| b["type"] == "tool_call")
        .unwrap();
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is your name?".to_string(),
        })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, StoredContentBlock::ToolCall(_)))
        .collect();

    // Assert at least one tool call
    assert!(
        !tool_call_blocks.is_empty(),
        "Expected at least one tool call in {output:?}"
    );

    for tool_call_block in tool_call_blocks {
        match tool_call_block {
            StoredContentBlock::ToolCall(tool_call) => {
                assert_eq!(tool_call.name, "get_temperature");
                serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
            }
            _ => panic!("Unreachable"),
        }
    }
}

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

    // NOTE - Gemini 2.5 produces 'UNEXPECTED_TOOL_CALL' here
    // See https://github.com/tensorzero/tensorzero/issues/2329
    if provider.model_name == "gcp-gemini-2.5-pro" {
        return;
    }

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "extra_headers": extra_headers.extra_headers,
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

// Test that the model doesn't emit a tool call when tool_choice is none
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
        // Gemini 2.5 Pro will sometimes emit 'executableCode' blocks, which we turn into 'unknown' blocks
        .find(|block| block["type"] == "text" || block["type"] == "unknown")
        .unwrap();
    if content_block["type"] == "unknown" {
        assert!(content_block.get("data").is_some());
    } else {
        assert!(content_block.get("text").unwrap().as_str().is_some());
    }

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![
            "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    let first = output.first().unwrap();
    match first {
        StoredContentBlock::Text(_) | StoredContentBlock::Unknown(_) => {}
        _ => {
            panic!("Expected a text or unknown block, got {first:?}");
        }
    }
}

pub async fn test_tool_use_tool_choice_none_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Gemini 2.5 Pro will produce 'executableCode' blocks for this test, which we don't support
    // in streaming mode (since we don't have "unknown" streaming chunks)
    if provider.model_name.contains("gemini-2.5-pro") {
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
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

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
        "extra_headers": extra_headers.extra_headers,
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
                "thought" => {
                    // Gemini models can return thoughts - ignore them
                }
                _ => {
                    panic!("Unexpected block type: {block_type}");
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![
            "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    let first = output.first().unwrap();
    match first {
        StoredContentBlock::Text(_text) => {}
        _ => {
            panic!("Expected a text block, got {first:?}");
        }
    }
}

pub async fn test_tool_use_tool_choice_specific_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // GCP Vertex AI, Groq, Mistral, and Together don't support ToolChoice::Specific.
    // (Together AI claims to support it, but we can't get it to behave strictly.)
    // In those cases, we use ToolChoice::Any with a single tool under the hood.
    // Even then, they seem to hallucinate a new tool.
    if provider.model_provider_name.contains("gcp_vertex")
        || provider.model_provider_name == "groq"
        || provider.model_provider_name == "mistral"
        || provider.model_provider_name == "together"
    {
        return;
    }

    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

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
        "extra_headers": extra_headers.extra_headers,
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
    let name = content_block.get("name").unwrap().as_str().unwrap();
    // We explicitly do not check the tool name, as xAI decides to call 'get_temperature'
    // instead of 'self_destruct'
    assert_eq!(name, raw_name);

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();

    assert_eq!(arguments, raw_arguments);

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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

    // We explicitly do *not* check `raw_response`, as model providers differ in whether or
    //not they actually call `self_destruct` (OpenAI will, but xAI does not).

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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![
            "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, StoredContentBlock::ToolCall(_)))
        .collect();

    // Assert at most one tool call (a model could decide to call no tools if to reads the `self_destruct` description).
    // Sglang likes to emit lots of tool calls
    if provider.model_provider_name != "sglang" {
        assert!(
            tool_call_blocks.len() <= 1,
            "Expected at most one tool call, found {}",
            tool_call_blocks.len()
        );
    }

    let tool_call_block = tool_call_blocks.first();
    match tool_call_block {
        Some(StoredContentBlock::ToolCall(tool_call)) => {
            // Don't check which tool was called, as xAI can sometimes call a tool other than `self_destruct`.
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        None => {}
        _ => panic!("Unreachable"),
    }
}

pub async fn test_tool_use_tool_choice_specific_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // - GCP Vertex AI, Mistral, Together and Groq don't support ToolChoice::Specific.
    // (Together AI claims to support it, but we can't get it to behave strictly.)
    // In those cases, we use ToolChoice::Any with a single tool under the hood.
    // Even then, they seem to hallucinate a new tool.
    if provider.model_provider_name.contains("gcp_vertex")
        || provider.model_provider_name == "mistral"
        || provider.model_provider_name == "together"
        || provider.model_provider_name == "groq"
    {
        return;
    }

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

    // Groq tool requests aren't always sent with the correct schema
    let prompt: String = if provider.model_provider_name == "groq" {
        "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool. Ensure that the request to the tool is sent with the correct json schema.".into()
    } else {
        "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool.".into()
    };

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
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
        "extra_headers": extra_headers.extra_headers,
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
                    // We explicitly do not check the tool name, as xAI decides to call 'get_temperature'
                    // instead of 'self_destruct'
                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    match &tool_id {
                        None => tool_id = Some(block_tool_id.to_string()),
                        Some(tool_id) => {
                            if provider.model_provider_name == "sglang" {
                                // Sglang likes to emit lots of duplicate tool calls
                                if tool_id != block_tool_id {
                                    continue;
                                }
                            } else {
                                assert_eq!(
                                    tool_id, block_tool_id,
                                    "Provider returned multiple tool calls"
                                );
                            }
                        }
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();
                    arguments.push_str(chunk_arguments);
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                "thought" => {
                    // Gemini models can return thoughts - ignore them
                }
                _ => {
                    panic!("Unexpected block type: {block_type}");
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
        }
    );

    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
    let content_block = output_clickhouse
        .iter()
        .find(|block| block.get("type").and_then(Value::as_str) == Some("tool_call"))
        .expect("No tool_call content block found in ClickHouse output");
    // The type check is implicitly handled by the find operation above.
    assert_eq!(content_block.get("id").unwrap().as_str().unwrap(), tool_id);
    // We explicitly do not check the tool name, as xAI decides to call 'get_temperature'
    // instead of 'self_destruct'
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        content_block.get("name").unwrap().as_str().unwrap()
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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
    // We explicitly do *not* check the content of `raw_response`, as model providers differ in whether or
    // not they actually call `self_destruct` (OpenAI will, but xAI does not).

    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![prompt.into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, StoredContentBlock::ToolCall(_)))
        .collect();

    // Assert at most one tool call (a model could decide to call no tools if to reads the `self_destruct` description).
    // Sglang likes to emit lots of tool calls
    if provider.model_provider_name != "sglang" {
        assert!(
            tool_call_blocks.len() <= 1,
            "Expected at most one tool call, found {}",
            tool_call_blocks.len()
        );
    }

    let tool_call_block = tool_call_blocks.first();
    match tool_call_block {
        Some(StoredContentBlock::ToolCall(tool_call)) => {
            // Don't check which tool was called, as xAI can sometimes call a tool other than `self_destruct`.
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        None => {}
        _ => panic!("Unreachable"),
    }
}

pub async fn test_tool_use_allowed_tools_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Groq isn't respecting allowed_tools and will call brave_search instead of get_humidity, for example
    if provider.model_provider_name == "groq" {
        return;
    }

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

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
        "extra_headers": extra_headers.extra_headers,
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."}]
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![
            "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, StoredContentBlock::ToolCall(_)))
        .collect();

    if provider.model_provider_name == "sglang" {
        // Sglang likes to emit lots of duplicate tool calls
        assert!(
            !tool_call_blocks.is_empty(),
            "Expected at least one tool call"
        );
    } else {
        // Assert exactly one tool call
        assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");
    }

    for tool_call_block in tool_call_blocks {
        match tool_call_block {
            StoredContentBlock::ToolCall(tool_call) => {
                assert_eq!(tool_call.name, "get_humidity");
                serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
            }
            _ => panic!("Unreachable"),
        }
    }
}

pub async fn test_tool_use_allowed_tools_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Together doesn't correctly produce streaming tool call chunks (it produces text chunks with the raw tool call).
    if provider.model_provider_name == "together" {
        return;
    }

    // Groq does not support streaming in JSON mode
    // (no reason given): https://console.groq.com/docs/text-chat#json-mode
    if provider.model_provider_name == "groq" {
        return;
    }

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
        "extra_headers": extra_headers.extra_headers,
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
    let mut tool_name = None;

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
                    // We support incremental streaming of raw names but currently don't believe any providers do this.
                    // If they start, we'd want to know.
                    // So we check that the raw name shows up once.
                    if let Some(block_raw_name) = block.get("raw_name") {
                        match tool_name {
                            Some(_) => {
                                // Sglang likes to emit lots of duplicate tool calls
                                if provider.model_provider_name != "sglang" {
                                    assert!(
                                        block_raw_name.as_str().unwrap().is_empty(),
                                        "Raw name already seen, got {block:#?}"
                                    );
                                }
                            }
                            None => {
                                tool_name = Some(block_raw_name.as_str().unwrap().to_string());
                            }
                        }
                    }

                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    match &tool_id {
                        None => tool_id = Some(block_tool_id.to_string()),
                        Some(tool_id) => {
                            if provider.model_provider_name == "sglang" {
                                // Sglang likes to emit lots of duplicate tool calls
                                if tool_id != block_tool_id {
                                    continue;
                                }
                            } else {
                                assert_eq!(tool_id, block_tool_id);
                            }
                        }
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();
                    arguments.push_str(chunk_arguments);
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                "thought" => {
                    // Gemini models can return thoughts - ignore them
                }
                _ => {
                    panic!("Unexpected block type: {block_type}");
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }
    assert!(tool_name.is_some());
    assert_eq!(tool_name.as_ref().unwrap(), "get_humidity");

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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well

    // Ignore other content blocks
    let content_block = output_clickhouse
        .iter()
        .find(|b| b["type"] == "tool_call")
        .unwrap();
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![
            "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, StoredContentBlock::ToolCall(_)))
        .collect();

    // Sglang likes to emit lots of tool calls
    if provider.model_provider_name == "sglang" {
        assert!(
            !tool_call_blocks.is_empty(),
            "Expected at least one tool call"
        );
    } else {
        // Assert exactly one tool call
        assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");
    }

    for tool_call_block in tool_call_blocks {
        match tool_call_block {
            StoredContentBlock::ToolCall(tool_call) => {
                assert_eq!(tool_call.name, "get_humidity");
                serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
            }
            _ => panic!("Unreachable"),
        }
    }
}

pub async fn test_tool_multi_turn_inference_request_with_provider(provider: E2ETestProvider) {
    // NOTE: The xAI API returns an error for multi-turn tool use when the assistant message only has tool_calls but no content.
    // The xAI team has acknowledged the issue and is working on a fix.
    // We skip this test for xAI until the fix is deployed.
    // https://gist.github.com/GabrielBianconi/47a4247cfd8b6689e7228f654806272d
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
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "123456789",
                            "name": "get_temperature",
                            "arguments": {"location": "Tokyo", "units": "celsius"}
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
        "extra_headers": if provider.is_modal_provider() { get_modal_extra_headers() } else { UnfilteredInferenceExtraHeaders::default() },
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "id": "123456789", "name": "get_temperature", "arguments": "{\"location\":\"Tokyo\",\"units\":\"celsius\"}"}]
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![
        StoredRequestMessage {
            role: Role::User,
            content: vec![
                "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        },
        StoredRequestMessage {
            role: Role::Assistant,
            content: vec![StoredContentBlock::ToolCall(ToolCall {
                id: "123456789".to_string(),
                name: "get_temperature".to_string(),
                arguments: "{\"location\":\"Tokyo\",\"units\":\"celsius\"}".to_string(),
            })],
        },
        StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::ToolResult(ToolResult {
                id: "123456789".to_string(),
                name: "get_temperature".to_string(),
                result: "70".to_string(),
            })],
        },
    ];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let first = output.first().unwrap();
    match first {
        StoredContentBlock::Text(text) => {
            assert!(text.text.to_lowercase().contains("tokyo"));
        }
        _ => {
            panic!("Expected a text block, got {first:?}");
        }
    }
}

pub async fn test_tool_multi_turn_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Together doesn't correctly produce streaming tool call chunks (it produces text chunks with the raw tool call).
    if provider.model_provider_name == "together" {
        return;
    }

    // NOTE: The xAI API returns an error for multi-turn tool use when the assistant message only has tool_calls but no content.
    // The xAI team has acknowledged the issue and is working on a fix.
    // We skip this test for xAI until the fix is deployed.
    // https://gist.github.com/GabrielBianconi/47a4247cfd8b6689e7228f654806272d
    if provider.model_provider_name == "xai" {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
       "function_name": "weather_helper",
        "episode_id": episode_id,
        "extra_headers": if provider.is_modal_provider() { get_modal_extra_headers() } else { UnfilteredInferenceExtraHeaders::default() },
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
                            "arguments": {"location": "Tokyo", "units": "celsius"}
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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "123456789",
                        "name": "get_temperature",
                        "arguments": "{\"location\":\"Tokyo\",\"units\":\"celsius\"}"
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());

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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![
        StoredRequestMessage {
            role: Role::User,
            content: vec![
                "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        },
        StoredRequestMessage {
            role: Role::Assistant,
            content: vec![StoredContentBlock::ToolCall(ToolCall {
                id: "123456789".to_string(),
                name: "get_temperature".to_string(),
                arguments: "{\"location\":\"Tokyo\",\"units\":\"celsius\"}".to_string(),
            })],
        },
        StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::ToolResult(ToolResult {
                id: "123456789".to_string(),
                name: "get_temperature".to_string(),
                result: "30".to_string(),
            })],
        },
    ];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let first = output.first().unwrap();
    match first {
        StoredContentBlock::Text(text) => {
            assert!(text.text.to_lowercase().contains("tokyo"));
        }
        _ => {
            panic!("Expected a text block, got {first:?}");
        }
    }
}

pub async fn test_stop_sequences_inference_request_with_provider(
    provider: E2ETestProvider,
    client: &tensorzero::Client,
) {
    // OpenAI Responses doesn't support stop sequences
    if provider.variant_name == "openai-responses" {
        return;
    }
    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

    let response = client
        .inference(tensorzero::ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            model_name: None,
            variant_name: Some(provider.variant_name.clone()),
            episode_id: Some(episode_id),
            input: tensorzero::ClientInput {
                system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                    "assistant_name".to_string(),
                    "Dr. Mehta".into(),
                )])))),
                messages: vec![tensorzero::ClientInputMessage {
                    role: Role::User,
                    content: vec![tensorzero::ClientInputMessageContent::Text(
                        TextKind::Text {
                            text: "Write me a short sentence ending with the word TensorZero. Don't say anything else."
                                .to_string(),
                        },
                    )],
                }],
            },
            extra_headers,
            params: tensorzero::InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    stop_sequences: Some(vec!["TensorZero".to_string()]),
                    ..Default::default()
                }
            },
            ..Default::default()
        })
        .await
        .unwrap();

    match response {
        tensorzero::InferenceOutput::NonStreaming(response) => {
            let InferenceResponse::Chat(response) = response else {
                panic!("Expected a chat response");
            };

            // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;

            // Check ClickHouse - ChatInference Table
            let clickhouse = get_clickhouse().await;

            println!("Response: {response:#?}");

            let model_inference =
                select_model_inference_clickhouse(&clickhouse, response.inference_id)
                    .await
                    .unwrap();
            println!("Model inference: {model_inference:#?}");

            let chat_inference =
                select_chat_inference_clickhouse(&clickhouse, response.inference_id)
                    .await
                    .unwrap();
            println!("Chat inference: {chat_inference:#?}");

            // Just check 'stop_sequences', as we check ModelInference and ChatInference in lots of other tests
            let inference_params = chat_inference
                .get("inference_params")
                .unwrap()
                .as_str()
                .unwrap();
            let inference_params: Value = serde_json::from_str(inference_params).unwrap();
            let inference_params = inference_params.get("chat_completion").unwrap();
            let stop_sequences = inference_params
                .get("stop_sequences")
                .unwrap()
                .as_array()
                .unwrap();
            assert_eq!(stop_sequences.len(), 1);
            assert_eq!(stop_sequences[0].as_str().unwrap(), "TensorZero");

            // Only some providers have a stop_sequence finish reason -
            // other providers will just give us 'stop'
            const MISSING_STOP_SEQUENCE_PROVIDERS: &[&str] = &[
                "fireworks",
                "vllm",
                "sglang",
                "mistral",
                "gcp_vertex_gemini",
                "google_ai_studio_gemini",
                "xai",
                "together",
                "deepseek",
                "openrouter",
                "openai",
                "azure",
                "groq",
                "hyperbolic",
            ];
            if MISSING_STOP_SEQUENCE_PROVIDERS.contains(&provider.model_provider_name.as_str())
                || provider.model_name == "gemma-3-1b-aws-sagemaker-openai"
                || provider.model_name == "deepseek-r1-aws-bedrock"
            {
                assert_eq!(response.finish_reason, Some(FinishReason::Stop));
            } else {
                assert_eq!(response.finish_reason, Some(FinishReason::StopSequence));
            }
            // TGI gives us a finish_reason of StopSequence, but still include the stop sequence in the response
            if !(provider.model_provider_name == "tgi"
                || provider.model_name == "gemma-3-1b-aws-sagemaker-tgi")
            {
                let json = serde_json::to_string(&response).unwrap();
                assert!(
                    !json.to_lowercase().contains("tensorzero"),
                    "TensorZero should not be in the response: `{json}`"
                );
            }
        }
        tensorzero::InferenceOutput::Streaming(_) => {
            panic!("Unexpected streaming response");
        }
    }
}

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
        extra_headers: if provider.model_provider_name == "vllm"
            || provider.model_provider_name == "sglang"
        {
            get_modal_extra_headers()
        } else {
            UnfilteredInferenceExtraHeaders::default()
        },
        input: tensorzero::ClientInput {
            system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                "assistant_name".to_string(),
                "Dr. Mehta".into(),
            )])))),
            messages: vec![tensorzero::ClientInputMessage {
                role: Role::User,
                content: vec![
                    tensorzero::ClientInputMessageContent::Text(
                        TextKind::Text {
                            text: "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function.".to_string()
                        }
                    )
                ],
            }],
        },
        stream: Some(false),
        dynamic_tool_params: tensorzero::DynamicToolParams {
            additional_tools: Some(vec![
                tensorzero::Tool::Function(tensorzero::FunctionTool {
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
                }),
            ]),
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."}]
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text { text: "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."
            .to_string() })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, StoredContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        StoredContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

pub async fn test_dynamic_tool_use_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
    client: &tensorzero::Client,
) {
    let episode_id = Uuid::now_v7();

    let input_function_name = "basic_test";

    let stream = client.inference(tensorzero::ClientInferenceParams {
        function_name: Some(input_function_name.to_string()),
        model_name: None,
        variant_name: Some(provider.variant_name.clone()),
        episode_id: Some(episode_id),
        input: tensorzero::ClientInput {
            system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                "assistant_name".to_string(),
                "Dr. Mehta".into(),
            )])))),
            messages: vec![tensorzero::ClientInputMessage {
                role: Role::User,
                content: vec![tensorzero::ClientInputMessageContent::Text(TextKind::Text { text: "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function.".to_string() })],
            }],
        },
        extra_headers: if provider.model_provider_name == "vllm"
            || provider.model_provider_name == "sglang"
        {
            get_modal_extra_headers()
        } else {
            UnfilteredInferenceExtraHeaders::default()
        },
        stream: Some(true),
        dynamic_tool_params: tensorzero::DynamicToolParams {
            additional_tools: Some(vec![
                tensorzero::Tool::Function(tensorzero::FunctionTool {
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
                }),
            ]),
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
    let mut tool_name = None;

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

        let blocks = chunk_json.get("content").unwrap().as_array().unwrap();
        for block in blocks {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    if let Some(block_raw_name) = block.get("raw_name") {
                        match tool_name {
                            Some(_) => {
                                assert!(
                                    block_raw_name.as_str().unwrap().is_empty(),
                                    "Raw name already seen, got {block:#?}"
                                );
                            }
                            None => {
                                tool_name = Some(block_raw_name.as_str().unwrap().to_string());
                            }
                        }
                    }

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
                "thought" => {
                    // Gemini models can return thoughts - ignore them
                }
                _ => {
                    panic!("Unexpected block type: {block_type}");
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    assert!(tool_name.is_some());
    assert_eq!(tool_name.as_ref().unwrap(), "get_temperature");

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
    println!("Collected arguments: {arguments}");
    serde_json::from_str::<Value>(&arguments).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well

    // Ignore other content blocks
    let content_block = output_clickhouse
        .iter()
        .find(|b| b["type"] == "tool_call")
        .unwrap();
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
    assert_eq!(tool_params["parallel_tool_calls"], Value::Null);

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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text { text: "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."
            .to_string() })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, StoredContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        StoredContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

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
        "extra_headers": if provider.is_modal_provider() { get_modal_extra_headers() } else { UnfilteredInferenceExtraHeaders::default() },
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
    check_parallel_tool_use_inference_response(
        response_json,
        &provider,
        Some(episode_id),
        false,
        Value::Bool(true),
    )
    .await;
}

pub async fn check_parallel_tool_use_inference_response(
    response_json: Value,
    provider: &E2ETestProvider,
    episode_id: Option<Uuid>,
    is_batch: bool,
    parallel_param: Value,
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
    assert_eq!(function_name, "weather_helper_parallel");

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
                        "text": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
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
    assert_eq!(tool_params["parallel_tool_calls"], parallel_param);

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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text { text: "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
            .to_string() })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();

    let is_openrouter = provider.model_provider_name == "openrouter";
    let is_groq = provider.model_provider_name == "groq";
    if is_openrouter || is_groq {
        // For Groq and OpenRouter, check that there are at least 2 tool calls
        // (these providers may include an empty text block)
        let tool_calls = output
            .iter()
            .filter(|block| matches!(block, StoredContentBlock::ToolCall(_)))
            .count();
        assert_eq!(
            tool_calls, 2,
            "Expected 2 tool calls for OpenRouter, got {tool_calls}"
        );
    } else {
        // For other providers, expect exactly 2 blocks total
        assert_eq!(
            output.len(),
            2,
            "Expected exactly 2 output blocks, got {}",
            output.len()
        );
    }

    let mut tool_call_names = vec![];
    for block in output {
        match block {
            StoredContentBlock::ToolCall(tool_call) => {
                tool_call_names.push(tool_call.name);
                serde_json::from_str::<Value>(&tool_call.arguments).unwrap();
            }
            StoredContentBlock::Text(text) if text.text.is_empty() && is_openrouter => {
                // Skip empty text blocks for OpenRouter
                continue;
            }
            StoredContentBlock::Text(text) if text.text.trim().is_empty() && is_groq => {
                // Skip empty text blocks for Groq
                continue;
            }
            _ => {
                panic!("Expected a tool call or empty text (for OpenRouter/Groq), got {block:?}");
            }
        }
    }
    assert!(tool_call_names.contains(&"get_temperature".to_string()));
    assert!(tool_call_names.contains(&"get_humidity".to_string()));
}

pub async fn test_parallel_tool_use_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Together doesn't correctly produce streaming tool call chunks (it produces text chunks with the raw tool call).
    // Groq also doesn't seem to produce the correct tool call chunks, but not sure what's happening there yet.
    if provider.model_provider_name == "together" || provider.model_provider_name == "groq" {
        return;
    }

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
        "extra_headers": if provider.is_modal_provider() { get_modal_extra_headers() } else { UnfilteredInferenceExtraHeaders::default() },
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
                    // We support incremental streaming of raw names but currently don't believe any providers do this.
                    // If they start, we'd want to know.
                    // So we check that the raw name shows up once for each tool.
                    if let Some(block_raw_name) = block.get("raw_name") {
                        match block_raw_name.as_str().unwrap() {
                            "get_temperature" => {
                                assert!(get_temperature_tool_id.is_none());
                                get_temperature_tool_id = Some(block_tool_id.to_string());
                            }
                            "get_humidity" => {
                                assert!(get_humidity_tool_id.is_none());
                                get_humidity_tool_id = Some(block_tool_id.to_string());
                            }
                            "" => {
                                // Do nothing with the empty string
                            }
                            _ => {
                                panic!("Unexpected tool name: {block_raw_name}");
                            }
                        }
                    } else {
                        assert!(
                            block.get("raw_name").is_none(),
                            "Expected no raw_name in non-first block"
                        );
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();

                    if block_tool_id == get_temperature_tool_id.as_ref().unwrap() {
                        get_temperature_arguments.push_str(chunk_arguments);
                    } else if block_tool_id == get_humidity_tool_id.as_ref().unwrap() {
                        get_humidity_arguments.push_str(chunk_arguments);
                    } else {
                        panic!("Unexpected tool id: {block_tool_id}");
                    }
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {block_type}");
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }
    assert!(get_temperature_tool_id.is_some());
    assert!(get_humidity_tool_id.is_some());

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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."}]
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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text { text: "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
            .to_string() })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    assert!(output.len() >= 2);
    let mut tool_call_names = vec![];
    for block in output {
        match block {
            StoredContentBlock::ToolCall(tool_call) => {
                tool_call_names.push(tool_call.name);
                serde_json::from_str::<Value>(&tool_call.arguments).unwrap();
            }
            // Sometimes vLLM returns a text block alongside the tool calls
            StoredContentBlock::Text(_) => {
                // Do nothing
            }
            _ => {
                panic!("Expected a tool call or text block, got {block:?}");
            }
        }
    }
    assert!(tool_call_names.contains(&"get_temperature".to_string()));
    assert!(tool_call_names.contains(&"get_humidity".to_string()));
}

pub async fn test_json_mode_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]},
        "stream": false,
        "extra_headers": extra_headers.extra_headers,
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is the name of the capital city of Japan?".to_string(),
        })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let mut output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    // We don't care about thoughts in this test
    output.retain(|block| !matches!(block, StoredContentBlock::Thought(_)));

    let is_openrouter = provider.model_provider_name == "openrouter";
    // Some providers may return both an empty text block and a tool_call block
    assert!(
        output.len() <= 2,
        "Expected at most 2 output blocks, got {}",
        output.len()
    );

    // Check for valid content in the output
    let mut found_valid_content = false;
    for output_block in &output {
        match output_block {
            StoredContentBlock::Text(text) if text.text.trim().is_empty() => {
                // Skip empty text blocks
                continue;
            }
            StoredContentBlock::Text(text) => {
                let _: Value = serde_json::from_str(&text.text).unwrap();
                assert!(text.text.to_lowercase().contains("tokyo"));
                found_valid_content = true;
            }
            StoredContentBlock::ToolCall(tool_call) => {
                // OpenRouter may use tool_call format
                assert_eq!(tool_call.name, "respond");
                let arguments: Value = serde_json::from_str(&tool_call.arguments).unwrap();
                let answer = arguments.get("answer").unwrap().as_str().unwrap();
                assert!(answer.to_lowercase().contains("tokyo"));
                found_valid_content = true;
            }
            _ => {
                panic!("Expected a text block or tool_call, got {output_block:?}");
            }
        }
    }

    // OpenRouter must have at least one valid content block
    if is_openrouter {
        assert!(
            found_valid_content,
            "No valid JSON content found in OpenRouter response"
        );
    }
}

pub async fn test_dynamic_json_mode_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]},
        "stream": false,
        "output_schema": output_schema.clone(),
        "extra_headers": extra_headers.extra_headers,
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
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
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

    if !is_batch {
        let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
        assert!(processing_time_ms > 0);
    }

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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is the name of the capital city of Japan?".to_string(),
        })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let mut output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    // We don't care about thoughts in this test
    output.retain(|block| !matches!(block, StoredContentBlock::Thought(_)));

    let is_openrouter = provider.model_provider_name == "openrouter";
    // Some providers return both an empty text block and a tool_call block
    assert!(
        output.len() <= 2,
        "Expected at most 2 output blocks for OpenRouter, got {}",
        output.len()
    );

    // Check for valid content in the output
    let mut found_valid_content = false;
    for output_block in &output {
        match output_block {
            StoredContentBlock::Text(text) if text.text.trim().is_empty() => {
                // Skip empty text blocks
                continue;
            }
            StoredContentBlock::Text(text) => {
                let _: Value = serde_json::from_str(&text.text).unwrap();
                assert!(&text.text.to_lowercase().contains("tokyo"));
                found_valid_content = true;
            }
            StoredContentBlock::ToolCall(tool_call) => {
                // OpenRouter may use tool_call format
                assert_eq!(tool_call.name, "respond");
                let arguments: Value = serde_json::from_str(&tool_call.arguments).unwrap();
                let response = arguments.get("response").unwrap().as_str().unwrap();
                assert!(response.to_lowercase().contains("tokyo"));
                found_valid_content = true;
            }
            _ => {
                panic!("Expected a text block or tool_call (for OpenRouter), got {output_block:?}");
            }
        }
    }

    // OpenRouter must have at least one valid content block
    // We do this check because OpenRouter sometimes responds with empty content blocks
    if is_openrouter {
        assert!(
            found_valid_content,
            "No valid JSON content found in OpenRouter response"
        );
    }
}

pub async fn test_json_mode_streaming_inference_request_with_provider(provider: E2ETestProvider) {
    if provider.variant_name.contains("tgi")
        || provider.variant_name.contains("cot")
        || provider.model_provider_name == "groq"
    {
        // TGI does not support streaming in JSON mode (because it doesn't support streaming tools)
        // Groq does not support streaming in JSON mode (no reason given): https://console.groq.com/docs/text-chat#json-mode)
        return;
    }

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]},
        "stream": true,
        "extra_headers": extra_headers.extra_headers,
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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
                "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
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
    let max_tokens = if provider.model_name.contains("gemini-2.5-pro") {
        500
    } else if provider.model_name.starts_with("o1") {
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

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is the name of the capital city of Japan?".to_string(),
        })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        StoredContentBlock::Text(text) => {
            let parsed: Value = serde_json::from_str(&text.text).unwrap();
            let answer = parsed.get("answer").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        StoredContentBlock::ToolCall(tool_call) => {
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

pub async fn test_short_inference_request_with_provider(provider: E2ETestProvider) {
    // We currently host ollama on sagemaker, and use a wrapped 'openai' provider
    // in our tensorzero.toml. ollama doesn't support 'max_completion_tokens', so this test
    // currently fails. It's fine to skip it, since we really care about testing the sagemaker
    // wrapper code, not whatever container we happen to be wrapping.
    if provider.model_provider_name == "aws_sagemaker" {
        return;
    }

    // The 2.5 Pro model always seems to think before responding, even with
    // {"generationConfig": {"thinkingConfig": {"thinkingBudget": 0 }}
    // This also happens for DeepSeek R1
    // This prevents us from setting a low max_tokens, since the thinking tokens will
    // use up all of the output tokens before an actual response is generated.
    if provider.model_name.contains("gemini-2.5-pro")
        || provider.model_name.contains("deepseek-r1-aws-bedrock")
    {
        return;
    }

    // The OpenAI Responses API has a minimum value of 16
    let max_tokens = if provider.model_name.starts_with("responses-") {
        16
    } else {
        1
    };

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

    // Include randomness in the prompt to force a cache miss for the first request
    let randomness = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": format!("Dr. Mehta: {randomness}")},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan? Explain your answer."
                }
            ]},
        "stream": false,
        "tags": {"foo": "bar"},
        "cache_options": {"enabled": "on", "lookback_s": 10},
        "params": {
            "chat_completion": {
                "max_tokens": max_tokens
            }
        },
        "extra_headers": extra_headers.extra_headers,
    });
    if provider.variant_name.contains("openai") && provider.variant_name.contains("o1") {
        // Can't pin a single token for o1
        return;
    }

    if provider.variant_name.contains("openrouter") {
        // OpenRouter claims gpt4.1-mini needs a minimum of 16 output tokens
        return;
    }

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

    check_short_inference_response(
        randomness,
        response_json,
        Some(episode_id),
        &provider,
        max_tokens,
        false,
    )
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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

    check_short_inference_response(
        randomness,
        response_json,
        Some(episode_id),
        &provider,
        max_tokens,
        true,
    )
    .await;
}

async fn check_short_inference_response(
    randomness: Uuid,
    response_json: Value,
    episode_id: Option<Uuid>,
    provider: &E2ETestProvider,
    max_tokens: u32,
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
    // We don't check the content here since there's only 1 token allowed

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    if should_be_cached {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert_eq!(output_tokens, max_tokens as u64);
    }
    let finish_reason = response_json
        .get("finish_reason")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(finish_reason, "length");

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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
        "system": {"assistant_name": format!("Dr. Mehta: {randomness}")},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the name of the capital city of Japan? Explain your answer."}]
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

    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.get("foo").unwrap().as_str().unwrap(), "bar");

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    assert!(tool_params.is_empty());

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
        max_tokens as u64
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
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();
    assert!(input_tokens.as_u64().unwrap() > 0);
    assert!(output_tokens.as_u64().unwrap() > 0);
    if !should_be_cached {
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }
    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        format!("You are a helpful and friendly assistant named Dr. Mehta: {randomness}")
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is the name of the capital city of Japan? Explain your answer.".to_string(),
        })],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let finish_reason = result.get("finish_reason").unwrap().as_str().unwrap();
    assert_eq!(finish_reason, "length");

    // Check the InferenceTag Table
    select_inference_tags_clickhouse(
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
    assert_eq!(
        result.get("cached").unwrap().as_bool().unwrap(),
        should_be_cached
    );

    // Check that our cache entry was only written once
    if should_be_cached {
        // Allow some time for an incorrect second cache write to happen
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Count the number of cache entries for this raw_request
        // Note that this can have false negatives (ClickHouse might decide to merge parts
        // immediately after a duplicate insert)
        let count = clickhouse
            .run_query_synchronous(
                "SELECT COUNT(*) FROM ModelInferenceCache WHERE raw_request = {raw_request:String} "
                    .to_string(),
                &[("raw_request", result["raw_request"].as_str().unwrap())]
                    .into_iter()
                    .collect(),
            )
            .await
            .unwrap()
            .response
            .trim()
            .parse::<u64>()
            .unwrap();

        assert_eq!(count, 1);
    }
}

pub async fn test_multi_turn_parallel_tool_use_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Together's model is too dumb to figure out multi-turn tool + parallel tool calls... It keeps calling the same tool over and over.
    if provider.model_provider_name == "together" {
        return;
    }

    let episode_id = Uuid::now_v7();

    let mut payload = json!({
        "function_name": "weather_helper_parallel",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Fahrenheit)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
                }
            ]},
        "parallel_tool_calls": true,
        "stream": false,
        "extra_headers": if provider.is_modal_provider() { get_modal_extra_headers() } else { UnfilteredInferenceExtraHeaders::default() },
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

    // Extract the tool results from the response
    let mut redacted_tool_calls = Vec::new();
    let mut tool_results = Vec::new();

    for content_block in response_json.get("content").unwrap().as_array().unwrap() {
        let content_block_type = content_block.get("type").unwrap().as_str().unwrap();

        // Special handling for OpenRouter empty text blocks
        let is_openrouter = provider.model_provider_name == "openrouter";
        if content_block_type == "text" && is_openrouter {
            // For OpenRouter, skip empty text blocks
            let text = content_block.get("text").unwrap().as_str().unwrap();
            if text.is_empty() {
                continue;
            } else {
                panic!("Unexpected text block with non-empty content: {text}");
            }
        }

        // Special handling for groq text blocks
        if content_block_type == "text" && provider.model_provider_name == "groq" {
            let text = content_block.get("text").unwrap().as_str().unwrap();
            // Groq produces a text block containing '\n'
            if text.trim().is_empty() {
                continue;
            } else {
                panic!("Unexpected text block with non-empty content: {text}");
            }
        }

        assert_eq!(
            content_block_type, "tool_call",
            "Expected tool_call, got {content_block_type}"
        );

        if content_block.get("name").unwrap().as_str().unwrap() == "get_temperature" {
            tool_results.push(json!(
                {
                    "type": "tool_result",
                    "id": content_block.get("id").unwrap().as_str().unwrap(),
                    "name": "get_temperature",
                    "result": "70",
                }
            ));
        } else if content_block.get("name").unwrap().as_str().unwrap() == "get_humidity" {
            tool_results.push(json!(
                {
                    "type": "tool_result",
                    "id": content_block.get("id").unwrap().as_str().unwrap(),
                    "name": "get_humidity",
                    "result": "30",
                }
            ));
        } else {
            panic!(
                "Unknown tool call: {}",
                content_block.get("name").unwrap().as_str().unwrap()
            );
        }
        redacted_tool_calls.push(content_block);
    }

    // Build the payload for the second inference request
    let assistant_message = json!({
        "role": "assistant",
        "content": redacted_tool_calls,
    });

    let user_message = json!({
        "role": "user",
        "content": tool_results,
    });

    payload["input"]["messages"]
        .as_array_mut()
        .unwrap()
        .extend([assistant_message, user_message]);

    println!(
        "Second Payload: {}",
        serde_json::to_string_pretty(&payload).unwrap()
    );

    // Make the second inference request
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("Second API response: {response_json:#?}");

    check_multi_turn_parallel_tool_use_inference_response(
        response_json,
        &provider,
        Some(episode_id),
        false,
    )
    .await;
}

pub async fn check_multi_turn_parallel_tool_use_inference_response(
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

    // Validate that the assistant message is correct
    assert_eq!(content.len(), 1);
    let content_block = content.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content_text = content_block.get("text").unwrap().as_str().unwrap();
    assert!(content_text.to_lowercase().contains("70"));
    assert!(content_text.to_lowercase().contains("30"));

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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

    let last_input_message = input["messages"].as_array().unwrap().last().unwrap();
    assert_eq!(last_input_message["role"], "user");
    let last_input_message_content = last_input_message["content"].as_array().unwrap();
    assert_eq!(last_input_message_content.len(), 2);
    for tool_result in last_input_message_content {
        assert_eq!(tool_result["type"], "tool_result");
    }

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    let output_content = serde_json::to_value(content).unwrap();
    println!("Output clickhouse: {output_clickhouse:#?}");
    println!("Output content: {output_content:#?}");
    assert_eq!(output_clickhouse, *output_content.as_array().unwrap());

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], true);

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
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("70"));
    assert!(raw_response.to_lowercase().contains("30"));

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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let last_input_message = input_messages.last().unwrap();
    assert_eq!(last_input_message.role, Role::User);
    let last_input_message_content = &last_input_message.content;
    assert_eq!(last_input_message_content.len(), 2);
    for tool_result in last_input_message_content {
        match tool_result {
            StoredContentBlock::ToolResult(tool_result) => {
                assert!(
                    tool_result.name == "get_temperature" || tool_result.name == "get_humidity"
                );
            }
            _ => {
                panic!("Expected a tool call, got {tool_result:?}");
            }
        }
    }
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let output_content = output.first().unwrap();
    match output_content {
        StoredContentBlock::Text(text) => {
            assert!(text.text.to_lowercase().contains("70"));
            assert!(text.text.to_lowercase().contains("30"));
        }
        _ => {
            panic!("Expected a text block, got {output_content:?}");
        }
    }
}

pub async fn test_multi_turn_parallel_tool_use_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Together's model is too dumb to figure out multi-turn tool + parallel tool calls... It keeps calling the same tool over and over.
    if provider.model_provider_name == "together" {
        return;
    }

    let episode_id = Uuid::now_v7();

    let mut payload = json!({
        "function_name": "weather_helper_parallel",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Fahrenheit)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
                }
            ]},
        "parallel_tool_calls": true,
        "stream": false,
        "variant_name": provider.variant_name,
        "extra_headers": if provider.is_modal_provider() { get_modal_extra_headers() } else { UnfilteredInferenceExtraHeaders::default() },
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

    // Extract the tool results from the response
    let mut redacted_tool_calls = Vec::new();
    let mut tool_results = Vec::new();

    for content_block in response_json.get("content").unwrap().as_array().unwrap() {
        let content_block_type = content_block.get("type").unwrap().as_str().unwrap();

        // Special handling for OpenRouter empty text blocks
        let is_openrouter = provider.model_provider_name == "openrouter";
        if content_block_type == "text" && is_openrouter {
            // For OpenRouter, skip empty text blocks
            let text = content_block.get("text").unwrap().as_str().unwrap();
            if text.is_empty() {
                continue;
            } else {
                panic!("Unexpected text block with non-empty content: {text}");
            }
        }

        // Special handling for groq text blocks
        if content_block_type == "text" && provider.model_provider_name == "groq" {
            let text = content_block.get("text").unwrap().as_str().unwrap();
            // Groq produces a text block containing '\n'
            if text.trim().is_empty() {
                continue;
            } else {
                panic!("Unexpected text block with non-empty content: {text}");
            }
        }

        assert_eq!(
            content_block_type, "tool_call",
            "Expected tool_call, got {content_block_type}"
        );

        if content_block.get("name").unwrap().as_str().unwrap() == "get_temperature" {
            tool_results.push(json!(
                {
                    "type": "tool_result",
                    "id": content_block.get("id").unwrap().as_str().unwrap(),
                    "name": "get_temperature",
                    "result": "70",
                }
            ));
        } else if content_block.get("name").unwrap().as_str().unwrap() == "get_humidity" {
            tool_results.push(json!(
                {
                    "type": "tool_result",
                    "id": content_block.get("id").unwrap().as_str().unwrap(),
                    "name": "get_humidity",
                    "result": "30",
                }
            ));
        } else {
            panic!(
                "Unknown tool call: {}",
                content_block.get("name").unwrap().as_str().unwrap()
            );
        }

        redacted_tool_calls.push(content_block);
    }

    // Build the payload for the second inference request
    let assistant_message = json!({
        "role": "assistant",
        "content": redacted_tool_calls,
    });

    let user_message = json!({
        "role": "user",
        "content": tool_results,
    });

    // Update the payload with the user message
    payload["input"]["messages"]
        .as_array_mut()
        .unwrap()
        .extend([assistant_message, user_message]);

    println!(
        "Second Payload: {}",
        serde_json::to_string_pretty(&payload).unwrap()
    );

    // Make the payload stream=true
    payload["stream"] = json!(true);

    // Make the second inference request
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
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    let mut output_content = String::new();

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
                "text" => {
                    output_content.push_str(block.get("text").unwrap().as_str().unwrap());
                }
                _ => {
                    panic!("Unexpected block type: {block_type}");
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

    // Check that the output contains the values
    println!("Output content: {output_content:#?}");
    assert!(output_content.contains("70"));
    assert!(output_content.contains("30"));

    // Check inference_id
    let inference_id = inference_id.unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

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

    let last_input_message = input["messages"].as_array().unwrap().last().unwrap();
    assert_eq!(last_input_message["role"], "user");
    let last_input_message_content = last_input_message["content"].as_array().unwrap();
    assert_eq!(last_input_message_content.len(), 2);
    for tool_result in last_input_message_content {
        assert_eq!(tool_result["type"], "tool_result");
    }

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    println!("Output clickhouse: {output_clickhouse:#?}");
    println!("Output content: {output_content:#?}");
    assert_eq!(output_clickhouse[0]["text"], output_content);

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
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap();
    let output_tokens = result.get("output_tokens").unwrap();

    // NB: Azure OpenAI Service doesn't support input/output tokens during streaming, but Azure AI Foundry does
    if provider.variant_name.contains("azure")
        && !provider.variant_name.contains("azure-ai-foundry")
    {
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
    let input_messages: Vec<StoredRequestMessage> = serde_json::from_str(input_messages).unwrap();
    let last_input_message = input_messages.last().unwrap();
    assert_eq!(last_input_message.role, Role::User);
    let last_input_message_content = &last_input_message.content;
    assert_eq!(last_input_message_content.len(), 2);
    for tool_result in last_input_message_content {
        match tool_result {
            StoredContentBlock::ToolResult(tool_result) => {
                assert!(
                    tool_result.name == "get_temperature" || tool_result.name == "get_humidity"
                );
            }
            _ => {
                panic!("Expected a tool call, got {tool_result:?}");
            }
        }
    }
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let output_content = output.first().unwrap();
    match output_content {
        StoredContentBlock::Text(text) => {
            assert!(text.text.to_lowercase().contains("70"));
            assert!(text.text.to_lowercase().contains("30"));
        }
        _ => {
            panic!("Expected a text block, got {output_content:?}");
        }
    }
}

pub async fn test_json_mode_off_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

    let payload = json!({
        "function_name": "json_success",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "AskJeeves"},
               "messages": [
                   {
                       "role": "user",
                       "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                   }
               ]
            },
        "params": {
            "chat_completion": {
                "json_mode": "off",
            }
        },
        "stream": false,
        "extra_headers": extra_headers.extra_headers,
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

    println!("API response: {response_json}");

    // Assert the output isn't JSON
    let output = response_json.get("output").unwrap().as_object().unwrap();
    let parsed = output.get("parsed").unwrap().as_object();
    assert_eq!(parsed, None);
    let raw = output.get("raw").unwrap().as_str().unwrap();
    assert!(serde_json::from_str::<Value>(raw).is_err());

    // Assert that the answer is correct
    assert!(
        raw.to_lowercase().contains("tokyo"),
        "Unexpected raw output: {raw}"
    );

    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    // First, check Inference table
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    // Check that correctly parsed output is present
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Value = serde_json::from_str(output).unwrap();
    let raw = output.get("raw").unwrap().as_str().unwrap();
    assert!(raw.to_lowercase().contains("tokyo"));

    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    // Check the processing time
    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check that we saved the correct json mode to ClickHouse
    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let clickhouse_json_mode = inference_params
        .get("chat_completion")
        .unwrap()
        .get("json_mode")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!("off", clickhouse_json_mode);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));

    // Check that raw_request is valid JSON
    let raw_request_val: Value =
        serde_json::from_str(raw_request).expect("raw_request should be valid JSON");

    // Check that we're not sending `response_format` or `generationConfig`
    if provider.model_provider_name == "google_ai_studio_gemini"
        || provider.model_provider_name == "gcp_vertex_gemini"
    {
        assert!(raw_request_val["generationConfig"]
            .get("response_mime_type")
            .is_none());
    } else {
        assert!(raw_request_val.get("response_format").is_none());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 5);

    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 5);

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    assert!(result.get("ttft_ms").unwrap().is_null());
}

pub async fn test_multiple_text_blocks_in_message_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };
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
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the name of the capital city"
                        },
                        {
                            "type": "text",
                            "text": "of Japan?"
                        }
                    ]
                }
            ]},
        "stream": false,
        "tags": {"foo": "bar"},
        "extra_headers": extra_headers.extra_headers,
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

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    // Some providers always produce thought blocks - we don't care about them in this test
    let mut content = response_json
        .get("content")
        .unwrap()
        .as_array()
        .unwrap()
        .clone();
    content.retain(|c| c.get("type").unwrap().as_str().unwrap() != "thought");
    assert_eq!(content.len(), 1);
    let content_block = content.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert!(content.to_lowercase().contains("tokyo"));
}
