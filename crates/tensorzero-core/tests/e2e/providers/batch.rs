#![expect(clippy::print_stdout)]
/// This file contains 3 types of test for batch inference:
/// 1. Start a batch inference. Should do whatever is necessary for a provider to start a batch inference for a particular kind of inference
///    Should also set up bookkeeping so that we can poll for the batch inference to complete later on.
/// 2. Poll for a batch inference from Pending. Take a currently Pending batch inference and poll for it to complete.
/// 3. Poll for a batch inference from Completed. Take a currently Completed batch inference (so we know there are results),
///    change it so that it looks like it is Pending, and then poll for it to be completed again.
use std::collections::HashMap;

use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use std::collections::HashSet;
use tensorzero_core::db::{
    batch_inference::BatchInferenceQueries, delegating_connection::DelegatingDatabaseConnection,
    test_helpers::TestDatabaseHelpers,
};
use tensorzero_core::inference::types::batch::{BatchRequestRow, BatchStatus};
use tensorzero_core::inference::types::{StoredContentBlock, StoredRequestMessage};
use tensorzero_core::tool::Tool;
use tensorzero_core::{
    endpoints::batch_inference::PollPathParams,
    inference::types::{Role, Text},
    tool::{ToolCall, ToolResult},
};
use tokio::time::{Duration, sleep};
use url::Url;
use uuid::Uuid;

use crate::providers::common::{
    check_dynamic_json_mode_inference_response, check_dynamic_tool_use_inference_response,
    check_json_mode_inference_response, check_multi_turn_parallel_tool_use_inference_response,
    check_parallel_tool_use_inference_response, check_tool_use_multi_turn_inference_response,
    check_tool_use_tool_choice_allowed_tools_inference_response,
    check_tool_use_tool_choice_auto_unused_inference_response,
    check_tool_use_tool_choice_auto_used_inference_response,
    check_tool_use_tool_choice_none_inference_response,
    check_tool_use_tool_choice_required_inference_response,
    check_tool_use_tool_choice_specific_inference_response,
};
use crate::{
    common::get_gateway_endpoint,
    providers::common::{check_inference_params_response, check_simple_image_inference_response},
};

use super::common::E2ETestProvider;

#[macro_export]
macro_rules! generate_batch_inference_tests {
    ($func:ident) => {
        use $crate::providers::batch::test_dynamic_json_mode_batch_inference_request_with_provider;
        use $crate::providers::batch::test_dynamic_tool_use_batch_inference_request_with_provider;
        use $crate::providers::batch::test_start_inference_params_batch_inference_request_with_provider;
        use $crate::providers::batch::test_json_mode_batch_inference_request_with_provider;
        use $crate::providers::batch::test_parallel_tool_use_batch_inference_request_with_provider;
        use $crate::providers::batch::test_start_simple_image_batch_inference_request_with_provider;
        use $crate::providers::batch::test_tool_multi_turn_batch_inference_request_with_provider;
        use $crate::providers::batch::test_tool_use_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_existing_simple_image_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_completed_simple_image_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_existing_inference_params_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_completed_inference_params_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_existing_tool_choice_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_completed_tool_use_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_existing_multi_turn_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_completed_multi_turn_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_existing_dynamic_tool_use_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_completed_dynamic_tool_use_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_existing_parallel_tool_use_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_completed_parallel_tool_use_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_existing_json_mode_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_completed_json_mode_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_existing_dynamic_json_mode_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_completed_dynamic_json_mode_batch_inference_request_with_provider;
        use $crate::providers::batch::test_allowed_tools_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_existing_allowed_tools_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_completed_allowed_tools_batch_inference_request_with_provider;
        use $crate::providers::batch::test_multi_turn_parallel_tool_use_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_existing_multi_turn_parallel_batch_inference_request_with_provider;
        use $crate::providers::batch::test_poll_completed_multi_turn_parallel_batch_inference_request_with_provider;

        #[tokio::test]
        async fn test_start_simple_image_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_start_simple_image_batch_inference_request_with_provider(provider).await;
                }
            }
        }

        #[tokio::test]
        async fn test_poll_existing_simple_image_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_existing_simple_image_batch_inference_request_with_provider(provider).await;
                }
            }
        }

        #[tokio::test]
        async fn test_poll_completed_simple_image_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_completed_simple_image_batch_inference_request_with_provider(provider).await;
                }
            }
        }

        #[tokio::test]
        async fn test_start_inference_params_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.inference_params_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_start_inference_params_batch_inference_request_with_provider(provider).await;
                }
            }
        }

        #[tokio::test]
        async fn test_poll_existing_inference_params_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.inference_params_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_existing_inference_params_batch_inference_request_with_provider(provider).await;
                }
            }
        }

        #[tokio::test]
        async fn test_poll_completed_inference_params_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.inference_params_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_completed_inference_params_batch_inference_request_with_provider(provider).await;
                }
            }
        }

        #[tokio::test]
        async fn test_start_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_tool_use_batch_inference_request_with_provider(provider).await;
                }
            }
        }

        #[tokio::test]
        async fn test_poll_existing_tool_choice_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_existing_tool_choice_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_completed_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_completed_tool_use_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_start_tool_multi_turn_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.tool_multi_turn_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_tool_multi_turn_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_existing_multi_turn_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.tool_multi_turn_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_existing_multi_turn_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_completed_multi_turn_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.tool_multi_turn_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_completed_multi_turn_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_start_multi_turn_parallel_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.parallel_tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_multi_turn_parallel_tool_use_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_existing_multi_turn_parallel_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.parallel_tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_existing_multi_turn_parallel_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_completed_multi_turn_parallel_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.parallel_tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_completed_multi_turn_parallel_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_start_dynamic_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.dynamic_tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_dynamic_tool_use_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_existing_dynamic_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.dynamic_tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_existing_dynamic_tool_use_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_completed_dynamic_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.dynamic_tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_completed_dynamic_tool_use_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_start_parallel_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.parallel_tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_parallel_tool_use_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_existing_parallel_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.parallel_tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_existing_parallel_tool_use_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_completed_parallel_tool_use_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.parallel_tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_completed_parallel_tool_use_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_start_json_mode_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.json_mode_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_json_mode_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_existing_json_mode_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.json_mode_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_existing_json_mode_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_completed_json_mode_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.json_mode_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_completed_json_mode_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_start_dynamic_json_mode_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.json_mode_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_dynamic_json_mode_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_existing_dynamic_json_mode_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.json_mode_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_existing_dynamic_json_mode_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_completed_dynamic_json_mode_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.json_mode_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_completed_dynamic_json_mode_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_start_allowed_tools_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_allowed_tools_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_existing_allowed_tools_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_existing_allowed_tools_batch_inference_request_with_provider(provider).await;

                }
            }
        }

        #[tokio::test]
        async fn test_poll_completed_allowed_tools_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_poll_completed_allowed_tools_batch_inference_request_with_provider(provider).await;

                }
            }
        }
    };
}

pub async fn check_batch_request_status(
    conn: &DelegatingDatabaseConnection,
    batch_id: Uuid,
    provider: &E2ETestProvider,
    expected_status: BatchStatus,
) {
    let result = conn
        .get_batch_request(batch_id, None)
        .await
        .expect("Failed to get batch request")
        .expect("Batch request not found");

    println!("BatchRequest: {result:#?}");

    assert_eq!(result.batch_id, batch_id);
    // We can't check that the batch params are exactly the same because they vary per-provider
    // We will check that they are valid by using them instead.
    assert_eq!(result.model_name.as_ref(), provider.model_name);
    assert_eq!(
        result.model_provider_name.as_ref(),
        provider.model_provider_name
    );
    assert_eq!(result.status, expected_status);
    assert!(result.errors.is_empty());
}

pub fn get_poll_batch_inference_url(query: PollPathParams) -> Url {
    let mut url = get_gateway_endpoint("/batch_inference");
    match query {
        PollPathParams {
            batch_id,
            inference_id: None,
        } => {
            url.path_segments_mut().unwrap().push(&batch_id.to_string());
            url
        }
        PollPathParams {
            batch_id,
            inference_id: Some(inference_id),
        } => {
            url.path_segments_mut()
                .unwrap()
                .push(&batch_id.to_string())
                .push("inference")
                .push(&inference_id.to_string());
            url
        }
    }
}

/// Find the latest batch inference matching the given criteria.
/// This uses a direct ClickHouse query since there's no trait method for this kind of lookup.
async fn get_latest_batch_inference(
    conn: &DelegatingDatabaseConnection,
    function_name: &str,
    variant_name: &str,
    status: &str,
    tags: Option<HashMap<String, String>>,
) -> Option<BatchRequestRow<'static>> {
    assert!(
        status == "pending" || status == "completed",
        "Status must be either 'pending' or 'completed'"
    );
    let tags = tags.unwrap_or_default();
    let tag_conditions = tags
        .iter()
        .map(|(k, v)| format!("tags['{k}'] = '{v}'"))
        .collect::<Vec<_>>()
        .join(" AND ");

    let tag_filter = if tags.is_empty() {
        String::new()
    } else {
        format!("AND bmi.{tag_conditions}")
    };

    let query = format!(
        r"
            SELECT DISTINCT
                br.batch_id,
                br.id,
                br.batch_params,
                br.model_name,
                br.model_provider_name,
                br.status,
                br.function_name,
                br.variant_name,
                br.raw_request,
                br.raw_response,
                br.errors
            FROM BatchRequest br
            INNER JOIN BatchModelInference bmi ON br.batch_id = bmi.batch_id
            WHERE br.function_name = '{function_name}'
                AND br.variant_name = '{variant_name}'
                AND br.status = '{status}'
                {tag_filter}
            ORDER BY br.timestamp DESC
            LIMIT 1
            FORMAT JSONEachRow
        "
    );
    let response = conn
        .clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    if response.response.is_empty() {
        return None;
    }
    let batch_request = serde_json::from_str::<BatchRequestRow>(&response.response).unwrap();
    Some(batch_request)
}

pub struct InsertedFakeDataIds {
    batch_id: Uuid,
    inference_id: Uuid,
}

/// If there are already completed batch inferences for the given function and variant,
/// this will create new pending batch inferences with new batch and inference IDs.
/// This will test polling in a short-term way if possible (there is already data in the DB with valid params).
///
/// If `tags` is provided, it will only look up batch inferences that match the provided tags.
/// The new `BatchRequest` will be written out with different tags, to ensure that it doesn't
/// affect tests running in parallel that want to check the original batch inference.
async fn insert_fake_pending_batch_inference_data(
    conn: &DelegatingDatabaseConnection,
    function_name: &str,
    variant_name: &str,
    tags: Option<HashMap<String, String>>,
) -> Option<InsertedFakeDataIds> {
    let batch_request =
        get_latest_batch_inference(conn, function_name, variant_name, "completed", tags).await;
    let mut batch_request = match batch_request {
        None => {
            return None;
        }
        Some(batch_request) => batch_request,
    };
    let mut batch_inferences = conn
        .get_batch_model_inferences(batch_request.batch_id, &[])
        .await
        .expect("Failed to get batch model inferences");
    if batch_inferences.is_empty() {
        return None;
    }
    let new_batch_id = Uuid::now_v7();
    batch_request.batch_id = new_batch_id;
    for inference in &mut batch_inferences {
        inference.batch_id = new_batch_id;
        inference.tags = HashMap::from([(
            "fake_pending".to_string(),
            serde_json::to_string(&serde_json::Value::Object(
                inference
                    .tags
                    .clone()
                    .into_iter()
                    .map(|(k, v)| (k, v.into()))
                    .collect::<serde_json::Map<String, Value>>(),
            ))
            .unwrap(),
        )]);
    }

    conn.write_batch_model_inferences(batch_inferences.as_slice())
        .await
        .expect("Failed to write batch model inferences");
    conn.write_batch_request(&batch_request)
        .await
        .expect("Failed to write batch request");

    Some(InsertedFakeDataIds {
        batch_id: new_batch_id,
        inference_id: batch_inferences[0].inference_id,
    })
}

pub async fn test_start_simple_image_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_ids": [episode_id],
        "inputs":
            [{
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What kind of animal is in this image?"},
                        {
                            "type": "image",
                            "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png"
                        },
                    ]
                },
            ]}],
        "tags": [{"foo": "bar", "test_type": "batch_simple_image_v2", "variant_name": provider.variant_name}],
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Wait for trailing writes from the API to be visible
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let results = conn
        .get_batch_model_inferences(batch_id, &[inference_id])
        .await
        .expect("Failed to get batch model inferences");
    assert_eq!(results.len(), 1);
    let result = &results[0];

    println!("BatchModelInference: {result:#?}");

    assert_eq!(result.inference_id, inference_id);
    assert_eq!(result.batch_id, batch_id);
    assert_eq!(result.function_name.as_ref(), "basic_test");
    assert_eq!(result.variant_name.as_ref(), provider.variant_name);
    assert_eq!(result.episode_id, episode_id);

    let input = serde_json::to_value(result.input.as_ref().unwrap()).unwrap();
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

    let input_messages = result.input_messages.as_ref().unwrap();
    assert_eq!(input_messages.len(), 1);
    assert_eq!(input_messages[0].role, Role::User);
    assert_eq!(
        input_messages[0].content[0],
        StoredContentBlock::Text(Text {
            text: "What kind of animal is in this image?".to_string(),
        })
    );
    assert!(
        matches!(input_messages[0].content[1], StoredContentBlock::File(_)),
        "Unexpected input: {input_messages:?}"
    );
    assert_eq!(input_messages[0].content.len(), 2);

    assert_eq!(
        result.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named Dr. Mehta"
    );

    assert!(result.tool_params.is_none());

    let inference_params = result.inference_params.as_ref().unwrap();
    assert!(inference_params.chat_completion.temperature.is_none());
    assert!(inference_params.chat_completion.seed.is_none());
    let expected_max_tokens: u32 = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };
    assert_eq!(
        inference_params.chat_completion.max_tokens.unwrap(),
        expected_max_tokens
    );

    assert_eq!(result.model_name.as_ref(), provider.model_name);
    assert_eq!(
        result.model_provider_name.as_ref(),
        provider.model_provider_name
    );
    assert!(result.output_schema.is_none());
    assert_eq!(result.tags.get("foo").unwrap(), "bar");

    let raw_request = result.raw_request.as_ref().unwrap();
    assert!(!raw_request.is_empty());

    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Pending).await;
}

/// If there is a pending batch inference for the function, variant, and tags
/// that are used for the simple batch inference tests,
/// this will poll the batch inference and check that the response is correct.
///
/// This test polls by `batch_id` then by inference id.
pub async fn test_poll_existing_simple_image_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "basic_test";
    let latest_pending_batch_inference = get_latest_batch_inference(
        &conn,
        function_name,
        &provider.variant_name,
        "pending",
        Some(HashMap::from([(
            "test_type".to_string(),
            "batch_simple_image_v2".to_string(),
        )])),
    )
    .await;
    let batch_inference = match latest_pending_batch_inference {
        None => return, // No pending batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    let batch_id = batch_inference.batch_id;

    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => return,
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    // Check the response from polling by `batch_id`
    check_simple_image_inference_response(inferences_json[0].clone(), None, &provider, true, false)
        .await;

    // Check the response from polling by inference_id
    let inference_id = inferences_json[0]
        .get("inference_id")
        .unwrap()
        .as_str()
        .unwrap();
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: Some(Uuid::parse_str(inference_id).unwrap()),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");

    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    check_simple_image_inference_response(inferences_json[0].clone(), None, &provider, true, false)
        .await;
    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Completed).await;
}

/// If there is a completed batch inference for the function, variant, and tags
/// that are used for the simple batch inference tests,
/// this test will create fake pending data that uses the same API params but with new IDs
/// and then poll the batch inference and check that the response is correct.
///
/// This way the gateway will actually poll the inference data from the inference provider.
///
/// This test polls by `inference_id` then by `batch_id`.
pub async fn test_poll_completed_simple_image_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "basic_test";
    let latest_pending_batch_inference = insert_fake_pending_batch_inference_data(
        &conn,
        function_name,
        &provider.variant_name,
        Some(HashMap::from([(
            "test_type".to_string(),
            "batch_simple_image_v2".to_string(),
        )])),
    )
    .await;
    let ids = match latest_pending_batch_inference {
        None => return, // No completed batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    sleep(Duration::from_millis(200)).await;
    test_poll_completed_simple_image_batch_inference_request_with_provider_and_ids(provider, ids)
        .await;
}

pub async fn test_poll_completed_simple_image_batch_inference_request_with_provider_and_ids(
    provider: E2ETestProvider,
    ids: InsertedFakeDataIds,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    // Poll by inference_id
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: Some(ids.inference_id),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_simple_image_inference_response(inferences_json[0].clone(), None, &provider, true, false)
        .await;

    // Poll by `batch_id`
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_simple_image_inference_response(inferences_json[0].clone(), None, &provider, true, false)
        .await;

    check_batch_request_status(&conn, ids.batch_id, &provider, BatchStatus::Completed).await;
}

pub async fn test_start_inference_params_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_ids": [episode_id],
        "inputs":
            [{
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": [{"type": "raw_text", "value": "What is the name of the capital city of Japan?"}]
                }
            ]}],
        "params": {
            "chat_completion": {
                "temperature": [0.9],
                "seed": [1337],
                "max_tokens": [120],
                "top_p": [0.9],
                "presence_penalty": [0.1],
                "frequency_penalty": [0.2],
            },
            "fake_variant_type": {
                "temperature": [0.8],
                "seed": [7331],
                "max_tokens": [80],
            }
        },
        "tags": [{
            "test_type": "batch_inference_params"
        }],
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Wait for trailing writes from the API to be visible
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let results = conn
        .get_batch_model_inferences(batch_id, &[inference_id])
        .await
        .expect("Failed to get batch model inferences");
    assert_eq!(results.len(), 1);
    let result = &results[0];

    println!("BatchModelInference: {result:#?}");

    assert_eq!(result.inference_id, inference_id);
    assert_eq!(result.batch_id, batch_id);
    assert_eq!(result.function_name.as_ref(), "basic_test");
    assert_eq!(result.variant_name.as_ref(), provider.variant_name);
    assert_eq!(result.episode_id, episode_id);

    let input = serde_json::to_value(result.input.as_ref().unwrap()).unwrap();
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

    let input_messages = result.input_messages.as_ref().unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![
            "What is the name of the capital city of Japan?"
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, &expected_input_messages);

    assert_eq!(
        result.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named Dr. Mehta"
    );

    assert!(result.tool_params.is_none());

    let inference_params = result.inference_params.as_ref().unwrap();
    assert_eq!(inference_params.chat_completion.temperature.unwrap(), 0.9);
    assert_eq!(inference_params.chat_completion.seed.unwrap(), 1337);
    assert_eq!(inference_params.chat_completion.max_tokens.unwrap(), 120);
    assert_eq!(inference_params.chat_completion.top_p.unwrap(), 0.9);
    assert_eq!(
        inference_params.chat_completion.presence_penalty.unwrap(),
        0.1
    );
    assert_eq!(
        inference_params.chat_completion.frequency_penalty.unwrap(),
        0.2
    );

    assert_eq!(result.model_name.as_ref(), provider.model_name);
    assert_eq!(
        result.model_provider_name.as_ref(),
        provider.model_provider_name
    );
    assert!(result.output_schema.is_none());

    let raw_request = result.raw_request.as_ref().unwrap();
    assert!(!raw_request.is_empty());

    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Pending).await;
}

/// If there is a pending batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this will poll the batch inference and check that the response is correct.
///
/// This test polls by `batch_id` then by inference id.
pub async fn test_poll_existing_inference_params_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "basic_test";
    let latest_pending_batch_inference = get_latest_batch_inference(
        &conn,
        function_name,
        &provider.variant_name,
        "pending",
        Some(HashMap::from([(
            "test_type".to_string(),
            "batch_inference_params".to_string(),
        )])),
    )
    .await;
    let batch_inference = match latest_pending_batch_inference {
        None => return, // No pending batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    let batch_id = batch_inference.batch_id;
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => return,
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    // Check the response from polling by `batch_id`
    check_inference_params_response(inferences_json[0].clone(), &provider, None, true).await;

    // Check the response from polling by inference_id
    let inference_id = inferences_json[0]
        .get("inference_id")
        .unwrap()
        .as_str()
        .unwrap();
    let batch_id = batch_inference.batch_id;
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: Some(Uuid::parse_str(inference_id).unwrap()),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");

    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    check_inference_params_response(inferences_json[0].clone(), &provider, None, true).await;
    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Completed).await;
}

/// If there is a completed batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this test will create fake pending data that uses the same API params but with new IDs
/// and then poll the batch inference and check that the response is correct.
///
/// This way the gateway will actually poll the inference data from the inference provider.
///
/// This test polls by `inference_id` then by `batch_id`.
pub async fn test_poll_completed_inference_params_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "basic_test";
    let latest_pending_batch_inference = insert_fake_pending_batch_inference_data(
        &conn,
        function_name,
        &provider.variant_name,
        Some(HashMap::from([(
            "test_type".to_string(),
            "batch_inference_params".to_string(),
        )])),
    )
    .await;
    let ids = match latest_pending_batch_inference {
        None => return, // No completed batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    sleep(Duration::from_millis(200)).await;
    test_poll_completed_inference_params_batch_inference_request_with_provider_and_ids(
        provider, ids,
    )
    .await;
}

pub async fn test_poll_completed_inference_params_batch_inference_request_with_provider_and_ids(
    provider: E2ETestProvider,
    ids: InsertedFakeDataIds,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    // Poll by inference_id
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: Some(ids.inference_id),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_inference_params_response(inferences_json[0].clone(), &provider, None, true).await;

    // Poll by `batch_id`
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_inference_params_response(inferences_json[0].clone(), &provider, None, true).await;
    check_batch_request_status(&conn, ids.batch_id, &provider, BatchStatus::Completed).await;
}

/// Tests that the tool use works as expected in a batch inference request.
/// Each element is a different test case from the e2e test suite for the synchronous API.
pub async fn test_tool_use_batch_inference_request_with_provider(provider: E2ETestProvider) {
    let mut episode_ids = Vec::new();
    for _ in 0..5 {
        episode_ids.push(Uuid::now_v7());
    }

    let payload = json!({
        "function_name": "weather_helper",
        "episode_ids": episode_ids,
        "inputs":
            [{
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
            {
                "system": {"assistant_name": "Dr. Mehta"},
                "messages": [
                 {
                     "role": "user",
                     "content": "What is your name?"
                 }
             ]},
             {
                "system": { "assistant_name": "Dr. Mehta" },
                "messages": [
                    {
                        "role": "user",
                        "content": "What is your name?"
                    }
                ]
            },
            {
                "system": {"assistant_name": "Dr. Mehta"},
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    }
                ]},
                {
                    "system": {"assistant_name": "Dr. Mehta"},
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                        }
                    ]},
             ],
        "tool_choice": [null, null, "required", "none", {"specific": "self_destruct"}],
        "additional_tools": [null, null, null, null, [{
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
        }]],
        "allowed_tools": [null, null, null, null, null],
        "variant_name": provider.variant_name,
        "tags": [{"test_type": "auto_used"}, {"test_type": "auto_unused"}, {"test_type": "required"}, {"test_type": "none"}, {"test_type": "specific"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    let status = response.status();
    let response_text = response.text().await.unwrap();

    println!("API response: {response_text:#?}");
    assert_eq!(status, StatusCode::OK);
    let response_json: Value = serde_json::from_str(&response_text).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 5);

    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    // Parse all 5 inference IDs
    let inference_ids: Vec<Uuid> = inference_ids
        .iter()
        .map(|id| Uuid::parse_str(id.as_str().unwrap()).unwrap())
        .collect();
    assert_eq!(inference_ids.len(), 5);
    let mut inference_id_to_index: HashMap<Uuid, usize> =
        inference_ids.iter().copied().zip(0..).collect();

    let response_episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(response_episode_ids.len(), 5);

    // Parse and verify all 5 episode IDs match
    let response_episode_ids: Vec<Uuid> = response_episode_ids
        .iter()
        .map(|id| Uuid::parse_str(id.as_str().unwrap()).unwrap())
        .collect();

    // Verify each episode ID matches the expected episode ID
    for (episode_id_response, expected_episode_id) in response_episode_ids.iter().zip(&episode_ids)
    {
        assert_eq!(episode_id_response, expected_episode_id);
    }

    // Wait for trailing writes from the API to be visible
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let correct_inputs = json!([
        {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        },
        {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "What is your name?"}]}]
        },
        {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "What is your name?"}]}]
        },
        {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]}]
        },
        {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]}]
        }
    ]);
    let expected_input_messages = [
        [StoredRequestMessage {
            role: Role::User,
            content: vec![
                "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        }],
        [StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::Text(Text {
                text: "What is your name?".to_string(),
            })],
        }],
        [StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::Text(Text {
                text: "What is your name?".to_string(),
            })],
        }],
        [StoredRequestMessage {
            role: Role::User,
            content: vec![
                "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        }],
        [StoredRequestMessage {
            role: Role::User,
            content: vec![
                "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        }]
    ];

    let expected_systems = [
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").",
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").",
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").",
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").",
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").",
    ];
    let expected_tool_params = [
        json!({
            "tool_choice": "auto",
            "parallel_tool_calls": null,
            "tools_available": [{
                "name": "get_temperature",
                "description": "Get the current temperature in a given location",
                "strict": false,
                "parameters": {
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
                }
            }]
        }),
        json!({
            "tools_available": [{
                "description": "Get the current temperature in a given location",
                "parameters": {
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
                },
                "name": "get_temperature",
                "strict": false
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": null
        }),
        json!({
            "tools_available": [{
                "description": "Get the current temperature in a given location",
                "parameters": {
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
                },
                "name": "get_temperature",
                "strict": false
            }],
            "tool_choice": "required",
            "parallel_tool_calls": null
        }),
        json!({
            "tools_available": [{
                "description": "Get the current temperature in a given location",
                "parameters": {
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
                },
                "name": "get_temperature",
                "strict": false
            }],
            "tool_choice": "none",
            "parallel_tool_calls": null
        }),
        json!({
            "tools_available": [{
                "description": "Get the current temperature in a given location",
                "parameters": {
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
                },
                "name": "get_temperature",
                "strict": false
            }, {
                "description": "Do not call this function under any circumstances.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fast": {
                            "type": "boolean",
                            "description": "Whether to use a fast method to self-destruct."
                        }
                    },
                    "required": ["fast"],
                    "additionalProperties": false
                },
                "name": "self_destruct",
                "strict": false
            }],
            "tool_choice": {
                "specific": "self_destruct"
            },
            "parallel_tool_calls": null
        }),
    ];

    let expected_max_tokens = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };

    let expected_inference_params = [
        json!({
            "chat_completion": {
                "max_tokens": expected_max_tokens,
            }
        }),
        json!({"chat_completion": {"max_tokens": expected_max_tokens}}),
        json!({"chat_completion": {"max_tokens": expected_max_tokens}}),
        json!({"chat_completion": {"max_tokens": expected_max_tokens}}),
        json!({"chat_completion": {"max_tokens": expected_max_tokens}}),
    ];

    let all_results = conn
        .get_batch_model_inferences(batch_id, &inference_ids)
        .await
        .expect("Failed to get batch model inferences");
    assert_eq!(all_results.len(), 5);

    for result in &all_results {
        let i = inference_id_to_index.remove(&result.inference_id).unwrap();
        println!("BatchModelInference (#{i}): {result:#?}");

        assert_eq!(result.batch_id, batch_id);
        assert_eq!(result.function_name.as_ref(), "weather_helper");
        assert_eq!(result.variant_name.as_ref(), provider.variant_name);
        assert_eq!(result.episode_id, episode_ids[i]);

        let input = serde_json::to_value(result.input.as_ref().unwrap()).unwrap();
        assert_eq!(input, correct_inputs[i]);

        let input_messages = result.input_messages.as_ref().unwrap();
        assert_eq!(input_messages, &expected_input_messages[i]);

        assert_eq!(result.system.as_deref().unwrap(), expected_systems[i]);

        // Compare tool_params as JSON for complex structure comparison
        let tool_params_value = serde_json::to_value(result.tool_params.as_ref().unwrap()).unwrap();
        assert_eq!(tool_params_value, expected_tool_params[i]);

        // Verify decomposed tool call storage format
        let tool_params_ref = result.tool_params.as_ref().unwrap();
        if i < 4 {
            assert!(tool_params_ref.dynamic_tools.is_empty());
        } else {
            assert_eq!(tool_params_ref.dynamic_tools.len(), 1);
            let Tool::Function(tool) = &tool_params_ref.dynamic_tools[0] else {
                panic!("Expected Function tool");
            };
            assert_eq!(tool.name, "self_destruct");
        }

        assert!(
            tool_params_ref.dynamic_provider_tools.is_empty(),
            "dynamic_provider_tools should be empty"
        );

        assert!(
            tool_params_ref.parallel_tool_calls.is_none(),
            "parallel_tool_calls should be None"
        );

        let inference_params_value =
            serde_json::to_value(result.inference_params.as_ref().unwrap()).unwrap();
        assert_eq!(inference_params_value, expected_inference_params[i]);

        assert_eq!(result.model_name.as_ref(), provider.model_name);
        assert_eq!(
            result.model_provider_name.as_ref(),
            provider.model_provider_name
        );
        assert!(result.output_schema.is_none());

        assert_eq!(
            result.tags.get("test_type").unwrap(),
            payload["tags"][i]["test_type"].as_str().unwrap()
        );

        let raw_request = result.raw_request.as_ref().unwrap();
        assert!(!raw_request.is_empty());
    }

    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Pending).await;
}

/// For a given batch id, get the tags for all inferences in the batch
/// Returns a map from inference id to tags
async fn get_tags_for_batch_inferences(
    conn: &DelegatingDatabaseConnection,
    batch_id: Uuid,
) -> Option<HashMap<Uuid, HashMap<String, String>>> {
    let batch_model_inferences = conn
        .get_batch_model_inferences(batch_id, &[])
        .await
        .expect("Failed to get batch model inferences");
    if batch_model_inferences.is_empty() {
        return None;
    }
    let mut inference_tags = HashMap::new();
    for bmi in batch_model_inferences {
        inference_tags.insert(bmi.inference_id, bmi.tags);
    }
    Some(inference_tags)
}

/// If there is a pending batch inference for the function, variant, and tags
/// that are used for the tool choice auto used tests,
/// this will poll the batch inference and check that the response is correct.
///
/// This test polls by `batch_id` then by inference id.
pub async fn test_poll_existing_tool_choice_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "weather_helper";
    let latest_pending_batch_inference = get_latest_batch_inference(
        &conn,
        function_name,
        &provider.variant_name,
        "pending",
        Some(HashMap::from([(
            "test_type".to_string(),
            "auto_used".to_string(),
        )])),
    )
    .await;
    println!("latest_pending_batch_inference: {latest_pending_batch_inference:#?}");
    let batch_inference = match latest_pending_batch_inference {
        None => return, // No pending batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    let batch_id = batch_inference.batch_id;
    let inference_tags = get_tags_for_batch_inferences(&conn, batch_id)
        .await
        .unwrap();
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => return,
        "completed" => (),
        status => panic!("Bad batch inference status: {status}"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();

    let mut test_types_seen = HashSet::new();
    for inference_json in inferences_json {
        let inference_id = inference_json
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap();
        let inference_id = Uuid::parse_str(inference_id).unwrap();
        let tags = inference_tags.get(&inference_id).unwrap();
        let test_type = tags.get("test_type").unwrap();
        match test_type.as_str() {
            "auto_used" => {
                check_tool_use_tool_choice_auto_used_inference_response(
                    inference_json.clone(),
                    &provider,
                    None,
                    true,
                )
                .await;
            }
            "auto_unused" => {
                check_tool_use_tool_choice_auto_unused_inference_response(
                    inference_json.clone(),
                    &provider,
                    None,
                    true,
                )
                .await;
            }
            "required" => {
                check_tool_use_tool_choice_required_inference_response(
                    inference_json.clone(),
                    &provider,
                    None,
                    true,
                )
                .await;
            }
            "none" => {
                check_tool_use_tool_choice_none_inference_response(
                    inference_json.clone(),
                    &provider,
                    None,
                    true,
                )
                .await;
            }
            "specific" => {
                check_tool_use_tool_choice_specific_inference_response(
                    inference_json.clone(),
                    &provider,
                    None,
                    true,
                )
                .await;
            }
            _ => panic!("Unknown test type"),
        }
        test_types_seen.insert(test_type.clone());
    }

    assert_eq!(test_types_seen.len(), 5);
    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Completed).await;
}

/// If there is a completed batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this test will create fake pending data that uses the same API params but with new IDs
/// and then poll the batch inference and check that the response is correct.
///
/// This way the gateway will actually poll the inference data from the inference provider.
///
/// This test polls by `inference_id` then by `batch_id`.
pub async fn test_poll_completed_tool_use_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "weather_helper";
    let latest_pending_batch_inference = insert_fake_pending_batch_inference_data(
        &conn,
        function_name,
        &provider.variant_name,
        Some(HashMap::from([(
            "test_type".to_string(),
            "auto_used".to_string(),
        )])),
    )
    .await;
    let ids = match latest_pending_batch_inference {
        None => return, // No completed batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    sleep(Duration::from_millis(200)).await;
    test_poll_completed_tool_use_batch_inference_request_with_provider_and_ids(provider, ids).await;
}

pub async fn test_poll_completed_tool_use_batch_inference_request_with_provider_and_ids(
    provider: E2ETestProvider,
    ids: InsertedFakeDataIds,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let batch_id = ids.batch_id;
    let inference_tags = get_tags_for_batch_inferences(&conn, batch_id)
        .await
        .unwrap();
    // Poll by `batch_id`
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => return,
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();

    let mut test_types_seen = HashSet::new();
    for inference_json in inferences_json {
        let inference_id = inference_json
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap();
        let inference_id = Uuid::parse_str(inference_id).unwrap();
        let tags = inference_tags.get(&inference_id).unwrap();
        // `insert_fake_pending_batch_inference_data` wraps the original tags to avoid clashing
        // with other tests
        let original_tags: HashMap<String, String> = serde_json::from_str(
            tags.get("fake_pending")
                .expect("Missing 'fake_pending' tag"),
        )
        .unwrap();
        let test_type = original_tags.get("test_type").unwrap();
        match test_type.as_str() {
            "auto_used" => {
                check_tool_use_tool_choice_auto_used_inference_response(
                    inference_json.clone(),
                    &provider,
                    None,
                    true,
                )
                .await;
            }
            "auto_unused" => {
                check_tool_use_tool_choice_auto_unused_inference_response(
                    inference_json.clone(),
                    &provider,
                    None,
                    true,
                )
                .await;
            }
            "required" => {
                check_tool_use_tool_choice_required_inference_response(
                    inference_json.clone(),
                    &provider,
                    None,
                    true,
                )
                .await;
            }
            "none" => {
                check_tool_use_tool_choice_none_inference_response(
                    inference_json.clone(),
                    &provider,
                    None,
                    true,
                )
                .await;
            }
            "specific" => {
                check_tool_use_tool_choice_specific_inference_response(
                    inference_json.clone(),
                    &provider,
                    None,
                    true,
                )
                .await;
            }
            _ => panic!("Unknown test type"),
        }
        test_types_seen.insert(test_type.clone());
    }

    assert_eq!(test_types_seen.len(), 5);
    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Completed).await;
}

pub async fn test_allowed_tools_batch_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_ids": [episode_id],
        "inputs":[{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."
                }
            ]}],
        "tool_choice": ["required"],
        "allowed_tools": [["get_humidity"]],
        "variant_name": provider.variant_name,
        "tags": [{"test_type": "allowed_tools"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Wait for trailing writes from the API to be visible
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let results = conn
        .get_batch_model_inferences(batch_id, &[inference_id])
        .await
        .expect("Failed to get batch model inferences");
    assert_eq!(results.len(), 1);
    let result = &results[0];

    println!("BatchModelInference: {result:#?}");

    assert_eq!(result.inference_id, inference_id);
    assert_eq!(result.batch_id, batch_id);
    assert_eq!(result.function_name.as_ref(), "basic_test");
    assert_eq!(result.variant_name.as_ref(), provider.variant_name);
    assert_eq!(result.episode_id, episode_id);

    let input = serde_json::to_value(result.input.as_ref().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let input_messages = result.input_messages.as_ref().unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec!["What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, &expected_input_messages);

    assert_eq!(
        result.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named Dr. Mehta"
    );

    let tool_params_value = serde_json::to_value(result.tool_params.as_ref().unwrap()).unwrap();
    println!("Tool params: {tool_params_value:#?}");
    let expected_tool_params = json!({
        "tools_available": [{
            "description": "Get the current humidity in a given location",
            "parameters": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the humidity for (e.g. \"New York\")"
                    }
                },
                "required": ["location"],
                "additionalProperties": false
            },
            "name": "get_humidity",
            "strict": false
        }],
        "tool_choice": "required",
        "parallel_tool_calls": null
    });
    assert_eq!(tool_params_value, expected_tool_params);

    let inference_params = result.inference_params.as_ref().unwrap();
    let expected_max_tokens: u32 = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };
    assert_eq!(
        inference_params.chat_completion.max_tokens.unwrap(),
        expected_max_tokens
    );

    assert_eq!(result.model_name.as_ref(), provider.model_name);
    assert_eq!(
        result.model_provider_name.as_ref(),
        provider.model_provider_name
    );
    assert!(result.output_schema.is_none());

    let raw_request = result.raw_request.as_ref().unwrap();
    assert!(!raw_request.is_empty());

    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Pending).await;
}

/// If there is a pending batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this will poll the batch inference and check that the response is correct.
///
/// This test polls by `batch_id` then by inference id.
pub async fn test_poll_existing_allowed_tools_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "basic_test";
    let latest_pending_batch_inference = get_latest_batch_inference(
        &conn,
        function_name,
        &provider.variant_name,
        "pending",
        Some(HashMap::from([(
            "test_type".to_string(),
            "allowed_tools".to_string(),
        )])),
    )
    .await;
    let batch_inference = match latest_pending_batch_inference {
        None => return, // No pending batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    let batch_id = batch_inference.batch_id;
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => return,
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    // Check the response from polling by `batch_id`
    check_tool_use_tool_choice_allowed_tools_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
    )
    .await;

    // Check the response from polling by inference_id
    let inference_id = inferences_json[0]
        .get("inference_id")
        .unwrap()
        .as_str()
        .unwrap();
    let url = get_poll_batch_inference_url(PollPathParams {
        inference_id: Some(Uuid::parse_str(inference_id).unwrap()),
        batch_id,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");

    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    check_tool_use_tool_choice_allowed_tools_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
    )
    .await;
    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Completed).await;
}

/// If there is a completed batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this test will create fake pending data that uses the same API params but with new IDs
/// and then poll the batch inference and check that the response is correct.
///
/// This way the gateway will actually poll the inference data from the inference provider.
///
/// This test polls by `inference_id` then by `batch_id`.
pub async fn test_poll_completed_allowed_tools_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "basic_test";
    let latest_pending_batch_inference = insert_fake_pending_batch_inference_data(
        &conn,
        function_name,
        &provider.variant_name,
        Some(HashMap::from([(
            "test_type".to_string(),
            "allowed_tools".to_string(),
        )])),
    )
    .await;
    let ids = match latest_pending_batch_inference {
        None => return, // No completed batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    sleep(Duration::from_millis(200)).await;
    test_poll_completed_allowed_tools_batch_inference_request_with_provider_and_ids(provider, ids)
        .await;
}

pub async fn test_poll_completed_allowed_tools_batch_inference_request_with_provider_and_ids(
    provider: E2ETestProvider,
    ids: InsertedFakeDataIds,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    // Poll by inference_id
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: Some(ids.inference_id),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_tool_use_tool_choice_allowed_tools_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
    )
    .await;

    // Poll by `batch_id`
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_tool_use_tool_choice_allowed_tools_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
    )
    .await;

    check_batch_request_status(&conn, ids.batch_id, &provider, BatchStatus::Completed).await;
}

pub async fn test_multi_turn_parallel_tool_use_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!(
        {
      "function_name": "weather_helper_parallel",
      "episode_ids": [episode_id],
      "inputs": [{
        "system": {
          "assistant_name": "Dr. Mehta"
        },
        "messages": [
          {
            "role": "user",
            "content": "What is the weather like in Tokyo (in Fahrenheit)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
          },
          {
            "role": "assistant",
            "content": [
              {
                "type": "tool_call",
                "arguments": {"location":"Tokyo","units":"fahrenheit"},
                "id": "1234",
                "name": "get_temperature"
              },
              {
                "type": "tool_call",
                "arguments": {"location":"Tokyo"},
                "id": "5678",
                "name": "get_humidity"
              }
            ]
          },
          {
            "role": "user",
            "content": [
              {
                "type": "tool_result",
                "id": "1234",
                "name": "get_temperature",
                "result": "70"
              },
              {
                "type": "tool_result",
                "id": "5678",
                "name": "get_humidity",
                "result": "30"
              }
            ]
          }
        ]
      }],
      "parallel_tool_calls": [true],
      "variant_name": provider.variant_name,
      "tags": [{"test": "multi_turn_parallel_tool_use"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Wait for trailing writes from the API to be visible
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let results = conn
        .get_batch_model_inferences(batch_id, &[inference_id])
        .await
        .expect("Failed to get batch model inferences");
    assert_eq!(results.len(), 1);
    let result = &results[0];

    println!("BatchModelInference: {result:#?}");

    assert_eq!(result.inference_id, inference_id);
    assert_eq!(result.batch_id, batch_id);
    assert_eq!(
        result.function_name.as_ref(),
        payload["function_name"].as_str().unwrap()
    );
    assert_eq!(result.variant_name.as_ref(), provider.variant_name);
    assert_eq!(result.episode_id, episode_id);

    let input = serde_json::to_value(result.input.as_ref().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Fahrenheit)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."}]
            },
            {
                "role": "assistant",
                "content": [
                  {
                    "type": "tool_call",
                    "arguments": "{\"location\":\"Tokyo\",\"units\":\"fahrenheit\"}",
                    "id": "1234",
                    "name": "get_temperature"
                  },
                  {
                    "type": "tool_call",
                    "arguments": "{\"location\":\"Tokyo\"}",
                    "id": "5678",
                    "name": "get_humidity"
                  }
                ]
              },
            {
                "role": "user",
                "content": [
                  {
                    "type": "tool_result",
                    "id": "1234",
                    "name": "get_temperature",
                    "result": "70"
                  },
                  {
                    "type": "tool_result",
                    "id": "5678",
                    "name": "get_humidity",
                    "result": "30"
                  }
                ]
              }
        ],
    });
    assert_eq!(input, correct_input);

    let input_messages = result.input_messages.as_ref().unwrap();
    let expected_input_messages = vec![
        StoredRequestMessage {
            role: Role::User,
            content: vec![
                StoredContentBlock::Text(Text {
                    text: "What is the weather like in Tokyo (in Fahrenheit)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions.".to_string(),
                }),
            ],
        },
        StoredRequestMessage {
            role: Role::Assistant,
            content: vec![StoredContentBlock::ToolCall(ToolCall {
                name: "get_temperature".to_string(),
                arguments: "{\"location\":\"Tokyo\",\"units\":\"fahrenheit\"}".to_string(),
                id: "1234".to_string(),
            }), StoredContentBlock::ToolCall(ToolCall {
                name: "get_humidity".to_string(),
                arguments: "{\"location\":\"Tokyo\"}".to_string(),
                id: "5678".to_string(),
            })],
        },
        StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::ToolResult(ToolResult {
                name: "get_temperature".to_string(),
                result: "70".to_string(),
                id: "1234".to_string(),
            }), StoredContentBlock::ToolResult(ToolResult {
                name: "get_humidity".to_string(),
                result: "30".to_string(),
                id: "5678".to_string(),
            })],
        },
    ];
    assert_eq!(input_messages, &expected_input_messages);

    assert_eq!(
        result.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with two tool calls. Use BOTH the \"get_temperature\" and \"get_humidity\" tools.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit with 50% humidity.\")."
    );

    let tool_params_value = serde_json::to_value(result.tool_params.as_ref().unwrap()).unwrap();
    let expected_tool_params = json!({
        "tools_available":[
            {"description":"Get the current temperature in a given location","parameters":{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{"location":{"type":"string","description":"The location to get the temperature for (e.g. \"New York\")"},"units":{"type":"string","description":"The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")","enum":["fahrenheit","celsius"]}},"required":["location"],"additionalProperties":false},"name":"get_temperature","strict":false},
            {"description": "Get the current humidity in a given location", "parameters": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "properties": {"location": {"type": "string", "description": "The location to get the humidity for (e.g. \"New York\")"}}, "required": ["location"], "additionalProperties": false}, "name": "get_humidity", "strict": false}
        ],
        "tool_choice":"auto",
        "parallel_tool_calls": true
    });
    assert_eq!(tool_params_value, expected_tool_params);

    let inference_params = result.inference_params.as_ref().unwrap();
    assert!(inference_params.chat_completion.temperature.is_none());
    assert!(inference_params.chat_completion.seed.is_none());
    let expected_max_tokens: u32 = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };
    assert_eq!(
        inference_params.chat_completion.max_tokens.unwrap(),
        expected_max_tokens
    );

    assert_eq!(result.model_name.as_ref(), provider.model_name);

    assert_eq!(
        result.model_provider_name.as_ref(),
        provider.model_provider_name
    );

    assert!(result.output_schema.is_none());

    assert_eq!(
        result.tags.get("test").unwrap().as_str(),
        "multi_turn_parallel_tool_use"
    );

    let raw_request = result.raw_request.as_ref().unwrap();
    assert!(!raw_request.is_empty());

    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Pending).await;
}

pub async fn test_tool_multi_turn_batch_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
       "function_name": "weather_helper",
        "episode_ids": [episode_id],
        "inputs":[{
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
                            "result": "70"
                        }
                    ]
                }
            ]}],
        "variant_name": provider.variant_name,
        "tags": [{"test": "multi_turn"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Wait for trailing writes from the API to be visible
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let results = conn
        .get_batch_model_inferences(batch_id, &[inference_id])
        .await
        .expect("Failed to get batch model inferences");
    assert_eq!(results.len(), 1);
    let result = &results[0];

    println!("BatchModelInference: {result:#?}");

    assert_eq!(result.inference_id, inference_id);
    assert_eq!(result.batch_id, batch_id);
    assert_eq!(
        result.function_name.as_ref(),
        payload["function_name"].as_str().unwrap()
    );
    assert_eq!(result.variant_name.as_ref(), provider.variant_name);
    assert_eq!(result.episode_id, episode_id);

    let input = serde_json::to_value(result.input.as_ref().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "name": "get_temperature", "arguments": "{\"location\": \"Tokyo\"}", "id": "123456789"}]
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "name": "get_temperature", "result": "70", "id": "123456789"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let input_messages = result.input_messages.as_ref().unwrap();
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
                name: "get_temperature".to_string(),
                arguments: "{\"location\": \"Tokyo\"}".to_string(),
                id: "123456789".to_string(),
            })],
        },
        StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::ToolResult(ToolResult {
                name: "get_temperature".to_string(),
                result: "70".to_string(),
                id: "123456789".to_string(),
            })],
        },
    ];
    assert_eq!(input_messages, &expected_input_messages);

    assert_eq!(
        result.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );

    let tool_params_value = serde_json::to_value(result.tool_params.as_ref().unwrap()).unwrap();
    let expected_tool_params = json!({"tools_available":[{"description":"Get the current temperature in a given location","parameters":{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{"location":{"type":"string","description":"The location to get the temperature for (e.g. \"New York\")"},"units":{"type":"string","description":"The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")","enum":["fahrenheit","celsius"]}},"required":["location"],"additionalProperties":false},"name":"get_temperature","strict":false}],"tool_choice":"auto","parallel_tool_calls":null});
    assert_eq!(tool_params_value, expected_tool_params);

    let inference_params = result.inference_params.as_ref().unwrap();
    assert!(inference_params.chat_completion.temperature.is_none());
    assert!(inference_params.chat_completion.seed.is_none());
    let expected_max_tokens: u32 = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };
    assert_eq!(
        inference_params.chat_completion.max_tokens.unwrap(),
        expected_max_tokens
    );

    assert_eq!(result.model_name.as_ref(), provider.model_name);

    assert_eq!(
        result.model_provider_name.as_ref(),
        provider.model_provider_name
    );

    assert!(result.output_schema.is_none());

    assert_eq!(result.tags.get("test").unwrap().as_str(), "multi_turn");

    let raw_request = result.raw_request.as_ref().unwrap();
    assert!(!raw_request.is_empty());

    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Pending).await;
}

/// If there is a pending batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this will poll the batch inference and check that the response is correct.
///
/// This test polls by `batch_id` then by inference id.
pub async fn test_poll_existing_multi_turn_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "weather_helper";
    let latest_pending_batch_inference = get_latest_batch_inference(
        &conn,
        function_name,
        &provider.variant_name,
        "pending",
        Some(HashMap::from([(
            "test_type".to_string(),
            "multi_turn".to_string(),
        )])),
    )
    .await;
    let batch_inference = match latest_pending_batch_inference {
        None => return, // No pending batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    let batch_id = batch_inference.batch_id;
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => return,
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    // Check the response from polling by `batch_id`
    check_tool_use_multi_turn_inference_response(inferences_json[0].clone(), &provider, None, true)
        .await;

    // Check the response from polling by inference_id
    let inference_id = inferences_json[0]
        .get("inference_id")
        .unwrap()
        .as_str()
        .unwrap();
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: Some(Uuid::parse_str(inference_id).unwrap()),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");

    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    check_tool_use_multi_turn_inference_response(inferences_json[0].clone(), &provider, None, true)
        .await;
    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Completed).await;
}

pub async fn test_poll_existing_multi_turn_parallel_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "weather_helper_parallel";
    let latest_pending_batch_inference = get_latest_batch_inference(
        &conn,
        function_name,
        &provider.variant_name,
        "pending",
        Some(HashMap::from([(
            "test".to_string(),
            "multi_turn_parallel_tool_use".to_string(),
        )])),
    )
    .await;
    let batch_inference = match latest_pending_batch_inference {
        None => return, // No pending batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    let batch_id = batch_inference.batch_id;
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => return,
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    // Check the response from polling by `batch_id`
    check_multi_turn_parallel_tool_use_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
    )
    .await;

    // Check the response from polling by inference_id
    let inference_id = inferences_json[0]
        .get("inference_id")
        .unwrap()
        .as_str()
        .unwrap();
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: Some(Uuid::parse_str(inference_id).unwrap()),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");

    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    check_multi_turn_parallel_tool_use_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
    )
    .await;
    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Completed).await;
}

pub async fn test_poll_completed_multi_turn_parallel_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "weather_helper_parallel";
    let latest_pending_batch_inference = insert_fake_pending_batch_inference_data(
        &conn,
        function_name,
        &provider.variant_name,
        Some(HashMap::from([(
            "test".to_string(),
            "multi_turn_parallel_tool_use".to_string(),
        )])),
    )
    .await;
    let ids = match latest_pending_batch_inference {
        None => return, // No completed batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    sleep(Duration::from_millis(200)).await;
    test_poll_completed_multi_turn_parallel_batch_inference_request_with_provider_and_ids(
        provider, ids,
    )
    .await;
}

pub async fn test_poll_completed_multi_turn_parallel_batch_inference_request_with_provider_and_ids(
    provider: E2ETestProvider,
    ids: InsertedFakeDataIds,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    // Poll by inference_id
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: Some(ids.inference_id),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_multi_turn_parallel_tool_use_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
    )
    .await;

    // Poll by `batch_id`
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_multi_turn_parallel_tool_use_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
    )
    .await;

    check_batch_request_status(&conn, ids.batch_id, &provider, BatchStatus::Completed).await;
}

/// If there is a completed batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this test will create fake pending data that uses the same API params but with new IDs
/// and then poll the batch inference and check that the response is correct.
///
/// This way the gateway will actually poll the inference data from the inference provider.
///
/// This test polls by `inference_id` then by `batch_id`.
pub async fn test_poll_completed_multi_turn_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "weather_helper";
    let latest_pending_batch_inference = insert_fake_pending_batch_inference_data(
        &conn,
        function_name,
        &provider.variant_name,
        Some(HashMap::from([(
            "test_type".to_string(),
            "multi_turn".to_string(),
        )])),
    )
    .await;
    let ids = match latest_pending_batch_inference {
        None => return, // No completed batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    sleep(Duration::from_millis(200)).await;
    test_poll_completed_multi_turn_batch_inference_request_with_provider_and_ids(provider, ids)
        .await;
}

pub async fn test_poll_completed_multi_turn_batch_inference_request_with_provider_and_ids(
    provider: E2ETestProvider,
    ids: InsertedFakeDataIds,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    // Poll by inference_id
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: Some(ids.inference_id),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_tool_use_multi_turn_inference_response(inferences_json[0].clone(), &provider, None, true)
        .await;

    // Poll by `batch_id`
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_tool_use_multi_turn_inference_response(inferences_json[0].clone(), &provider, None, true)
        .await;

    check_batch_request_status(&conn, ids.batch_id, &provider, BatchStatus::Completed).await;
}

pub async fn test_dynamic_tool_use_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_ids": [episode_id],
        "inputs":[{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."
                }
            ]}],
        "additional_tools": [[
            {
                "name": "get_temperature",
                "description": "Get the current temperature in a given location",
                "parameters": {
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
                }
            }
        ]],
        "variant_name": provider.variant_name,
        "tags": [{"test_type": "dynamic_tool_use"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Wait for trailing writes from the API to be visible
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let results = conn
        .get_batch_model_inferences(batch_id, &[inference_id])
        .await
        .expect("Failed to get batch model inferences");
    assert_eq!(results.len(), 1);
    let result = &results[0];

    println!("BatchModelInference: {result:#?}");

    assert_eq!(result.inference_id, inference_id);
    assert_eq!(result.batch_id, batch_id);
    assert_eq!(
        result.function_name.as_ref(),
        payload["function_name"].as_str().unwrap()
    );
    assert_eq!(result.variant_name.as_ref(), provider.variant_name);
    assert_eq!(result.episode_id, episode_id);

    let input = serde_json::to_value(result.input.as_ref().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let input_messages = result.input_messages.as_ref().unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function.".to_string(),
        })],
    }];
    assert_eq!(input_messages, &expected_input_messages);

    assert_eq!(
        result.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named Dr. Mehta"
    );

    let tool_params_value = serde_json::to_value(result.tool_params.as_ref().unwrap()).unwrap();
    let expected_tool_params = json!({
        "tools_available": [{
            "description": "Get the current temperature in a given location",
            "parameters": {
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
            },
            "name": "get_temperature",
            "strict": false
        }],
        "tool_choice": "auto",
        "parallel_tool_calls": null
    });
    assert_eq!(tool_params_value, expected_tool_params);

    let inference_params = result.inference_params.as_ref().unwrap();
    let expected_max_tokens: u32 = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };
    assert_eq!(
        inference_params.chat_completion.max_tokens.unwrap(),
        expected_max_tokens
    );

    assert_eq!(result.model_name.as_ref(), provider.model_name);

    assert_eq!(
        result.model_provider_name.as_ref(),
        provider.model_provider_name
    );

    assert!(result.output_schema.is_none());

    assert!(!result.tags.is_empty());

    let raw_request = result.raw_request.as_ref().unwrap();
    assert!(!raw_request.is_empty());

    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Pending).await;
}

/// If there is a pending batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this will poll the batch inference and check that the response is correct.
///
/// This test polls by `batch_id` then by inference id.
pub async fn test_poll_existing_dynamic_tool_use_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "basic_test";
    let latest_pending_batch_inference = get_latest_batch_inference(
        &conn,
        function_name,
        &provider.variant_name,
        "pending",
        Some(HashMap::from([(
            "test_type".to_string(),
            "dynamic_tool_use".to_string(),
        )])),
    )
    .await;
    let batch_inference = match latest_pending_batch_inference {
        None => return, // No pending batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    let batch_id = batch_inference.batch_id;
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => return,
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    // Check the response from polling by `batch_id`
    check_dynamic_tool_use_inference_response(inferences_json[0].clone(), &provider, None, true)
        .await;

    // Check the response from polling by inference_id
    let inference_id = inferences_json[0]
        .get("inference_id")
        .unwrap()
        .as_str()
        .unwrap();
    let url = get_poll_batch_inference_url(PollPathParams {
        inference_id: Some(Uuid::parse_str(inference_id).unwrap()),
        batch_id,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");

    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    check_dynamic_tool_use_inference_response(inferences_json[0].clone(), &provider, None, true)
        .await;
    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Completed).await;
}

/// If there is a completed batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this test will create fake pending data that uses the same API params but with new IDs
/// and then poll the batch inference and check that the response is correct.
///
/// This way the gateway will actually poll the inference data from the inference provider.
///
/// This test polls by `inference_id` then by `batch_id`.
pub async fn test_poll_completed_dynamic_tool_use_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "basic_test";
    let latest_pending_batch_inference = insert_fake_pending_batch_inference_data(
        &conn,
        function_name,
        &provider.variant_name,
        Some(HashMap::from([(
            "test_type".to_string(),
            "dynamic_tool_use".to_string(),
        )])),
    )
    .await;
    let ids = match latest_pending_batch_inference {
        None => return, // No completed batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    sleep(Duration::from_millis(200)).await;
    test_poll_completed_dynamic_tool_use_batch_inference_request_with_provider_and_ids(
        provider, ids,
    )
    .await;
}

pub async fn test_poll_completed_dynamic_tool_use_batch_inference_request_with_provider_and_ids(
    provider: E2ETestProvider,
    ids: InsertedFakeDataIds,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    // Poll by inference_id
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: Some(ids.inference_id),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_dynamic_tool_use_inference_response(inferences_json[0].clone(), &provider, None, true)
        .await;

    // Poll by `batch_id`
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_dynamic_tool_use_inference_response(inferences_json[0].clone(), &provider, None, true)
        .await;

    check_batch_request_status(&conn, ids.batch_id, &provider, BatchStatus::Completed).await;
}

pub async fn test_parallel_tool_use_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper_parallel",
        "episode_ids": [episode_id],
        "inputs":[{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
                }
            ]}],
        "parallel_tool_calls": [true],
        "variant_name": provider.variant_name,
        "tags": [{"test_type": "parallel_tool_use"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Wait for trailing writes from the API to be visible
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let results = conn
        .get_batch_model_inferences(batch_id, &[inference_id])
        .await
        .expect("Failed to get batch model inferences");
    assert_eq!(results.len(), 1);
    let result = &results[0];

    println!("BatchModelInference: {result:#?}");

    assert_eq!(result.inference_id, inference_id);
    assert_eq!(result.batch_id, batch_id);
    assert_eq!(
        result.function_name.as_ref(),
        payload["function_name"].as_str().unwrap()
    );
    assert_eq!(result.variant_name.as_ref(), provider.variant_name);
    assert_eq!(result.episode_id, episode_id);

    let input = serde_json::to_value(result.input.as_ref().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let input_messages = result.input_messages.as_ref().unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions.".to_string(),
        })],
    }];
    assert_eq!(input_messages, &expected_input_messages);

    assert_eq!(
        result.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with two tool calls. Use BOTH the \"get_temperature\" and \"get_humidity\" tools.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit with 50% humidity.\")."
    );

    let tool_params_value = serde_json::to_value(result.tool_params.as_ref().unwrap()).unwrap();
    let expected_tool_params = json!({
        "tools_available": [
            {
                "description": "Get the current temperature in a given location",
                "parameters": {
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
                },
                "name": "get_temperature",
                "strict": false
            },
            {
                "description": "Get the current humidity in a given location",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the humidity for (e.g. \"New York\")"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                },
                "name": "get_humidity",
                "strict": false
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": true
    });
    assert_eq!(tool_params_value, expected_tool_params);

    let inference_params = result.inference_params.as_ref().unwrap();
    assert_eq!(inference_params.chat_completion.max_tokens.unwrap(), 100);

    assert_eq!(result.model_name.as_ref(), provider.model_name);

    assert_eq!(
        result.model_provider_name.as_ref(),
        provider.model_provider_name
    );

    assert!(result.output_schema.is_none());

    assert!(!result.tags.is_empty());

    let raw_request = result.raw_request.as_ref().unwrap();
    assert!(!raw_request.is_empty());

    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Pending).await;
}

/// If there is a pending batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this will poll the batch inference and check that the response is correct.
///
/// This test polls by `batch_id` then by inference id.
pub async fn test_poll_existing_parallel_tool_use_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "weather_helper_parallel";
    let latest_pending_batch_inference = get_latest_batch_inference(
        &conn,
        function_name,
        &provider.variant_name,
        "pending",
        Some(HashMap::from([(
            "test_type".to_string(),
            "parallel_tool_use".to_string(),
        )])),
    )
    .await;
    let batch_inference = match latest_pending_batch_inference {
        None => return, // No pending batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    let batch_id = batch_inference.batch_id;
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => return,
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    // Check the response from polling by `batch_id`
    check_parallel_tool_use_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
        true.into(),
    )
    .await;

    // Check the response from polling by inference_id
    let inference_id = inferences_json[0]
        .get("inference_id")
        .unwrap()
        .as_str()
        .unwrap();
    // Poll by inference_id
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: Some(Uuid::parse_str(inference_id).unwrap()),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");

    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    check_parallel_tool_use_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
        true.into(),
    )
    .await;
    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Completed).await;
}

/// If there is a completed batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this test will create fake pending data that uses the same API params but with new IDs
/// and then poll the batch inference and check that the response is correct.
///
/// This way the gateway will actually poll the inference data from the inference provider.
///
/// This test polls by `inference_id` then by `batch_id`.
pub async fn test_poll_completed_parallel_tool_use_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "weather_helper_parallel";
    let latest_pending_batch_inference = insert_fake_pending_batch_inference_data(
        &conn,
        function_name,
        &provider.variant_name,
        Some(HashMap::from([(
            "test_type".to_string(),
            "parallel_tool_use".to_string(),
        )])),
    )
    .await;
    let ids = match latest_pending_batch_inference {
        None => return, // No completed batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    sleep(Duration::from_millis(200)).await;
    test_poll_completed_parallel_tool_use_batch_inference_request_with_provider_and_ids(
        provider, ids,
    )
    .await;
}

pub async fn test_poll_completed_parallel_tool_use_batch_inference_request_with_provider_and_ids(
    provider: E2ETestProvider,
    ids: InsertedFakeDataIds,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    // Poll by inference_id
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: Some(ids.inference_id),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_parallel_tool_use_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
        true.into(),
    )
    .await;

    // Poll by `batch_id`
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_parallel_tool_use_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
        true.into(),
    )
    .await;

    check_batch_request_status(&conn, ids.batch_id, &provider, BatchStatus::Completed).await;
}

pub async fn test_json_mode_batch_inference_request_with_provider(provider: E2ETestProvider) {
    if provider.variant_name.ends_with("cot") {
        // Don't test chain of thought variants with batch mode
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": provider.variant_name,
        "episode_ids": [episode_id],
        "inputs": [{
            "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]}],
        "tags": [{"test_type": "json_mode_v2"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    let response_status = response.status();
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    // Check if the API response is fine
    assert_eq!(response_status, StatusCode::OK);

    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Wait for trailing writes from the API to be visible
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let results = conn
        .get_batch_model_inferences(batch_id, &[inference_id])
        .await
        .expect("Failed to get batch model inferences");
    assert_eq!(results.len(), 1);
    let result = &results[0];

    println!("BatchModelInference: {result:#?}");

    assert_eq!(result.inference_id, inference_id);
    assert_eq!(result.batch_id, batch_id);
    assert_eq!(
        result.function_name.as_ref(),
        payload["function_name"].as_str().unwrap()
    );
    assert_eq!(result.variant_name.as_ref(), provider.variant_name);
    assert_eq!(result.episode_id, episode_id);

    let input = serde_json::to_value(result.input.as_ref().unwrap()).unwrap();
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

    let input_messages = result.input_messages.as_ref().unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![
            "What is the name of the capital city of Japan?"
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, &expected_input_messages);

    assert_eq!(
        result.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}"
    );

    assert!(result.tool_params.is_none());
    let inference_params = result.inference_params.as_ref().unwrap();
    let expected_max_tokens: u32 = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };
    assert_eq!(
        inference_params.chat_completion.max_tokens.unwrap(),
        expected_max_tokens
    );

    assert_eq!(result.model_name.as_ref(), provider.model_name);

    assert_eq!(
        result.model_provider_name.as_ref(),
        provider.model_provider_name
    );

    let output_schema = result.output_schema.as_ref().unwrap();
    let output_schema: Value = serde_json::from_str(output_schema).unwrap();
    let expected_output_schema = json!({
        "type": "object",
        "properties": {
            "answer": {"type": "string"}
        },
        "required": ["answer"],
        "additionalProperties": false
    });
    assert_eq!(output_schema, expected_output_schema);

    assert!(!result.tags.is_empty());

    let raw_request = result.raw_request.as_ref().unwrap();
    assert!(!raw_request.is_empty());

    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Pending).await;
}

/// If there is a pending batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this will poll the batch inference and check that the response is correct.
///
/// This test polls by `batch_id` then by inference id.
pub async fn test_poll_existing_json_mode_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    if provider.variant_name.ends_with("cot") {
        // Don't test chain of thought variants with batch mode
        return;
    }

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "json_success";
    let latest_pending_batch_inference = get_latest_batch_inference(
        &conn,
        function_name,
        &provider.variant_name,
        "pending",
        Some(HashMap::from([(
            "test_type".to_string(),
            "json_mode_v2".to_string(),
        )])),
    )
    .await;
    let batch_inference = match latest_pending_batch_inference {
        None => return, // No pending batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    let batch_id = batch_inference.batch_id;
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => return,
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    // Check the response from polling by `batch_id`
    check_json_mode_inference_response(inferences_json[0].clone(), &provider, None, true).await;

    // Check the response from polling by inference_id
    let inference_id = inferences_json[0]
        .get("inference_id")
        .unwrap()
        .as_str()
        .unwrap();
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: Some(Uuid::parse_str(inference_id).unwrap()),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");

    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    check_json_mode_inference_response(inferences_json[0].clone(), &provider, None, true).await;
    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Completed).await;
}

/// If there is a completed batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this test will create fake pending data that uses the same API params but with new IDs
/// and then poll the batch inference and check that the response is correct.
///
/// This way the gateway will actually poll the inference data from the inference provider.
///
/// This test polls by `inference_id` then by `batch_id`.
pub async fn test_poll_completed_json_mode_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    if provider.variant_name.ends_with("cot") {
        // Don't test chain of thought variants with batch mode
        return;
    }

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "json_success";
    let latest_pending_batch_inference = insert_fake_pending_batch_inference_data(
        &conn,
        function_name,
        &provider.variant_name,
        Some(HashMap::from([(
            "test_type".to_string(),
            "json_mode_v2".to_string(),
        )])),
    )
    .await;
    let ids = match latest_pending_batch_inference {
        None => return, // No completed batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    sleep(Duration::from_millis(200)).await;
    test_poll_completed_json_mode_batch_inference_request_with_provider_and_ids(provider, ids)
        .await;
}

pub async fn test_poll_completed_json_mode_batch_inference_request_with_provider_and_ids(
    provider: E2ETestProvider,
    ids: InsertedFakeDataIds,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    // Poll by inference_id
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: Some(ids.inference_id),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_json_mode_inference_response(inferences_json[0].clone(), &provider, None, true).await;

    // Poll by `batch_id`
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_json_mode_inference_response(inferences_json[0].clone(), &provider, None, true).await;
    check_batch_request_status(&conn, ids.batch_id, &provider, BatchStatus::Completed).await;
}

pub async fn test_dynamic_json_mode_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    if provider.variant_name.ends_with("cot") {
        // Don't test chain of thought variants with batch mode
        return;
    }

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
        "episode_ids": [episode_id],
        "inputs": [
            {
               "system": {"assistant_name": "Dr. Mehta", "schema": serialized_output_schema},
               "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]}],
        "output_schemas": [output_schema.clone()],
        "tags": [{"test_type": "dynamic_json_mode"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Wait for trailing writes from the API to be visible
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let results = conn
        .get_batch_model_inferences(batch_id, &[inference_id])
        .await
        .expect("Failed to get batch model inferences");
    assert_eq!(results.len(), 1);
    let result = &results[0];

    println!("BatchModelInference: {result:#?}");

    assert_eq!(result.inference_id, inference_id);
    assert_eq!(result.batch_id, batch_id);
    assert_eq!(
        result.function_name.as_ref(),
        payload["function_name"].as_str().unwrap()
    );
    assert_eq!(result.variant_name.as_ref(), provider.variant_name);
    assert_eq!(result.episode_id, episode_id);

    let input = serde_json::to_value(result.input.as_ref().unwrap()).unwrap();
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

    let input_messages = result.input_messages.as_ref().unwrap();
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![
            "What is the name of the capital city of Japan?"
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, &expected_input_messages);

    assert_eq!(
        result.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nDo not include any other text than the JSON object.  Do not include \"```json\" or \"```\" or anything else.\n\nPlease answer the questions in a JSON with the following schema:\n\n{\"type\":\"object\",\"properties\":{\"response\":{\"type\":\"string\"}},\"required\":[\"response\"],\"additionalProperties\":false}"
    );

    assert!(result.tool_params.is_none());
    let inference_params = result.inference_params.as_ref().unwrap();
    let expected_max_tokens: u32 = if provider.model_name.starts_with("o1") {
        1000
    } else {
        100
    };
    assert_eq!(
        inference_params.chat_completion.max_tokens.unwrap(),
        expected_max_tokens
    );

    assert_eq!(result.model_name.as_ref(), provider.model_name);

    assert_eq!(
        result.model_provider_name.as_ref(),
        provider.model_provider_name
    );

    let output_schema = result.output_schema.as_ref().unwrap();
    let output_schema: Value = serde_json::from_str(output_schema).unwrap();
    let expected_output_schema = json!({
        "type": "object",
        "properties": {
            "response": {"type": "string"}
        },
        "required": ["response"],
        "additionalProperties": false
    });
    assert_eq!(output_schema, expected_output_schema);

    assert!(!result.tags.is_empty());

    let raw_request = result.raw_request.as_ref().unwrap();
    assert!(!raw_request.is_empty());

    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Pending).await;
}

/// If there is a pending batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this will poll the batch inference and check that the response is correct.
///
/// This test polls by `batch_id` then by inference id.
pub async fn test_poll_existing_dynamic_json_mode_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    if provider.variant_name.ends_with("cot") {
        // Don't test chain of thought variants with batch mode
        return;
    }

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "json_success";
    let latest_pending_batch_inference = get_latest_batch_inference(
        &conn,
        function_name,
        &provider.variant_name,
        "pending",
        Some(HashMap::from([(
            "test_type".to_string(),
            "dynamic_json_mode".to_string(),
        )])),
    )
    .await;
    let batch_inference = match latest_pending_batch_inference {
        None => return, // No pending batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    let batch_id = batch_inference.batch_id;
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => return,
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    // Check the response from polling by `batch_id`
    check_dynamic_json_mode_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        None,
        true,
    )
    .await;

    // Check the response from polling by inference_id
    let inference_id = inferences_json[0]
        .get("inference_id")
        .unwrap()
        .as_str()
        .unwrap();
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: Some(Uuid::parse_str(inference_id).unwrap()),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");

    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);
    check_dynamic_json_mode_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        None,
        true,
    )
    .await;
    check_batch_request_status(&conn, batch_id, &provider, BatchStatus::Completed).await;
}

/// If there is a completed batch inference for the function, variant, and tags
/// that are used for the inference params batch inference tests,
/// this test will create fake pending data that uses the same API params but with new IDs
/// and then poll the batch inference and check that the response is correct.
///
/// This way the gateway will actually poll the inference data from the inference provider.
///
/// This test polls by `inference_id` then by `batch_id`.
pub async fn test_poll_completed_dynamic_json_mode_batch_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    if provider.variant_name.ends_with("cot") {
        // Don't test chain of thought variants with batch mode
        return;
    }

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let function_name = "json_success";
    let latest_pending_batch_inference = insert_fake_pending_batch_inference_data(
        &conn,
        function_name,
        &provider.variant_name,
        Some(HashMap::from([(
            "test_type".to_string(),
            "dynamic_json_mode".to_string(),
        )])),
    )
    .await;
    let ids = match latest_pending_batch_inference {
        None => return, // No completed batch inference found, so we can't test polling
        Some(batch_inference) => batch_inference,
    };
    sleep(Duration::from_millis(200)).await;
    test_poll_completed_dynamic_json_mode_batch_inference_request_with_provider_and_ids(
        provider, ids,
    )
    .await;
}

pub async fn test_poll_completed_dynamic_json_mode_batch_inference_request_with_provider_and_ids(
    provider: E2ETestProvider,
    ids: InsertedFakeDataIds,
) {
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    // Poll by inference_id
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: Some(ids.inference_id),
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_dynamic_json_mode_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        None,
        true,
    )
    .await;

    // Poll by `batch_id`
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id: ids.batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");
    match response_json.get("status").unwrap().as_str().unwrap() {
        "pending" => panic!("Batch inference is pending"),
        "completed" => (),
        _ => panic!("Batch inference failed"),
    }
    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, ids.batch_id);

    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_dynamic_json_mode_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        None,
        true,
    )
    .await;
    check_batch_request_status(&conn, ids.batch_id, &provider, BatchStatus::Completed).await;
}
