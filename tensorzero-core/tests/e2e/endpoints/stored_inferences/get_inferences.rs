/// Tests for the /v1/inferences/list_inferences and /v1/inferences/get_inferences endpoints.
use reqwest::Client;
use serde_json::{json, Value};
use tensorzero::InferenceOutputSource;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Helper function to call list_inferences via HTTP
async fn list_inferences(request: Value) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let http_client = Client::new();
    let resp = http_client
        .post(get_gateway_endpoint("/v1/inferences/list_inferences"))
        .json(&request)
        .send()
        .await?;

    assert!(
        resp.status().is_success(),
        "list_inferences request failed: status={:?}, body={:?}",
        resp.status(),
        resp.text().await
    );

    let resp_json: Value = resp.json().await?;
    let inferences = resp_json["inferences"]
        .as_array()
        .expect("Expected 'inferences' array in response")
        .clone();

    Ok(inferences)
}

/// Helper function to call get_inferences via HTTP
async fn get_inferences_by_ids(
    ids: Vec<Uuid>,
    output_source: InferenceOutputSource,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let http_client = Client::new();
    let id_strings: Vec<String> = ids.iter().map(std::string::ToString::to_string).collect();

    let resp = http_client
        .post(get_gateway_endpoint("/v1/inferences/get_inferences"))
        .json(&json!({
            "ids": id_strings,
            "output_source": json!(output_source)
        }))
        .send()
        .await?;

    assert!(
        resp.status().is_success(),
        "get_inferences request failed: status={:?}, body={:?}",
        resp.status(),
        resp.text().await
    );

    let resp_json: Value = resp.json().await?;
    let inferences = resp_json["inferences"]
        .as_array()
        .expect("Expected 'inferences' array in response")
        .clone();

    Ok(inferences)
}

// Tests for list_inferences endpoint

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_simple_query_json_function() {
    let request = json!({
        "function_name": "extract_entities",
        "output_source": "inference",
        "limit": 2,
        "order_by": [
            {
                "by": "timestamp",
                "direction": "descending"
            }
        ]
    });

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 2);

    for inference in &res {
        assert_eq!(inference["type"], "json");
        assert_eq!(inference["function_name"], "extract_entities");
        assert_eq!(
            inference["dispreferred_outputs"].as_array().unwrap().len(),
            0
        );
    }

    // Verify ORDER BY timestamp DESC - check that timestamps are in descending order
    let mut prev_timestamp: Option<String> = None;
    for inference in &res {
        let timestamp = inference["timestamp"].as_str().unwrap().to_string();
        if let Some(prev) = &prev_timestamp {
            assert!(
                timestamp <= *prev,
                "Timestamps should be in descending order. Got: {timestamp} <= {prev}"
            );
        }
        prev_timestamp = Some(timestamp);
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_simple_query_chat_function() {
    let request = json!({
        "function_name": "write_haiku",
        "output_source": "demonstration",
        "limit": 3,
        "offset": 3,
        "order_by": [
            {
                "by": "timestamp",
                "direction": "ascending"
            }
        ]
    });

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 3);

    for inference in &res {
        assert_eq!(inference["type"], "chat");
        assert_eq!(inference["function_name"], "write_haiku");
        assert_eq!(
            inference["dispreferred_outputs"].as_array().unwrap().len(),
            1
        );
    }

    // Verify ORDER BY timestamp ASC - check that timestamps are in ascending order
    let mut prev_timestamp: Option<String> = None;
    for inference in &res {
        let timestamp = inference["timestamp"].as_str().unwrap().to_string();
        if let Some(prev) = &prev_timestamp {
            assert!(
                timestamp >= *prev,
                "Timestamps should be in ascending order. Got: {timestamp} >= {prev}"
            );
        }
        prev_timestamp = Some(timestamp);
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_query_with_float_filter() {
    let request = json!({
        "function_name": "extract_entities",
        "output_source": "inference",
        "limit": 3,
        "filter": {
            "type": "float_metric",
            "metric_name": "jaccard_similarity",
            "value": 0.5,
            "comparison_operator": ">"
        },
        "order_by": [
            {
                "by": "metric",
                "name": "jaccard_similarity",
                "direction": "descending"
            }
        ]
    });

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 3);

    for inference in &res {
        assert_eq!(inference["type"], "json");
        assert_eq!(inference["function_name"], "extract_entities");
        assert_eq!(
            inference["dispreferred_outputs"].as_array().unwrap().len(),
            0
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_demonstration_output_source() {
    let request = json!({
        "function_name": "extract_entities",
        "output_source": "demonstration",
        "limit": 5,
        "offset": 1
    });

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 5);

    for inference in &res {
        assert_eq!(inference["type"], "json");
        assert_eq!(inference["function_name"], "extract_entities");
        assert_eq!(
            inference["dispreferred_outputs"].as_array().unwrap().len(),
            1
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_boolean_metric_filter() {
    let request = json!({
        "function_name": "extract_entities",
        "output_source": "inference",
        "limit": 5,
        "offset": 1,
        "filter": {
            "type": "boolean_metric",
            "metric_name": "exact_match",
            "value": true
        }
    });

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 5);

    for inference in &res {
        assert_eq!(inference["type"], "json");
        assert_eq!(inference["function_name"], "extract_entities");
        assert_eq!(
            inference["dispreferred_outputs"].as_array().unwrap().len(),
            0
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_and_filter_multiple_float_metrics() {
    let request = json!({
        "function_name": "extract_entities",
        "output_source": "inference",
        "limit": 1,
        "filter": {
            "type": "and",
            "children": [
                {
                    "type": "float_metric",
                    "metric_name": "jaccard_similarity",
                    "value": 0.5,
                    "comparison_operator": ">"
                },
                {
                    "type": "float_metric",
                    "metric_name": "jaccard_similarity",
                    "value": 0.8,
                    "comparison_operator": "<"
                }
            ]
        }
    });

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 1);

    for inference in &res {
        assert_eq!(inference["type"], "json");
        assert_eq!(inference["function_name"], "extract_entities");
        assert_eq!(
            inference["dispreferred_outputs"].as_array().unwrap().len(),
            0
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_or_filter_mixed_metrics() {
    let request = json!({
        "function_name": "extract_entities",
        "output_source": "inference",
        "limit": 1,
        "filter": {
            "type": "or",
            "children": [
                {
                    "type": "float_metric",
                    "metric_name": "jaccard_similarity",
                    "value": 0.8,
                    "comparison_operator": ">="
                },
                {
                    "type": "boolean_metric",
                    "metric_name": "exact_match",
                    "value": true
                },
                {
                    "type": "boolean_metric",
                    "metric_name": "goal_achieved",
                    "value": true
                }
            ]
        }
    });

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 1);

    for inference in &res {
        assert_eq!(inference["type"], "json");
        assert_eq!(inference["function_name"], "extract_entities");
        assert_eq!(
            inference["dispreferred_outputs"].as_array().unwrap().len(),
            0
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_not_filter() {
    let request = json!({
        "function_name": "extract_entities",
        "output_source": "inference",
        "filter": {
            "type": "not",
            "child": {
                "type": "or",
                "children": [
                    {
                        "type": "boolean_metric",
                        "metric_name": "exact_match",
                        "value": true
                    },
                    {
                        "type": "boolean_metric",
                        "metric_name": "exact_match",
                        "value": false
                    }
                ]
            }
        }
    });

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_simple_time_filter() {
    let request = json!({
        "function_name": "extract_entities",
        "output_source": "inference",
        "limit": 5,
        "filter": {
            "type": "time",
            "time": "2023-01-01T00:00:00Z",
            "comparison_operator": ">"
        },
        "order_by": [
            {
                "by": "metric",
                "name": "exact_match",
                "direction": "descending"
            },
            {
                "by": "timestamp",
                "direction": "ascending"
            }
        ]
    });

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 5);

    for inference in &res {
        assert_eq!(inference["type"], "json");
        assert_eq!(inference["function_name"], "extract_entities");
    }

    // Verify ORDER BY timestamp ASC (secondary sort)
    let mut prev_timestamp: Option<String> = None;
    for inference in &res {
        let timestamp = inference["timestamp"].as_str().unwrap().to_string();
        if let Some(prev) = &prev_timestamp {
            assert!(
                timestamp >= *prev,
                "Timestamps should be in ascending order for secondary sort. Got: {timestamp} >= {prev}"
            );
        }
        prev_timestamp = Some(timestamp);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_simple_tag_filter() {
    let request = json!({
        "function_name": "extract_entities",
        "output_source": "inference",
        "limit": 200,
        "filter": {
            "type": "tag",
            "key": "tensorzero::evaluation_name",
            "value": "entity_extraction",
            "comparison_operator": "="
        }
    });

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 200);

    for inference in &res {
        assert_eq!(inference["type"], "json");
        assert_eq!(inference["function_name"], "extract_entities");
        assert_eq!(
            inference["tags"]["tensorzero::evaluation_name"],
            "entity_extraction"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_combined_time_and_tag_filter() {
    let request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 50,
        "filter": {
            "type": "and",
            "children": [
                {
                    "type": "time",
                    "time": "2025-04-14T23:30:00Z",
                    "comparison_operator": ">="
                },
                {
                    "type": "tag",
                    "key": "tensorzero::evaluation_name",
                    "value": "haiku",
                    "comparison_operator": "="
                }
            ]
        }
    });

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 50);

    for inference in &res {
        assert_eq!(inference["type"], "chat");
        assert_eq!(inference["function_name"], "write_haiku");
        assert_eq!(inference["tags"]["tensorzero::evaluation_name"], "haiku");

        let timestamp = inference["timestamp"].as_str().unwrap();
        assert!(timestamp >= "2025-04-14T23:30:00Z");
    }
}

// Tests for get_inferences endpoint (by ID)

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_by_ids_json_only() {
    // First, list some JSON inference IDs
    let list_request = json!({
        "function_name": "extract_entities",
        "output_source": "inference",
        "limit": 3
    });

    let initial_res = list_inferences(list_request).await.unwrap();
    assert_eq!(initial_res.len(), 3);

    // Extract the IDs
    let ids: Vec<Uuid> = initial_res
        .iter()
        .map(|inf| Uuid::parse_str(inf["inference_id"].as_str().unwrap()).unwrap())
        .collect();

    // Now get by IDs
    let res = get_inferences_by_ids(ids.clone(), InferenceOutputSource::Inference)
        .await
        .unwrap();

    // Should get back the same 3 inferences
    assert_eq!(res.len(), 3);

    for inference in &res {
        assert_eq!(inference["type"], "json");
        let inference_id = Uuid::parse_str(inference["inference_id"].as_str().unwrap()).unwrap();
        assert!(ids.contains(&inference_id));
        assert_eq!(inference["function_name"], "extract_entities");
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_by_ids_chat_only() {
    // First, list some Chat inference IDs
    let list_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 2
    });

    let initial_res = list_inferences(list_request).await.unwrap();
    assert_eq!(initial_res.len(), 2);

    // Extract the IDs
    let ids: Vec<Uuid> = initial_res
        .iter()
        .map(|inf| Uuid::parse_str(inf["inference_id"].as_str().unwrap()).unwrap())
        .collect();

    // Now get by IDs
    let res = get_inferences_by_ids(ids.clone(), InferenceOutputSource::Inference)
        .await
        .unwrap();

    // Should get back the same 2 inferences
    assert_eq!(res.len(), 2);

    for inference in &res {
        assert_eq!(inference["type"], "chat");
        let inference_id = Uuid::parse_str(inference["inference_id"].as_str().unwrap()).unwrap();
        assert!(ids.contains(&inference_id));
        assert_eq!(inference["function_name"], "write_haiku");
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_by_ids_unknown_id_returns_empty() {
    // Get by an unknown ID
    let unknown_ids = vec![Uuid::now_v7()];
    let res = get_inferences_by_ids(unknown_ids, InferenceOutputSource::Inference)
        .await
        .unwrap();

    assert!(res.is_empty(), "Expected empty result for unknown ID");
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_by_ids_mixed_types() {
    // Get some JSON inference IDs
    let json_request = json!({
        "function_name": "extract_entities",
        "output_source": "inference",
        "limit": 2
    });
    let json_res = list_inferences(json_request).await.unwrap();

    // Get some Chat inference IDs
    let chat_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 2
    });
    let chat_res = list_inferences(chat_request).await.unwrap();

    // Combine the IDs
    let mut ids: Vec<Uuid> = json_res
        .iter()
        .map(|inf| Uuid::parse_str(inf["inference_id"].as_str().unwrap()).unwrap())
        .collect();
    ids.extend(
        chat_res
            .iter()
            .map(|inf| Uuid::parse_str(inf["inference_id"].as_str().unwrap()).unwrap()),
    );

    // Now get by mixed IDs
    let res = get_inferences_by_ids(ids.clone(), InferenceOutputSource::Inference)
        .await
        .unwrap();

    // Should get back 4 inferences (2 JSON + 2 Chat)
    assert_eq!(res.len(), 4);

    let mut json_count = 0;
    let mut chat_count = 0;

    for inference in &res {
        let inference_id = Uuid::parse_str(inference["inference_id"].as_str().unwrap()).unwrap();
        assert!(ids.contains(&inference_id));

        match inference["type"].as_str().unwrap() {
            "json" => {
                assert_eq!(inference["function_name"], "extract_entities");
                json_count += 1;
            }
            "chat" => {
                assert_eq!(inference["function_name"], "write_haiku");
                chat_count += 1;
            }
            other => panic!("Unexpected inference type: {other}"),
        }
    }

    assert_eq!(json_count, 2);
    assert_eq!(chat_count, 2);
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_by_ids_empty_list() {
    // Get by empty list of IDs should return empty result
    let res = get_inferences_by_ids(vec![], InferenceOutputSource::Inference)
        .await
        .unwrap();
    assert!(res.is_empty(), "Expected empty result for empty ID list");
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_by_ids_duplicate_ids() {
    // First, get one inference ID
    let list_request = json!({
        "function_name": "extract_entities",
        "output_source": "inference",
        "limit": 1
    });

    let initial_res = list_inferences(list_request).await.unwrap();
    assert_eq!(initial_res.len(), 1);

    let id = Uuid::parse_str(initial_res[0]["inference_id"].as_str().unwrap()).unwrap();

    // Query with the same ID duplicated
    let duplicate_ids = vec![id, id, id];
    let res = get_inferences_by_ids(duplicate_ids, InferenceOutputSource::Inference)
        .await
        .unwrap();

    // Should still only get back 1 inference (deduplicated by ClickHouse)
    assert_eq!(res.len(), 1);
    assert_eq!(res[0]["type"], "json");
    let returned_id = Uuid::parse_str(res[0]["inference_id"].as_str().unwrap()).unwrap();
    assert_eq!(returned_id, id);
}

// Tests for search_query_experimental

#[tokio::test(flavor = "multi_thread")]
async fn test_search_query_simple_search() {
    let request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 10,
        // We arbitrarily choose a query term in the data fixture
        "search_query_experimental": "formamide"
    });

    let res = list_inferences(request).await.unwrap();
    assert!(
        !res.is_empty(),
        "Expected at least one result for 'formamide' query"
    );

    for inference in &res {
        assert_eq!(
            inference["function_name"], "write_haiku",
            "Function name filter should be applied"
        );
        let input_string = serde_json::to_string(&inference["input"])
            .unwrap()
            .to_lowercase();
        let output_string = serde_json::to_string(&inference["output"])
            .unwrap()
            .to_lowercase();
        assert!(
            input_string.contains("formamide") || output_string.contains("formamide"),
            "Input or output should contain 'formamide'"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_query_case_insensitive() {
    let request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 5,
        "search_query_experimental": "FORMAMIDE"
    });

    let res = list_inferences(request).await.unwrap();

    assert!(!res.is_empty(), "Expected results for 'FORMAMIDE' query");

    // There is no inference with all-caps 'FORMAMIDE', but there are ones with lowercase ones.
    for inference in &res {
        let input_string = serde_json::to_string(&inference["input"]).unwrap();
        let output_string = serde_json::to_string(&inference["output"]).unwrap();
        assert!(
            !input_string.contains("FORMAMIDE") && !output_string.contains("FORMAMIDE"),
            "Input or output should not contain all-caps 'FORMAMIDE'"
        );
        assert!(
            input_string.to_lowercase().contains("formamide")
                || output_string.to_lowercase().contains("formamide"),
            "Input or output should contain 'formamide' in a case-insensitive match"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_query_no_results() {
    // Search for something that definitely doesn't exist
    let request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 10,
        "search_query_experimental": "xyzzyqwertyzzznonexistent"
    });

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 0, "Expected no results for non-existent term");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_query_with_other_filters() {
    // Test that text query works in combination with other filters
    let request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 5,
        "search_query_experimental": "nature",
        "filter": {
            "type": "time",
            "time": "2023-01-01T00:00:00Z",
            "comparison_operator": ">"
        }
    });

    let res = list_inferences(request).await.unwrap();

    // Should only return results that match both the text query AND the time filter
    for inference in &res {
        assert_eq!(inference["function_name"], "write_haiku");

        let input_string = serde_json::to_string(&inference["input"])
            .unwrap()
            .to_lowercase();
        let output_string = serde_json::to_string(&inference["output"])
            .unwrap()
            .to_lowercase();
        assert!(
            input_string.contains("nature") || output_string.contains("nature"),
            "Input or output should contain 'nature'"
        );

        let timestamp = inference["timestamp"].as_str().unwrap();
        assert!(
            timestamp > "2023-01-01T00:00:00Z",
            "Timestamp should be after filter"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_query_order_by_search_relevance() {
    // Test ordering by term frequency in descending order
    let request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 10,
        "search_query_experimental": "formamide",
        "order_by": [
            {
                "by": "search_relevance",
                "direction": "descending"
            }
        ]
    });

    let res = list_inferences(request).await.unwrap();

    assert!(!res.is_empty(), "Expected results for 'formamide' query");

    // Verify that results are ordered by search relevance (currently term frequency) in descending order
    let mut prev_relevance = None;
    for inference in &res {
        assert_eq!(inference["function_name"], "write_haiku");
        let input_string = serde_json::to_string(&inference["input"])
            .unwrap()
            .to_lowercase();
        let output_string = serde_json::to_string(&inference["output"])
            .unwrap()
            .to_lowercase();
        let relevance =
            input_string.matches("formamide").count() + output_string.matches("formamide").count();
        if let Some(prev) = &prev_relevance {
            assert!(
                relevance <= *prev,
                "Search relevance should be in descending order. Got: {relevance} > {prev}"
            );
            prev_relevance = Some(relevance);
        }
    }
}

// Tests for cursor-based pagination

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_inferences_with_before_cursor() {
    // First, get some inferences without cursor
    let initial_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 5,
        "order_by": [
            {
                "by": "timestamp",
                "direction": "descending"
            }
        ]
    });

    let initial_res = list_inferences(initial_request).await.unwrap();
    assert!(!initial_res.is_empty(), "Expected some inferences");

    // Get the ID of the third inference to use as cursor
    let cursor_id = initial_res[2]["inference_id"].as_str().unwrap();

    // Now request inferences before this cursor
    // Note: cannot use order_by with cursor pagination
    let cursor_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 5,
        "before": cursor_id,
    });

    let cursor_res = list_inferences(cursor_request).await.unwrap();

    // Verify we got results
    assert!(
        !cursor_res.is_empty(),
        "Expected results with before cursor"
    );

    // Verify all returned inferences have IDs less than the cursor
    let cursor_uuid = Uuid::parse_str(cursor_id).unwrap();
    for inference in &cursor_res {
        let inference_id = Uuid::parse_str(inference["inference_id"].as_str().unwrap()).unwrap();
        assert!(
            inference_id < cursor_uuid,
            "Inference ID should be less than cursor ID"
        );
    }

    // Verify results are still in descending timestamp order
    let mut prev_timestamp: Option<String> = None;
    for inference in &cursor_res {
        let timestamp = inference["timestamp"].as_str().unwrap().to_string();
        if let Some(prev) = &prev_timestamp {
            assert!(
                timestamp <= *prev,
                "Timestamps should be in descending order"
            );
        }
        prev_timestamp = Some(timestamp);
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_inferences_with_after_cursor() {
    // First, get some inferences without cursor
    let initial_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 10,
        "order_by": [
            {
                "by": "timestamp",
                "direction": "descending"
            }
        ]
    });

    let initial_res = list_inferences(initial_request).await.unwrap();
    assert_eq!(initial_res.len(), 10, "Expected 10 write_haiku inferences");

    // Get the ID of the (0-indexed) 4th inference to use as cursor, so we return the 5th-9th inferences.
    let cursor_id = initial_res[4]["inference_id"].as_str().unwrap();

    // Now request inferences after this cursor
    // Note: cannot use order_by with cursor pagination
    let cursor_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 5,
        "after": cursor_id,
    });

    let cursor_res = list_inferences(cursor_request).await.unwrap();

    assert!(!cursor_res.is_empty(), "Expected results with after cursor");

    // Verify all returned inferences have IDs greater than the cursor
    let cursor_uuid = Uuid::parse_str(cursor_id).unwrap();
    for inference in &cursor_res {
        let inference_id = Uuid::parse_str(inference["inference_id"].as_str().unwrap()).unwrap();
        assert!(
            inference_id > cursor_uuid,
            "Inference ID should be greater than cursor ID"
        );
    }

    // Verify results are still in descending timestamp order
    let mut prev_timestamp: Option<String> = None;
    for inference in &cursor_res {
        let timestamp = inference["timestamp"].as_str().unwrap().to_string();
        if let Some(prev) = &prev_timestamp {
            assert!(
                timestamp <= *prev,
                "Timestamps should be in descending order"
            );
        }
        prev_timestamp = Some(timestamp);
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_inferences_cursor_pagination_prevents_offset() {
    let http_client = Client::new();

    // First, get an inference ID to use as cursor
    let initial_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 1,
    });

    let initial_res = list_inferences(initial_request).await.unwrap();
    assert!(!initial_res.is_empty(), "Expected at least one inference");
    let cursor_id = initial_res[0]["inference_id"].as_str().unwrap();

    // Try to use both cursor and offset - should fail
    let invalid_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 5,
        "before": cursor_id,
        "offset": 10, // This should cause an error
    });

    let resp = http_client
        .post(get_gateway_endpoint("/v1/inferences/list_inferences"))
        .json(&invalid_request)
        .send()
        .await
        .unwrap();

    // Should return an error status
    assert!(
        !resp.status().is_success(),
        "Request with both cursor and offset should fail"
    );

    let error_body = resp.text().await.unwrap();
    assert!(
        error_body.contains("Cannot use 'offset' with cursor pagination"),
        "Error message should mention cursor/offset conflict"
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_inferences_cursor_mutually_exclusive() {
    let http_client = Client::new();

    // First, get inference IDs to use as cursors
    let initial_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 2,
    });

    let initial_res = list_inferences(initial_request).await.unwrap();
    assert!(
        initial_res.len() >= 2,
        "Expected at least two inferences for test"
    );
    let cursor_id_1 = initial_res[0]["inference_id"].as_str().unwrap();
    let cursor_id_2 = initial_res[1]["inference_id"].as_str().unwrap();

    // Try to use both before and after - should fail
    let invalid_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 5,
        "before": cursor_id_1,
        "after": cursor_id_2, // This should cause an error
    });

    let resp = http_client
        .post(get_gateway_endpoint("/v1/inferences/list_inferences"))
        .json(&invalid_request)
        .send()
        .await
        .unwrap();

    // Should return an error status
    assert!(
        !resp.status().is_success(),
        "Request with both before and after should fail"
    );

    let error_body = resp.text().await.unwrap();
    assert!(
        error_body.contains("Cannot specify both 'before' and 'after'"),
        "Error message should mention before/after conflict"
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_inferences_cursor_with_filters() {
    // Test that cursor pagination works with filters
    let initial_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 3,
        "filter": {
            "type": "tag",
            "key": "tensorzero::dataset_name",
            "value": "foo",
            "comparison_operator": "="
        },
        "order_by": [
            {
                "by": "timestamp",
                "direction": "descending"
            }
        ]
    });

    let initial_res = list_inferences(initial_request).await.unwrap();
    assert!(!initial_res.is_empty(), "Expected at least one inference");

    // Get the ID of the first inference to use as cursor
    let cursor_id = initial_res[0]["inference_id"].as_str().unwrap();

    // Request inferences before this cursor with the same filter
    let cursor_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 3,
        "before": cursor_id,
        "filter": {
            "type": "tag",
            "key": "tensorzero::dataset_name",
            "value": "foo",
            "comparison_operator": "="
        },
        "order_by": [
            {
                "by": "timestamp",
                "direction": "descending"
            }
        ]
    });

    let cursor_res = list_inferences(cursor_request).await.unwrap();

    assert!(
        !cursor_res.is_empty(),
        "Expected results with before cursor and filter"
    );

    // Verify all returned inferences have IDs less than the cursor
    let cursor_uuid = Uuid::parse_str(cursor_id).unwrap();
    for inference in &cursor_res {
        let inference_id = Uuid::parse_str(inference["inference_id"].as_str().unwrap()).unwrap();
        assert!(
            inference_id < cursor_uuid,
            "Inference ID should be less than cursor ID"
        );
        // Verify the filter is still applied
        assert_eq!(inference["function_name"], "write_haiku");
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_inferences_cursor_with_timestamp_ordering_succeeds() {
    // First, get some inferences to use as cursor
    let initial_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 5,
        "order_by": [
            {
                "by": "timestamp",
                "direction": "descending"
            }
        ]
    });

    let initial_res = list_inferences(initial_request).await.unwrap();
    assert!(!initial_res.is_empty(), "Expected at least one inference");
    let cursor_id = initial_res[2]["inference_id"].as_str().unwrap();

    // Timestamp ordering should work with cursor pagination
    let cursor_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 5,
        "before": cursor_id,
        "order_by": [
            {
                "by": "timestamp",
                "direction": "descending"
            }
        ]
    });

    let cursor_res = list_inferences(cursor_request).await.unwrap();

    // Should get results successfully
    assert!(
        !cursor_res.is_empty(),
        "Should get results with timestamp ordering and cursor"
    );

    // Verify results are ordered by timestamp descending
    let mut prev_timestamp: Option<String> = None;
    for inference in &cursor_res {
        let timestamp = inference["timestamp"].as_str().unwrap().to_string();
        if let Some(prev) = &prev_timestamp {
            assert!(
                timestamp <= *prev,
                "Timestamps should be in descending order"
            );
        }
        prev_timestamp = Some(timestamp);
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_inferences_cursor_with_metric_ordering_fails() {
    let http_client = Client::new();

    // First, get an inference ID to use as cursor
    let initial_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 1,
    });

    let initial_res = list_inferences(initial_request).await.unwrap();
    assert!(!initial_res.is_empty(), "Expected at least one inference");
    let cursor_id = initial_res[0]["inference_id"].as_str().unwrap();

    // Try to use cursor with metric ordering - should fail
    let invalid_request = json!({
        "function_name": "write_haiku",
        "output_source": "inference",
        "limit": 5,
        "before": cursor_id,
        "order_by": [
            {
                "by": "metric",
                "name": "test_metric",
                "direction": "descending"
            }
        ]
    });

    let resp = http_client
        .post(get_gateway_endpoint("/v1/inferences/list_inferences"))
        .json(&invalid_request)
        .send()
        .await
        .unwrap();

    // Should return an error status
    assert!(
        !resp.status().is_success(),
        "Request with cursor and metric ordering should fail"
    );
}
