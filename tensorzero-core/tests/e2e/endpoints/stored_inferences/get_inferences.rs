/// Tests for the /v1/inferences/list_inferences and /v1/inferences/get_inferences endpoints.
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde_json::Value;
use tensorzero::InferenceOutputSource;
use tensorzero_core::endpoints::stored_inferences::v1::types::{
    BooleanMetricFilter, FloatComparisonOperator, FloatMetricFilter, GetInferencesRequest,
    InferenceFilter, ListInferencesRequest, OrderBy, OrderByTerm, OrderDirection,
    TagComparisonOperator, TagFilter, TimeComparisonOperator, TimeFilter,
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Helper function to call list_inferences via HTTP
async fn list_inferences(
    request: ListInferencesRequest,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
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

    let request = GetInferencesRequest {
        ids,
        function_name: None,
        output_source,
    };

    let resp = http_client
        .post(get_gateway_endpoint("/v1/inferences/get_inferences"))
        .json(&request)
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
    let request = ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(2),
        offset: None,
        filter: None,
        order_by: Some(vec![OrderBy {
            term: OrderByTerm::Timestamp,
            direction: OrderDirection::Desc,
        }]),
        search_query_experimental: None,
    };

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
    let request = ListInferencesRequest {
        function_name: Some("write_haiku".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Demonstration,
        limit: Some(3),
        offset: Some(3),
        filter: None,
        order_by: Some(vec![OrderBy {
            term: OrderByTerm::Timestamp,
            direction: OrderDirection::Asc,
        }]),
        search_query_experimental: None,
    };

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
    let request = ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(3),
        offset: None,
        filter: Some(InferenceFilter::FloatMetric(FloatMetricFilter {
            metric_name: "jaccard_similarity".to_string(),
            value: 0.5,
            comparison_operator: FloatComparisonOperator::GreaterThan,
        })),
        order_by: Some(vec![OrderBy {
            term: OrderByTerm::Metric {
                name: "jaccard_similarity".to_string(),
            },
            direction: OrderDirection::Desc,
        }]),
        search_query_experimental: None,
    };

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
    let request = ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Demonstration,
        limit: Some(5),
        offset: Some(1),
        filter: None,
        order_by: None,
        search_query_experimental: None,
    };

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
    let request = ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(5),
        offset: Some(1),
        filter: Some(InferenceFilter::BooleanMetric(BooleanMetricFilter {
            metric_name: "exact_match".to_string(),
            value: true,
        })),
        order_by: None,
        search_query_experimental: None,
    };

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
    let request = ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(1),
        offset: None,
        filter: Some(InferenceFilter::And {
            children: vec![
                InferenceFilter::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.5,
                    comparison_operator: FloatComparisonOperator::GreaterThan,
                }),
                InferenceFilter::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.8,
                    comparison_operator: FloatComparisonOperator::LessThan,
                }),
            ],
        }),
        order_by: None,
        search_query_experimental: None,
    };

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
    let request = ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(1),
        offset: None,
        filter: Some(InferenceFilter::Or {
            children: vec![
                InferenceFilter::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.8,
                    comparison_operator: FloatComparisonOperator::GreaterThanOrEqual,
                }),
                InferenceFilter::BooleanMetric(BooleanMetricFilter {
                    metric_name: "exact_match".to_string(),
                    value: true,
                }),
                InferenceFilter::BooleanMetric(BooleanMetricFilter {
                    metric_name: "goal_achieved".to_string(),
                    value: true,
                }),
            ],
        }),
        order_by: None,
        search_query_experimental: None,
    };

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
    let request = ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: None,
        offset: None,
        filter: Some(InferenceFilter::Not {
            child: Box::new(InferenceFilter::Or {
                children: vec![
                    InferenceFilter::BooleanMetric(BooleanMetricFilter {
                        metric_name: "exact_match".to_string(),
                        value: true,
                    }),
                    InferenceFilter::BooleanMetric(BooleanMetricFilter {
                        metric_name: "exact_match".to_string(),
                        value: false,
                    }),
                ],
            }),
        }),
        order_by: None,
        search_query_experimental: None,
    };

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_simple_time_filter() {
    let request = ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(5),
        offset: None,
        filter: Some(InferenceFilter::Time(TimeFilter {
            time: "2023-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap(),
            comparison_operator: TimeComparisonOperator::GreaterThan,
        })),
        order_by: Some(vec![
            OrderBy {
                term: OrderByTerm::Metric {
                    name: "exact_match".to_string(),
                },
                direction: OrderDirection::Desc,
            },
            OrderBy {
                term: OrderByTerm::Timestamp,
                direction: OrderDirection::Asc,
            },
        ]),
        search_query_experimental: None,
    };

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
    let request = ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(200),
        offset: None,
        filter: Some(InferenceFilter::Tag(TagFilter {
            key: "tensorzero::evaluation_name".to_string(),
            value: "entity_extraction".to_string(),
            comparison_operator: TagComparisonOperator::Equal,
        })),
        order_by: None,
        search_query_experimental: None,
    };

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
    let request = ListInferencesRequest {
        function_name: Some("write_haiku".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(50),
        offset: None,
        filter: Some(InferenceFilter::And {
            children: vec![
                InferenceFilter::Time(TimeFilter {
                    time: "2025-04-14T23:30:00Z".parse::<DateTime<Utc>>().unwrap(),
                    comparison_operator: TimeComparisonOperator::GreaterThanOrEqual,
                }),
                InferenceFilter::Tag(TagFilter {
                    key: "tensorzero::evaluation_name".to_string(),
                    value: "haiku".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
            ],
        }),
        order_by: None,
        search_query_experimental: None,
    };

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
    let list_request = ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(3),
        offset: None,
        filter: None,
        order_by: None,
        search_query_experimental: None,
    };

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
    let list_request = ListInferencesRequest {
        function_name: Some("write_haiku".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(2),
        offset: None,
        filter: None,
        order_by: None,
        search_query_experimental: None,
    };

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
    let json_request = ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(2),
        offset: None,
        filter: None,
        order_by: None,
        search_query_experimental: None,
    };
    let json_res = list_inferences(json_request).await.unwrap();

    // Get some Chat inference IDs
    let chat_request = ListInferencesRequest {
        function_name: Some("write_haiku".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(2),
        offset: None,
        filter: None,
        order_by: None,
        search_query_experimental: None,
    };
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
    let list_request = ListInferencesRequest {
        function_name: Some("extract_entities".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(1),
        offset: None,
        filter: None,
        order_by: None,
        search_query_experimental: None,
    };

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
    let request = ListInferencesRequest {
        function_name: Some("write_haiku".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(10),
        offset: None,
        filter: None,
        order_by: None,
        // We arbitrarily choose a query term in the data fixture
        search_query_experimental: Some("formamide".to_string()),
    };

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
    let request = ListInferencesRequest {
        function_name: Some("write_haiku".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(5),
        offset: None,
        filter: None,
        order_by: None,
        search_query_experimental: Some("FORMAMIDE".to_string()),
    };

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
    let request = ListInferencesRequest {
        function_name: Some("write_haiku".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(10),
        offset: None,
        filter: None,
        order_by: None,
        search_query_experimental: Some("xyzzyqwertyzzznonexistent".to_string()),
    };

    let res = list_inferences(request).await.unwrap();
    assert_eq!(res.len(), 0, "Expected no results for non-existent term");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_query_with_other_filters() {
    // Test that text query works in combination with other filters
    let request = ListInferencesRequest {
        function_name: Some("write_haiku".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(5),
        offset: None,
        filter: Some(InferenceFilter::Time(TimeFilter {
            time: "2023-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap(),
            comparison_operator: TimeComparisonOperator::GreaterThan,
        })),
        order_by: None,
        search_query_experimental: Some("nature".to_string()),
    };

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
    let request = ListInferencesRequest {
        function_name: Some("write_haiku".to_string()),
        variant_name: None,
        episode_id: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(10),
        offset: None,
        filter: None,
        order_by: Some(vec![OrderBy {
            term: OrderByTerm::SearchRelevance,
            direction: OrderDirection::Desc,
        }]),
        search_query_experimental: Some("formamide".to_string()),
    };

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
