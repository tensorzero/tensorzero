#![expect(deprecated)]
/// E2E tests for the legacy experimental_list_inferences client method.
/// TODO: fully deprecate and remove when v1 list_inferences is fully released.
use chrono::DateTime;
use tensorzero::test_helpers::make_embedded_gateway;
use tensorzero::{
    BooleanMetricFilter, ClientExt, FloatComparisonOperator, FloatMetricFilter, InferenceFilter,
    InferenceOutputSource, ListInferencesParams, StoredInference, TagComparisonOperator, TagFilter,
    TimeComparisonOperator, TimeFilter,
};
use tensorzero_core::db::clickhouse::query_builder::{OrderBy, OrderByTerm, OrderDirection};
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread")]
pub async fn test_simple_query_json_function() {
    let client = make_embedded_gateway().await;
    let order_by = vec![OrderBy {
        term: OrderByTerm::Timestamp,
        direction: OrderDirection::Desc,
    }];
    let opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        limit: 2,
        order_by: Some(&order_by),
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 2);

    for inference in &res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
        assert!(json_inference.dispreferred_outputs.is_empty());
    }

    // Verify ORDER BY timestamp DESC - check that timestamps are in descending order
    let mut prev_timestamp = None;
    for inference in &res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        if let Some(prev) = prev_timestamp {
            assert!(
                json_inference.timestamp <= prev,
                "Timestamps should be in descending order. Got: {} <= {}",
                json_inference.timestamp,
                prev
            );
        }
        prev_timestamp = Some(json_inference.timestamp);
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_simple_query_chat_function() {
    let client = make_embedded_gateway().await;
    let order_by = vec![OrderBy {
        term: OrderByTerm::Timestamp,
        direction: OrderDirection::Asc,
    }];
    let opts = ListInferencesParams {
        function_name: Some("write_haiku"),
        output_source: InferenceOutputSource::Demonstration,
        limit: 3,
        offset: 3,
        order_by: Some(&order_by),
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 3);

    for inference in &res {
        let StoredInference::Chat(chat_inference) = inference else {
            panic!("Expected a Chat inference");
        };
        assert_eq!(chat_inference.function_name, "write_haiku");
        assert_eq!(chat_inference.dispreferred_outputs.len(), 1);
    }

    // Verify ORDER BY timestamp ASC - check that timestamps are in ascending order
    let mut prev_timestamp = None;
    for inference in &res {
        let StoredInference::Chat(chat_inference) = inference else {
            panic!("Expected a Chat inference");
        };
        if let Some(prev) = prev_timestamp {
            assert!(
                chat_inference.timestamp >= prev,
                "Timestamps should be in ascending order. Got: {} >= {}",
                chat_inference.timestamp,
                prev
            );
        }
        prev_timestamp = Some(chat_inference.timestamp);
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_simple_query_with_float_filter() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilter::FloatMetric(FloatMetricFilter {
        metric_name: "jaccard_similarity".to_string(),
        value: 0.5,
        comparison_operator: FloatComparisonOperator::GreaterThan,
    });
    let order_by = vec![OrderBy {
        term: OrderByTerm::Metric {
            name: "jaccard_similarity".to_string(),
        },
        direction: OrderDirection::Desc,
    }];
    let opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter_node),
        limit: 3,
        order_by: Some(&order_by),
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 3);
    for inference in &res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
        assert!(json_inference.dispreferred_outputs.is_empty());
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_demonstration_output_source() {
    let client = make_embedded_gateway().await;
    let opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        output_source: InferenceOutputSource::Demonstration,
        limit: 5,
        offset: 1,
        ..Default::default()
    };

    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 5);
    for inference in &res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
        assert_eq!(json_inference.dispreferred_outputs.len(), 1);
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_boolean_metric_filter() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilter::BooleanMetric(BooleanMetricFilter {
        metric_name: "exact_match".to_string(),
        value: true,
    });
    let opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter_node),
        limit: 5,
        offset: 1,
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 5);
    for inference in &res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
        assert!(json_inference.dispreferred_outputs.is_empty());
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_and_filter_multiple_float_metrics() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilter::And {
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
    };
    let opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter_node),
        limit: 1,
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 1);
    for inference in &res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
        assert!(json_inference.dispreferred_outputs.is_empty());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_or_filter_mixed_metrics() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilter::Or {
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
                // Episode-level metric
                metric_name: "goal_achieved".to_string(),
                value: true,
            }),
        ],
    };
    let opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter_node),
        limit: 1,
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 1);
    for inference in &res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
        assert!(json_inference.dispreferred_outputs.is_empty());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_not_filter() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilter::Not {
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
    };
    let opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter_node),
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_simple_time_filter() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilter::Time(TimeFilter {
        time: DateTime::from_timestamp(1672531200, 0).unwrap(), // 2023-01-01 00:00:00 UTC
        comparison_operator: TimeComparisonOperator::GreaterThan,
    });
    let order_by = vec![
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
    ];
    let opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter_node),
        limit: 5,
        order_by: Some(&order_by),
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 5);

    for inference in &res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
    }

    // Verify ORDER BY timestamp ASC (secondary sort) - check that for same metric values, timestamps are ascending
    let mut prev_timestamp = None;
    for inference in &res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        if let Some(prev) = prev_timestamp {
            assert!(
                json_inference.timestamp >= prev,
                "Timestamps should be in ascending order for secondary sort. Got: {} >= {}",
                json_inference.timestamp,
                prev
            );
        }
        prev_timestamp = Some(json_inference.timestamp);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_simple_tag_filter() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilter::Tag(TagFilter {
        key: "tensorzero::evaluation_name".to_string(),
        value: "entity_extraction".to_string(),
        comparison_operator: TagComparisonOperator::Equal,
    });
    let opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter_node),
        limit: 200,
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 200);
    for inference in &res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
        assert_eq!(
            json_inference.tags["tensorzero::evaluation_name"],
            "entity_extraction"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_combined_time_and_tag_filter() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilter::And {
        children: vec![
            InferenceFilter::Time(TimeFilter {
                // 2025-04-14 23:30:00 UTC (should exclude some of these elements)
                time: DateTime::from_timestamp(1744673400, 0).unwrap(),
                comparison_operator: TimeComparisonOperator::GreaterThanOrEqual,
            }),
            InferenceFilter::Tag(TagFilter {
                key: "tensorzero::evaluation_name".to_string(),
                value: "haiku".to_string(),
                comparison_operator: TagComparisonOperator::Equal,
            }),
        ],
    };
    let opts = ListInferencesParams {
        function_name: Some("write_haiku"),
        filters: Some(&filter_node),
        limit: 50,
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 50);
    for inference in &res {
        let StoredInference::Chat(chat_inference) = inference else {
            panic!("Expected a Chat inference");
        };
        assert_eq!(chat_inference.function_name, "write_haiku");
        assert_eq!(chat_inference.tags["tensorzero::evaluation_name"], "haiku");
        assert!(chat_inference.timestamp >= DateTime::from_timestamp(1744673400, 0).unwrap());
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_query_by_ids_json_only() {
    let client = make_embedded_gateway().await;

    // First, get some JSON inference IDs
    let opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        limit: 3,
        ..Default::default()
    };
    let initial_res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(initial_res.len(), 3);

    // Extract the IDs
    let ids: Vec<_> = initial_res
        .iter()
        .map(|inf| match inf {
            StoredInference::Json(j) => j.inference_id,
            StoredInference::Chat(_) => panic!("Expected JSON inference"),
        })
        .collect();

    // Now query by IDs without function_name
    let opts = ListInferencesParams {
        function_name: None,
        ids: Some(&ids),
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();

    // Should get back the same 3 inferences
    assert_eq!(res.len(), 3);
    for inference in &res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected JSON inference");
        };
        assert!(ids.contains(&json_inference.inference_id));
        assert_eq!(json_inference.function_name, "extract_entities");
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_query_by_ids_chat_only() {
    let client = make_embedded_gateway().await;

    // First, get some Chat inference IDs
    let opts = ListInferencesParams {
        function_name: Some("write_haiku"),
        limit: 2,
        ..Default::default()
    };
    let initial_res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(initial_res.len(), 2);

    // Extract the IDs
    let ids: Vec<_> = initial_res
        .iter()
        .map(|inf| match inf {
            StoredInference::Chat(c) => c.inference_id,
            StoredInference::Json(_) => panic!("Expected Chat inference"),
        })
        .collect();

    // Now query by IDs without function_name
    let opts = ListInferencesParams {
        ids: Some(&ids),
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();

    // Should get back the same 2 inferences
    assert_eq!(res.len(), 2);
    for inference in &res {
        let StoredInference::Chat(chat_inference) = inference else {
            panic!("Expected Chat inference");
        };
        assert!(ids.contains(&chat_inference.inference_id));
        assert_eq!(chat_inference.function_name, "write_haiku");
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_query_by_ids_unknown_id_returns_empty() {
    let client = make_embedded_gateway().await;

    // Query by an unknown ID
    let unknown_ids = [Uuid::now_v7()];
    let opts = ListInferencesParams {
        ids: Some(&unknown_ids),
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();

    assert!(res.is_empty(), "Expected empty result for unknown ID");
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_query_by_ids_mixed_types() {
    let client = make_embedded_gateway().await;

    // Get some JSON inference IDs
    let json_opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        limit: 2,
        ..Default::default()
    };
    let json_res = client
        .experimental_list_inferences(json_opts)
        .await
        .unwrap();

    // Get some Chat inference IDs
    let chat_opts = ListInferencesParams {
        function_name: Some("write_haiku"),
        limit: 2,
        ..Default::default()
    };
    let chat_res = client
        .experimental_list_inferences(chat_opts)
        .await
        .unwrap();

    // Combine the IDs
    let mut ids: Vec<_> = json_res
        .iter()
        .map(|inf| match inf {
            StoredInference::Json(j) => j.inference_id,
            StoredInference::Chat(_) => panic!("Expected JSON inference"),
        })
        .collect();
    ids.extend(chat_res.iter().map(|inf| match inf {
        StoredInference::Chat(c) => c.inference_id,
        StoredInference::Json(_) => panic!("Expected Chat inference"),
    }));

    // Now query by mixed IDs without function_name
    let opts = ListInferencesParams {
        ids: Some(&ids),
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();

    // Should get back 4 inferences (2 JSON + 2 Chat)
    assert_eq!(res.len(), 4);

    let mut json_count = 0;
    let mut chat_count = 0;
    for inference in &res {
        match inference {
            StoredInference::Json(json_inference) => {
                assert!(ids.contains(&json_inference.inference_id));
                assert_eq!(json_inference.function_name, "extract_entities");
                json_count += 1;
            }
            StoredInference::Chat(chat_inference) => {
                assert!(ids.contains(&chat_inference.inference_id));
                assert_eq!(chat_inference.function_name, "write_haiku");
                chat_count += 1;
            }
        }
    }
    assert_eq!(json_count, 2);
    assert_eq!(chat_count, 2);
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_query_by_ids_with_order_by_timestamp() {
    let client = make_embedded_gateway().await;

    // Get some mixed inference IDs
    let json_opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        limit: 3,
        ..Default::default()
    };
    let json_res = client
        .experimental_list_inferences(json_opts)
        .await
        .unwrap();

    let chat_opts = ListInferencesParams {
        function_name: Some("write_haiku"),
        limit: 3,
        ..Default::default()
    };
    let chat_res = client
        .experimental_list_inferences(chat_opts)
        .await
        .unwrap();

    let mut ids: Vec<_> = json_res
        .iter()
        .map(|inf| match inf {
            StoredInference::Json(j) => j.inference_id,
            StoredInference::Chat(_) => panic!("Expected JSON inference"),
        })
        .collect();
    ids.extend(chat_res.iter().map(|inf| match inf {
        StoredInference::Chat(c) => c.inference_id,
        StoredInference::Json(_) => panic!("Expected Chat inference"),
    }));

    // Query with ORDER BY timestamp DESC
    let order_by = vec![OrderBy {
        term: OrderByTerm::Timestamp,
        direction: OrderDirection::Desc,
    }];
    let opts = ListInferencesParams {
        ids: Some(&ids),
        order_by: Some(&order_by),
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();

    assert_eq!(res.len(), 6);

    // Verify timestamps are in descending order
    let mut prev_timestamp = None;
    for inference in &res {
        let timestamp = match inference {
            StoredInference::Json(j) => j.timestamp,
            StoredInference::Chat(c) => c.timestamp,
        };
        if let Some(prev) = prev_timestamp {
            assert!(
                timestamp <= prev,
                "Timestamps should be in descending order. Got: {timestamp} <= {prev}"
            );
        }
        prev_timestamp = Some(timestamp);
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_query_by_ids_with_order_by_metric_errors() {
    let client = make_embedded_gateway().await;

    // Get some JSON inference IDs
    let opts = ListInferencesParams {
        function_name: Some("extract_entities"),
        limit: 2,
        ..Default::default()
    };
    let initial_res = client.experimental_list_inferences(opts).await.unwrap();

    let ids: Vec<_> = initial_res
        .iter()
        .map(|inf| match inf {
            StoredInference::Json(j) => j.inference_id,
            StoredInference::Chat(_) => panic!("Expected JSON inference"),
        })
        .collect();

    // Try to ORDER BY a metric without function_name - should error
    let order_by = vec![OrderBy {
        term: OrderByTerm::Metric {
            name: "jaccard_similarity".to_string(),
        },
        direction: OrderDirection::Desc,
    }];
    let opts = ListInferencesParams {
        ids: Some(&ids),
        order_by: Some(&order_by),
        ..Default::default()
    };
    let res = client.experimental_list_inferences(opts).await;

    // Should error because ORDER BY metric is not supported without function_name
    assert!(res.is_err());
    let err_msg = format!("{:?}", res.unwrap_err());
    assert!(err_msg.contains("not supported"));
}
