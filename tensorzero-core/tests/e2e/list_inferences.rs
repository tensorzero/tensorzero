use chrono::DateTime;
use tensorzero::{
    BooleanMetricFilter, FloatComparisonOperator, FloatMetricFilter, InferenceFilterTreeNode,
    InferenceOutputSource, ListInferencesParams, StoredInference, TagComparisonOperator, TagFilter,
    TimeComparisonOperator, TimeFilter,
};
use tensorzero_core::db::clickhouse::{
    query_builder::{OrderBy, OrderByTerm, OrderDirection},
    ClickhouseFormat,
};

#[tokio::test(flavor = "multi_thread")]
pub async fn test_simple_query_json_function() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let order_by = vec![OrderBy {
        term: OrderByTerm::Timestamp,
        direction: OrderDirection::Desc,
    }];
    let opts = ListInferencesParams {
        function_name: "extract_entities",
        variant_name: None,
        filters: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(2),
        offset: None,
        order_by: Some(&order_by),
        format: ClickhouseFormat::JsonEachRow,
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
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let order_by = vec![OrderBy {
        term: OrderByTerm::Timestamp,
        direction: OrderDirection::Asc,
    }];
    let opts = ListInferencesParams {
        function_name: "write_haiku",
        variant_name: None,
        filters: None,
        output_source: InferenceOutputSource::Demonstration,
        limit: Some(3),
        offset: Some(3),
        order_by: Some(&order_by),
        format: ClickhouseFormat::JsonEachRow,
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
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
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
        function_name: "extract_entities",
        variant_name: None,
        filters: Some(&filter_node),
        output_source: InferenceOutputSource::Inference,
        limit: Some(3),
        offset: None,
        order_by: Some(&order_by),
        format: ClickhouseFormat::JsonEachRow,
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
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let opts = ListInferencesParams {
        function_name: "extract_entities",
        variant_name: None,
        filters: None,
        output_source: InferenceOutputSource::Demonstration,
        limit: Some(5),
        offset: Some(1),
        order_by: None,
        format: ClickhouseFormat::JsonEachRow,
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
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::BooleanMetric(BooleanMetricFilter {
        metric_name: "exact_match".to_string(),
        value: true,
    });
    let opts = ListInferencesParams {
        function_name: "extract_entities",
        variant_name: None,
        filters: Some(&filter_node),
        output_source: InferenceOutputSource::Inference,
        limit: Some(5),
        offset: Some(1),
        order_by: None,
        format: ClickhouseFormat::JsonEachRow,
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
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::And {
        children: vec![
            InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                metric_name: "jaccard_similarity".to_string(),
                value: 0.5,
                comparison_operator: FloatComparisonOperator::GreaterThan,
            }),
            InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                metric_name: "jaccard_similarity".to_string(),
                value: 0.8,
                comparison_operator: FloatComparisonOperator::LessThan,
            }),
        ],
    };
    let opts = ListInferencesParams {
        function_name: "extract_entities",
        variant_name: None,
        filters: Some(&filter_node),
        output_source: InferenceOutputSource::Inference,
        limit: Some(1),
        offset: None,
        order_by: None,
        format: ClickhouseFormat::JsonEachRow,
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
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::Or {
        children: vec![
            InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                metric_name: "jaccard_similarity".to_string(),
                value: 0.8,
                comparison_operator: FloatComparisonOperator::GreaterThanOrEqual,
            }),
            InferenceFilterTreeNode::BooleanMetric(BooleanMetricFilter {
                metric_name: "exact_match".to_string(),
                value: true,
            }),
            InferenceFilterTreeNode::BooleanMetric(BooleanMetricFilter {
                // Episode-level metric
                metric_name: "goal_achieved".to_string(),
                value: true,
            }),
        ],
    };
    let opts = ListInferencesParams {
        function_name: "extract_entities",
        variant_name: None,
        filters: Some(&filter_node),
        output_source: InferenceOutputSource::Inference,
        limit: Some(1),
        offset: None,
        order_by: None,
        format: ClickhouseFormat::JsonEachRow,
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
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::Not {
        child: Box::new(InferenceFilterTreeNode::Or {
            children: vec![
                InferenceFilterTreeNode::BooleanMetric(BooleanMetricFilter {
                    metric_name: "exact_match".to_string(),
                    value: true,
                }),
                InferenceFilterTreeNode::BooleanMetric(BooleanMetricFilter {
                    metric_name: "exact_match".to_string(),
                    value: false,
                }),
            ],
        }),
    };
    let opts = ListInferencesParams {
        function_name: "extract_entities",
        variant_name: None,
        filters: Some(&filter_node),
        output_source: InferenceOutputSource::Inference,
        limit: None,
        offset: None,
        order_by: None,
        format: ClickhouseFormat::JsonEachRow,
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_simple_time_filter() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::Time(TimeFilter {
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
        function_name: "extract_entities",
        variant_name: None,
        filters: Some(&filter_node),
        output_source: InferenceOutputSource::Inference,
        limit: Some(5),
        offset: None,
        order_by: Some(&order_by),
        format: ClickhouseFormat::JsonEachRow,
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
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::Tag(TagFilter {
        key: "tensorzero::evaluation_name".to_string(),
        value: "entity_extraction".to_string(),
        comparison_operator: TagComparisonOperator::Equal,
    });
    let opts = ListInferencesParams {
        function_name: "extract_entities",
        variant_name: None,
        filters: Some(&filter_node),
        output_source: InferenceOutputSource::Inference,
        limit: Some(200),
        offset: None,
        order_by: None,
        format: ClickhouseFormat::JsonEachRow,
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
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::And {
        children: vec![
            InferenceFilterTreeNode::Time(TimeFilter {
                // 2025-04-14 23:30:00 UTC (should exclude some of these elements)
                time: DateTime::from_timestamp(1744673400, 0).unwrap(),
                comparison_operator: TimeComparisonOperator::GreaterThanOrEqual,
            }),
            InferenceFilterTreeNode::Tag(TagFilter {
                key: "tensorzero::evaluation_name".to_string(),
                value: "haiku".to_string(),
                comparison_operator: TagComparisonOperator::Equal,
            }),
        ],
    };
    let opts = ListInferencesParams {
        function_name: "write_haiku",
        variant_name: None,
        filters: Some(&filter_node),
        output_source: InferenceOutputSource::Inference,
        limit: Some(50),
        offset: None,
        order_by: None,
        format: ClickhouseFormat::JsonEachRow,
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
