use tensorzero::{
    BooleanMetricNode, FloatComparisonOperator, FloatMetricNode, InferenceFilterTreeNode,
    InferenceOutputSource, ListInferencesParams, StoredInference,
};
use tensorzero_internal::clickhouse::ClickhouseFormat;

use crate::providers::common::make_embedded_gateway;

#[tokio::test(flavor = "multi_thread")]
pub async fn test_simple_query_json_function() {
    let client = make_embedded_gateway().await;
    let opts = ListInferencesParams {
        function_name: "extract_entities",
        variant_name: None,
        filters: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(2),
        offset: None,
        format: ClickhouseFormat::JsonEachRow,
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 2);
    for inference in res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_simple_query_chat_function() {
    let client = make_embedded_gateway().await;
    let opts = ListInferencesParams {
        function_name: "write_haiku",
        variant_name: Some("better_prompt_haiku_3_5"),
        filters: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(3),
        offset: Some(3),
        format: ClickhouseFormat::JsonEachRow,
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 3);
    for inference in res {
        let StoredInference::Chat(chat_inference) = inference else {
            panic!("Expected a Chat inference");
        };
        assert_eq!(chat_inference.function_name, "write_haiku");
        assert_eq!(chat_inference.variant_name, "better_prompt_haiku_3_5");
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_simple_query_with_float_filter() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
        metric_name: "jaccard_similarity".to_string(),
        value: 0.5,
        comparison_operator: FloatComparisonOperator::GreaterThan,
    });
    let opts = ListInferencesParams {
        function_name: "extract_entities",
        variant_name: None,
        filters: Some(&filter_node),
        output_source: InferenceOutputSource::Inference,
        limit: Some(1),
        offset: None,
        format: ClickhouseFormat::JsonEachRow,
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 1);
    for inference in res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_demonstration_output_source() {
    let client = make_embedded_gateway().await;
    let opts = ListInferencesParams {
        function_name: "extract_entities",
        variant_name: None,
        filters: None,
        output_source: InferenceOutputSource::Demonstration,
        limit: Some(5),
        offset: Some(1),
        format: ClickhouseFormat::JsonEachRow,
    };

    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 5);
    for inference in res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_boolean_metric_filter() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::BooleanMetric(BooleanMetricNode {
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
        format: ClickhouseFormat::JsonEachRow,
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 5);
    for inference in res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_and_filter_multiple_float_metrics() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::And {
        children: vec![
            InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
                metric_name: "jaccard_similarity".to_string(),
                value: 0.5,
                comparison_operator: FloatComparisonOperator::GreaterThan,
            }),
            InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
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
        format: ClickhouseFormat::JsonEachRow,
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 1);
    for inference in res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_or_filter_mixed_metrics() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::Or {
        children: vec![
            InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
                metric_name: "jaccard_similarity".to_string(),
                value: 0.8,
                comparison_operator: FloatComparisonOperator::GreaterThanOrEqual,
            }),
            InferenceFilterTreeNode::BooleanMetric(BooleanMetricNode {
                metric_name: "exact_match".to_string(),
                value: true,
            }),
            InferenceFilterTreeNode::BooleanMetric(BooleanMetricNode {
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
        format: ClickhouseFormat::JsonEachRow,
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 1);
    for inference in res {
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a JSON inference");
        };
        assert_eq!(json_inference.function_name, "extract_entities");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_not_filter() {
    let client = make_embedded_gateway().await;
    let filter_node = InferenceFilterTreeNode::Not {
        child: Box::new(InferenceFilterTreeNode::Or {
            children: vec![
                InferenceFilterTreeNode::BooleanMetric(BooleanMetricNode {
                    metric_name: "exact_match".to_string(),
                    value: true,
                }),
                InferenceFilterTreeNode::BooleanMetric(BooleanMetricNode {
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
        format: ClickhouseFormat::JsonEachRow,
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 0);
}
