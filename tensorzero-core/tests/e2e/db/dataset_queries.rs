use tensorzero::{
    DatasetQueryParams, FloatComparisonOperator, GetDatasetMetadataParams, GetDatasetRowsParams,
};
use tensorzero_core::config::{MetricConfigLevel, MetricConfigType};
use tensorzero_core::db::clickhouse::dataset_queries::{
    DatasetMetadata, DatasetOutputSource, DatasetQueries, MetricFilter,
};
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::endpoints::datasets::DatapointKind;

#[tokio::test]
async fn test_count_rows_for_chat_dataset_with_write_haiku_function() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: None,
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing chat inferences");
}

#[tokio::test]
async fn test_count_rows_for_json_dataset_with_extract_entities_function_and_variant() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Json,
        function_name: Some("extract_entities".to_string()),
        dataset_name: None,
        variant_name: Some("llama_8b_initial_prompt".to_string()),
        extra_where: None,
        extra_params: None,
        metric_filter: None,
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing json inferences");
}

#[tokio::test]
async fn test_count_rows_with_float_metric_filter() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "haiku_rating".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing inferences with float metric filter");
}

#[tokio::test]
async fn test_count_rows_chat_datapoints_with_boolean_metric_filter() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "haiku_score".to_string(),
            metric_type: MetricConfigType::Boolean,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.0,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::Inference,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing inferences with boolean metric filter");
}

#[tokio::test]
async fn test_count_rows_json_datapoints_with_boolean_metric_filter() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Json,
        function_name: Some("extract_entities".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "exact_match".to_string(),
            metric_type: MetricConfigType::Boolean,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.0,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::Inference,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing inferences with boolean metric filter");
}

#[tokio::test]
async fn test_count_rows_chat_datapoints_with_boolean_metric_filter_at_episode_level() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "haiku_score_episode".to_string(),
            metric_type: MetricConfigType::Boolean,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.0,
            join_on: MetricConfigLevel::Episode,
        }),
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing inferences with boolean metric filter");
}

#[tokio::test]
async fn test_count_rows_json_datapoints_with_boolean_metric_filter_at_episode_level() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Json,
        function_name: Some("extract_entities".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "exact_match_episode".to_string(),
            metric_type: MetricConfigType::Boolean,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.0,
            join_on: MetricConfigLevel::Episode,
        }),
        output_source: DatasetOutputSource::Inference,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing inferences with boolean metric filter");
}

#[tokio::test]
async fn test_count_rows_chat_datapoints_with_float_metric_filter_at_inference_level() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "haiku_rating".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing inferences with float metric filter");
}

#[tokio::test]
async fn test_count_rows_json_datapoints_with_float_metric_filter_at_inference_level() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Json,
        function_name: Some("extract_entities".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "jaccard_similarity".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing inferences with float metric filter");
}

#[tokio::test]
async fn test_count_rows_chat_datapoints_with_float_metric_filter_at_episode_level() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "haiku_rating_episode".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Episode,
        }),
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing inferences with float metric filter");
}

#[tokio::test]
async fn test_count_rows_json_datapoints_with_float_metric_filter_at_episode_level() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Json,
        function_name: Some("extract_entities".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "jaccard_similarity_episode".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Episode,
        }),
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing inferences with float metric filter");
}

#[tokio::test]
async fn test_count_rows_chat_datapoints_with_metric_filter_and_demonstration_join() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "haiku_rating".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::Demonstration,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing inferences with float metric filter");
}

#[tokio::test]
async fn test_count_rows_json_datapoints_with_float_metric_filter_and_demonstration_join() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Json,
        function_name: Some("extract_entities".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "jaccard_similarity".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::Demonstration,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert_eq!(count, 0, "Should have 0 inferences");
}

#[tokio::test]
async fn test_get_dataset_metadata_returns_correct_counts_for_all_datasets() {
    let params = GetDatasetMetadataParams {
        function_name: None,
        page_size: None,
        offset: None,
    };
    let metadata = get_clickhouse()
        .await
        .get_dataset_metadata(&params)
        .await
        .unwrap();

    // We only assert that the result contains the expected datasets
    // Because other tests insert into the table, there could be additional datasets
    assert!(metadata.contains(&DatasetMetadata {
        dataset_name: "foo".to_string(),
        count: 119,
        last_updated: "2025-04-15T02:33:58Z".to_string(),
    }));
    assert!(metadata.contains(&DatasetMetadata {
        dataset_name: "bar".to_string(),
        count: 6,
        last_updated: "2025-03-14T17:38:09Z".to_string(),
    }));
}

#[tokio::test]
async fn test_get_dataset_metadata_returns_correct_counts_for_specific_function() {
    let params = GetDatasetMetadataParams {
        function_name: Some("write_haiku".to_string()),
        page_size: None,
        offset: None,
    };
    let metadata = get_clickhouse()
        .await
        .get_dataset_metadata(&params)
        .await
        .unwrap();

    // We only assert that the result contains the expected dataset
    // Because other tests insert into the table, there could be additional datasets
    assert!(metadata.contains(&DatasetMetadata {
        dataset_name: "foo".to_string(),
        count: 77,
        last_updated: "2025-03-23T20:03:59Z".to_string(),
    }));
}

#[tokio::test]
async fn test_get_dataset_rows_returns_correct_rows_for_specific_dataset() {
    let params = GetDatasetRowsParams {
        dataset_name: "notadataset".to_string(),
        page_size: 10,
        offset: 0,
    };

    let rows = get_clickhouse()
        .await
        .get_dataset_rows(&params)
        .await
        .unwrap();

    assert_eq!(rows.len(), 0, "Should have 0 rows");
}

#[tokio::test]
async fn test_get_dataset_rows_pages_correctly() {
    let mut all_rows = Vec::new();
    let mut offset = 0;
    let page_size = 10;

    loop {
        let params = GetDatasetRowsParams {
            dataset_name: "foo".to_string(),
            page_size,
            offset,
        };
        let rows = get_clickhouse()
            .await
            .get_dataset_rows(&params)
            .await
            .unwrap();
        let is_last_page = rows.len() != page_size as usize;

        all_rows.extend(rows);
        offset += page_size;

        if is_last_page {
            break;
        }
    }

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(all_rows.len() > 0, "Should have existing rows");
}
