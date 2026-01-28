use std::path::Path;

use tensorzero_core::{
    config::{Config, ConfigFileGlob, MetricConfigLevel},
    db::{
        clickhouse::{query_builder::InferenceFilter, test_helpers::get_clickhouse},
        feedback::{FeedbackQueries, FeedbackRow},
        inferences::{FunctionInfo, InferenceOutputSource, InferenceQueries, ListInferencesParams},
    },
    endpoints::stored_inferences::v1::types::DemonstrationFeedbackFilter,
    inference::types::FunctionType,
};
use uuid::Uuid;

async fn get_e2e_config() -> Config {
    Config::load_from_path_optional_verify_credentials(
        &ConfigFileGlob::new_from_path(Path::new("tests/e2e/config/tensorzero.*.toml")).unwrap(),
        false,
    )
    .await
    .unwrap()
    .into_config_without_writing_for_tests()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_inferences_filtered_to_has_demonstrations() {
    let config = get_e2e_config().await;
    let clickhouse = get_clickhouse().await;
    let filter = InferenceFilter::DemonstrationFeedback(DemonstrationFeedbackFilter {
        has_demonstration: true,
    });

    let inferences = clickhouse
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 3,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should return inferences with demonstration feedback"
    );

    for inference in &inferences {
        let inference_id = inference.id();
        let feedback = clickhouse
            .query_feedback_by_target_id(inference_id, None, None, Some(50))
            .await
            .unwrap();

        let has_demonstration = feedback
            .iter()
            .any(|row| matches!(row, FeedbackRow::Demonstration(_)));

        assert!(
            has_demonstration,
            "Expected demonstration feedback for inference {inference_id}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_inferences_filtered_to_no_demonstrations() {
    let config = get_e2e_config().await;
    let clickhouse = get_clickhouse().await;
    let filter = InferenceFilter::DemonstrationFeedback(DemonstrationFeedbackFilter {
        has_demonstration: false,
    });

    let inferences = clickhouse
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 3,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should return inferences without demonstration feedback"
    );

    for inference in &inferences {
        let inference_id = match inference {
            tensorzero_core::stored_inference::StoredInferenceDatabase::Json(json_inf) => {
                json_inf.inference_id
            }
            tensorzero_core::stored_inference::StoredInferenceDatabase::Chat(chat_inf) => {
                chat_inf.inference_id
            }
        };
        let feedback = clickhouse
            .query_feedback_by_target_id(inference_id, None, None, Some(50))
            .await
            .unwrap();
        let has_demonstration = feedback
            .iter()
            .any(|row| matches!(row, FeedbackRow::Demonstration(_)));

        assert!(
            !has_demonstration,
            "Expected no demonstration feedback for inference {inference_id}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_output_chat_inference() {
    let config = get_e2e_config().await;
    let clickhouse = get_clickhouse().await;

    // First, list some chat inferences to get a valid inference_id
    let inferences = clickhouse
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("write_haiku"),
                output_source: InferenceOutputSource::Inference,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should have at least one chat inference"
    );

    let inference = &inferences[0];
    let inference_id = inference.id();

    // Get function info for this inference
    let function_info = clickhouse
        .get_function_info(&inference_id, MetricConfigLevel::Inference)
        .await
        .unwrap()
        .expect("Should find function info for existing inference");

    assert_eq!(
        function_info.function_type,
        FunctionType::Chat,
        "write_haiku should be a Chat function"
    );

    // Now test get_inference_output
    let output = clickhouse
        .get_inference_output(&function_info, inference_id)
        .await
        .unwrap();

    assert!(
        output.is_some(),
        "Should return output for existing inference"
    );
    let output_str = output.unwrap();
    assert!(
        !output_str.is_empty(),
        "Output should not be empty for existing inference"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_output_json_inference() {
    let config = get_e2e_config().await;
    let clickhouse = get_clickhouse().await;

    // First, list some json inferences to get a valid inference_id
    let inferences = clickhouse
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                output_source: InferenceOutputSource::Inference,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should have at least one json inference"
    );

    let inference = &inferences[0];
    let inference_id = inference.id();

    // Get function info for this inference
    let function_info = clickhouse
        .get_function_info(&inference_id, MetricConfigLevel::Inference)
        .await
        .unwrap()
        .expect("Should find function info for existing inference");

    assert_eq!(
        function_info.function_type,
        FunctionType::Json,
        "extract_entities should be a Json function"
    );

    // Now test get_inference_output
    let output = clickhouse
        .get_inference_output(&function_info, inference_id)
        .await
        .unwrap();

    assert!(
        output.is_some(),
        "Should return output for existing json inference"
    );
    let output_str = output.unwrap();
    assert!(
        !output_str.is_empty(),
        "Output should not be empty for existing json inference"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_output_not_found() {
    let clickhouse = get_clickhouse().await;

    // Create a fake function_info with a non-existent inference_id
    let fake_function_info = FunctionInfo {
        function_name: "write_haiku".to_string(),
        function_type: FunctionType::Chat,
        variant_name: "test_variant".to_string(),
        episode_id: Uuid::now_v7(),
    };

    let non_existent_inference_id = Uuid::now_v7();

    let output = clickhouse
        .get_inference_output(&fake_function_info, non_existent_inference_id)
        .await
        .unwrap();

    assert!(
        output.is_none(),
        "Should return None for non-existent inference"
    );
}
