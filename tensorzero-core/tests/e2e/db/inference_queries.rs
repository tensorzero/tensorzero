use std::path::Path;

use tensorzero_core::{
    config::{Config, ConfigFileGlob},
    db::{
        clickhouse::{query_builder::InferenceFilter, test_helpers::get_clickhouse},
        feedback::{FeedbackQueries, FeedbackRow},
        inferences::{InferenceOutputSource, InferenceQueries, ListInferencesParams},
    },
    endpoints::stored_inferences::v1::types::DemonstrationFeedbackFilter,
};

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
