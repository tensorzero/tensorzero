//! Handler for checking if human feedback exists for an inference evaluation.

use axum::Json;
use axum::extract::{Path, State};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::db::evaluation_queries::{EvaluationQueries, InferenceEvaluationHumanFeedbackRow};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Request body for checking human feedback.
#[derive(Debug, Deserialize)]
pub struct GetHumanFeedbackRequest {
    /// The name of the metric being evaluated.
    pub metric_name: String,
    /// The serialized inference output to match against.
    pub output: String,
}

/// Response for the check human feedback endpoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct GetHumanFeedbackResponse {
    /// The human feedback result, if it exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feedback: Option<InferenceEvaluationHumanFeedbackRow>,
}

/// Handler for `POST /internal/evaluations/datapoints/{datapoint_id}/get_human_feedback`
///
/// Checks if human feedback exists for a given combination of metric name,
/// datapoint ID, and output.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "evaluations.get_human_feedback", skip_all)]
pub async fn get_human_feedback_handler(
    State(app_state): AppState,
    Path(datapoint_id): Path<Uuid>,
    Json(request): Json<GetHumanFeedbackRequest>,
) -> Result<Json<GetHumanFeedbackResponse>, Error> {
    let response = get_human_feedback(
        &app_state.clickhouse_connection_info,
        &request.metric_name,
        &datapoint_id,
        &request.output,
    )
    .await?;

    Ok(Json(response))
}

/// Core business logic for checking human feedback.
pub async fn get_human_feedback(
    clickhouse: &impl EvaluationQueries,
    metric_name: &str,
    datapoint_id: &Uuid,
    output: &str,
) -> Result<GetHumanFeedbackResponse, Error> {
    let feedback = clickhouse
        .get_inference_evaluation_human_feedback(metric_name, datapoint_id, output)
        .await?;

    Ok(GetHumanFeedbackResponse { feedback })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::evaluation_queries::MockEvaluationQueries;

    #[tokio::test]
    async fn get_human_feedback_returns_feedback_when_exists() {
        let datapoint_id = Uuid::now_v7();
        let evaluator_inference_id = Uuid::now_v7();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_inference_evaluation_human_feedback()
            .withf(move |metric_name, dp_id, output| {
                metric_name == "test_metric"
                    && *dp_id == datapoint_id
                    && output == r#"{"raw":"test output"}"#
            })
            .times(1)
            .returning(move |_, _, _| {
                Box::pin(async move {
                    Ok(Some(InferenceEvaluationHumanFeedbackRow {
                        value: serde_json::json!(0.95),
                        evaluator_inference_id,
                    }))
                })
            });

        let result = get_human_feedback(
            &mock_clickhouse,
            "test_metric",
            &datapoint_id,
            r#"{"raw":"test output"}"#,
        )
        .await
        .unwrap();

        assert!(result.feedback.is_some());
        let feedback = result.feedback.unwrap();
        assert_eq!(feedback.value, serde_json::json!(0.95));
        assert_eq!(feedback.evaluator_inference_id, evaluator_inference_id);
    }

    #[tokio::test]
    async fn get_human_feedback_returns_none_when_not_exists() {
        let datapoint_id = Uuid::now_v7();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_inference_evaluation_human_feedback()
            .times(1)
            .returning(|_, _, _| Box::pin(async move { Ok(None) }));

        let result = get_human_feedback(
            &mock_clickhouse,
            "nonexistent_metric",
            &datapoint_id,
            "test output",
        )
        .await
        .unwrap();

        assert!(result.feedback.is_none());
    }

    #[tokio::test]
    async fn get_human_feedback_with_boolean_value() {
        let datapoint_id = Uuid::now_v7();
        let evaluator_inference_id = Uuid::now_v7();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_inference_evaluation_human_feedback()
            .times(1)
            .returning(move |_, _, _| {
                Box::pin(async move {
                    Ok(Some(InferenceEvaluationHumanFeedbackRow {
                        value: serde_json::json!(true),
                        evaluator_inference_id,
                    }))
                })
            });

        let result = get_human_feedback(&mock_clickhouse, "test_metric", &datapoint_id, "output")
            .await
            .unwrap();

        assert!(result.feedback.is_some());
        let feedback = result.feedback.unwrap();
        assert_eq!(feedback.value, serde_json::json!(true));
    }

    #[tokio::test]
    async fn get_human_feedback_with_object_value() {
        let datapoint_id = Uuid::now_v7();
        let evaluator_inference_id = Uuid::now_v7();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_inference_evaluation_human_feedback()
            .times(1)
            .returning(move |_, _, _| {
                Box::pin(async move {
                    Ok(Some(InferenceEvaluationHumanFeedbackRow {
                        value: serde_json::json!({"score": 0.8, "reason": "good"}),
                        evaluator_inference_id,
                    }))
                })
            });

        let result = get_human_feedback(&mock_clickhouse, "test_metric", &datapoint_id, "output")
            .await
            .unwrap();

        assert!(result.feedback.is_some());
        let feedback = result.feedback.unwrap();
        assert_eq!(
            feedback.value,
            serde_json::json!({"score": 0.8, "reason": "good"})
        );
    }
}
