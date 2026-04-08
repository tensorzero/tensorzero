use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{MultipleChoiceOption, UserQuestionAnswer};

/// A block of rich content displayed alongside an autoeval example.
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[ts(export, tag = "type", rename_all = "snake_case")]
pub enum AutoEvalContentBlock {
    /// Rendered as formatted markdown.
    Markdown {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        label: Option<String>,
    },
    /// Rendered as a formatted JSON viewer.
    Json {
        data: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        label: Option<String>,
    },
}

// =============================================================================
// AutoEval Example Labeling Types
// =============================================================================

/// Payload for an autoeval example labeling event.
///
/// Groups labeled examples together, each with rich context blocks
/// (e.g. prompt/response) and associated labeling questions.
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct EventPayloadAutoEvalExampleLabeling {
    pub examples: Vec<AutoEvalExampleLabeling>,
}

/// A single example to label, with context and a structured labeling question.
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct AutoEvalExampleLabeling {
    /// Optional since some behaviors only depend on the response
    pub maybe_excerpted_prompt: Option<AutoEvalContentBlock>,
    /// Mandatory since we always need the response for evals
    pub maybe_excerpted_response: AutoEvalContentBlock,
    pub source: AutoEvalExampleSource,
    /// The multiple-choice labeling question for this example.
    pub label_question: AutoEvalLabelQuestion,
    /// An optional free-response explanation question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub explanation_question: Option<AutoEvalExplanationQuestion>,
}

#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AutoEvalExampleSource {
    Inference(AutoEvalExampleSourceInference),
    Synthetic(AutoEvalExampleSourceSynthetic),
}

#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct AutoEvalExampleSourceInference {
    /// The inference ID of the historical datapoint for auto evals
    pub id: Uuid,
}

#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct AutoEvalExampleSourceSynthetic {
    pub full_prompt: Option<AutoEvalContentBlock>,
    pub full_response: AutoEvalContentBlock,
}

/// A multiple-choice labeling question within an autoeval example.
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct AutoEvalLabelQuestion {
    pub id: Uuid,
    pub header: String,
    pub question: String,
    pub options: Vec<MultipleChoiceOption>,
}

/// A free-response explanation question within an autoeval example.
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct AutoEvalExplanationQuestion {
    pub id: Uuid,
    pub header: String,
    pub question: String,
}

/// Minimal input payload for submitting autoeval example labeling answers.
/// The server enriches this with context from the original labeling event before storing.
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct CreateEventPayloadAutoEvalExampleLabelingAnswers {
    /// Map from question UUID to response.
    pub responses: HashMap<Uuid, UserQuestionAnswer>,
    /// The event ID of the original `AutoEvalExampleLabeling` event these answers correspond to.
    pub auto_eval_example_labeling_event_id: Uuid,
}

/// Self-contained read-only payload for labeled autoeval examples.
/// Includes the full context blocks so the UI can render everything
/// without looking up the original labeling event.
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct EventPayloadAutoEvalExampleLabelingAnswers {
    pub examples: Vec<AutoEvalLabeledExample>,
    /// The event ID of the original `AutoEvalExampleLabeling` event these answers correspond to.
    pub auto_eval_example_labeling_event_id: Uuid,
}

/// A labeled example with its full context and submitted answers.
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct AutoEvalLabeledExample {
    /// Optional since some behaviors only depend on the response
    pub maybe_excerpted_prompt: Option<AutoEvalContentBlock>,
    /// Mandatory since we always need the response for evals
    pub maybe_excerpted_response: AutoEvalContentBlock,
    pub source: AutoEvalExampleSource,
    /// The multiple-choice labeling question for this example.
    pub label_question: AutoEvalLabelQuestion,
    /// The user's answer to the label question.
    pub label_answer: UserQuestionAnswer,
    /// An optional free-response explanation question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub explanation_question: Option<AutoEvalExplanationQuestion>,
    /// The user's answer to the explanation question, if one was present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub explanation_answer: Option<UserQuestionAnswer>,
}

// =============================================================================
// AutoEval Behavior Spec Types
// =============================================================================

/// Payload for an autoeval behavior spec event.
///
/// Contains two required free-response fields: a target behavior description
/// and additional context, both with optional pre-filled defaults.
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct EventPayloadAutoEvalBehaviorSpec {
    /// The target behavior question.
    pub target_behavior: AutoEvalBehaviorSpecQuestion,
    /// The additional context question.
    pub additional_context: AutoEvalBehaviorSpecQuestion,
}

/// A free-response question within an autoeval behavior spec.
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct AutoEvalBehaviorSpecQuestion {
    pub id: Uuid,
    pub header: String,
    pub question: String,
    /// Pre-filled text shown in the input. The user can edit or submit as-is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub default_value: Option<String>,
}

/// Minimal input payload for submitting autoeval behavior spec answers.
/// The server enriches this with context from the original event before storing.
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct CreateEventPayloadAutoEvalBehaviorSpecAnswers {
    /// Map from question UUID to response.
    pub responses: HashMap<Uuid, UserQuestionAnswer>,
    /// The event ID of the original `AutoEvalBehaviorSpec` event these answers correspond to.
    pub auto_eval_behavior_spec_event_id: Uuid,
}

/// Self-contained read-only payload for answered autoeval behavior spec.
/// Includes the full questions so the UI can render everything
/// without looking up the original event.
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize)]
#[ts(export)]
pub struct EventPayloadAutoEvalBehaviorSpecAnswers {
    /// The target behavior question.
    pub target_behavior: AutoEvalBehaviorSpecQuestion,
    /// The user's answer to the target behavior question.
    pub target_behavior_answer: UserQuestionAnswer,
    /// The additional context question.
    pub additional_context: AutoEvalBehaviorSpecQuestion,
    /// The user's answer to the additional context question.
    pub additional_context_answer: UserQuestionAnswer,
    /// The event ID of the original `AutoEvalBehaviorSpec` event these answers correspond to.
    pub auto_eval_behavior_spec_event_id: Uuid,
}
