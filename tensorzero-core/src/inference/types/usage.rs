use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The type of API used for a model inference.
/// Used in raw usage reporting to help consumers interpret provider-specific usage data.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, ts_rs::TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum ApiType {
    ChatCompletions,
    Responses,
    Embeddings,
}

/// A single entry in the raw usage array, representing usage data from one model inference.
/// This preserves the original provider-specific usage object for fields that TensorZero
/// normalizes away (e.g., OpenAI's `reasoning_tokens`, Anthropic's `cache_read_input_tokens`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct RawUsageEntry {
    pub model_inference_id: Uuid,
    pub provider_type: String,
    pub api_type: ApiType,
    pub data: serde_json::Value,
}

pub fn raw_usage_entries_from_value(
    model_inference_id: Uuid,
    provider_type: &str,
    api_type: ApiType,
    usage: serde_json::Value,
) -> Vec<RawUsageEntry> {
    vec![RawUsageEntry {
        model_inference_id,
        provider_type: provider_type.to_string(),
        api_type,
        data: usage,
    }]
}

pub fn raw_usage_entries_from_usage<T: Serialize>(
    model_inference_id: Uuid,
    provider_type: &str,
    api_type: ApiType,
    usage: &T,
) -> Option<Vec<RawUsageEntry>> {
    serde_json::to_value(usage).ok().map(|usage| {
        raw_usage_entries_from_value(model_inference_id, provider_type, api_type, usage)
    })
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

/// Usage with optional raw provider-specific usage data.
/// This is used at the API response boundary to include `raw_usage` when requested.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct UsageWithRaw {
    #[serde(flatten)]
    #[ts(flatten)]
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub raw_usage: Option<Vec<RawUsageEntry>>,
}

impl From<Usage> for UsageWithRaw {
    fn from(usage: Usage) -> Self {
        Self {
            usage,
            raw_usage: None,
        }
    }
}

impl Usage {
    pub fn zero() -> Usage {
        Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
        }
    }

    pub fn total_tokens(&self) -> Option<u32> {
        match (self.input_tokens, self.output_tokens) {
            (Some(input), Some(output)) => Some(input + output),
            _ => None,
        }
    }

    /// Sum an iterator of Usage values.
    /// If any usage has None for a field, the sum for that field becomes None.
    pub fn sum_iter_strict<I: Iterator<Item = Usage>>(iter: I) -> Usage {
        iter.fold(Usage::zero(), |acc, usage| Usage {
            input_tokens: match (acc.input_tokens, usage.input_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
            output_tokens: match (acc.output_tokens, usage.output_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
        })
    }

    /// Sum two `Usage` instances.
    /// `None` contaminates on both sides.
    pub fn sum_strict(&mut self, other: &Usage) {
        self.input_tokens = match (self.input_tokens, other.input_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };

        self.output_tokens = match (self.output_tokens, other.output_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_sum_all_some() {
        let usages = vec![
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            },
            Usage {
                input_tokens: Some(5),
                output_tokens: Some(15),
            },
            Usage {
                input_tokens: Some(3),
                output_tokens: Some(7),
            },
        ];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(sum.input_tokens, Some(18));
        assert_eq!(sum.output_tokens, Some(42));
    }

    #[test]
    fn test_usage_sum_with_none() {
        // If any usage has None for a field, the sum for that field becomes None
        let usages = vec![
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            },
            Usage {
                input_tokens: None,
                output_tokens: Some(15),
            },
            Usage {
                input_tokens: Some(3),
                output_tokens: Some(7),
            },
        ];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(sum.input_tokens, None); // None because one usage had None
        assert_eq!(sum.output_tokens, Some(42)); // All had Some, so sum is Some
    }

    #[test]
    fn test_usage_sum_both_fields_none() {
        let usages = vec![
            Usage {
                input_tokens: Some(10),
                output_tokens: None,
            },
            Usage {
                input_tokens: None,
                output_tokens: Some(15),
            },
        ];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(sum.input_tokens, None);
        assert_eq!(sum.output_tokens, None);
    }

    #[test]
    fn test_usage_sum_empty() {
        let usages: Vec<Usage> = vec![];
        let sum = Usage::sum_iter_strict(usages.into_iter());
        // Empty sum should return Some(0) for both fields since we start with Some(0)
        assert_eq!(sum.input_tokens, Some(0));
        assert_eq!(sum.output_tokens, Some(0));
    }

    #[test]
    fn test_usage_sum_single() {
        let usages = vec![Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
        }];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(sum.input_tokens, Some(10));
        assert_eq!(sum.output_tokens, Some(20));
    }

    #[test]
    fn test_api_type_serialization() {
        assert_eq!(
            serde_json::to_string(&ApiType::ChatCompletions).unwrap(),
            "\"chat_completions\""
        );
        assert_eq!(
            serde_json::to_string(&ApiType::Responses).unwrap(),
            "\"responses\""
        );
        assert_eq!(
            serde_json::to_string(&ApiType::Embeddings).unwrap(),
            "\"embeddings\""
        );
    }

    #[test]
    fn test_raw_usage_entry_serialization() {
        let entry = RawUsageEntry {
            model_inference_id: Uuid::nil(),
            provider_type: "openai".to_string(),
            api_type: ApiType::ChatCompletions,
            data: serde_json::json!({
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "reasoning_tokens": 30
            }),
        };
        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["provider_type"], "openai");
        assert_eq!(json["api_type"], "chat_completions");
        assert_eq!(json["data"]["prompt_tokens"], 100);
        assert_eq!(json["data"]["reasoning_tokens"], 30);
    }

    #[test]
    fn test_raw_usage_entry_null_data() {
        let entry = RawUsageEntry {
            model_inference_id: Uuid::nil(),
            provider_type: "openai".to_string(),
            api_type: ApiType::ChatCompletions,
            data: serde_json::Value::Null,
        };
        let json = serde_json::to_value(&entry).unwrap();
        // data field should be present as null
        assert_eq!(json["data"], serde_json::Value::Null, "data should be null");
    }

    #[test]
    fn test_usage_with_raw_serialization() {
        let usage_with_raw = UsageWithRaw {
            usage: Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
            },
            raw_usage: Some(vec![RawUsageEntry {
                model_inference_id: Uuid::nil(),
                provider_type: "openai".to_string(),
                api_type: ApiType::ChatCompletions,
                data: serde_json::json!({"prompt_tokens": 100}),
            }]),
        };
        let json = serde_json::to_value(&usage_with_raw).unwrap();
        // Flattened fields should be at top level
        assert_eq!(json["input_tokens"], 100);
        assert_eq!(json["output_tokens"], 50);
        assert!(json["raw_usage"].is_array());
        assert_eq!(json["raw_usage"][0]["provider_type"], "openai");
    }

    #[test]
    fn test_usage_with_raw_no_raw_usage() {
        let usage_with_raw = UsageWithRaw {
            usage: Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
            },
            raw_usage: None,
        };
        let json = serde_json::to_string(&usage_with_raw).unwrap();
        // raw_usage should be omitted when None
        assert!(!json.contains("raw_usage"));
        // But input/output tokens should be present
        assert!(json.contains("input_tokens"));
        assert!(json.contains("output_tokens"));
    }
}
