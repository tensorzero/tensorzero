use super::{deserialize_delete, serialize_delete};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(transparent)]
pub struct ExtraHeadersConfig {
    pub data: Vec<ExtraHeader>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
pub struct ExtraHeader {
    pub name: String,
    #[serde(flatten)]
    pub kind: ExtraHeaderKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ExtraHeaderKind {
    Value(String),
    // We only allow `"delete": true` to be set - deserializing `"delete": false` will error
    #[serde(
        serialize_with = "serialize_delete",
        deserialize_with = "deserialize_delete"
    )]
    Delete,
}

/// The 'InferenceExtraHeaders' options provided directly in an inference request
/// These have not yet been filtered by variant name
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(transparent)]
pub struct UnfilteredInferenceExtraHeaders {
    pub extra_headers: Vec<InferenceExtraHeader>,
}

impl UnfilteredInferenceExtraHeaders {
    pub fn is_empty(&self) -> bool {
        self.extra_headers.is_empty()
    }

    /// Get a reference to the extra_headers vector
    pub fn as_slice(&self) -> &[InferenceExtraHeader] {
        &self.extra_headers
    }

    /// Filter the 'InferenceExtraHeader' options by variant name
    /// If the variant name is `None`, then all variant-specific extra header options are removed
    pub fn filter(self, variant_name: &str) -> FilteredInferenceExtraHeaders {
        FilteredInferenceExtraHeaders {
            data: self
                .extra_headers
                .into_iter()
                .filter(|config| config.should_apply_variant(variant_name))
                .collect(),
        }
    }
}

/// The result of filtering `InferenceExtraHeader` by variant name.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(transparent)]
pub struct FilteredInferenceExtraHeaders {
    pub data: Vec<InferenceExtraHeader>,
}

/// Holds the config-level and inference-level extra headers options
#[derive(Clone, Debug, Default, PartialEq, Serialize, ts_rs::TS)]
pub struct FullExtraHeadersConfig {
    pub variant_extra_headers: Option<ExtraHeadersConfig>,
    pub inference_extra_headers: FilteredInferenceExtraHeaders,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(untagged)]
pub enum InferenceExtraHeader {
    Provider {
        model_provider_name: String,
        name: String,
        #[serde(flatten)]
        kind: ExtraHeaderKind,
    },
    Variant {
        variant_name: String,
        name: String,
        #[serde(flatten)]
        kind: ExtraHeaderKind,
    },
    Always {
        name: String,
        #[serde(flatten)]
        kind: ExtraHeaderKind,
    },
}

impl InferenceExtraHeader {
    pub fn should_apply_variant(&self, variant_name: &str) -> bool {
        match self {
            InferenceExtraHeader::Provider { .. } => true,
            InferenceExtraHeader::Variant {
                variant_name: v, ..
            } => v == variant_name,
            InferenceExtraHeader::Always { .. } => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_extra_header_all_deserialize() {
        let json = r#"{"name": "X-Custom-Header", "value": "custom-value"}"#;
        let result: InferenceExtraHeader = serde_json::from_str(json).unwrap();
        match result {
            InferenceExtraHeader::Always { name, kind } => {
                assert_eq!(name, "X-Custom-Header");
                match kind {
                    ExtraHeaderKind::Value(v) => {
                        assert_eq!(v, "custom-value");
                    }
                    ExtraHeaderKind::Delete => panic!("Expected Value kind"),
                }
            }
            _ => panic!("Expected Always variant"),
        }
    }

    #[test]
    fn test_inference_extra_header_all_with_delete() {
        let json = r#"{"name": "X-Custom-Header", "delete": true}"#;
        let result: InferenceExtraHeader = serde_json::from_str(json).unwrap();
        match result {
            InferenceExtraHeader::Always { name, kind } => {
                assert_eq!(name, "X-Custom-Header");
                assert_eq!(kind, ExtraHeaderKind::Delete);
            }
            _ => panic!("Expected Always variant"),
        }
    }

    #[test]
    fn test_should_apply_variant_all() {
        let all_variant = InferenceExtraHeader::Always {
            name: "X-Custom-Header".to_string(),
            kind: ExtraHeaderKind::Value("value".to_string()),
        };

        // Always should apply to any variant
        assert!(all_variant.should_apply_variant("variant1"));
        assert!(all_variant.should_apply_variant("variant2"));
        assert!(all_variant.should_apply_variant("any_variant"));
    }

    #[test]
    fn test_should_apply_variant_provider() {
        let provider_variant = InferenceExtraHeader::Provider {
            model_provider_name: "openai".to_string(),
            name: "Authorization".to_string(),
            kind: ExtraHeaderKind::Value("Bearer token".to_string()),
        };

        // Provider should apply to any variant
        assert!(provider_variant.should_apply_variant("variant1"));
        assert!(provider_variant.should_apply_variant("variant2"));
    }

    #[test]
    fn test_should_apply_variant_variant_match() {
        let variant = InferenceExtraHeader::Variant {
            variant_name: "variant1".to_string(),
            name: "X-Variant-Header".to_string(),
            kind: ExtraHeaderKind::Value("value".to_string()),
        };

        // Should apply to matching variant
        assert!(variant.should_apply_variant("variant1"));
    }

    #[test]
    fn test_should_apply_variant_variant_no_match() {
        let variant = InferenceExtraHeader::Variant {
            variant_name: "variant1".to_string(),
            name: "X-Variant-Header".to_string(),
            kind: ExtraHeaderKind::Value("value".to_string()),
        };

        // Should NOT apply to non-matching variant
        assert!(!variant.should_apply_variant("variant2"));
    }

    #[test]
    fn test_filter_includes_all_variant() {
        let unfiltered = UnfilteredInferenceExtraHeaders {
            extra_headers: vec![
                InferenceExtraHeader::Variant {
                    variant_name: "variant1".to_string(),
                    name: "X-V1".to_string(),
                    kind: ExtraHeaderKind::Value("v1".to_string()),
                },
                InferenceExtraHeader::Always {
                    name: "X-All".to_string(),
                    kind: ExtraHeaderKind::Value("all".to_string()),
                },
                InferenceExtraHeader::Variant {
                    variant_name: "variant2".to_string(),
                    name: "X-V2".to_string(),
                    kind: ExtraHeaderKind::Value("v2".to_string()),
                },
            ],
        };

        let filtered = unfiltered.filter("variant1");
        assert_eq!(filtered.data.len(), 2); // variant1 + Always

        // Verify Always is included
        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, InferenceExtraHeader::Always { .. })));
    }

    #[test]
    fn test_filter_mixed_variants() {
        let unfiltered = UnfilteredInferenceExtraHeaders {
            extra_headers: vec![
                InferenceExtraHeader::Provider {
                    model_provider_name: "openai".to_string(),
                    name: "X-Provider".to_string(),
                    kind: ExtraHeaderKind::Value("provider".to_string()),
                },
                InferenceExtraHeader::Variant {
                    variant_name: "variant1".to_string(),
                    name: "X-V1".to_string(),
                    kind: ExtraHeaderKind::Value("v1".to_string()),
                },
                InferenceExtraHeader::Always {
                    name: "X-All".to_string(),
                    kind: ExtraHeaderKind::Value("all".to_string()),
                },
                InferenceExtraHeader::Variant {
                    variant_name: "variant2".to_string(),
                    name: "X-V2".to_string(),
                    kind: ExtraHeaderKind::Value("v2".to_string()),
                },
            ],
        };

        let filtered = unfiltered.filter("variant1");
        // Should include: Provider, Variant(variant1), Always
        assert_eq!(filtered.data.len(), 3);

        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, InferenceExtraHeader::Provider { .. })));
        assert!(filtered
            .data
            .iter()
            .any(|item| matches!(item, InferenceExtraHeader::Always { .. })));
        assert!(filtered.data.iter().any(|item| match item {
            InferenceExtraHeader::Variant { variant_name, .. } => variant_name == "variant1",
            _ => false,
        }));
    }

    #[test]
    fn test_inference_extra_header_all_roundtrip() {
        let original = InferenceExtraHeader::Always {
            name: "X-Test-Header".to_string(),
            kind: ExtraHeaderKind::Value("test-value".to_string()),
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: InferenceExtraHeader = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }
}
