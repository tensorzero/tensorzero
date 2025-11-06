use serde::{Deserialize, Serialize};

/// Service tier for inference requests.
///
/// Controls the priority and latency characteristics of the request.
/// Different providers map these values differently to their own service tiers.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    #[default]
    Auto,
    Default,
    Priority,
    Flex,
}

impl std::fmt::Display for ServiceTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ServiceTier::Auto => "auto",
            ServiceTier::Default => "default",
            ServiceTier::Priority => "priority",
            ServiceTier::Flex => "flex",
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_tier_serialization() {
        assert_eq!(
            serde_json::to_string(&ServiceTier::Auto).unwrap(),
            r#""auto""#
        );
        assert_eq!(
            serde_json::to_string(&ServiceTier::Default).unwrap(),
            r#""default""#
        );
        assert_eq!(
            serde_json::to_string(&ServiceTier::Priority).unwrap(),
            r#""priority""#
        );
        assert_eq!(
            serde_json::to_string(&ServiceTier::Flex).unwrap(),
            r#""flex""#
        );
    }

    #[test]
    fn test_service_tier_deserialization() {
        assert_eq!(
            serde_json::from_str::<ServiceTier>(r#""auto""#).unwrap(),
            ServiceTier::Auto
        );
        assert_eq!(
            serde_json::from_str::<ServiceTier>(r#""default""#).unwrap(),
            ServiceTier::Default
        );
        assert_eq!(
            serde_json::from_str::<ServiceTier>(r#""priority""#).unwrap(),
            ServiceTier::Priority
        );
        assert_eq!(
            serde_json::from_str::<ServiceTier>(r#""flex""#).unwrap(),
            ServiceTier::Flex
        );
    }
}
