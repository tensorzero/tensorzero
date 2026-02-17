use crate::error::{Error, ErrorDetails};

/// A validated namespace identifier.
///
/// Namespace identifiers must be non-empty strings.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export, type = "string"))]
pub struct Namespace(String);

impl Namespace {
    /// Creates a new Namespace, validating that it is non-empty.
    pub fn new(namespace: impl Into<String>) -> Result<Self, Error> {
        let namespace = namespace.into();
        if namespace.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Namespace identifier cannot be empty".to_string(),
            }));
        }
        Ok(Self(namespace))
    }

    /// Returns the namespace as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the Namespace and returns the inner String.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::fmt::Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for Namespace {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl serde::Serialize for Namespace {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Namespace {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Namespace::new(s).map_err(|e| serde::de::Error::custom(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_valid() {
        let cases = ["abc", "my_namespace_2", "a", "UPPER", "with-hyphens"];
        for case in cases {
            let ns = Namespace::new(case);
            assert!(ns.is_ok(), "Namespace `{case}` should be valid");
            assert_eq!(ns.unwrap().as_str(), case);
        }
    }

    #[test]
    fn test_namespace_empty_rejected() {
        let ns = Namespace::new("");
        assert!(ns.is_err(), "Empty namespace should be rejected");
    }

    #[test]
    fn test_namespace_serde_roundtrip() {
        let ns = Namespace::new("my_namespace").unwrap();
        let serialized = serde_json::to_string(&ns).unwrap();
        assert_eq!(serialized, "\"my_namespace\"");
        let deserialized: Namespace = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            ns, deserialized,
            "Namespace should survive serialize + deserialize roundtrip"
        );
    }

    #[test]
    fn test_namespace_deserialize_empty_rejected() {
        let result = serde_json::from_str::<Namespace>("\"\"");
        assert!(result.is_err(), "Deserializing an empty string should fail");
    }
}
