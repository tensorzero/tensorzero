use serde::Deserialize;
use serde_json::Value;

/// Deserializes a "doubly-serialized" field of a struct.
/// If you have a struct like this:
/// ```ignore
/// #[derive(Deserialize)]
/// struct Inner {
///     foo: u32,
///     bar: String,
/// }
///
/// #[derive(Deserialize)]
/// struct Outer {
///     #[serde(deserialize_with = "deserialize_json_string")]
///     inner: Inner,
/// }
/// ```
///
/// And the inner struct is itself a JSON serialized string, you can deserialize it like this:
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": \"{\\"foo\\": 1, \\"bar\\": \\"baz\\"}\"}")?;
/// assert_eq!(outer.inner.foo, 1);
/// assert_eq!(outer.inner.bar, "baz");
/// ```
pub fn deserialize_json_string<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let json_str = String::deserialize(deserializer)?;
    serde_json::from_str(&json_str).map_err(serde::de::Error::custom)
}

/// Deserializes an optional "doubly-serialized" field of a struct.
/// If you have a struct like this:
/// ```ignore
/// #[derive(Deserialize)]
/// struct Inner {
///     foo: u32,
///     bar: String,
/// }
///
/// #[derive(Deserialize)]
/// struct Outer {
///     #[serde(deserialize_with = "deserialize_json_string")]
///     inner: Option<Inner>,
/// }
/// ```
///
/// And the inner struct is itself a JSON serialized string, you can deserialize it like this:
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": \"{\\"foo\\": 1, \\"bar\\": \\"baz\\"}\"}")?;
/// assert_eq!(outer.inner.foo, 1);
/// assert_eq!(outer.inner.bar, "baz");
/// ```
///
/// In ClickHouse we also run into case where the data is the empty string as an indicator of null.
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": \"\"}")?;
/// assert_eq!(outer.inner, None);
/// ```
pub fn deserialize_optional_json_string<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let opt_json_str: Option<String> = Option::deserialize(deserializer)?;
    match opt_json_str {
        Some(json_str) => {
            if json_str.is_empty() {
                return Ok(None);
            }
            serde_json::from_str(&json_str)
                .map(Some)
                .map_err(serde::de::Error::custom)
        }
        None => Ok(None),
    }
}

/// Deserializes a "maybe-doubly-serialized" field of a struct.
/// If you have a struct like this:
/// ```ignore
/// #[derive(Deserialize)]
/// struct Inner {
///     foo: u32,
///     bar: String,
/// }
///
/// #[derive(Deserialize)]
/// struct Outer {
///     #[serde(deserialize_with = "deserialize_json_string")]
///     inner: Inner,
/// }
/// ```
///
/// And the inner struct is itself a JSON serialized string, you can deserialize it like this:
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": \"{\\"foo\\": 1, \\"bar\\": \\"baz\\"}\"}")?;
/// assert_eq!(outer.inner.foo, 1);
/// assert_eq!(outer.inner.bar, "baz");
/// ```
///
/// You might also need to deserialize a normal version of the struct:
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": {\"foo\": 1, \"bar\": \"baz\"}}")?;
/// assert_eq!(outer.inner.foo, 1);
/// assert_eq!(outer.inner.bar, "baz");
/// ```
pub fn deserialize_string_or_parsed_json<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let value: Value = Deserialize::deserialize(deserializer)?;
    match value {
        Value::String(s) => serde_json::from_str(&s).map_err(serde::de::Error::custom),
        _ => serde_json::from_value(value).map_err(serde::de::Error::custom),
    }
}

/// Deserializes an optional "maybe-doubly-serialized" field of a struct.
/// If you have a struct like this:
/// ```ignore
/// #[derive(Deserialize)]
/// struct Inner {
///     foo: u32,
///     bar: String,
/// }
///
/// #[derive(Deserialize)]
/// struct Outer {
///     #[serde(deserialize_with = "deserialize_json_string")]
///     inner: Inner,
/// }
/// ```
///
/// And the inner struct is itself a JSON serialized string, you can deserialize it like this:
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": \"{\\"foo\\": 1, \\"bar\\": \\"baz\\"}\"}")?;
/// assert_eq!(outer.inner.foo, 1);
/// assert_eq!(outer.inner.bar, "baz");
/// ```
///
/// You might also need to deserialize a normal version of the struct:
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": {\"foo\": 1, \"bar\": \"baz\"}}")?;
/// assert_eq!(outer.inner.foo, 1);
/// assert_eq!(outer.inner.bar, "baz");
/// ```
///
/// In ClickHouse we also run into case where the data is the empty string as an indicator of null.
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": \"\"}")?;
/// assert_eq!(outer.inner, None);
/// ```
pub fn deserialize_optional_string_or_parsed_json<'de, D, T>(
    deserializer: D,
) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let value: Value = Deserialize::deserialize(deserializer)?;
    match value {
        Value::Null => Ok(None),
        // If the value is a string, parse it as JSON then deserialize it into the target type
        Value::String(s) => {
            if s.is_empty() {
                return Ok(None);
            }
            Ok(Some(
                serde_json::from_str(&s).map_err(serde::de::Error::custom)?,
            ))
        }
        // If the value is a JSON object, deserialize it into the target type
        value => Ok(Some(
            serde_json::from_value(value).map_err(serde::de::Error::custom)?,
        )),
    }
}

/// Deserializes a defaulted "maybe-doubly-serialized" field of a struct.
/// If you have a struct like this:
/// ```ignore
/// #[derive(Deserialize, Default)]
/// struct Inner {
///     foo: u32,
///     bar: String,
/// }
///
///
/// #[derive(Deserialize)]
/// struct Outer {
///     #[serde(deserialize_with = "deserialize_json_string")]
///     inner: Inner,
/// }
/// ```
///
/// And the inner struct is itself a JSON serialized string, you can deserialize it like this:
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": \"{\\"foo\\": 1, \\"bar\\": \\"baz\\"}\"}")?;
/// assert_eq!(outer.inner.foo, 1);
/// assert_eq!(outer.inner.bar, "baz");
/// ```
///
/// You might also need to deserialize a normal version of the struct:
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": {\"foo\": 1, \"bar\": \"baz\"}}")?;
/// assert_eq!(outer.inner.foo, 1);
/// assert_eq!(outer.inner.bar, "baz");
/// ```
///
/// In ClickHouse we also run into case where the data is the empty string as an indicator of null.
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": \"\"}")?;
/// assert_eq!(outer.inner, Inner { foo: 0, bar: "".to_string() });
/// ```
pub fn deserialize_defaulted_string_or_parsed_json<'de, D, T>(
    deserializer: D,
) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned + Default,
{
    let value: Value = Deserialize::deserialize(deserializer)?;
    match value {
        Value::String(s) => {
            if s.is_empty() {
                return Ok(T::default());
            }
            Ok(serde_json::from_str(&s).map_err(serde::de::Error::custom)?)
        }
        _ => Ok(serde_json::from_value(value).map_err(serde::de::Error::custom)?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, Serialize, PartialEq, Default)]
    struct TestStruct {
        foo: u32,
        bar: String,
    }

    #[derive(Debug, Deserialize)]
    struct TestOuter {
        #[serde(deserialize_with = "deserialize_json_string")]
        inner: TestStruct,
    }

    #[derive(Debug, Deserialize)]
    struct TestOptionalOuter {
        #[serde(deserialize_with = "deserialize_optional_json_string")]
        inner: Option<TestStruct>,
    }

    #[derive(Debug, Deserialize)]
    struct TestStringOrParsedOuter {
        #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
        inner: TestStruct,
    }

    #[derive(Debug, Deserialize)]
    struct TestOptionalStringOrParsedOuter {
        #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
        inner: Option<TestStruct>,
    }

    #[derive(Debug, Deserialize)]
    struct TestDefaultedStringOrParsedOuter {
        #[serde(deserialize_with = "deserialize_defaulted_string_or_parsed_json")]
        inner: TestStruct,
    }

    #[test]
    fn test_deserialize_json_string_success() {
        let json = r#"{"inner": "{\"foo\": 42, \"bar\": \"test\"}"}"#;
        let result: TestOuter = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner.foo, 42);
        assert_eq!(result.inner.bar, "test");
    }

    #[test]
    fn test_deserialize_json_string_invalid_json() {
        let json = r#"{"inner": "{\"foo\": 42, \"bar\": invalid"}"#;
        let result: Result<TestOuter, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_json_string_default_values() {
        let json = r#"{"inner": "{\"foo\": 0, \"bar\": \"\"}"}"#;
        let result: TestOuter = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner.foo, 0);
        assert_eq!(result.inner.bar, "");
    }

    #[test]
    fn test_deserialize_optional_json_string_some() {
        let json = r#"{"inner": "{\"foo\": 42, \"bar\": \"test\"}"}"#;
        let result: TestOptionalOuter = serde_json::from_str(json).unwrap();
        assert!(result.inner.is_some());
        let inner = result.inner.unwrap();
        assert_eq!(inner.foo, 42);
        assert_eq!(inner.bar, "test");
    }

    #[test]
    fn test_deserialize_optional_json_string_none() {
        let json = r#"{"inner": null}"#;
        let result: TestOptionalOuter = serde_json::from_str(json).unwrap();
        assert!(result.inner.is_none());
    }

    #[test]
    fn test_deserialize_optional_json_string_empty_string() {
        let json = r#"{"inner": ""}"#;
        let result: TestOptionalOuter = serde_json::from_str(json).unwrap();
        assert!(result.inner.is_none());
    }

    #[test]
    fn test_deserialize_optional_json_string_invalid_json() {
        let json = r#"{"inner": "{\"foo\": 42, \"bar\": invalid"}"#;
        let result: Result<TestOptionalOuter, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_string_or_parsed_json_from_string() {
        let json = r#"{"inner": "{\"foo\": 42, \"bar\": \"test\"}"}"#;
        let result: TestStringOrParsedOuter = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner.foo, 42);
        assert_eq!(result.inner.bar, "test");
    }

    #[test]
    fn test_deserialize_string_or_parsed_json_from_object() {
        let json = r#"{"inner": {"foo": 42, "bar": "test"}}"#;
        let result: TestStringOrParsedOuter = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner.foo, 42);
        assert_eq!(result.inner.bar, "test");
    }

    #[test]
    fn test_deserialize_string_or_parsed_json_invalid_string() {
        let json = r#"{"inner": "{\"foo\": 42, \"bar\": invalid"}"#;
        let result: Result<TestStringOrParsedOuter, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_optional_string_or_parsed_json_from_string() {
        let json = r#"{"inner": "{\"foo\": 42, \"bar\": \"test\"}"}"#;
        let result: TestOptionalStringOrParsedOuter = serde_json::from_str(json).unwrap();
        assert!(result.inner.is_some());
        let inner = result.inner.unwrap();
        assert_eq!(inner.foo, 42);
        assert_eq!(inner.bar, "test");
    }

    #[test]
    fn test_deserialize_optional_string_or_parsed_json_from_object() {
        let json = r#"{"inner": {"foo": 42, "bar": "test"}}"#;
        let result: TestOptionalStringOrParsedOuter = serde_json::from_str(json).unwrap();
        assert!(result.inner.is_some());
        let inner = result.inner.unwrap();
        assert_eq!(inner.foo, 42);
        assert_eq!(inner.bar, "test");
    }

    #[test]
    fn test_deserialize_optional_string_or_parsed_json_null() {
        let json = r#"{"inner": null}"#;
        let result: TestOptionalStringOrParsedOuter = serde_json::from_str(json).unwrap();
        assert!(result.inner.is_none());
    }

    #[test]
    fn test_deserialize_optional_string_or_parsed_json_empty_string() {
        let json = r#"{"inner": ""}"#;
        let result: TestOptionalStringOrParsedOuter = serde_json::from_str(json).unwrap();
        assert!(result.inner.is_none());
    }

    #[test]
    fn test_deserialize_optional_string_or_parsed_json_invalid_string() {
        let json = r#"{"inner": "{\"foo\": 42, \"bar\": invalid"}"#;
        let result: Result<TestOptionalStringOrParsedOuter, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_defaulted_string_or_parsed_json_from_string() {
        let json = r#"{"inner": "{\"foo\": 42, \"bar\": \"test\"}"}"#;
        let result: TestDefaultedStringOrParsedOuter = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner.foo, 42);
        assert_eq!(result.inner.bar, "test");
    }

    #[test]
    fn test_deserialize_defaulted_string_or_parsed_json_from_object() {
        let json = r#"{"inner": {"foo": 42, "bar": "test"}}"#;
        let result: TestDefaultedStringOrParsedOuter = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner.foo, 42);
        assert_eq!(result.inner.bar, "test");
    }

    #[test]
    fn test_deserialize_defaulted_string_or_parsed_json_empty_string() {
        let json = r#"{"inner": ""}"#;
        let result: TestDefaultedStringOrParsedOuter = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner, TestStruct::default());
    }

    #[test]
    fn test_deserialize_defaulted_string_or_parsed_json_invalid_string() {
        let json = r#"{"inner": "{\"foo\": 42, \"bar\": invalid"}"#;
        let result: Result<TestDefaultedStringOrParsedOuter, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }
}
