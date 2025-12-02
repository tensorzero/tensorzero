use serde::de::IntoDeserializer;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::HashMap;

/// Serializes a value as a JSON string (for "doubly-serialized" fields).
/// This is the inverse of `deserialize_json_string`.
pub fn serialize_json_string<S, T>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize,
{
    let json_str = serde_json::to_string(value).map_err(serde::ser::Error::custom)?;
    serializer.serialize_str(&json_str)
}

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

/// Deserializes a "doubly-serialized" field of a struct, but allows the string "" or null to be the default.
/// If you have a struct like this:
/// ```ignore
/// #[derive(Deserialize, Default)]
/// struct Inner {
///     foo: u32,
///     bar: String,
/// }
///
/// #[derive(Deserialize)]
/// struct Outer {
///     #[serde(deserialize_with = "deserialize_defaulted_json_string")]
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
/// you can also do this:
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": \"\"}")?;
/// assert_eq!(outer.inner, Inner { foo: 0, bar: "".to_string() });
/// ```
///
/// or this:
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": null}")?;
/// assert_eq!(outer.inner, Inner { foo: 0, bar: "".to_string() });
/// ```
pub fn deserialize_defaulted_json_string<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned + Default,
{
    let value: Value = Deserialize::deserialize(deserializer)?;
    match value {
        Value::Null => Ok(T::default()),
        Value::String(s) => {
            if s.is_empty() {
                return Ok(T::default());
            }
            serde_json::from_str(&s).map_err(serde::de::Error::custom)
        }
        _ => Err(serde::de::Error::custom(
            "expected a string or null for deserialize_defaulted_json_string",
        )),
    }
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

/// Deserializes an `Option<Option<T>>`, distinguishing between an omitted field (`None`),
/// an explicit JSON `null` (`Some(None)`), and a concrete value (`Some(Some(T))`).
///
/// This is useful for API structs where we need to distinguish between an omitted field, JSON null, and a concrete value.
/// Use it like this:
/// ```ignore
/// #[derive(Deserialize)]
/// struct ParamsStruct {
///     #[serde(default, deserialize_with = "deserialize_double_option")]
///     maybe_null_field: Option<Option<String>>,
/// }
/// ```
pub fn deserialize_double_option<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let value = Value::deserialize(deserializer)?;
    if value.is_null() {
        Ok(Some(None))
    } else {
        let inner = T::deserialize(value.into_deserializer()).map_err(serde::de::Error::custom)?;
        Ok(Some(Some(inner)))
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

/// Deserializes a defaulted "doubly-serialized" field of a struct.
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
///
/// In ClickHouse we also run into case where the data is the empty string as an indicator of null.
/// ```ignore
/// let outer = serde_json::from_str::<Outer>("{\"inner\": \"\"}")?;
/// assert_eq!(outer.inner, Inner { foo: 0, bar: "".to_string() });
/// ```
pub fn deserialize_defaulted_string<'de, D, T>(deserializer: D) -> Result<T, D::Error>
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
        _ => Err(serde::de::Error::custom("expected a string")),
    }
}

/// Like `deserialize_option_u64`, but requires a number to be present.
pub fn deserialize_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Helper {
        String(String),
        Number(u64),
    }

    match Helper::deserialize(deserializer)? {
        Helper::String(s) => {
            if s.is_empty() {
                Err(D::Error::custom("empty string is not a valid u64"))
            } else {
                s.parse::<u64>()
                    .map_err(|_| D::Error::custom(format!("invalid u64 string: '{s}'")))
            }
        }
        Helper::Number(n) => Ok(n),
    }
}

/// In JSON, large numbers cannot be represented as numbers so we instead represent them as strings.
/// This function deserializes them as strings and then parses them as u64s.
/// It also handles the case where the value is null or a number as usual.
pub fn deserialize_option_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Helper {
        String(String),
        Number(u64),
        Null,
    }

    match Helper::deserialize(deserializer)? {
        Helper::String(s) => {
            if s.is_empty() {
                Err(D::Error::custom("empty string is not a valid u64"))
            } else {
                s.parse::<u64>()
                    .map(Some)
                    .map_err(|_| D::Error::custom(format!("invalid u64 string: '{s}'")))
            }
        }
        Helper::Number(n) => Ok(Some(n)),
        Helper::Null => Ok(None),
    }
}

/// Serializes an optional value, returning an empty string if the value is None.
/// This is useful for ClickHouse compatibility where empty strings represent null for certain fields.
pub fn serialize_none_as_empty_string<S, T>(
    value: &Option<T>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize,
{
    match value {
        Some(v) => v.serialize(serializer),
        None => serializer.serialize_str(""),
    }
}

/// Serializes an optional HashMap, returning an empty map if the value is None.
/// This is useful for ClickHouse compatibility where empty maps represent null for map fields.
pub fn serialize_none_as_empty_map<S, K, V>(
    value: &Option<HashMap<K, V>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    K: Serialize,
    V: Serialize,
{
    match value {
        Some(map) => map.serialize(serializer),
        None => {
            let map = serializer.serialize_map(Some(0))?;
            map.end()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
    struct TestStruct {
        foo: u32,
        bar: String,
    }

    #[derive(Debug, Deserialize)]
    struct TestOuter {
        #[serde(deserialize_with = "deserialize_json_string")]
        inner: TestStruct,
    }

    #[derive(Debug, Deserialize, Default)]
    struct TestDefaultedOuter {
        #[serde(deserialize_with = "deserialize_defaulted_string")]
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

    #[derive(Debug, Deserialize, Default, PartialEq)]
    struct TestDefaultedStruct {
        foo: u32,
        bar: String,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestDefaultedJsonOuter {
        #[serde(deserialize_with = "deserialize_defaulted_json_string")]
        inner: TestDefaultedStruct,
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

    #[derive(Debug, Deserialize)]
    struct TestOptionU64Outer {
        #[serde(deserialize_with = "deserialize_option_u64")]
        inner: Option<u64>,
    }

    #[derive(Debug, Deserialize)]
    struct TestU64Outer {
        #[serde(deserialize_with = "deserialize_u64")]
        inner: u64,
    }

    #[test]
    fn test_deserialize_option_u64_string() {
        let json = r#"{"inner": "1234567890"}"#;
        let result: TestOptionU64Outer = serde_json::from_str(json).unwrap();
        assert!(result.inner.is_some());
        let inner = result.inner.unwrap();
        assert_eq!(inner, 1234567890);
    }

    #[test]
    fn test_deserialize_option_u64_number() {
        let json = r#"{"inner": 1234567890}"#;
        let result: TestOptionU64Outer = serde_json::from_str(json).unwrap();
        assert!(result.inner.is_some());
        let inner = result.inner.unwrap();
        assert_eq!(inner, 1234567890);
    }

    #[test]
    fn test_deserialize_option_u64_null() {
        let json = r#"{"inner": null}"#;
        let result: TestOptionU64Outer = serde_json::from_str(json).unwrap();
        assert!(result.inner.is_none());
    }

    #[test]
    fn test_deserialize_u64_string() {
        let json = r#"{"inner": "1234567890"}"#;
        let result: TestU64Outer = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner, 1234567890);
    }

    #[test]
    fn test_deserialize_u64_number() {
        let json = r#"{"inner": 1234567890}"#;
        let result: TestU64Outer = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner, 1234567890);
    }

    #[test]
    fn test_deserialize_defaulted_json_string_empty_string() {
        let json = r#"{"inner": ""}"#;
        let result: TestDefaultedJsonOuter = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner, TestDefaultedStruct::default());
    }

    #[test]
    fn test_deserialize_defaulted_json_string_invalid_json() {
        let json = r#"{"inner": "{\"foo\": 42, \"bar\": invalid"}"#;
        let result: Result<TestDefaultedOuter, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_defaulted_json_string_valid() {
        let json = r#"{"inner": "{\"foo\": 1, \"bar\": \"test\"}"}"#;
        let result: TestDefaultedJsonOuter = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner.foo, 1);
        assert_eq!(result.inner.bar, "test");
    }

    #[test]
    fn test_deserialize_defaulted_json_string_null() {
        let json = r#"{"inner": null}"#;
        let result: TestDefaultedJsonOuter = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner, TestDefaultedStruct::default());
    }

    #[test]
    fn test_deserialize_defaulted_string_null() {
        let json = r#"{"inner": null}"#;
        let result: Result<TestDefaultedOuter, _> = serde_json::from_str(json);
        assert!(result.is_err());

        let json = r#"{"inner": "{\"foo\": 21, \"bar\": \"datboi\"}"}"#;
        let result: TestDefaultedOuter = serde_json::from_str(json).unwrap();
        assert_eq!(
            result.inner,
            TestStruct {
                foo: 21,
                bar: "datboi".to_string()
            }
        );

        let json = r#"{"inner": ""}"#;
        let result: TestDefaultedOuter = serde_json::from_str(json).unwrap();
        assert_eq!(result.inner, TestStruct::default());
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestDoubleOptionStruct<T: for<'a> Deserialize<'a>> {
        #[serde(default, deserialize_with = "deserialize_double_option")]
        maybe_null_field: Option<Option<T>>,
    }

    #[derive(Debug, Default, Deserialize, PartialEq)]
    struct TestDoubleOptionStructInnerStruct {
        inner_field: i32,
    }

    #[derive(Debug, Default, Deserialize, PartialEq)]
    struct TestDoubleOptionStructInnerOptionStruct {
        #[serde(default)]
        inner_option_field: Option<i32>,
    }

    #[test]
    fn test_deserialize_double_option_for_omitted_field() {
        let json = r"{}";
        let result: TestDoubleOptionStruct<String> = serde_json::from_str(json).unwrap();
        assert_eq!(result.maybe_null_field, None);
    }

    #[test]
    fn test_deserialize_double_option_for_null_field() {
        let json = r#"{"maybe_null_field": null}"#;
        let result: TestDoubleOptionStruct<String> = serde_json::from_str(json).unwrap();
        assert_eq!(result.maybe_null_field, Some(None));
    }

    #[test]
    fn test_deserialize_double_option_for_concrete_value() {
        let json = r#"{"maybe_null_field": "test"}"#;
        let result: TestDoubleOptionStruct<String> = serde_json::from_str(json).unwrap();
        assert_eq!(result.maybe_null_field, Some(Some("test".to_string())));
    }

    #[test]
    fn test_deserialize_double_option_for_nested_struct_with_null() {
        let json = r#"{"maybe_null_field": null}"#;
        let result: TestDoubleOptionStruct<TestDoubleOptionStructInnerStruct> =
            serde_json::from_str(json).unwrap();
        assert_eq!(result.maybe_null_field, Some(None));
    }

    #[test]
    fn test_deserialize_double_option_for_nested_struct() {
        let json = r#"{"maybe_null_field": {"inner_field": 123}}"#;
        let result: TestDoubleOptionStruct<TestDoubleOptionStructInnerStruct> =
            serde_json::from_str(json).unwrap();
        assert_eq!(
            result.maybe_null_field,
            Some(Some(TestDoubleOptionStructInnerStruct { inner_field: 123 }))
        );
    }

    #[test]
    fn test_deserialize_double_option_does_not_affect_inner_option() {
        let json = r#"{"maybe_null_field": {"inner_option_field": null}}"#;
        let result: TestDoubleOptionStruct<TestDoubleOptionStructInnerOptionStruct> =
            serde_json::from_str(json).unwrap();
        assert_eq!(
            result.maybe_null_field,
            Some(Some(TestDoubleOptionStructInnerOptionStruct {
                inner_option_field: None
            }))
        );
    }

    #[derive(Debug, Serialize, PartialEq)]
    struct TestSerializeNoneAsEmptyString {
        #[serde(serialize_with = "serialize_none_as_empty_string")]
        field: Option<TestStruct>,
    }

    #[test]
    fn test_serialize_none_as_empty_string_with_some() {
        let obj = TestSerializeNoneAsEmptyString {
            field: Some(TestStruct {
                foo: 1,
                bar: "test".to_string(),
            }),
        };
        let json = serde_json::to_string(&obj).unwrap();
        assert_eq!(json, r#"{"field":{"foo":1,"bar":"test"}}"#);
    }

    #[test]
    fn test_serialize_none_as_empty_string_with_none() {
        let obj = TestSerializeNoneAsEmptyString { field: None };
        let json = serde_json::to_string(&obj).unwrap();
        assert_eq!(json, r#"{"field":""}"#);
    }

    // Tests for serialize_none_as_empty_map
    #[derive(Debug, Serialize, PartialEq)]
    struct TestSerializeNoneAsEmptyMap {
        #[serde(serialize_with = "serialize_none_as_empty_map")]
        field: Option<HashMap<String, String>>,
    }

    #[test]
    fn test_serialize_none_as_empty_map_with_some() {
        let mut map = HashMap::new();
        map.insert("key1".to_string(), "value1".to_string());
        map.insert("key2".to_string(), "value2".to_string());

        let obj = TestSerializeNoneAsEmptyMap { field: Some(map) };
        let json = serde_json::to_string(&obj).unwrap();

        // Parse back to verify it's a valid map with the right contents
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let field_map = parsed["field"].as_object().unwrap();
        assert_eq!(field_map.len(), 2);
        assert_eq!(field_map["key1"], "value1");
        assert_eq!(field_map["key2"], "value2");
    }

    #[test]
    fn test_serialize_none_as_empty_map_with_none() {
        let obj = TestSerializeNoneAsEmptyMap { field: None };
        let json = serde_json::to_string(&obj).unwrap();
        assert_eq!(json, r#"{"field":{}}"#);
    }

    #[test]
    fn test_serialize_none_as_empty_map_with_empty_map() {
        let obj = TestSerializeNoneAsEmptyMap {
            field: Some(HashMap::new()),
        };
        let json = serde_json::to_string(&obj).unwrap();
        assert_eq!(json, r#"{"field":{}}"#);
    }
}
