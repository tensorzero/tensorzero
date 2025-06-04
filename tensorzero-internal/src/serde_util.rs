use serde::Deserialize;
use serde_json::Value;

pub fn deserialize_json_string<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let json_str = String::deserialize(deserializer)?;
    serde_json::from_str(&json_str).map_err(serde::de::Error::custom)
}

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

pub fn deserialize_optional_string_or_parsed_json<'de, D, T>(
    deserializer: D,
) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let value: Value = Deserialize::deserialize(deserializer)?;
    match value {
        Value::String(s) => {
            if s.is_empty() {
                return Ok(None);
            }
            Ok(Some(
                serde_json::from_str(&s).map_err(serde::de::Error::custom)?,
            ))
        }
        _ => Ok(Some(
            serde_json::from_value(value).map_err(serde::de::Error::custom)?,
        )),
    }
}

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
