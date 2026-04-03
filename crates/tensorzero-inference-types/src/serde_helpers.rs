use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

pub(crate) fn serialize_delete<S>(s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    true.serialize(s)
}

pub(crate) fn deserialize_delete<'de, D>(d: D) -> Result<(), D::Error>
where
    D: Deserializer<'de>,
{
    let val = bool::deserialize(d)?;
    if !val {
        return Err(D::Error::custom(
            "Error deserializing replacement config: `delete` must be `true`, or not set",
        ));
    }
    Ok(())
}

// Field-aware versions for struct fields (not enum variants)
#[expect(clippy::trivially_copy_pass_by_ref)]
pub(crate) fn serialize_delete_field<S>(_: &(), s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    true.serialize(s)
}

pub(crate) fn deserialize_delete_field<'de, D>(d: D) -> Result<(), D::Error>
where
    D: Deserializer<'de>,
{
    let val = bool::deserialize(d)?;
    if !val {
        return Err(D::Error::custom(
            "Error deserializing replacement config: `delete` must be `true`, or not set",
        ));
    }
    Ok(())
}

pub(crate) fn schema_for_delete_field(_gen: &mut schemars::SchemaGenerator) -> schemars::Schema {
    let mut map = Map::new();
    map.insert("type".to_owned(), Value::String("boolean".to_owned()));
    map.insert("const".to_owned(), Value::Bool(true));
    schemars::Schema::from(map)
}
