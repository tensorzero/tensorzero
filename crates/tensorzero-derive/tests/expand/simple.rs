#[derive(tensorzero_derive::TensorZeroDeserialize)]
#[serde(tag = "type")]
enum MyEnum {
    UnitVariant,
    StructVariant { field1: i32, field2: bool },
    TupleVariant(i32, bool),
}
