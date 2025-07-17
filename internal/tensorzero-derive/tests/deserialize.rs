#[derive(tensorzero_derive::TensorZeroDeserialize)]
#[serde(tag = "mytag")]
#[serde(rename_all = "snake_case")]
#[derive(PartialEq, Debug)]
enum MyTaggedEnum {
    FirstVariant { field1: NestedStruct, field2: bool },
    SecondVariant(bool, u8),
    Third,
}

#[derive(serde::Deserialize, PartialEq, Debug)]
struct NestedStruct {
    some_field: String,
}

#[test]
fn test_good_deserialize() {
    let val = serde_json::json!(
        {
            "mytag": "first_variant",
            "field1": {
                "some_field": "hello"
            },
            "field2": true
        }
    );
    let res: MyTaggedEnum = serde_json::from_value(val).unwrap();
    assert_eq!(
        res,
        MyTaggedEnum::FirstVariant {
            field1: NestedStruct {
                some_field: "hello".to_string()
            },
            field2: true,
        }
    );
}

#[test]
fn test_bad_deserialize() {
    let val = serde_json::json!(
        {
            "mytag": "first_variant",
            "field1": {
                "some_field": 123
            },
            "field2": true,
        }
    );
    let res: Result<MyTaggedEnum, _> = serde_json::from_value(val);
    let err = res.unwrap_err().to_string();
    assert_eq!(
        err,
        "field1.some_field: invalid type: integer `123`, expected a string"
    );
}
