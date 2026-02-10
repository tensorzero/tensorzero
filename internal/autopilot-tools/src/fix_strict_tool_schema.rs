use schemars::{
    Schema,
    transform::{RecursiveTransform, Transform},
};

/// Applies various fixes to make a tool schema compatible with OpenAI and Anthropic strict mode:
/// * Removes the 'minimum' field from 'uint' schemas (which are added by schemars)
///
/// This should be applied as-needed to make the `test_list_tools_` tests pass
pub fn fix_strict_tool_schema(mut schema: Schema) -> Schema {
    let mut transform = RecursiveTransform(|schema: &mut Schema| {
        if let Some(obj) = schema.as_object_mut()
            && obj
                .get("format")
                .is_some_and(|f| f == "uint" || f == "uint32" || f == "uint64")
        {
            obj.remove("minimum");
        }
    });
    transform.transform(&mut schema);
    schema
}
