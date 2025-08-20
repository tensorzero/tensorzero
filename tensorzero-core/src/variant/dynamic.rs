use crate::config::{ErrorContext, SchemaData, UninitializedVariantInfo};
use crate::error::Error;

use super::VariantInfo;

pub fn load_dynamic_variant_info(
    config: UninitializedVariantInfo,
    schemas: &SchemaData,
    function_name: String,
) -> Result<VariantInfo, Error> {
    let mut variant_info = config.load(
        schemas,
        &ErrorContext {
            function_name,
            variant_name: "tensorzero::dynamic_variant".to_string(),
        },
    )?;
    variant_info.set_weight(Some(1.0));

    Ok(variant_info)
}
