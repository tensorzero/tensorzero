use crate::config_parser::UninitializedVariantInfo;
use crate::error::Error;

use super::VariantInfo;

pub fn load_dynamic_variant_info(config: UninitializedVariantInfo) -> Result<VariantInfo, Error> {
    let mut variant_info = config.load()?;
    variant_info.set_weight(Some(1.0));

    Ok(variant_info)
}
