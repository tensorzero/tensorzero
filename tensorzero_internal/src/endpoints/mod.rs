use std::collections::HashMap;

use crate::error::{Error, ErrorDetails};

pub mod batch_inference;
pub mod fallback;
pub mod feedback;
pub mod inference;
pub mod openai_compatible;
pub mod status;

fn check_tags(tags: &HashMap<String, String>) -> Result<(), Error> {
    for tag_name in tags.keys() {
        if tag_name.starts_with("tensorzero::") {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Tag name cannot start with 'tensorzero::': {tag_name}"),
            }));
        }
    }
    Ok(())
}
