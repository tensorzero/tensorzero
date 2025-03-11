use std::collections::HashMap;

use crate::error::{Error, ErrorDetails};

pub mod batch_inference;
pub mod datasets;
pub mod fallback;
pub mod feedback;
pub mod inference;
pub mod object_storage;
pub mod openai_compatible;
pub mod status;

fn validate_tags(tags: &HashMap<String, String>, internal: bool) -> Result<(), Error> {
    if internal {
        return Ok(());
    }
    for tag_name in tags.keys() {
        if tag_name.starts_with("tensorzero::") {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Tag name cannot start with 'tensorzero::': {tag_name}"),
            }));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_tags() {
        let mut tags = HashMap::new();
        assert!(validate_tags(&tags, false).is_ok());
        tags.insert("tensorzero::test".to_string(), "test".to_string());
        assert!(validate_tags(&tags, false).is_err());
        // once we're in internal mode, we can have tags that start with "tensorzero::"
        assert!(validate_tags(&tags, true).is_ok());
    }
}
