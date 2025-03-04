use std::collections::HashMap;

use crate::error::{Error, ErrorDetails};

pub mod batch_inference;
pub mod datasets;
pub mod fallback;
pub mod feedback;
pub mod inference;
pub mod openai_compatible;
pub mod status;

fn validate_tags(tags: &HashMap<String, String>) -> Result<(), Error> {
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
        assert!(validate_tags(&tags).is_ok());
        tags.insert("tensorzero::test".to_string(), "test".to_string());
        assert!(validate_tags(&tags).is_err());
    }
}
