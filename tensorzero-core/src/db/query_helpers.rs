use crate::error::Error;

/// Escapes a string for JSON without quotes.
///
/// This is used to escape the text query when doing a substring match on input and output strings, because
/// input and output strings are JSON-escaped in ClickHouse (and json::text is also JSON-escaped in Postgres).
pub fn json_escape_string_without_quotes(s: &str) -> Result<String, Error> {
    let mut json_escaped = serde_json::to_string(s)?;
    json_escaped.remove(0);
    json_escaped.pop();
    Ok(json_escaped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_escape_string_without_quotes() {
        assert_eq!(
            json_escape_string_without_quotes("").unwrap(),
            String::new()
        );
        assert_eq!(
            json_escape_string_without_quotes("test").unwrap(),
            "test".to_string()
        );
        assert_eq!(
            json_escape_string_without_quotes("123").unwrap(),
            "123".to_string()
        );
        assert_eq!(
            json_escape_string_without_quotes("he's").unwrap(),
            "he's".to_string()
        );
    }

    #[test]
    fn test_json_escape_string_escapes_correctly() {
        assert_eq!(
            json_escape_string_without_quotes(r#""test""#).unwrap(),
            r#"\"test\""#.to_string()
        );

        assert_eq!(
            json_escape_string_without_quotes(r"end of line\next line").unwrap(),
            r"end of line\\next line".to_string()
        );
    }
}
