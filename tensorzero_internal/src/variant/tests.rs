#[cfg(test)]
mod tests {
    use super::*;
    use toml;

    #[test]
    fn test_missing_variant_type() {
        let toml_str = r#"
            [functions.my_function]
            type = "chat"

            [functions.my_function.variants.my_variant]
            model = "anthropic::claude-3-haiku-20240307"
        "#;

        let result = toml::from_str::<Config>(toml_str);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("missing field `type` in `functions.my_function.variants.my_variant`"));
    }
}