use tracing;

pub fn check_quoted_env_var(key: &str) {
    if let Ok(value) = std::env::var(key) {
        if (value.starts_with('"') && value.ends_with('"')) || 
           (value.starts_with('\'') && value.ends_with('\'')) {
            tracing::warn!(
                "Environment variable {} contains quotes. This may cause issues. Remove the quotes from the value in your environment configuration.",
                key
            );
        }
    }
}

pub fn validate_environment_variables() {
    let env_vars = vec![
        "TENSORZERO_CLICKHOUSE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "FIREWORKS_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY"
    ];

    for var in env_vars {
        check_quoted_env_var(var);
    }
} 