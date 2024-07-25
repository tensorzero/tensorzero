use minijinja::{Environment, UndefinedBehavior};
use once_cell::sync::OnceCell;
use serde_json::Value;
use std::path::PathBuf;

use crate::error::Error;

static ENV: OnceCell<Environment> = OnceCell::new();

pub fn initialize_templates(template_paths: &[&PathBuf]) {
    let mut env = Environment::new();
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    for path in template_paths {
        let template_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| panic!("Invalid template file name: {}", path.display()))
            .to_owned();
        let template_content = std::fs::read_to_string(path).expect("Failed to read template file");
        env.add_template_owned(template_name, template_content)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to add template at {} because of {}",
                    path.display(),
                    e
                )
            });
    }
    ENV.set(env).expect("Jinja environment already initialized");
}

#[allow(dead_code)]
pub fn template_message(
    template_name: &str,
    context: &Value,
    environment: Option<&Environment>,
) -> Result<String, Error> {
    let env = match environment {
        Some(env) => env,
        None => ENV.get().expect("Jinja environment not initialized"),
    };
    let template =
        env.get_template(template_name)
            .map_err(|_| Error::MiniJinjaTemplateMissing {
                template_name: template_name.to_string(),
            })?;
    let maybe_message = template.render(context);
    match maybe_message {
        Ok(message) => Ok(message),
        Err(err) => {
            let mut message = err.to_string();
            let mut err = &err as &dyn std::error::Error;
            while let Some(next_err) = err.source() {
                message.push_str("\nCaused by: ");
                message.push_str(&next_err.to_string());
                err = next_err;
            }
            Err(Error::MiniJinjaTemplateRender {
                template_name: template_name.to_string(),
                message,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_template() {
        let template = "hello, {{name}}!";
        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        write!(temp_file, "{}", template).expect("Failed to write to temp file");
        initialize_templates(&[&temp_file.path().to_path_buf()]);
    }

    #[test]
    #[should_panic(expected = "Failed to read template file")]
    fn test_nonexistent_file() {
        let nonexistent_path = PathBuf::from("nonexistent_file.txt");
        initialize_templates(&[&nonexistent_path]);
    }

    #[test]
    #[should_panic(expected = "Failed to add template")]
    fn test_malformed_template() {
        let malformed_template = "{{ unclosed_bracket";
        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        write!(temp_file, "{}", malformed_template).expect("Failed to write to temp file");
        initialize_templates(&[&temp_file.path().to_path_buf()]);
    }

    #[test]
    fn test_template_message() {
        let template = "Hello, {{ name }}! You are {{ age }} years old.";
        let env = setup_environment(template);

        // Test with correct inputs
        let context = serde_json::json!({
            "name": "Alice",
            "age": 30
        });
        let result = template_message("test_template", &context, Some(&env));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, Alice! You are 30 years old.");

        // Test with missing input
        let context = serde_json::json!({
            "name": "Bob"
        });
        let result = template_message("test_template", &context, Some(&env));
        assert!(result.is_err());

        // Test with incorrect input type
        let context = serde_json::json!({
            "name": "Charlie",
            "age": "thirty"
        });
        let result = template_message("test_template", &context, Some(&env));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, Charlie! You are thirty years old.");
    }

    fn setup_environment(template: &str) -> Environment {
        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Strict);
        env.add_template_owned("test_template", template).unwrap();
        env
    }
}
