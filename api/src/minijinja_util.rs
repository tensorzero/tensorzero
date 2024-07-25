use minijinja::{Environment, UndefinedBehavior};
use once_cell::sync::OnceCell;
use serde_json::Value;
use std::path::PathBuf;

use crate::error::Error;

static MINIJINJA_ENV: OnceCell<Environment> = OnceCell::new();

/// Initializes the ENV with the given templates, given as a list of paths to the templates.
/// The name of each template in the environment will simply be the path to that template.
/// This should be called once at startup.
pub fn initialize_templates(template_paths: &[&PathBuf]) {
    let mut env = Environment::new();
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    for path in template_paths {
        let template_name = path
            .to_str()
            .unwrap_or_else(|| panic!("Invalid template file path: {}", path.display()))
            .to_owned();
        println!("Adding template: {}", template_name);
        let template_content = std::fs::read_to_string(path).expect("Failed to read template file");
        env.add_template_owned(template_name, template_content)
            .unwrap_or_else(|e| panic!("Failed to add template at {}: {}", path.display(), e));
    }
    MINIJINJA_ENV
        .set(env)
        .expect("Jinja environment already initialized");
}

#[allow(dead_code)]
// Templates a message with a MiniJinja template.
pub fn template_message(template_name: &str, context: &Value) -> Result<String, Error> {
    let env = MINIJINJA_ENV
        .get()
        .expect("Jinja environment not initialized");
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
        let template2 = "Hello, {{ name }}! You are {{ age }} years old.";
        let mut temp_file2 = NamedTempFile::new().expect("Failed to create temporary file");
        write!(temp_file2, "{}", template2).expect("Failed to write to temp file");
        // Add both templates to make sure they're both added.
        initialize_templates(&[
            &temp_file.path().to_path_buf(),
            &temp_file.path().to_path_buf(),
            &temp_file2.path().to_path_buf(),
        ]);
        let env = MINIJINJA_ENV
            .get()
            .expect("Jinja environment not initialized");
        let templates = env.templates().collect::<Vec<_>>();
        assert_eq!(
            templates.len(),
            2,
            "Expected two templates in the environment"
        );
        let template_name = temp_file.path().to_str().unwrap();
        assert!(templates.iter().any(|(name, _)| name == &template_name));

        let context = serde_json::json!({"name": "world"});
        let result = template_message(template_name, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "hello, world!");

        // Test a little templating:
        let test_template_name = temp_file2.path().to_str().unwrap();

        // Test with correct inputs
        let context = serde_json::json!({
            "name": "Alice",
            "age": 30
        });
        let result = template_message(test_template_name, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, Alice! You are 30 years old.");

        // Test with missing input
        let context = serde_json::json!({
            "name": "Bob"
        });
        let result = template_message(test_template_name, &context);
        assert!(result.is_err());

        // Test with incorrect input type
        let context = serde_json::json!({
            "name": "Charlie",
            "age": "thirty"
        });
        let result = template_message(test_template_name, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, Charlie! You are thirty years old.");
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
}
