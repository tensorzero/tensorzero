use minijinja::{Environment, UndefinedBehavior};
use serde_json::Value;
#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
#[cfg(test)]
use tempfile::NamedTempFile;

use crate::error::Error;

static MINIJINJA_ENV: OnceLock<Environment> = OnceLock::new();

/// Initializes the MINIJINJA_ENV with the given templates, given as a list of paths to the templates.
/// The name of each template in the environment will simply be the path to that template.
/// This should be called once at startup.
pub fn initialize_templates(template_paths: &[&PathBuf]) -> Result<(), Error> {
    let mut env = Environment::new();
    env.set_undefined_behavior(UndefinedBehavior::Strict);

    for path in template_paths {
        let template_name = path
            .to_str()
            .ok_or(Error::MiniJinjaTemplate {
                template_name: path.display().to_string(),
                message: "Template path is invalid".to_string(),
            })?
            .to_owned();

        let template_content =
            std::fs::read_to_string(path).map_err(|e| Error::MiniJinjaTemplate {
                template_name: path.display().to_string(),
                message: format!("Failed to read template file: {}", e),
            })?;

        env.add_template_owned(template_name, template_content)
            .map_err(|e| Error::MiniJinjaTemplate {
                template_name: path.display().to_string(),
                message: format!("Failed to add template: {}", e),
            })?;
    }

    MINIJINJA_ENV
        .set(env)
        .map_err(|_| Error::MiniJinjaEnvironment {
            message: "Jinja environment already initialized".to_string(),
        })?;

    Ok(())
}

#[allow(dead_code)]
// Templates a message with a MiniJinja template.
pub fn template_message(template_name: &str, context: &Value) -> Result<String, Error> {
    let env = MINIJINJA_ENV.get().ok_or(Error::MiniJinjaEnvironment {
        message: "Jinja environment not initialized".to_string(),
    })?;

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
pub(crate) mod tests {
    use super::*;

    #[test]
    fn test_template_good() {
        let templates = idempotent_initialize_test_templates();
        let template_name = templates.get("greeting").unwrap().to_str().unwrap();

        let context = serde_json::json!({"name": "world"});
        let result = template_message(template_name, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "hello, world!");

        // Test a little templating:
        let template_name = templates
            .get("greeting_with_age")
            .unwrap()
            .to_str()
            .unwrap();

        // Test with correct inputs
        let context = serde_json::json!({
            "name": "Alice",
            "age": 30
        });
        let result = template_message(template_name, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, Alice! You are 30 years old.");

        // Test with missing input
        let context = serde_json::json!({
            "name": "Bob"
        });
        let result = template_message(template_name, &context);
        assert!(result.is_err());

        // Test with incorrect input type
        let context = serde_json::json!({
            "name": "Charlie",
            "age": "thirty"
        });
        let result = template_message(template_name, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, Charlie! You are thirty years old.");
    }

    #[test]
    fn test_template_nonexistent_file() {
        let nonexistent_path = PathBuf::from("nonexistent_file.txt");
        let result = initialize_templates(&[&nonexistent_path]);
        assert_eq!(
            result.unwrap_err(),
            Error::MiniJinjaTemplate {
                template_name: nonexistent_path.display().to_string(),
                message: "Failed to read template file: No such file or directory (os error 2)"
                    .to_string(),
            }
        );
    }

    #[test]
    fn test_template_malformed_template() {
        let malformed_template = "{{ unclosed_bracket";
        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        write!(temp_file, "{}", malformed_template).expect("Failed to write to temp file");
        let result = initialize_templates(&[&temp_file.path().to_path_buf()]);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to add template"));
    }
    // At test time, we want to initialize the templates once, and then use them in all tests.
    // This is a bit of a hack, but it works for now.
    // Idea is that TEST_TEMPLATES is initialized once, and then we use it in all tests.
    // Same with MINIJINJA_ENV. We can put all the minijinja templates used in cargo tests in this initializer,
    // then every test that uses minijinja can use them by calling this function and getting the template paths
    // from the map.
    static TEST_TEMPLATES: OnceLock<HashMap<&'static str, PathBuf>> = OnceLock::new();

    pub fn idempotent_initialize_test_templates() -> &'static HashMap<&'static str, PathBuf> {
        TEST_TEMPLATES.get_or_init(|| {
            let mut templates = HashMap::new();

            // Template 1
            let template1 = "hello, {{name}}!";
            let temp_file1 = create_temp_file(template1);
            templates.insert("greeting", temp_file1.path().to_path_buf());

            // Template 2
            let template2 = "Hello, {{ name }}! You are {{ age }} years old.";
            let temp_file2 = create_temp_file(template2);
            templates.insert("greeting_with_age", temp_file2.path().to_path_buf());

            // System template
            let system_template =
                "You are a helpful and friendly assistant named {{ assistant_name }}";
            let temp_file3 = create_temp_file(system_template);
            templates.insert("system", temp_file3.path().to_path_buf());

            // Assistant Template
            let assistant_template =
                "I'm sorry but I can't help you with that because of {{ reason }}";
            let temp_file4 = create_temp_file(assistant_template);
            templates.insert("assistant", temp_file4.path().to_path_buf());

            // Initialize templates
            initialize_templates(&templates.values().collect::<Vec<_>>()).unwrap();

            templates
        })
    }

    fn create_temp_file(content: &str) -> NamedTempFile {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        write!(temp_file, "{}", content).expect("Failed to write to temp file");
        temp_file
    }
}
