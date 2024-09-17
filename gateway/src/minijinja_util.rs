use minijinja::{Environment, UndefinedBehavior};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::Error;

#[derive(Debug)]
pub struct TemplateConfig<'c> {
    env: minijinja::Environment<'c>,
}

impl<'c> TemplateConfig<'c> {
    pub fn new() -> Self {
        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Strict);
        Self { env }
    }

    /// Initializes the TemplateConfig with the given templates, given as a map from template names
    /// to template paths.
    pub fn initialize(&mut self, template_paths: HashMap<String, PathBuf>) -> Result<(), Error> {
        self.env.set_undefined_behavior(UndefinedBehavior::Strict);

        for (template_name, path) in template_paths {
            let template_content =
                std::fs::read_to_string(&path).map_err(|e| Error::MiniJinjaTemplate {
                    template_name: path.display().to_string(),
                    message: format!("Failed to read template file: {}", e),
                })?;

            self.env
                .add_template_owned(template_name, template_content)
                .map_err(|e| Error::MiniJinjaTemplate {
                    template_name: path.display().to_string(),
                    message: format!("Failed to add template: {}", e),
                })?;
        }
        self.add_hardcoded_templates()?;
        Ok(())
    }

    // Templates a message with a MiniJinja template.
    pub fn template_message(&self, template_name: &str, context: &Value) -> Result<String, Error> {
        let template =
            self.env
                .get_template(template_name)
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

    // Checks if a template needs any variables (i.e. needs a schema)
    pub fn template_needs_variables(&self, template_name: &str) -> Result<bool, Error> {
        let template =
            self.env
                .get_template(template_name)
                .map_err(|_| Error::MiniJinjaTemplateMissing {
                    template_name: template_name.to_string(),
                })?;

        let template_needs_variables = !template.undeclared_variables(true).is_empty();

        Ok(template_needs_variables)
    }

    pub fn add_hardcoded_templates(&mut self) -> Result<(), Error> {
        self.env
            .add_template("t0:best_of_n_evaluator_system", BEST_OF_N_EVALUATOR_SYSTEM)
            .map_err(|e| Error::MiniJinjaTemplate {
                template_name: "t0:best_of_n_evaluator_system".to_string(),
                message: format!("Failed to add template: {}", e),
            })?;
        self.env
            .add_template(
                "t0:best_of_n_evaluator_candidates",
                BEST_OF_N_EVALUATOR_CANDIDATES,
            )
            .map_err(|e| Error::MiniJinjaTemplate {
                template_name: "t0:best_of_n_evaluator_candidates".to_string(),
                message: format!("Failed to add template: {}", e),
            })?;
        Ok(())
    }
}

impl<'c> Default for TemplateConfig<'c> {
    fn default() -> Self {
        Self::new()
    }
}

const BEST_OF_N_EVALUATOR_SYSTEM: &str = r#"{%- if inner_system_message is defined -%}You are an assistant tasked with re-ranking candidate answers to the following problem:
{{ inner_system_message }}

{%- else -%}
You are an assistant tasked with re-ranking candidate answers to a problem.

{%- endif -%}
The messages below are the conversation history between the user and the assistant along with a final message giving a set of candidate responses.
Please evaluate the following candidate responses and provide your reasoning along with the index of the best candidate in the following JSON format:
{
    "thinking": "your reasoning here",
    "best_candidate_index": int  // Range: 0 to n-1
}"#;

const BEST_OF_N_EVALUATOR_CANDIDATES: &str = r#"Here are the candidate answers (with the index and a row of ------ separating):

{%- for candidate in candidates -%}
{{ loop.index0 }}: {{ candidate }}
------
{%- endfor -%}

Please evaluate these candidates and provide the index of the best one."#;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_template_good() {
        let templates = get_test_template_config();
        let template_name = "greeting";

        let context = serde_json::json!({"name": "world"});
        let result = templates.template_message(template_name, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "hello, world!");

        // Test a little templating:
        let template_name = "greeting_with_age";

        // Test with correct inputs
        let context = serde_json::json!({
            "name": "Alice",
            "age": 30
        });
        let result = templates.template_message(template_name, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, Alice! You are 30 years old.");

        // Test with missing input
        let context = serde_json::json!({
            "name": "Bob"
        });
        let result = templates.template_message(template_name, &context);
        assert!(result.is_err());

        // Test with incorrect input type
        let context = serde_json::json!({
            "name": "Charlie",
            "age": "thirty"
        });
        let result = templates.template_message(template_name, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, Charlie! You are thirty years old.");
    }

    #[test]
    fn test_template_nonexistent_file() {
        let nonexistent_path = PathBuf::from("nonexistent_file.txt");
        let mut template_config = TemplateConfig::new();
        let template_paths =
            HashMap::from([("nonexistent_file".to_string(), nonexistent_path.clone())]);
        let result = template_config.initialize(template_paths);
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
        let mut template_config = TemplateConfig::new();
        let template_paths = HashMap::from([(
            "malformed_template".to_string(),
            temp_file.path().to_path_buf(),
        )]);
        let result = template_config.initialize(template_paths);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to add template"));
    }

    pub fn get_test_template_config<'a>() -> TemplateConfig<'a> {
        let mut templates = HashMap::new();

        // Template 1
        let template1 = "hello, {{name}}!";
        let temp_file1 = create_temp_file(template1);
        templates.insert("greeting".to_string(), temp_file1.path().to_path_buf());

        // Template 2
        let template2 = "Hello, {{ name }}! You are {{ age }} years old.";
        let temp_file2 = create_temp_file(template2);
        templates.insert(
            "greeting_with_age".to_string(),
            temp_file2.path().to_path_buf(),
        );

        // System template
        let system_template = "You are a helpful and friendly assistant named {{ assistant_name }}";
        let temp_file3 = create_temp_file(system_template);
        templates.insert("system".to_string(), temp_file3.path().to_path_buf());

        // Filled in system template
        let system_template = "You are a helpful and friendly assistant named ChatGPT";
        let temp_file4 = create_temp_file(system_template);
        templates.insert("system_filled".to_string(), temp_file4.path().to_path_buf());

        // Assistant Template
        let assistant_template = "I'm sorry but I can't help you with that because of {{ reason }}";
        let temp_file5 = create_temp_file(assistant_template);
        templates.insert("assistant".to_string(), temp_file5.path().to_path_buf());

        // Filled in assistant template
        let assistant_template = "I'm sorry but I can't help you with that because of it's against my ethical guidelines";
        let temp_file6 = create_temp_file(assistant_template);
        templates.insert(
            "assistant_filled".to_string(),
            temp_file6.path().to_path_buf(),
        );

        // Filled in user template
        let user_template = "What's the capital of Japan?";
        let temp_file7 = create_temp_file(user_template);
        templates.insert("user_filled".to_string(), temp_file7.path().to_path_buf());

        // Initialize templates
        let mut template_config = TemplateConfig::new();
        let _ = template_config.initialize(templates.clone());
        template_config
    }

    fn create_temp_file(content: &str) -> NamedTempFile {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        write!(temp_file, "{}", content).expect("Failed to write to temp file");
        temp_file
    }
}
