use minijinja::{Environment, UndefinedBehavior};
use minijinja_utils::collect_all_template_paths;
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use crate::error::{Error, ErrorDetails};

#[derive(Clone, Debug)]
pub struct TemplateConfig<'c> {
    env: minijinja::Environment<'c>,
}

impl TemplateConfig<'_> {
    pub fn new() -> Self {
        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Strict);
        Self { env }
    }

    /// Initializes the TemplateConfig with the given templates, given as a map from template names
    /// to template content.
    /// If `template_base_directory` is provided, we'll walk the templates explicitly configured,
    /// find all files that we can tell would be loaded, and eagerly load them.
    /// Returns a `HashMap` of all additional templates that were loaded during the walk.
    /// The key is the path to template / name in the minijinja `Environment`, and the
    /// value is the contents.
    pub async fn initialize(
        &mut self,
        configured_templates: HashMap<String, String>,
        template_base_directory: Option<&Path>,
    ) -> Result<HashMap<String, String>, Error> {
        self.env.set_undefined_behavior(UndefinedBehavior::Strict);

        // Phase 1: Load explicitly configured templates
        for (template_name, template_content) in &configured_templates {
            self.env
                .add_template_owned(template_name.clone(), template_content.clone())
                .map_err(|e| {
                    Error::new(ErrorDetails::MiniJinjaTemplate {
                        template_name: template_name.clone(),
                        message: format!("Failed to add template: {e}"),
                    })
                })?;
        }

        // Phase 2: Load hardcoded templates
        self.add_hardcoded_templates()?;

        // Cache for storing loaded templates - will be used in future PR
        let mut all_template_load_data = HashMap::new();
        // Phase 3: If filesystem access is enabled, eagerly load all referenced templates
        if let Some(base_path) = template_base_directory {
            // Create a validation environment with a path loader to discover transitive dependencies
            let mut validation_env = minijinja::Environment::new();
            validation_env.set_undefined_behavior(UndefinedBehavior::Strict);
            validation_env.set_loader(minijinja::path_loader(base_path));

            // Add configured templates to the validation environment
            for (name, content) in &configured_templates {
                validation_env
                    .add_template_owned(name.clone(), content.clone())
                    .map_err(|e| {
                        Error::new(ErrorDetails::MiniJinjaTemplate {
                            template_name: name.clone(),
                            message: format!(
                                "Failed to add template to validation environment: {e}"
                            ),
                        })
                    })?;
            }

            // Analyze each configured template to discover all transitive dependencies
            let mut all_discovered_templates = HashSet::new();
            for template_name in configured_templates.keys() {
                let discovered_templates =
                    collect_all_template_paths(&validation_env, template_name)?;

                all_discovered_templates.extend(discovered_templates);
            }

            // Load discovered templates from filesystem into production environment
            for template_path in all_discovered_templates {
                let template_name = template_path.to_string_lossy();

                // Skip any template that is already in the environment
                if self.env.get_template(&template_name).is_ok() {
                    continue;
                }

                // Safely join the base directory with the template path
                let absolute_template_path = match safe_join(base_path, &template_name) {
                    Some(path) => path,
                    None => {
                        tracing::warn!(
                            "Could not safely join base path with template '{}'",
                            template_path.display()
                        );
                        continue;
                    }
                };

                // Read template content from filesystem
                let template_content =
                    match tokio::fs::read_to_string(&absolute_template_path).await {
                        Ok(content) => content,
                        Err(e) => {
                            tracing::warn!(
                                "Failed to read template at {}: {}. Skipping.",
                                absolute_template_path.display(),
                                e
                            );
                            continue;
                        }
                    };

                // Add to production environment
                self.env
                    .add_template_owned(template_name.to_string(), template_content.clone())
                    .map_err(|e| {
                        Error::new(ErrorDetails::MiniJinjaTemplate {
                            template_name: template_name.to_string(),
                            message: format!("Failed to add template: {e}"),
                        })
                    })?;

                all_template_load_data.insert(template_name.to_string(), template_content);
            }

            self.env.set_loader(|name| {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("Could not load template '{name}': the file was missing."),
                    // NOTE: this could also fail if our earlier code for catching dynamic includes failed to catch one
                ))
            });
        } else {
            // If we did not have a filesystem_path, the only templates minijinja could load would have
            // been explicitly specified in the config.
            self.env.set_loader(|name| {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("Could not load template '{name}' - if this a dynamic template included from the filesystem, please set `gateway.template_filesystem_access.enabled` to `true`")
                ))
            });
        }
        Ok(all_template_load_data)
    }

    pub fn add_template(
        &mut self,
        template_name: String,
        template_content: String,
    ) -> Result<(), Error> {
        self.env
            .add_template_owned(template_name.clone(), template_content)
            .map_err(|e| {
                Error::new(ErrorDetails::MiniJinjaTemplate {
                    template_name: template_name.to_string(),
                    message: format!("Failed to add template: {e}"),
                })
            })
    }

    // Templates a message with a MiniJinja template.
    pub fn template_message<S: Serialize>(
        &self,
        template_name: &str,
        context: &S,
    ) -> Result<String, Error> {
        let template = self.env.get_template(template_name).map_err(|_| {
            Error::new(ErrorDetails::MiniJinjaTemplateMissing {
                template_name: template_name.to_string(),
            })
        })?;
        let maybe_message = template.render(context);
        match maybe_message {
            Ok(message) => Ok(message),
            Err(err) => {
                let mut message = format!("Could not render template: {err:#}");
                let mut err = &err as &dyn std::error::Error;
                while let Some(next_err) = err.source() {
                    message.push_str(&format!("\nCaused by: {next_err:#}"));
                    err = next_err;
                }
                Err(ErrorDetails::MiniJinjaTemplateRender {
                    template_name: template_name.to_string(),
                    message,
                }
                .into())
            }
        }
    }

    pub fn get_undeclared_variables(&self, template_name: &str) -> Result<HashSet<String>, Error> {
        let template = self.env.get_template(template_name).map_err(|_| {
            Error::new(ErrorDetails::MiniJinjaTemplateMissing {
                template_name: template_name.to_string(),
            })
        })?;
        Ok(template.undeclared_variables(true))
    }

    // Checks if a template needs any variables (i.e. needs a schema)
    pub fn template_needs_variables(&self, template_name: &str) -> Result<bool, Error> {
        Ok(!self.get_undeclared_variables(template_name)?.is_empty())
    }

    pub fn contains_template(&self, template_name: &str) -> bool {
        self.env.get_template(template_name).is_ok()
    }

    pub fn add_hardcoded_templates(&mut self) -> Result<(), Error> {
        self.env
            .add_template("t0:best_of_n_evaluator_system", BEST_OF_N_EVALUATOR_SYSTEM)
            .map_err(|e| {
                Error::new(ErrorDetails::MiniJinjaTemplate {
                    template_name: "t0:best_of_n_evaluator_system".to_string(),
                    message: format!("Failed to add template: {e}"),
                })
            })?;
        self.env
            .add_template(
                "t0:best_of_n_evaluator_candidates",
                BEST_OF_N_EVALUATOR_CANDIDATES,
            )
            .map_err(|e| {
                Error::new(ErrorDetails::MiniJinjaTemplate {
                    template_name: "t0:best_of_n_evaluator_candidates".to_string(),
                    message: format!("Failed to add template: {e}"),
                })
            })?;
        self.env
            .add_template("t0:mixture_of_n_fuser_system", MIXTURE_OF_N_FUSER_SYSTEM)
            .map_err(|e| {
                Error::new(ErrorDetails::MiniJinjaTemplate {
                    template_name: "t0:mixture_of_n_fuser_system".to_string(),
                    message: format!("Failed to add template: {e}"),
                })
            })?;
        self.env
            .add_template(
                "t0:mixture_of_n_fuser_candidates",
                MIXTURE_OF_N_FUSER_CANDIDATES,
            )
            .map_err(|e| {
                Error::new(ErrorDetails::MiniJinjaTemplate {
                    template_name: "t0:mixture_of_n_fuser_candidates".to_string(),
                    message: format!("Failed to add template: {e}"),
                })
            })?;
        Ok(())
    }
}

/// Safely joins two paths.
/// Taken from `minijinja`
pub fn safe_join(base: &Path, template: &str) -> Option<PathBuf> {
    let mut rv = base.to_path_buf();
    for segment in template.split('/') {
        if segment.starts_with('.') || segment.contains('\\') {
            return None;
        }
        rv.push(segment);
    }
    Some(rv)
}

impl Default for TemplateConfig<'_> {
    fn default() -> Self {
        Self::new()
    }
}

const BEST_OF_N_EVALUATOR_SYSTEM: &str = r#"{%- if inner_system_message is defined -%}You are an assistant tasked with re-ranking candidate answers to the following problem:
------
{{ inner_system_message }}
------
{%- else -%}
You are an assistant tasked with re-ranking candidate answers to a problem.

{%- endif %}
The messages below are the conversation history between the user and the assistant along with a final message giving a set of candidate responses.
Please evaluate the following candidate responses and provide your reasoning along with the index of the best candidate in the following JSON format:
{
    "thinking": "your reasoning here",
    "answer_choice": int  // Range: 0 to {{ max_index }}
}
In the "thinking" block:
First, you should analyze each response itself against the conversation history and determine if it is a good response or not.
Then you should think out loud about which is best and most faithful to instructions.
In the "answer_choice" block: you should output the index of the best response."#;

const BEST_OF_N_EVALUATOR_CANDIDATES: &str = r"Here are the candidate answers (with the index and a row of ------ separating):{% for candidate in candidates %}
{{ loop.index0 }}: {{ candidate }}
------
{%- endfor %}
Please evaluate these candidates and provide the index of the best one.";

// Lightly edited from Table 6 in the [Archon paper](https://arxiv.org/abs/2409.15254).
const MIXTURE_OF_N_FUSER_SYSTEM: &str = r"{%- if inner_system_message is defined -%}You have been provided with a set of responses from various models to the following problem:
------
{{ inner_system_message }}
------
{%- else -%}
You have been provided with a set of responses from various models to the latest user
query.

{%- endif %}
Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses.";

const MIXTURE_OF_N_FUSER_CANDIDATES: &str = r"Here are the candidate answers (with the index and a row of ------ separating):{% for candidate in candidates %}
{{ loop.index0 }}:
{{ candidate }}
------
{%- endfor %}";

#[cfg(test)]
pub(crate) mod tests {
    use std::path::PathBuf;

    use crate::{config::path::ResolvedTomlPathData, jsonschema_util::StaticJSONSchema};

    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_template_good() {
        let templates = get_test_template_config().await;
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
        let err = result.unwrap_err();
        match err.get_details() {
            ErrorDetails::MiniJinjaTemplateRender { message, .. } => {
                assert!(message.contains("Referenced variables"));
            }
            _ => {
                panic!("Should be a MiniJinjaTemplateRender error");
            }
        }

        // Test with incorrect input type
        let context = serde_json::json!({
            "name": "Charlie",
            "age": "thirty"
        });
        let result = templates.template_message(template_name, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, Charlie! You are thirty years old.");
    }

    #[tokio::test]
    async fn test_template_malformed_template() {
        let malformed_template = "{{ unclosed_bracket";
        let mut template_config = TemplateConfig::new();
        let template_paths = HashMap::from([(
            "malformed_template".to_string(),
            malformed_template.to_string(),
        )]);
        let result = template_config.initialize(template_paths, None).await;
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to add template"));
    }

    #[tokio::test]
    async fn test_to_json_filter() {
        let templates = get_test_template_config().await;
        let context = serde_json::json!({"input": ["hello", "world"]});
        let result = templates.template_message("user_with_tojson", &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, [\"hello\",\"world\"]");
    }

    #[tokio::test]
    async fn test_join_filter() {
        let templates = get_test_template_config().await;
        let context = serde_json::json!({"input": ["hello", "hello", "world"]});
        let result = templates.template_message("user_with_join", &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, hello, hello, world!");
    }

    pub fn test_system_template_schema() -> StaticJSONSchema {
        StaticJSONSchema::from_value(json!({
            "type": "object",
            "properties": {
                "assistant_name": {
                    "type": "string"
                }
            },
            "required": ["assistant_name"]
        }))
        .unwrap()
    }

    pub fn test_user_template_schema() -> StaticJSONSchema {
        StaticJSONSchema::from_value(json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                },
                "age": {
                    "type": "number"
                }
            },
            "required": ["name", "age"]
        }))
        .unwrap()
    }

    pub fn test_assistant_template_schema() -> StaticJSONSchema {
        StaticJSONSchema::from_value(json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string"
                }
            },
            "required": ["reason"]
        }))
        .unwrap()
    }

    // Filled in system template
    pub fn get_system_filled_template() -> ResolvedTomlPathData {
        ResolvedTomlPathData::new_for_tests(
            PathBuf::from("system_filled"),
            Some("You are a helpful and friendly assistant named ChatGPT".to_string()),
        )
    }

    // Filled in user template
    pub fn get_user_filled_template() -> ResolvedTomlPathData {
        ResolvedTomlPathData::new_for_tests(
            PathBuf::from("user_filled"),
            Some("What's the capital of Japan?".to_string()),
        )
    }

    // Filled in assistant template
    pub fn get_assistant_filled_template() -> ResolvedTomlPathData {
        ResolvedTomlPathData::new_for_tests(
            PathBuf::from("assistant_filled"),
            Some("I'm sorry but I can't help you with that because of it's against my ethical guidelines".to_string()),
        )
    }

    // System template
    pub fn get_system_template() -> ResolvedTomlPathData {
        ResolvedTomlPathData::new_for_tests(
            PathBuf::from("system"),
            Some("You are a helpful and friendly assistant named {{ assistant_name }}".to_string()),
        )
    }

    pub fn get_assistant_template() -> ResolvedTomlPathData {
        ResolvedTomlPathData::new_for_tests(
            PathBuf::from("assistant"),
            Some("I'm sorry but I can't help you with that because of {{ reason }}".to_string()),
        )
    }

    pub fn get_greeting_with_age_template() -> ResolvedTomlPathData {
        ResolvedTomlPathData::new_for_tests(
            PathBuf::from("greeting_with_age"),
            Some("Hello, {{ name }}! You are {{ age }} years old.".to_string()),
        )
    }

    pub async fn get_test_template_config<'a>() -> TemplateConfig<'a> {
        let mut templates = HashMap::new();

        // Template 1
        templates.insert("greeting".to_string(), "hello, {{name}}!".to_string());

        // Template 2
        templates.insert(
            "greeting_with_age".to_string(),
            get_greeting_with_age_template().data().to_string(),
        );

        // System template
        templates.insert(
            "system".to_string(),
            get_system_template().data().to_string(),
        );

        // Filled in system template
        templates.insert(
            "system_filled".to_string(),
            get_system_filled_template().data().to_string(),
        );

        // Assistant Template
        templates.insert(
            "assistant".to_string(),
            get_assistant_template().data().to_string(),
        );

        // Filled in assistant template
        templates.insert(
            "assistant_filled".to_string(),
            get_assistant_filled_template().data().to_string(),
        );

        // Filled in user template
        templates.insert(
            "user_filled".to_string(),
            get_user_filled_template().data().to_string(),
        );

        // Template with tojson filter
        templates.insert(
            "user_with_tojson".to_string(),
            "Hello, {{ input | tojson }}".to_string(),
        );

        // Template with join filter
        templates.insert(
            "user_with_join".to_string(),
            "Hello, {{ input | join(', ') }}!".to_string(),
        );

        // Initialize templates
        let mut template_config = TemplateConfig::new();
        let _ = template_config.initialize(templates, None).await;
        template_config
    }

    #[tokio::test]
    async fn test_hardcoded_best_of_n_evaluator_system() {
        let mut config = TemplateConfig::new();
        config.initialize(HashMap::new(), None).await.unwrap();

        // 1. Test with inner_system_message and max_index = 3
        let context_with_message = json!({
            "inner_system_message": "some system message",
            "max_index": 3
        });
        let output_with_message = config
            .template_message("t0:best_of_n_evaluator_system", &context_with_message)
            .expect("Should render with inner_system_message and max_index");

        // Because Jinja has whitespace controls `{%- ... -%}`, you must match its exact spacing
        // and newlines. Below is an example of what the actual rendered string should look like.
        // Verify you have no accidental leading or trailing spaces or newlines unless they exist
        // in the template.

        let expected_with_message = r#"You are an assistant tasked with re-ranking candidate answers to the following problem:
------
some system message
------
The messages below are the conversation history between the user and the assistant along with a final message giving a set of candidate responses.
Please evaluate the following candidate responses and provide your reasoning along with the index of the best candidate in the following JSON format:
{
    "thinking": "your reasoning here",
    "answer_choice": int  // Range: 0 to 3
}
In the "thinking" block:
First, you should analyze each response itself against the conversation history and determine if it is a good response or not.
Then you should think out loud about which is best and most faithful to instructions.
In the "answer_choice" block: you should output the index of the best response."#;

        assert_eq!(
            output_with_message, expected_with_message,
            "Rendered text does not match the exact expected text (with inner_system_message)."
        );

        // 2. Test without inner_system_message but with max_index = 2
        let context_no_message = json!({ "max_index": 2 });
        let output_no_message = config
            .template_message("t0:best_of_n_evaluator_system", &context_no_message)
            .expect("Should render without inner_system_message");

        // This matches the `else` branch of the template
        let expected_no_message = r#"You are an assistant tasked with re-ranking candidate answers to a problem.
The messages below are the conversation history between the user and the assistant along with a final message giving a set of candidate responses.
Please evaluate the following candidate responses and provide your reasoning along with the index of the best candidate in the following JSON format:
{
    "thinking": "your reasoning here",
    "answer_choice": int  // Range: 0 to 2
}
In the "thinking" block:
First, you should analyze each response itself against the conversation history and determine if it is a good response or not.
Then you should think out loud about which is best and most faithful to instructions.
In the "answer_choice" block: you should output the index of the best response."#;

        assert_eq!(
            output_no_message, expected_no_message,
            "Rendered text does not match the exact expected text (without inner_system_message)."
        );
    }

    #[tokio::test]
    async fn test_hardcoded_best_of_n_evaluator_candidates() {
        let mut config = TemplateConfig::new();
        config.initialize(HashMap::new(), None).await.unwrap();

        // Provide a list of candidates.
        let context = json!({
            "candidates": ["Candidate A", "Candidate B"]
        });
        let output = config
            .template_message("t0:best_of_n_evaluator_candidates", &context)
            .expect("Should render best_of_n_evaluator_candidates");

        // The loop uses:
        //  {% for candidate in candidates -%}
        //      {{ loop.index0 }}: {{ candidate }}
        //      ------
        //  {%- endfor %}
        //
        // That means for two candidates, we get:
        //  "0: Candidate A
        //   ------
        //   1: Candidate B
        //   ------"

        // Because there's no extra text beyond that, the exact text should be:
        let expected = r"Here are the candidate answers (with the index and a row of ------ separating):
0: Candidate A
------
1: Candidate B
------
Please evaluate these candidates and provide the index of the best one.";

        assert_eq!(
            output, expected,
            "best_of_n_evaluator_candidates did not match the exact expected text."
        );
    }

    #[tokio::test]
    async fn test_hardcoded_mixture_of_n_fuser_system() {
        let mut config = TemplateConfig::new();
        config.initialize(HashMap::new(), None).await.unwrap();

        // 1. With inner_system_message
        let context_with_message = json!({ "inner_system_message": "some system message" });
        let output_with_message = config
            .template_message("t0:mixture_of_n_fuser_system", &context_with_message)
            .expect("Should render mixture_of_n_fuser_system with inner_system_message");

        let expected_with_message = r"You have been provided with a set of responses from various models to the following problem:
------
some system message
------
Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses.";

        assert_eq!(
            output_with_message, expected_with_message,
            "Rendered text does not match exactly (with inner_system_message)."
        );

        // 2. Without inner_system_message
        let context_no_message = json!({});
        let output_no_message = config
            .template_message("t0:mixture_of_n_fuser_system", &context_no_message)
            .expect("Should render mixture_of_n_fuser_system without inner_system_message");

        let expected_no_message = r"You have been provided with a set of responses from various models to the latest user
query.
Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses.";

        assert_eq!(
            output_no_message, expected_no_message,
            "Rendered text does not match exactly (without inner_system_message)."
        );
    }

    #[tokio::test]
    async fn test_hardcoded_mixture_of_n_fuser_candidates() {
        let mut config = TemplateConfig::new();
        config.initialize(HashMap::new(), None).await.unwrap();

        let context = json!({
            "candidates": [
                "Candidate response #1",
                "Candidate response #2"
            ]
        });
        let output = config
            .template_message("t0:mixture_of_n_fuser_candidates", &context)
            .expect("Should render mixture_of_n_fuser_candidates with candidates");

        // This template is:
        //  "Here are the candidate answers (with the index and a row of ------ separating):
        //  {% for candidate in candidates -%}
        //  {{ loop.index0 }}:
        //  {{ candidate }}
        //  ------
        //  {%- endfor %}"

        let expected = r"Here are the candidate answers (with the index and a row of ------ separating):
0:
Candidate response #1
------
1:
Candidate response #2
------";

        assert_eq!(
            output, expected,
            "mixture_of_n_fuser_candidates did not match the exact expected text."
        );
    }

    // Tests for filesystem template loading feature

    #[tokio::test]
    async fn test_filesystem_template_loading_with_include() {
        // Setup: Create temp directory with template files
        let temp_dir = tempfile::TempDir::new().unwrap();
        let header_path = temp_dir.path().join("header.html");
        std::fs::write(&header_path, "<h1>{{ title }}</h1>").unwrap();

        // Configure main template that includes the file
        let mut config = TemplateConfig::new();
        let templates = HashMap::from([(
            "main".to_string(),
            "{% include 'header.html' %}{{ body }}".to_string(),
        )]);

        // Initialize with filesystem access
        config
            .initialize(templates, Some(temp_dir.path()))
            .await
            .unwrap();

        // Assert template renders correctly
        let result =
            config.template_message("main", &json!({"title": "Welcome", "body": "Content"}));
        assert_eq!(result.unwrap(), "<h1>Welcome</h1>Content");
    }

    #[tokio::test]
    async fn test_filesystem_template_loading_nested_includes() {
        // Create temp directory with multiple template files
        let temp_dir = tempfile::TempDir::new().unwrap();

        std::fs::write(
            temp_dir.path().join("base.html"),
            "<html>{% include 'header.html' %}</html>",
        )
        .unwrap();

        std::fs::write(
            temp_dir.path().join("header.html"),
            "<head>{% include 'title.html' %}</head>",
        )
        .unwrap();

        std::fs::write(
            temp_dir.path().join("title.html"),
            "<title>{{ page_title }}</title>",
        )
        .unwrap();

        // Configure main template that includes base
        let mut config = TemplateConfig::new();
        let templates =
            HashMap::from([("main".to_string(), "{% include 'base.html' %}".to_string())]);

        config
            .initialize(templates, Some(temp_dir.path()))
            .await
            .unwrap();

        // Assert all nested templates load and render correctly
        let result = config.template_message("main", &json!({"page_title": "My Page"}));
        assert_eq!(
            result.unwrap(),
            "<html><head><title>My Page</title></head></html>"
        );
    }

    #[tokio::test]
    async fn test_filesystem_template_loading_dynamic_include_error() {
        // Configure template with dynamic include (variable)
        let mut config = TemplateConfig::new();
        let templates = HashMap::from([(
            "main".to_string(),
            "{% include template_name %}".to_string(),
        )]);

        // Create temp directory (though it won't matter since we fail during static analysis)
        let temp_dir = tempfile::TempDir::new().unwrap();

        // Initialize should FAIL because collect_all_template_paths detects dynamic load
        let result = config.initialize(templates, Some(temp_dir.path())).await;
        assert!(result.is_err());

        // Verify error is DynamicTemplateLoad
        let err = result.unwrap_err();
        match err.get_details() {
            ErrorDetails::DynamicTemplateLoad { internal } => match internal {
                minijinja_utils::AnalysisError::DynamicLoadsFound(locations) => {
                    assert_eq!(locations.len(), 1);
                    assert_eq!(locations[0].reason, "variable");
                }
                minijinja_utils::AnalysisError::ParseError(_) => {
                    panic!("Expected DynamicLoadsFound")
                }
            },
            _ => panic!(
                "Expected DynamicTemplateLoad error, got: {:?}",
                err.get_details()
            ),
        }
    }

    #[tokio::test]
    async fn test_filesystem_template_loading_missing_file() {
        // Configure template with static include
        let mut config = TemplateConfig::new();
        let templates = HashMap::from([(
            "main".to_string(),
            "{% include 'header.html' %}Content".to_string(),
        )]);

        // Initialize with temp directory but don't create the header.html file
        let temp_dir = tempfile::TempDir::new().unwrap();

        // Initialize should succeed (file is skipped with warning)
        config
            .initialize(templates, Some(temp_dir.path()))
            .await
            .unwrap();

        // Attempting to render should fail because the included template wasn't loaded
        let result = config.template_message("main", &json!({}));
        assert!(result.is_err());

        // Verify it's a missing template error
        match result.unwrap_err().get_details() {
            ErrorDetails::MiniJinjaTemplateRender { message, .. } => {
                assert!(
                    message.contains("header.html") || message.contains("the file was missing")
                );
            }
            _ => panic!("Expected MiniJinjaTemplateRender error"),
        }
    }

    #[tokio::test]
    async fn test_filesystem_template_loading_unsafe_path() {
        // Configure template with path traversal attempt
        let mut config = TemplateConfig::new();
        let templates = HashMap::from([(
            "main".to_string(),
            "{% include '../../../etc/passwd' %}Content".to_string(),
        )]);

        // Initialize with temp directory
        let temp_dir = tempfile::TempDir::new().unwrap();

        // Initialize should succeed (unsafe path is skipped with warning)
        config
            .initialize(templates, Some(temp_dir.path()))
            .await
            .unwrap();

        // Attempting to render should fail because the template wasn't loaded
        let result = config.template_message("main", &json!({}));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_filesystem_template_loading_without_base_directory() {
        // Configure template with include
        let mut config = TemplateConfig::new();
        let templates = HashMap::from([(
            "main".to_string(),
            "{% include 'header.html' %}Content".to_string(),
        )]);

        // Initialize WITHOUT filesystem access
        config.initialize(templates, None).await.unwrap();

        // Attempting to render should fail with helpful error message
        let result = config.template_message("main", &json!({}));
        assert!(result.is_err());

        // Verify error message mentions enabling filesystem access
        match result.unwrap_err().get_details() {
            ErrorDetails::MiniJinjaTemplateRender { message, .. } => {
                assert!(
                    message.contains("template_filesystem_access")
                        || message.contains("header.html")
                );
            }
            _ => panic!("Expected MiniJinjaTemplateRender error"),
        }
    }

    #[tokio::test]
    async fn test_filesystem_template_loading_no_references() {
        // Configure simple template with no includes
        let mut config = TemplateConfig::new();
        let templates = HashMap::from([("simple".to_string(), "Hello {{ name }}".to_string())]);

        // Initialize with temp directory
        let temp_dir = tempfile::TempDir::new().unwrap();
        config
            .initialize(templates, Some(temp_dir.path()))
            .await
            .unwrap();

        // Assert template renders normally
        let result = config.template_message("simple", &json!({"name": "World"}));
        assert_eq!(result.unwrap(), "Hello World");
    }

    #[tokio::test]
    async fn test_filesystem_template_loading_subdirectories() {
        // Create temp directory with subdirectory structure
        let temp_dir = tempfile::TempDir::new().unwrap();
        let partials_dir = temp_dir.path().join("partials");
        std::fs::create_dir(&partials_dir).unwrap();

        std::fs::write(
            partials_dir.join("footer.html"),
            "<footer>{{ copyright }}</footer>",
        )
        .unwrap();

        // Configure template that includes from subdirectory
        let mut config = TemplateConfig::new();
        let templates = HashMap::from([(
            "main".to_string(),
            "{% include 'partials/footer.html' %}".to_string(),
        )]);

        config
            .initialize(templates, Some(temp_dir.path()))
            .await
            .unwrap();

        // Assert file loads correctly from subdirectory
        let result = config.template_message("main", &json!({"copyright": "2025"}));
        assert_eq!(result.unwrap(), "<footer>2025</footer>");
    }
}
