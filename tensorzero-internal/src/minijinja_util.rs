use minijinja::{Environment, UndefinedBehavior};
use serde_json::Value;
use std::{collections::HashMap, path::PathBuf};

use crate::error::{Error, ErrorDetails};

#[derive(Debug)]
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
    /// to template paths.
    /// If `filesystem_path` is provided, we'll register a minijinja filesystem loader with the specified path
    pub fn initialize(
        &mut self,
        template_paths: HashMap<String, String>,
        filesystem_path: Option<PathBuf>,
    ) -> Result<(), Error> {
        self.env.set_undefined_behavior(UndefinedBehavior::Strict);

        for (template_name, template_content) in template_paths {
            self.env
                .add_template_owned(template_name.clone(), template_content)
                .map_err(|e| {
                    Error::new(ErrorDetails::MiniJinjaTemplate {
                        template_name,
                        message: format!("Failed to add template: {}", e),
                    })
                })?;
        }
        self.add_hardcoded_templates()?;
        if let Some(path) = filesystem_path {
            self.env.set_loader(minijinja::path_loader(path));
        } else {
            self.env.set_loader(|name| {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("Could not load template '{name}' - if this a dynamic template included from the filesystem, please set [gateway.enable_template_filesystem_access] to `true`")
                ))
            });
        }
        Ok(())
    }

    pub fn add_template(
        &mut self,
        template_name: &str,
        template_content: &str,
    ) -> Result<(), Error> {
        self.env
            .add_template_owned(template_name.to_string(), template_content.to_string())
            .map_err(|e| {
                Error::new(ErrorDetails::MiniJinjaTemplate {
                    template_name: template_name.to_string(),
                    message: format!("Failed to add template: {}", e),
                })
            })
    }

    // Templates a message with a MiniJinja template.
    pub fn template_message(&self, template_name: &str, context: &Value) -> Result<String, Error> {
        let template = self.env.get_template(template_name).map_err(|_| {
            Error::new(ErrorDetails::MiniJinjaTemplateMissing {
                template_name: template_name.to_string(),
            })
        })?;
        let maybe_message = template.render(context);
        match maybe_message {
            Ok(message) => Ok(message),
            Err(err) => {
                let mut message = format!("Could not render template: {:#}", err);
                let mut err = &err as &dyn std::error::Error;
                while let Some(next_err) = err.source() {
                    message.push_str(&format!("\nCaused by: {:#}", next_err));
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

    // Checks if a template needs any variables (i.e. needs a schema)
    pub fn template_needs_variables(&self, template_name: &str) -> Result<bool, Error> {
        let template = self.env.get_template(template_name).map_err(|_| {
            Error::new(ErrorDetails::MiniJinjaTemplateMissing {
                template_name: template_name.to_string(),
            })
        })?;

        let template_needs_variables = !template.undeclared_variables(true).is_empty();

        Ok(template_needs_variables)
    }

    pub fn add_hardcoded_templates(&mut self) -> Result<(), Error> {
        self.env
            .add_template("t0:best_of_n_evaluator_system", BEST_OF_N_EVALUATOR_SYSTEM)
            .map_err(|e| {
                Error::new(ErrorDetails::MiniJinjaTemplate {
                    template_name: "t0:best_of_n_evaluator_system".to_string(),
                    message: format!("Failed to add template: {}", e),
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
                    message: format!("Failed to add template: {}", e),
                })
            })?;
        self.env
            .add_template("t0:mixture_of_n_fuser_system", MIXTURE_OF_N_FUSER_SYSTEM)
            .map_err(|e| {
                Error::new(ErrorDetails::MiniJinjaTemplate {
                    template_name: "t0:mixture_of_n_fuser_system".to_string(),
                    message: format!("Failed to add template: {}", e),
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
                    message: format!("Failed to add template: {}", e),
                })
            })?;
        Ok(())
    }
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

const BEST_OF_N_EVALUATOR_CANDIDATES: &str = r#"Here are the candidate answers (with the index and a row of ------ separating):{% for candidate in candidates %}
{{ loop.index0 }}: {{ candidate }}
------
{%- endfor %}
Please evaluate these candidates and provide the index of the best one."#;

// Lightly edited from Table 6 in the [Archon paper](https://arxiv.org/abs/2409.15254).
const MIXTURE_OF_N_FUSER_SYSTEM: &str = r#"{%- if inner_system_message is defined -%}You have been provided with a set of responses from various models to the following problem:
------
{{ inner_system_message }}
------
{%- else -%}
You have been provided with a set of responses from various models to the latest user
query.

{%- endif %}
Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses."#;

const MIXTURE_OF_N_FUSER_CANDIDATES: &str = r#"Here are the candidate answers (with the index and a row of ------ separating):{% for candidate in candidates %}
{{ loop.index0 }}:
{{ candidate }}
------
{%- endfor %}"#;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

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

    #[test]
    fn test_template_malformed_template() {
        let malformed_template = "{{ unclosed_bracket";
        let mut template_config = TemplateConfig::new();
        let template_paths = HashMap::from([(
            "malformed_template".to_string(),
            malformed_template.to_string(),
        )]);
        let result = template_config.initialize(template_paths, None);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to add template"));
    }

    #[test]
    fn test_to_json_filter() {
        let templates = get_test_template_config();
        let context = serde_json::json!({"input": ["hello", "world"]});
        let result = templates.template_message("user_with_tojson", &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, [\"hello\",\"world\"]");
    }

    #[test]
    fn test_join_filter() {
        let templates = get_test_template_config();
        let context = serde_json::json!({"input": ["hello", "hello", "world"]});
        let result = templates.template_message("user_with_join", &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, hello, hello, world!");
    }

    pub fn get_test_template_config<'a>() -> TemplateConfig<'a> {
        let mut templates = HashMap::new();

        // Template 1
        templates.insert("greeting".to_string(), "hello, {{name}}!".to_string());

        // Template 2
        templates.insert(
            "greeting_with_age".to_string(),
            "Hello, {{ name }}! You are {{ age }} years old.".to_string(),
        );

        // System template
        templates.insert(
            "system".to_string(),
            "You are a helpful and friendly assistant named {{ assistant_name }}".to_string(),
        );

        // Filled in system template
        templates.insert(
            "system_filled".to_string(),
            "You are a helpful and friendly assistant named ChatGPT".to_string(),
        );

        // Assistant Template
        templates.insert(
            "assistant".to_string(),
            "I'm sorry but I can't help you with that because of {{ reason }}".to_string(),
        );

        // Filled in assistant template
        templates.insert(
            "assistant_filled".to_string(),
            "I'm sorry but I can't help you with that because of it's against my ethical guidelines".to_string(),
        );

        // Filled in user template
        templates.insert(
            "user_filled".to_string(),
            "What's the capital of Japan?".to_string(),
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
        let _ = template_config.initialize(templates, None);
        template_config
    }

    #[test]
    fn test_hardcoded_best_of_n_evaluator_system() {
        let mut config = TemplateConfig::new();
        config.initialize(HashMap::new(), None).unwrap();

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

    #[test]
    fn test_hardcoded_best_of_n_evaluator_candidates() {
        let mut config = TemplateConfig::new();
        config.initialize(HashMap::new(), None).unwrap();

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
        let expected = r#"Here are the candidate answers (with the index and a row of ------ separating):
0: Candidate A
------
1: Candidate B
------
Please evaluate these candidates and provide the index of the best one."#;

        assert_eq!(
            output, expected,
            "best_of_n_evaluator_candidates did not match the exact expected text."
        );
    }

    #[test]
    fn test_hardcoded_mixture_of_n_fuser_system() {
        let mut config = TemplateConfig::new();
        config.initialize(HashMap::new(), None).unwrap();

        // 1. With inner_system_message
        let context_with_message = json!({ "inner_system_message": "some system message" });
        let output_with_message = config
            .template_message("t0:mixture_of_n_fuser_system", &context_with_message)
            .expect("Should render mixture_of_n_fuser_system with inner_system_message");

        let expected_with_message = r#"You have been provided with a set of responses from various models to the following problem:
------
some system message
------
Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses."#;

        assert_eq!(
            output_with_message, expected_with_message,
            "Rendered text does not match exactly (with inner_system_message)."
        );

        // 2. Without inner_system_message
        let context_no_message = json!({});
        let output_no_message = config
            .template_message("t0:mixture_of_n_fuser_system", &context_no_message)
            .expect("Should render mixture_of_n_fuser_system without inner_system_message");

        let expected_no_message = r#"You have been provided with a set of responses from various models to the latest user
query.
Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses."#;

        assert_eq!(
            output_no_message, expected_no_message,
            "Rendered text does not match exactly (without inner_system_message)."
        );
    }

    #[test]
    fn test_hardcoded_mixture_of_n_fuser_candidates() {
        let mut config = TemplateConfig::new();
        config.initialize(HashMap::new(), None).unwrap();

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

        let expected = r#"Here are the candidate answers (with the index and a row of ------ separating):
0:
Candidate response #1
------
1:
Candidate response #2
------"#;

        assert_eq!(
            output, expected,
            "mixture_of_n_fuser_candidates did not match the exact expected text."
        );
    }
}
