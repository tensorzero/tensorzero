use std::{
    collections::HashMap,
    fmt::{self, Display},
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::Deserialize;

use crate::{
    config_parser::PathWithContents,
    error::{Error, ErrorDetails},
    function::{FunctionConfig, FunctionConfigJson},
    jsonschema_util::JSONSchemaFromPath,
    tool::{ImplicitToolConfig, ToolCallConfig, ToolChoice, ToolConfig, IMPLICIT_TOOL_NAME},
    variant::{
        chat_completion::{ChatCompletionConfig, ExtraBodyConfig},
        JsonMode, RetryConfig, VariantConfig,
    },
};

pub const LLM_JUDGE_SYSTEM_SCHEMA_TEXT: &str = include_str!("llm_judge_system_schema.json");
pub const LLM_JUDGE_USER_SCHEMA_TEXT: &str = include_str!("llm_judge_user_schema.json");
pub const LLM_JUDGE_FLOAT_OUTPUT_SCHEMA_TEXT: &str =
    include_str!("llm_judge_float_output_schema.json");
pub const LLM_JUDGE_BOOLEAN_OUTPUT_SCHEMA_TEXT: &str =
    include_str!("llm_judge_boolean_output_schema.json");

#[derive(Debug)]
pub struct EvalConfig {
    pub evaluators: HashMap<String, EvaluatorConfig>,
    pub dataset_name: String,
    pub function_name: String,
}

#[derive(Debug)]
pub enum EvaluatorConfig {
    ExactMatch,
    LLMJudge(LLMJudgeConfig),
}

#[derive(Debug)]
pub struct LLMJudgeConfig {
    pub output_type: LLMJudgeOutputType,
    pub include_datapoint_output: bool,
    pub optimize: LLMJudgeOptimize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LLMJudgeOutputType {
    Float,
    Boolean,
}

impl Display for LLMJudgeOutputType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                LLMJudgeOutputType::Float => "float",
                LLMJudgeOutputType::Boolean => "boolean",
            }
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LLMJudgeOptimize {
    Min,
    Max,
}

pub fn get_llm_judge_function_name(eval_name: &str, evaluator_name: &str) -> String {
    format!("tensorzero::llm_judge::{}::{}", eval_name, evaluator_name)
}

#[derive(Debug, Deserialize)]
pub struct UninitializedEvalConfig {
    evaluators: HashMap<String, UninitializedEvaluatorConfig>,
    dataset_name: String,
    function_name: String,
}

impl UninitializedEvalConfig {
    pub fn load<P: AsRef<Path>>(
        self,
        functions: &HashMap<String, Arc<FunctionConfig>>,
        base_path: P,
        eval_name: &str,
    ) -> Result<(EvalConfig, HashMap<String, Arc<FunctionConfig>>), Error> {
        if !functions.contains_key(&self.function_name) {
            return Err(ErrorDetails::Config {
                message: format!(
                    "Function `{}` not found (referenced in `[evals.{eval_name}]`)",
                    self.function_name
                ),
            }
            .into());
        }
        // Eval names cannot have "::" in them since we use it as a delimiter
        if eval_name.contains("::") {
            return Err(ErrorDetails::Config {
                message: format!(
                    "Eval names cannot contain \"::\" (referenced in `[evals.{eval_name}]`)"
                ),
            }
            .into());
        }
        let evaluator_results = self
            .evaluators
            .into_iter()
            .map(|(name, config)| {
                config
                    .load(&base_path, eval_name, &name)
                    .map(|(eval_config, func_config)| (name, eval_config, func_config))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        // Create HashMaps from the results
        let mut evaluators = HashMap::new();
        let mut function_configs = HashMap::new();

        for (evaluator_name, evaluator_config, function_config) in evaluator_results {
            // Add to evaluators map
            evaluators.insert(evaluator_name.clone(), evaluator_config);

            // Add to function_configs map if Some
            if let Some(config) = function_config {
                function_configs.insert(
                    get_llm_judge_function_name(eval_name, &evaluator_name),
                    Arc::new(config),
                );
            }
        }
        Ok((
            EvalConfig {
                evaluators,
                dataset_name: self.dataset_name,
                function_name: self.function_name,
            },
            function_configs,
        ))
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum UninitializedEvaluatorConfig {
    ExactMatch,
    #[serde(rename = "llm_judge")]
    LLMJudge(UninitializedLLMJudgeConfig),
}

#[derive(Debug, Deserialize)]
struct UninitializedLLMJudgeConfig {
    variants: HashMap<String, UninitializedLLMJudgeVariantConfig>,
    output_type: LLMJudgeOutputType,
    optimize: LLMJudgeOptimize,
    include_datapoint_output: bool,
}

impl UninitializedEvaluatorConfig {
    pub fn load<P: AsRef<Path>>(
        self,
        base_path: &P,
        eval_name: &str,
        evaluator_name: &str,
    ) -> Result<(EvaluatorConfig, Option<FunctionConfig>), Error> {
        // Evaluator names cannot have "::" in them since we use it as a delimiter in our function names later on
        if evaluator_name.contains("::") {
            return Err(ErrorDetails::Config {
                message: format!(
                    "Evaluator names cannot contain \"::\" (referenced in `[evals.{eval_name}.{evaluator_name}]`)"
                ),
            }
            .into());
        }
        match self {
            UninitializedEvaluatorConfig::ExactMatch => Ok((EvaluatorConfig::ExactMatch, None)),
            UninitializedEvaluatorConfig::LLMJudge(params) => {
                let variants = params
                    .variants
                    .into_iter()
                    .map(|(name, variant)| {
                        variant
                            .load(base_path, eval_name, evaluator_name, &params.output_type)
                            .map(|v| (name, v))
                    })
                    .collect::<Result<HashMap<_, _>, Error>>()?;
                let nonzero_weights = variants
                    .iter()
                    .filter(|(_, variant)| variant.weight() > 0.0)
                    .count();
                if nonzero_weights != 1 {
                    // TODO (Viraj): test this
                    return Err(ErrorDetails::Config {
                        message: format!(
                            "Evaluator `{evaluator_name}` in `[evals.{eval_name}]` must have exactly 1 variant that is active. Found {nonzero_weights} variants with nonzero weights."
                        ),
                    }
                    .into());
                }
                let system_schema_value = serde_json::from_str(LLM_JUDGE_SYSTEM_SCHEMA_TEXT)
                    .map_err(|e| {
                        Error::new(ErrorDetails::JsonSchema {
                            message: format!("Failed to parse LLM judge system schema: {e}. This should never happen, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports."),
                        })
                    })?;
                let user_schema_value = serde_json::from_str(LLM_JUDGE_USER_SCHEMA_TEXT)
                    .map_err(|e| {
                        Error::new(ErrorDetails::JsonSchema {
                            message: format!("Failed to parse LLM judge user schema: {e}. This should never happen, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports."),
                        })
                    })?;
                let output_schema_str = match params.output_type {
                    LLMJudgeOutputType::Float => LLM_JUDGE_FLOAT_OUTPUT_SCHEMA_TEXT,
                    LLMJudgeOutputType::Boolean => LLM_JUDGE_BOOLEAN_OUTPUT_SCHEMA_TEXT,
                };
                let output_schema_value = serde_json::from_str(output_schema_str)
                    .map_err(|e| {
                        Error::new(ErrorDetails::JsonSchema {
                            message: format!("Failed to parse LLM judge output schema: {e}. This should never happen, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports."),
                        })
                    })?;
                let output_schema = JSONSchemaFromPath::from_value(&output_schema_value)?;
                // TODO (Viraj): deduplicate this code
                let implicit_tool = ToolConfig::Implicit(ImplicitToolConfig {
                    parameters: output_schema.clone(),
                });
                let implicit_tool_call_config = ToolCallConfig {
                    tools_available: vec![implicit_tool],
                    tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
                    parallel_tool_calls: false,
                };
                let function_config = FunctionConfig::Json(FunctionConfigJson {
                    variants,
                    system_schema: Some(JSONSchemaFromPath::from_value(&system_schema_value)?),
                    user_schema: Some(JSONSchemaFromPath::from_value(&user_schema_value)?),
                    assistant_schema: None,
                    output_schema,
                    implicit_tool_call_config,
                });
                Ok((
                    EvaluatorConfig::LLMJudge(LLMJudgeConfig {
                        output_type: params.output_type,
                        include_datapoint_output: params.include_datapoint_output,
                        optimize: params.optimize,
                    }),
                    Some(function_config),
                ))
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UninitializedLLMJudgeVariantConfig {
    ChatCompletion(UninitializedLLMJudgeChatCompletionVariantConfig),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedLLMJudgeChatCompletionVariantConfig {
    #[serde(default)]
    pub active: bool,
    pub model: Arc<str>,
    pub system_instructions: PathBuf,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub json_mode: JsonMode, // This is JSON
    #[serde(default)]
    pub retries: RetryConfig,
    #[serde(default)]
    pub extra_body: Option<ExtraBodyConfig>,
}

impl UninitializedLLMJudgeVariantConfig {
    pub fn load<P: AsRef<Path>>(
        self,
        base_path: &P,
        eval_name: &str,
        evaluator_name: &str,
        output_type: &LLMJudgeOutputType,
    ) -> Result<VariantConfig, Error> {
        match self {
            UninitializedLLMJudgeVariantConfig::ChatCompletion(params) => {
                let system_instructions =
                    read_system_instructions(params.system_instructions, base_path)?;
                let templated_system_instructions = format!(
                    include_str!("llm_judge_system_instructions.txt"),
                    system_instructions = system_instructions,
                    output_type = output_type
                );
                let system_template = PathWithContents {
                    // Not a real path but this is used as the handle everywhere as the content is already provided below
                    path: PathBuf::from(format!(
                        "tensorzero::llm_judge::{eval_name}::{evaluator_name}::system"
                    )),
                    contents: templated_system_instructions,
                };
                Ok(VariantConfig::ChatCompletion(ChatCompletionConfig {
                    weight: if params.active { 1.0 } else { 0.0 },
                    model: params.model,
                    system_template: Some(system_template),
                    user_template: None,
                    assistant_template: None,
                    temperature: params.temperature,
                    top_p: params.top_p,
                    max_tokens: params.max_tokens,
                    presence_penalty: params.presence_penalty,
                    frequency_penalty: params.frequency_penalty,
                    seed: params.seed,
                    json_mode: Some(params.json_mode),
                    retries: params.retries,
                    extra_body: params.extra_body,
                }))
            }
        }
    }
}

fn read_system_instructions<P1: AsRef<Path>, P2: AsRef<Path>>(
    path: P1,
    base_path: &P2,
) -> Result<String, Error> {
    let path = base_path.as_ref().join(path);
    let file = File::open(&path).map_err(|e| {
        Error::new(ErrorDetails::FileRead {
            message: format!("Failed to open system instructions file: {e}"),
            file_path: path.to_string_lossy().to_string(),
        })
    })?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).map_err(|e| {
        Error::new(ErrorDetails::FileRead {
            message: format!("Failed to read system instructions file: {e}"),
            file_path: path.to_string_lossy().to_string(),
        })
    })?;
    Ok(contents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_system_instructions() {
        let system_instructions = read_system_instructions(
            PathBuf::from("evals/eval1/llm_judge_bool/system_instructions.txt"),
            &PathBuf::from("fixtures/config"),
        )
        .unwrap();
        assert_eq!(
            system_instructions,
            "Return True if there is NSFW content in this generation.\n"
        );

        // Nonexistent file
        let result = read_system_instructions(
            PathBuf::from("evals/eval1/llm_judge_bool/nonexistent.txt"),
            &PathBuf::from("fixtures/config"),
        );
        assert_eq!(*result.unwrap_err().get_details(), ErrorDetails::FileRead {
            message: "Failed to open system instructions file: No such file or directory (os error 2)".to_string(),
            file_path: "fixtures/config/evals/eval1/llm_judge_bool/nonexistent.txt".to_string(),
        });
    }
}
