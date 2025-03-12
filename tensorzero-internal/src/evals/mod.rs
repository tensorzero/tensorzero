use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::Deserialize;

use crate::{
    config_parser::{
        MetricConfig, MetricConfigLevel, MetricConfigOptimize, MetricConfigType, PathWithContents,
    },
    error::{Error, ErrorDetails},
    function::{FunctionConfig, FunctionConfigJson},
    jsonschema_util::JSONSchemaFromPath,
    tool::create_implicit_tool_call_config,
    variant::{
        chat_completion::{ChatCompletionConfig, ExtraBodyConfig},
        JsonMode, RetryConfig, VariantConfig,
    },
};

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
    ExactMatch(ExactMatchConfig),
    LLMJudge(LLMJudgeConfig),
}

impl EvaluatorConfig {
    pub fn cutoff(&self) -> Option<f32> {
        match self {
            EvaluatorConfig::ExactMatch(config) => config.cutoff,
            EvaluatorConfig::LLMJudge(config) => config.cutoff,
        }
    }

    pub fn optimize(&self) -> MetricConfigOptimize {
        match self {
            EvaluatorConfig::ExactMatch(_) => MetricConfigOptimize::Max,
            EvaluatorConfig::LLMJudge(config) => config.optimize.into(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ExactMatchConfig {
    #[serde(default)]
    pub cutoff: Option<f32>,
}

#[derive(Debug)]
pub struct LLMJudgeConfig {
    pub output_type: LLMJudgeOutputType,
    pub include: LLMJudgeIncludeConfig,
    pub optimize: LLMJudgeOptimize,
    pub cutoff: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
pub struct LLMJudgeIncludeConfig {
    #[serde(default)]
    pub reference_output: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LLMJudgeOutputType {
    Float,
    Boolean,
}

impl From<LLMJudgeOutputType> for MetricConfigType {
    fn from(output_type: LLMJudgeOutputType) -> Self {
        match output_type {
            LLMJudgeOutputType::Float => MetricConfigType::Float,
            LLMJudgeOutputType::Boolean => MetricConfigType::Boolean,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LLMJudgeOptimize {
    Min,
    Max,
}

impl From<LLMJudgeOptimize> for MetricConfigOptimize {
    fn from(optimize: LLMJudgeOptimize) -> Self {
        match optimize {
            LLMJudgeOptimize::Min => MetricConfigOptimize::Min,
            LLMJudgeOptimize::Max => MetricConfigOptimize::Max,
        }
    }
}

pub fn get_llm_judge_function_name(eval_name: &str, evaluator_name: &str) -> String {
    format!("tensorzero::llm_judge::{}::{}", eval_name, evaluator_name)
}

pub fn get_evaluator_metric_name(eval_name: &str, evaluator_name: &str) -> String {
    format!(
        "tensorzero::eval_name::{}::evaluator_name::{}",
        eval_name, evaluator_name
    )
}

#[derive(Debug, Deserialize)]
pub struct UninitializedEvalConfig {
    evaluators: HashMap<String, UninitializedEvaluatorConfig>,
    dataset_name: String,
    function_name: String,
}

type EvalLoadResult = Result<
    (
        EvalConfig,                           // The eval itself
        HashMap<String, Arc<FunctionConfig>>, // All functions which the eval needs {function_name -> function_config}
        HashMap<String, MetricConfig>, // All metrics which the eval needs {metric_name -> metric_config}
    ),
    Error,
>;

impl UninitializedEvalConfig {
    pub fn load<P: AsRef<Path>>(
        self,
        functions: &HashMap<String, Arc<FunctionConfig>>,
        base_path: P,
        eval_name: &str,
    ) -> EvalLoadResult {
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
                config.load(&base_path, eval_name, &name).map(
                    |(eval_config, func_config, metric_config)| {
                        (name, eval_config, func_config, metric_config)
                    },
                )
            })
            .collect::<Result<Vec<_>, Error>>()?;

        // Create HashMaps from the results
        let mut evaluators = HashMap::new();
        let mut function_configs = HashMap::new();
        let mut metric_configs = HashMap::new();
        for (evaluator_name, evaluator_config, function_config, metric_config) in evaluator_results
        {
            // Add to evaluators map
            evaluators.insert(evaluator_name.clone(), evaluator_config);

            // Add to function_configs map if Some
            if let Some(config) = function_config {
                function_configs.insert(
                    get_llm_judge_function_name(eval_name, &evaluator_name),
                    Arc::new(config),
                );
            }

            // Add to metric_configs map
            metric_configs.insert(
                get_evaluator_metric_name(eval_name, &evaluator_name),
                metric_config,
            );
        }
        Ok((
            EvalConfig {
                evaluators,
                dataset_name: self.dataset_name,
                function_name: self.function_name,
            },
            function_configs,
            metric_configs,
        ))
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum UninitializedEvaluatorConfig {
    ExactMatch(ExactMatchConfig),
    #[serde(rename = "llm_judge")]
    LLMJudge(UninitializedLLMJudgeConfig),
}

#[derive(Debug, Deserialize)]
struct UninitializedLLMJudgeConfig {
    variants: HashMap<String, UninitializedLLMJudgeVariantConfig>,
    output_type: LLMJudgeOutputType,
    optimize: LLMJudgeOptimize,
    #[serde(default)]
    include: LLMJudgeIncludeConfig,
    #[serde(default)]
    cutoff: Option<f32>,
}

impl UninitializedEvaluatorConfig {
    pub fn load<P: AsRef<Path>>(
        self,
        base_path: &P,
        eval_name: &str,
        evaluator_name: &str,
    ) -> Result<(EvaluatorConfig, Option<FunctionConfig>, MetricConfig), Error> {
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
            UninitializedEvaluatorConfig::ExactMatch(params) => Ok((
                EvaluatorConfig::ExactMatch(params),
                None,
                MetricConfig {
                    r#type: MetricConfigType::Boolean,
                    optimize: MetricConfigOptimize::Max,
                    level: MetricConfigLevel::Inference,
                },
            )),
            UninitializedEvaluatorConfig::LLMJudge(params) => {
                let variants = params
                    .variants
                    .into_iter()
                    .map(|(name, variant)| {
                        variant
                            .load(base_path, eval_name, evaluator_name)
                            .map(|v| (name, v))
                    })
                    .collect::<Result<HashMap<_, _>, Error>>()?;
                let nonzero_weights = variants
                    .iter()
                    // Treat a None weight as 0.0 for this check - we only care if we have multiple variants with an explicit positive weight
                    .filter(|(_, variant)| variant.weight().unwrap_or(0.0) > 0.0)
                    .count();
                if nonzero_weights != 1 {
                    return Err(ErrorDetails::Config {
                        message: format!(
                            "Evaluator `{evaluator_name}` in `[evals.{eval_name}]` must have exactly 1 variant that is active. Found {nonzero_weights} variants with nonzero weights."
                        ),
                    }
                    .into());
                }
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
                let implicit_tool_call_config =
                    create_implicit_tool_call_config(output_schema.clone());
                let function_config = FunctionConfig::Json(FunctionConfigJson {
                    variants,
                    system_schema: None,
                    user_schema: Some(JSONSchemaFromPath::from_value(&user_schema_value)?),
                    assistant_schema: None,
                    output_schema,
                    implicit_tool_call_config,
                });
                Ok((
                    EvaluatorConfig::LLMJudge(LLMJudgeConfig {
                        output_type: params.output_type,
                        include: params.include,
                        optimize: params.optimize,
                        cutoff: params.cutoff,
                    }),
                    Some(function_config),
                    MetricConfig {
                        r#type: params.output_type.into(),
                        optimize: params.optimize.into(),
                        level: MetricConfigLevel::Inference,
                    },
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
    ) -> Result<VariantConfig, Error> {
        match self {
            UninitializedLLMJudgeVariantConfig::ChatCompletion(params) => {
                let system_instructions =
                    read_system_instructions(params.system_instructions, base_path)?;
                let templated_system_instructions = format!(
                    include_str!("llm_judge_system_instructions.txt"),
                    system_instructions = system_instructions,
                );
                let system_template = PathWithContents {
                    // Not a real path but this is used as the handle everywhere as the content is already provided below
                    path: PathBuf::from(format!(
                        "tensorzero::llm_judge::{eval_name}::{evaluator_name}::system"
                    )),
                    contents: templated_system_instructions,
                };
                let user_template = PathWithContents {
                    path: PathBuf::from(format!(
                        "tensorzero::llm_judge::{eval_name}::{evaluator_name}::user"
                    )),
                    contents: include_str!("llm_judge_user_template.minijinja").to_string(),
                };
                Ok(VariantConfig::ChatCompletion(ChatCompletionConfig {
                    weight: Some(if params.active { 1.0 } else { 0.0 }),
                    model: params.model,
                    system_template: Some(system_template),
                    user_template: Some(user_template),
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
    use std::collections::HashMap;
    use std::sync::Arc;

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

    #[test]
    fn test_uninitialized_eval_config_load() {
        // Setup test fixtures
        let base_path = PathBuf::from("fixtures/config");
        let eval_name = "test_eval";

        // Prepare function configs map with a function referenced in the eval
        let mut functions = HashMap::new();
        let function_name = "generate_draft";
        let function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema: create_test_schema(),
            implicit_tool_call_config: create_implicit_tool_call_config(create_test_schema()),
        });
        functions.insert(function_name.to_string(), Arc::new(function_config));

        // Test case 1: Successful loading with exact match evaluator
        {
            let mut evaluators = HashMap::new();
            evaluators.insert(
                "em_evaluator".to_string(),
                UninitializedEvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: Some(0.4) }),
            );

            let uninitialized_config = UninitializedEvalConfig {
                evaluators,
                dataset_name: "test_dataset".to_string(),
                function_name: function_name.to_string(),
            };

            let result = uninitialized_config.load(&functions, &base_path, eval_name);
            assert!(result.is_ok());

            let (config, additional_functions, metric_configs) = result.unwrap();
            assert_eq!(config.dataset_name, "test_dataset");
            assert_eq!(config.function_name, function_name);
            assert_eq!(config.evaluators.len(), 1);
            match config.evaluators.get("em_evaluator").unwrap() {
                EvaluatorConfig::ExactMatch(params) => assert_eq!(params.cutoff, Some(0.4)),
                _ => panic!("Expected ExactMatch evaluator"),
            }
            // No additional function configs for exact match
            assert_eq!(additional_functions.len(), 0);

            // Verify the metrics
            assert_eq!(metric_configs.len(), 1);

            // Check the metric name follows expected format
            let metric_config_name = get_evaluator_metric_name(eval_name, "em_evaluator");
            assert_eq!(
                metric_config_name,
                "tensorzero::eval_name::test_eval::evaluator_name::em_evaluator"
            );
            assert!(metric_configs.contains_key(&metric_config_name));

            // Verify all properties of the metric config
            let metric_config = metric_configs.get(&metric_config_name).unwrap();
            assert_eq!(metric_config.r#type, MetricConfigType::Boolean);
            assert_eq!(metric_config.optimize, MetricConfigOptimize::Max);
            assert_eq!(metric_config.level, MetricConfigLevel::Inference);
        }

        // Test case 2: Successful loading with LLM judge evaluator
        {
            let mut variants = HashMap::new();
            variants.insert(
                "test_variant".to_string(),
                UninitializedLLMJudgeVariantConfig::ChatCompletion(
                    UninitializedLLMJudgeChatCompletionVariantConfig {
                        active: true,
                        model: Arc::from("gpt-3.5-turbo"),
                        system_instructions: PathBuf::from(
                            "evals/eval1/llm_judge_bool/system_instructions.txt",
                        ),
                        temperature: Some(0.7),
                        top_p: None,
                        max_tokens: Some(100),
                        presence_penalty: None,
                        frequency_penalty: None,
                        seed: None,
                        json_mode: JsonMode::ImplicitTool,
                        retries: RetryConfig::default(),
                        extra_body: None,
                    },
                ),
            );

            let llm_judge_config = UninitializedLLMJudgeConfig {
                variants,
                output_type: LLMJudgeOutputType::Boolean,
                optimize: LLMJudgeOptimize::Min,
                include: LLMJudgeIncludeConfig {
                    reference_output: false,
                },
                cutoff: None,
            };

            let mut evaluators = HashMap::new();
            evaluators.insert(
                "llm_judge_eval".to_string(),
                UninitializedEvaluatorConfig::LLMJudge(llm_judge_config),
            );

            let uninitialized_config = UninitializedEvalConfig {
                evaluators,
                dataset_name: "test_dataset".to_string(),
                function_name: function_name.to_string(),
            };

            let result = uninitialized_config.load(&functions, &base_path, eval_name);
            assert!(result.is_ok());

            let (config, additional_functions, metric_configs) = result.unwrap();
            assert_eq!(config.evaluators.len(), 1);

            // Verify LLM judge evaluator config
            match config.evaluators.get("llm_judge_eval").unwrap() {
                EvaluatorConfig::LLMJudge(judge_config) => {
                    assert!(matches!(
                        judge_config.output_type,
                        LLMJudgeOutputType::Boolean
                    ));
                    assert!(matches!(judge_config.optimize, LLMJudgeOptimize::Min));
                    assert!(!judge_config.include.reference_output);
                }
                _ => panic!("Expected LLMJudge evaluator config"),
            }

            // Verify additional function config was created
            assert_eq!(additional_functions.len(), 1);
            let function_name = get_llm_judge_function_name(eval_name, "llm_judge_eval");
            assert!(additional_functions.contains_key(&function_name));

            // Verify the function config has the correct type
            match additional_functions[&function_name].as_ref() {
                FunctionConfig::Json(json_config) => {
                    assert_eq!(json_config.variants.len(), 1);
                    assert!(json_config.variants.contains_key("test_variant"));
                    assert!(json_config.system_schema.is_none());
                    assert!(json_config.user_schema.is_some());
                    assert!(json_config.output_schema.value.is_object());
                }
                _ => panic!("Expected Json function config"),
            }

            // Verify the metrics
            assert_eq!(metric_configs.len(), 1);

            // Check the metric name follows expected format
            let metric_config_name = get_evaluator_metric_name(eval_name, "llm_judge_eval");
            assert_eq!(
                metric_config_name,
                "tensorzero::eval_name::test_eval::evaluator_name::llm_judge_eval"
            );
            assert!(metric_configs.contains_key(&metric_config_name));

            // Verify all properties of the metric config
            let metric_config = metric_configs.get(&metric_config_name).unwrap();
            assert_eq!(metric_config.r#type, MetricConfigType::Boolean);
            assert_eq!(metric_config.optimize, MetricConfigOptimize::Min);
            assert_eq!(metric_config.level, MetricConfigLevel::Inference);

            // Verify the type conversion from LLMJudgeOutputType to MetricConfigType
            let llm_judge_eval = match config.evaluators.get("llm_judge_eval").unwrap() {
                EvaluatorConfig::LLMJudge(config) => config,
                _ => panic!("Expected LLMJudge evaluator"),
            };
            assert_eq!(
                MetricConfigType::from(llm_judge_eval.output_type),
                metric_config.r#type
            );

            // Verify the optimize conversion from LLMJudgeOptimize to MetricConfigOptimize
            assert_eq!(
                MetricConfigOptimize::from(llm_judge_eval.optimize),
                metric_config.optimize
            );
        }

        // Test case 2.1: Successful loading with LLM judge evaluator with Float output type
        {
            let mut variants = HashMap::new();
            variants.insert(
                "test_variant".to_string(),
                UninitializedLLMJudgeVariantConfig::ChatCompletion(
                    UninitializedLLMJudgeChatCompletionVariantConfig {
                        active: true,
                        model: Arc::from("gpt-3.5-turbo"),
                        system_instructions: PathBuf::from(
                            "evals/eval1/llm_judge_bool/system_instructions.txt",
                        ),
                        temperature: Some(0.7),
                        top_p: None,
                        max_tokens: Some(100),
                        presence_penalty: None,
                        frequency_penalty: None,
                        seed: None,
                        json_mode: JsonMode::ImplicitTool,
                        retries: RetryConfig::default(),
                        extra_body: None,
                    },
                ),
            );

            let llm_judge_config = UninitializedLLMJudgeConfig {
                variants,
                output_type: LLMJudgeOutputType::Float,
                optimize: LLMJudgeOptimize::Max,
                include: LLMJudgeIncludeConfig {
                    reference_output: true,
                },
                cutoff: None,
            };

            let mut evaluators = HashMap::new();
            evaluators.insert(
                "llm_judge_float".to_string(),
                UninitializedEvaluatorConfig::LLMJudge(llm_judge_config),
            );

            let uninitialized_config = UninitializedEvalConfig {
                evaluators,
                dataset_name: "test_dataset".to_string(),
                function_name: function_name.to_string(),
            };

            let result = uninitialized_config.load(&functions, &base_path, eval_name);
            assert!(result.is_ok());

            let (config, additional_functions, metric_configs) = result.unwrap();
            assert_eq!(config.evaluators.len(), 1);

            // Verify LLM judge evaluator config
            match config.evaluators.get("llm_judge_float").unwrap() {
                EvaluatorConfig::LLMJudge(judge_config) => {
                    assert!(matches!(
                        judge_config.output_type,
                        LLMJudgeOutputType::Float
                    ));
                    assert!(matches!(judge_config.optimize, LLMJudgeOptimize::Max));
                    assert!(judge_config.include.reference_output);
                }
                _ => panic!("Expected LLMJudge evaluator config"),
            }

            // Verify additional function config was created
            assert_eq!(additional_functions.len(), 1);
            let function_name = get_llm_judge_function_name(eval_name, "llm_judge_float");
            assert!(additional_functions.contains_key(&function_name));

            // Verify the metrics
            assert_eq!(metric_configs.len(), 1);

            // Check the metric name follows expected format
            let metric_config_name = get_evaluator_metric_name(eval_name, "llm_judge_float");
            assert_eq!(
                metric_config_name,
                "tensorzero::eval_name::test_eval::evaluator_name::llm_judge_float"
            );
            assert!(metric_configs.contains_key(&metric_config_name));

            // Verify all properties of the metric config
            let metric_config = metric_configs.get(&metric_config_name).unwrap();
            assert_eq!(metric_config.r#type, MetricConfigType::Float);
            assert_eq!(metric_config.optimize, MetricConfigOptimize::Max);
            assert_eq!(metric_config.level, MetricConfigLevel::Inference);

            // Verify the type conversion from LLMJudgeOutputType to MetricConfigType
            let llm_judge_eval = match config.evaluators.get("llm_judge_float").unwrap() {
                EvaluatorConfig::LLMJudge(config) => config,
                _ => panic!("Expected LLMJudge evaluator"),
            };
            assert_eq!(
                MetricConfigType::from(llm_judge_eval.output_type),
                metric_config.r#type
            );

            // Verify the optimize conversion from LLMJudgeOptimize to MetricConfigOptimize
            assert_eq!(
                MetricConfigOptimize::from(llm_judge_eval.optimize),
                metric_config.optimize
            );
        }

        // Test case 3: Error when function doesn't exist
        {
            let mut evaluators = HashMap::new();
            evaluators.insert(
                "em_evaluator".to_string(),
                UninitializedEvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }),
            );

            let uninitialized_config = UninitializedEvalConfig {
                evaluators,
                dataset_name: "test_dataset".to_string(),
                function_name: "nonexistent_function".to_string(),
            };

            let result = uninitialized_config.load(&functions, &base_path, eval_name);
            assert!(result.is_err());
            assert!(matches!(
                *result.unwrap_err().get_details(),
                ErrorDetails::Config { .. }
            ));
        }

        // Test case 4: Error when eval name contains "::"
        {
            let mut evaluators = HashMap::new();
            evaluators.insert(
                "em_evaluator".to_string(),
                UninitializedEvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }),
            );

            let uninitialized_config = UninitializedEvalConfig {
                evaluators,
                dataset_name: "test_dataset".to_string(),
                function_name: function_name.to_string(),
            };

            let result = uninitialized_config.load(&functions, &base_path, "invalid::eval::name");
            assert!(result.is_err());
            assert!(matches!(
                *result.unwrap_err().get_details(),
                ErrorDetails::Config { .. }
            ));
        }

        // Test case 5: Error when multiple variants are active in LLM judge
        {
            let mut test_variant1 = HashMap::new();
            test_variant1.insert(
                "test_variant1".to_string(),
                UninitializedLLMJudgeVariantConfig::ChatCompletion(
                    UninitializedLLMJudgeChatCompletionVariantConfig {
                        active: true,
                        model: Arc::from("gpt-3.5-turbo"),
                        system_instructions: PathBuf::from(
                            "evals/eval1/llm_judge_bool/system_instructions.txt",
                        ),
                        temperature: Some(0.7),
                        top_p: None,
                        max_tokens: Some(100),
                        presence_penalty: None,
                        frequency_penalty: None,
                        seed: None,
                        json_mode: JsonMode::ImplicitTool,
                        retries: RetryConfig::default(),
                        extra_body: None,
                    },
                ),
            );

            let mut test_variant2 = HashMap::new();
            test_variant2.insert(
                "test_variant2".to_string(),
                UninitializedLLMJudgeVariantConfig::ChatCompletion(
                    UninitializedLLMJudgeChatCompletionVariantConfig {
                        active: true,
                        model: Arc::from("gpt-4"),
                        system_instructions: PathBuf::from(
                            "evals/eval1/llm_judge_bool/system_instructions.txt",
                        ),
                        temperature: Some(0.5),
                        top_p: None,
                        max_tokens: Some(200),
                        presence_penalty: None,
                        frequency_penalty: None,
                        seed: None,
                        json_mode: JsonMode::ImplicitTool,
                        retries: RetryConfig::default(),
                        extra_body: None,
                    },
                ),
            );

            // Combine the two variants
            let mut variants = HashMap::new();
            for (k, v) in test_variant1 {
                variants.insert(k, v);
            }
            for (k, v) in test_variant2 {
                variants.insert(k, v);
            }

            let llm_judge_config = UninitializedLLMJudgeConfig {
                variants,
                output_type: LLMJudgeOutputType::Boolean,
                optimize: LLMJudgeOptimize::Min,
                include: LLMJudgeIncludeConfig {
                    reference_output: false,
                },
                cutoff: Some(0.3),
            };

            let mut evaluators = HashMap::new();
            evaluators.insert(
                "multiple_active_variants".to_string(),
                UninitializedEvaluatorConfig::LLMJudge(llm_judge_config),
            );

            let uninitialized_config = UninitializedEvalConfig {
                evaluators,
                dataset_name: "test_dataset".to_string(),
                function_name: function_name.to_string(),
            };

            let result = uninitialized_config.load(&functions, &base_path, eval_name);
            assert!(result.is_err());
            assert_eq!(
                *result.unwrap_err().get_details(),
                ErrorDetails::Config {
                    message: "Evaluator `multiple_active_variants` in `[evals.test_eval]` must have exactly 1 variant that is active. Found 2 variants with nonzero weights.".to_string(),
                }
            );
        }

        // Test case 6: Error when evaluator name contains "::"
        {
            let base_path = PathBuf::from(".");
            let eval_name = "test_eval";
            let function_name = "test_function";

            let mut functions = HashMap::new();
            functions.insert(
                function_name.to_string(),
                Arc::new(FunctionConfig::Json(FunctionConfigJson {
                    variants: HashMap::new(),
                    output_schema: create_test_schema(),
                    system_schema: None,
                    user_schema: None,
                    assistant_schema: None,
                    implicit_tool_call_config: create_implicit_tool_call_config(
                        create_test_schema(),
                    ),
                })),
            );

            let mut evaluators = HashMap::new();
            evaluators.insert(
                "foo::invalid_name".to_string(),
                UninitializedEvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }),
            );

            let uninitialized_config = UninitializedEvalConfig {
                evaluators,
                dataset_name: "test_dataset".to_string(),
                function_name: function_name.to_string(),
            };

            let result = uninitialized_config.load(&functions, &base_path, eval_name);
            assert!(result.is_err());
            assert_eq!(
                *result.unwrap_err().get_details(),
                ErrorDetails::Config {
                    message:
                        "Evaluator names cannot contain \"::\" (referenced in `[evals.test_eval.foo::invalid_name]`)"
                            .to_string(),
                }
            );
        }

        // Test case 7: Successful loading with LLM judge evaluator with reference_output = true
        {
            let mut variants = HashMap::new();
            variants.insert(
                "test_variant".to_string(),
                UninitializedLLMJudgeVariantConfig::ChatCompletion(
                    UninitializedLLMJudgeChatCompletionVariantConfig {
                        active: true,
                        model: Arc::from("gpt-3.5-turbo"),
                        system_instructions: PathBuf::from(
                            "evals/eval1/llm_judge_bool/system_instructions.txt",
                        ),
                        temperature: Some(0.7),
                        top_p: None,
                        max_tokens: Some(100),
                        presence_penalty: None,
                        frequency_penalty: None,
                        seed: None,
                        json_mode: JsonMode::ImplicitTool,
                        retries: RetryConfig::default(),
                        extra_body: None,
                    },
                ),
            );

            let llm_judge_config = UninitializedLLMJudgeConfig {
                variants,
                output_type: LLMJudgeOutputType::Boolean,
                optimize: LLMJudgeOptimize::Min,
                include: LLMJudgeIncludeConfig {
                    reference_output: true,
                },
                cutoff: None,
            };

            let mut evaluators = HashMap::new();
            evaluators.insert(
                "llm_judge_with_ref".to_string(),
                UninitializedEvaluatorConfig::LLMJudge(llm_judge_config),
            );

            let uninitialized_config = UninitializedEvalConfig {
                evaluators,
                dataset_name: "test_dataset".to_string(),
                function_name: function_name.to_string(),
            };

            let result = uninitialized_config.load(&functions, &base_path, eval_name);
            assert!(result.is_ok());

            let (config, _additional_functions, _metric_configs) = result.unwrap();

            // Verify LLM judge evaluator config with reference_output = true
            match config.evaluators.get("llm_judge_with_ref").unwrap() {
                EvaluatorConfig::LLMJudge(judge_config) => {
                    assert!(matches!(
                        judge_config.output_type,
                        LLMJudgeOutputType::Boolean
                    ));
                    assert!(matches!(judge_config.optimize, LLMJudgeOptimize::Min));
                    assert!(judge_config.include.reference_output);
                }
                _ => panic!("Expected LLMJudge evaluator config"),
            }
        }
    }

    // Helper functions for tests
    fn create_test_schema() -> JSONSchemaFromPath {
        let schema_value = serde_json::json!({
            "type": "object",
            "properties": {
                "result": {
                    "type": "string"
                }
            },
            "required": ["result"]
        });
        JSONSchemaFromPath::from_value(&schema_value).unwrap()
    }
}
