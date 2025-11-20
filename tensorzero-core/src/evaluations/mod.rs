use std::collections::HashSet;
use std::{collections::HashMap, sync::Arc};

use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::{Deserialize, Serialize};
use tensorzero_derive::TensorZeroDeserialize;

use crate::config::{ErrorContext, LoadableConfig, UninitializedSchemas};
use crate::experimentation::ExperimentationConfig;
use crate::utils::retries::RetryConfig;
use crate::variant::chat_completion::UninitializedChatCompletionConfig;
use crate::variant::Variant;
use crate::{
    config::{
        path::ResolvedTomlPathData, MetricConfig, MetricConfigLevel, MetricConfigOptimize,
        MetricConfigType, PathWithContents, SchemaData, TimeoutsConfig,
    },
    error::{Error, ErrorDetails},
    function::{FunctionConfig, FunctionConfigJson},
    inference::types::{
        chat_completion_inference_params::ServiceTier, extra_body::ExtraBodyConfig,
        extra_headers::ExtraHeadersConfig,
    },
    jsonschema_util::StaticJSONSchema,
    tool::create_json_mode_tool_call_config,
    variant::{
        best_of_n_sampling::{
            UninitializedBestOfNEvaluatorConfig, UninitializedBestOfNSamplingConfig,
        },
        chain_of_thought::ChainOfThoughtConfig,
        chat_completion::ChatCompletionConfig,
        dicl::UninitializedDiclConfig,
        mixture_of_n::{UninitializedFuserConfig, UninitializedMixtureOfNConfig},
        JsonMode, VariantConfig, VariantInfo,
    },
};

pub const LLM_JUDGE_USER_SCHEMA_TEXT: &str = include_str!("llm_judge_user_schema.json");
pub const LLM_JUDGE_FLOAT_OUTPUT_SCHEMA_TEXT: &str =
    include_str!("llm_judge_float_output_schema.json");
pub const LLM_JUDGE_BOOLEAN_OUTPUT_SCHEMA_TEXT: &str =
    include_str!("llm_judge_boolean_output_schema.json");

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct InferenceEvaluationConfig {
    pub evaluators: HashMap<String, EvaluatorConfig>,
    pub function_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Deprecated: Use `InferenceEvaluationConfig` instead
pub type StaticEvaluationConfig = InferenceEvaluationConfig;

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvaluationConfig {
    #[serde(alias = "static")]
    Inference(InferenceEvaluationConfig),
}

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvaluatorConfig {
    ExactMatch(ExactMatchConfig),
    #[serde(rename = "llm_judge")]
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

    /// Returns true if this evaluator produces Bernoulli (boolean) outputs
    pub fn is_bernoulli(&self) -> bool {
        match self {
            EvaluatorConfig::ExactMatch(_) => true,
            EvaluatorConfig::LLMJudge(config) => {
                matches!(config.output_type, LLMJudgeOutputType::Boolean)
            }
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
#[serde(deny_unknown_fields)]
pub struct ExactMatchConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cutoff: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
#[serde(deny_unknown_fields)]
pub struct LLMJudgeConfig {
    pub input_format: LLMJudgeInputFormat,
    pub output_type: LLMJudgeOutputType,
    pub include: LLMJudgeIncludeConfig,
    pub optimize: LLMJudgeOptimize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cutoff: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct LLMJudgeIncludeConfig {
    #[serde(default)]
    pub reference_output: bool,
}

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum LLMJudgeInputFormat {
    #[default]
    Serialized,
    Messages,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
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

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
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

pub fn get_llm_judge_function_name(evaluation_name: &str, evaluator_name: &str) -> String {
    format!("tensorzero::llm_judge::{evaluation_name}::{evaluator_name}")
}

pub fn get_evaluator_metric_name(evaluation_name: &str, evaluator_name: &str) -> String {
    format!("tensorzero::evaluation_name::{evaluation_name}::evaluator_name::{evaluator_name}")
}

#[derive(Debug)]
pub enum UninitializedEvaluationConfig {
    Inference(UninitializedInferenceEvaluationConfig),
}

impl UninitializedEvaluationConfig {
    pub fn load(
        self,
        functions: &HashMap<String, Arc<FunctionConfig>>,

        evaluation_name: &str,
    ) -> EvaluationLoadResult {
        match self {
            UninitializedEvaluationConfig::Inference(config) => {
                config.load(functions, evaluation_name)
            }
        }
    }
}

// Custom deserializer to log deprecation warning when "static" is used
impl<'de> Deserialize<'de> for UninitializedEvaluationConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct EvaluationConfigVisitor;

        impl<'de> Visitor<'de> for EvaluationConfigVisitor {
            type Value = UninitializedEvaluationConfig;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct UninitializedEvaluationConfig")
            }

            fn visit_map<V>(self, mut map: V) -> Result<UninitializedEvaluationConfig, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut type_value: Option<String> = None;
                let mut other_fields = serde_json::Map::new();

                while let Some(key) = map.next_key::<String>()? {
                    if key == "type" {
                        type_value = Some(map.next_value()?);
                    } else {
                        let value: serde_json::Value = map.next_value()?;
                        other_fields.insert(key, value);
                    }
                }

                let type_str = type_value.ok_or_else(|| de::Error::missing_field("type"))?;

                // Log deprecation warning if "static" is used
                if type_str == "static" {
                    crate::utils::deprecation_warning(
                        "The evaluation type 'static' is deprecated. Please use 'inference' instead. Support for 'static' will be removed in a future version."
                    );
                }

                // Validate type
                if type_str != "inference" && type_str != "static" {
                    return Err(de::Error::unknown_variant(&type_str, &["inference"]));
                }

                // Deserialize the config
                let config: UninitializedInferenceEvaluationConfig =
                    serde_json::from_value(serde_json::Value::Object(other_fields))
                        .map_err(de::Error::custom)?;

                Ok(UninitializedEvaluationConfig::Inference(config))
            }
        }

        deserializer.deserialize_struct(
            "UninitializedEvaluationConfig",
            &["type"],
            EvaluationConfigVisitor,
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedInferenceEvaluationConfig {
    evaluators: HashMap<String, UninitializedEvaluatorConfig>,
    function_name: String,
    #[serde(default)]
    description: Option<String>,
}

/// Deprecated: Use `UninitializedInferenceEvaluationConfig` instead
pub type UninitializedStaticEvaluationConfig = UninitializedInferenceEvaluationConfig;

type EvaluationLoadResult = Result<
    (
        InferenceEvaluationConfig,            // The evaluation itself
        HashMap<String, Arc<FunctionConfig>>, // All functions which the evaluation needs {function_name -> function_config}
        HashMap<String, MetricConfig>, // All metrics which the evaluation needs {metric_name -> metric_config}
    ),
    Error,
>;

impl UninitializedInferenceEvaluationConfig {
    pub fn load(
        self,
        functions: &HashMap<String, Arc<FunctionConfig>>,
        evaluation_name: &str,
    ) -> EvaluationLoadResult {
        if !functions.contains_key(&self.function_name) {
            return Err(ErrorDetails::Config {
                message: format!(
                    "Function `{}` not found (referenced in `[evaluations.{evaluation_name}]`)",
                    self.function_name
                ),
            }
            .into());
        }

        // evaluation names cannot have "::" in them since we use it as a delimiter
        if evaluation_name.contains("::") {
            return Err(ErrorDetails::Config {
                message: format!(
                    "evaluation names cannot contain \"::\" (referenced in `[evaluations.{evaluation_name}]`)"
                ),
            }
            .into());
        }
        let evaluator_results = self
            .evaluators
            .into_iter()
            .map(|(name, config)| {
                config.load(evaluation_name, &name).map(
                    |(evaluation_config, func_config, metric_config)| {
                        (name, evaluation_config, func_config, metric_config)
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
                    get_llm_judge_function_name(evaluation_name, &evaluator_name),
                    Arc::new(config),
                );
            }

            // Add to metric_configs map
            metric_configs.insert(
                get_evaluator_metric_name(evaluation_name, &evaluator_name),
                metric_config,
            );
        }
        Ok((
            InferenceEvaluationConfig {
                evaluators,
                function_name: self.function_name,
                description: self.description,
            },
            function_configs,
            metric_configs,
        ))
    }
}

#[derive(Debug, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum UninitializedEvaluatorConfig {
    ExactMatch(ExactMatchConfig),
    #[serde(rename = "llm_judge")]
    LLMJudge(UninitializedLLMJudgeConfig),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UninitializedLLMJudgeConfig {
    #[serde(default)]
    input_format: LLMJudgeInputFormat,
    variants: HashMap<String, UninitializedLLMJudgeVariantInfo>,
    output_type: LLMJudgeOutputType,
    optimize: LLMJudgeOptimize,
    #[serde(default)]
    include: LLMJudgeIncludeConfig,
    #[serde(default)]
    cutoff: Option<f32>,
    #[serde(default)]
    description: Option<String>,
}

impl UninitializedEvaluatorConfig {
    pub fn load(
        self,
        evaluation_name: &str,
        evaluator_name: &str,
    ) -> Result<(EvaluatorConfig, Option<FunctionConfig>, MetricConfig), Error> {
        // Evaluator names cannot have "::" in them since we use it as a delimiter in our function names later on
        if evaluator_name.contains("::") {
            return Err(ErrorDetails::Config {
                message: format!(
                    "Evaluator names cannot contain \"::\" (referenced in `[evaluations.{evaluation_name}.{evaluator_name}]`)"
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
                let user_schema_value: Option<serde_json::Value> = match params.input_format {
                    LLMJudgeInputFormat::Serialized => Some(serde_json::from_str(LLM_JUDGE_USER_SCHEMA_TEXT)
                        .map_err(|e| {
                            Error::new(ErrorDetails::JsonSchema {
                                message: format!("Failed to parse LLM judge user schema: {e}. This should never happen, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports."),
                            })
                        })?),
                    LLMJudgeInputFormat::Messages => None,
                };
                let user_schema = user_schema_value
                    .map(StaticJSONSchema::from_value)
                    .transpose()?;
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
                let output_schema = StaticJSONSchema::from_value(output_schema_value)?;
                let json_mode_tool_call_config =
                    create_json_mode_tool_call_config(output_schema.clone());

                let mut variants = params
                    .variants
                    .into_iter()
                    .map(|(name, variant)| {
                        variant
                            .load(
                                evaluation_name,
                                evaluator_name,
                                &params.input_format,
                                &name,
                                user_schema.clone(),
                            )
                            .map(|v| (name, v))
                    })
                    .collect::<Result<HashMap<_, _>, Error>>()?;
                let nonzero_weights = variants
                    .iter()
                    // Treat a None weight as 0.0 for this check - we only care if we have multiple variants with an explicit positive weight
                    .filter(|(_, variant)| variant.inner.weight().unwrap_or(0.0) > 0.0)
                    .count();
                if nonzero_weights != 1 && variants.len() > 1 {
                    return Err(ErrorDetails::Config {
                        message: format!(
                            "Evaluator `{evaluator_name}` in `[evaluations.{evaluation_name}]` must have exactly 1 variant that is active. Found {nonzero_weights} variants with nonzero weights."
                        ),
                    }
                    .into());
                } else if variants.len() == 1 {
                    // If there is only one variant, it should have weight 1.0
                    let Some((_, variant)) = variants.iter_mut().next() else {
                        return Err(ErrorDetails::Config {
                            message: "Failed to grab first variant from variants map. This should never happen, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.".to_string(),
                        }.into());
                    };
                    if let Some(weight) = variant.inner.weight() {
                        if weight == 0.0 {
                            return Err(ErrorDetails::Config {
                                message: format!("Evaluator `{evaluator_name}` in `[evaluations.{evaluation_name}]` must have exactly 1 variant that is active. You have specified a single inactive variant."),
                            }
                            .into());
                        }
                    }
                    match &mut variant.inner {
                        VariantConfig::ChatCompletion(variant) => {
                            variant.set_weight(Some(1.0));
                        }
                        VariantConfig::BestOfNSampling(variant) => {
                            variant.set_weight(Some(1.0));
                        }
                        VariantConfig::MixtureOfN(variant) => {
                            variant.set_weight(Some(1.0));
                        }
                        VariantConfig::Dicl(variant) => {
                            variant.set_weight(Some(1.0));
                        }
                        VariantConfig::ChainOfThought(variant) => {
                            variant.inner.set_weight(Some(1.0));
                        }
                    };
                }
                let variants: HashMap<_, _> = variants
                    .into_iter()
                    .map(|(name, variant)| (name, Arc::new(variant)))
                    .collect();
                let all_template_names: HashSet<String> = variants
                    .values()
                    .flat_map(|v| v.get_all_explicit_template_names())
                    .collect();
                let experimentation = ExperimentationConfig::legacy_from_variants_map(&variants);
                let function_config = FunctionConfig::Json(FunctionConfigJson {
                    variants,
                    schemas: SchemaData::load(
                        user_schema,
                        None,
                        None,
                        UninitializedSchemas::default(),
                        &format!("tensorzero::evaluator::{evaluator_name}"),
                    )?,
                    output_schema,
                    json_mode_tool_call_config,
                    description: None,
                    all_explicit_template_names: all_template_names,
                    experimentation,
                });
                Ok((
                    EvaluatorConfig::LLMJudge(LLMJudgeConfig {
                        input_format: params.input_format,
                        output_type: params.output_type,
                        include: params.include,
                        optimize: params.optimize,
                        cutoff: params.cutoff,
                        description: params.description,
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
struct UninitializedLLMJudgeVariantInfo {
    #[serde(flatten)]
    inner: UninitializedLLMJudgeVariantConfig,
    timeouts: Option<TimeoutsConfig>,
}

#[derive(Debug, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum UninitializedLLMJudgeVariantConfig {
    ChatCompletion(UninitializedLLMJudgeChatCompletionVariantConfig),
    #[serde(rename = "experimental_best_of_n_sampling")]
    BestOfNSampling(UninitializedLLMJudgeBestOfNVariantConfig),
    #[serde(rename = "experimental_mixture_of_n")]
    MixtureOfNSampling(UninitializedLLMJudgeMixtureOfNVariantConfig),
    #[serde(rename = "experimental_dynamic_in_context_learning")]
    Dicl(UninitializedLLMJudgeDiclVariantConfig),
    #[serde(rename = "experimental_chain_of_thought")]
    ChainOfThought(UninitializedLLMJudgeChainOfThoughtVariantConfig),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UninitializedLLMJudgeChatCompletionVariantConfig {
    #[serde(default)]
    active: Option<bool>,
    model: Arc<str>,
    system_instructions: ResolvedTomlPathData,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u32>,
    json_mode: JsonMode, // This is a JSON function
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verbosity: Option<String>,
    #[serde(default)]
    retries: RetryConfig,
    #[serde(default)]
    extra_body: Option<ExtraBodyConfig>,
    #[serde(default)]
    extra_headers: Option<ExtraHeadersConfig>,
}

/// Converts a chat completion judge variant config to a chat completion config.
/// This is factored out so that both the chain of thought and chat completion judges
/// can use the same implementation.
fn convert_chat_completion_judge_to_variant(
    evaluation_name: &str,
    evaluator_name: &str,
    variant_name: &str,
    input_format: &LLMJudgeInputFormat,
    params: UninitializedLLMJudgeChatCompletionVariantConfig,
    user_schema: Option<StaticJSONSchema>,
) -> Result<ChatCompletionConfig, Error> {
    let system_instructions = params.system_instructions.data();
    let templated_system_instructions = format!(
        include_str!("llm_judge_system_instructions.txt"),
        system_instructions = system_instructions,
    );
    let system_template_path = get_template_path(
        evaluation_name,
        evaluator_name,
        variant_name,
        "system",
        templated_system_instructions,
    );
    let system_template = PathWithContents::from_path(system_template_path)?;
    let user_template = match input_format {
        LLMJudgeInputFormat::Serialized => Some(PathWithContents::from_path(get_template_path(
            evaluation_name,
            evaluator_name,
            variant_name,
            "user",
            include_str!("llm_judge_user_template.minijinja").to_string(),
        ))?),
        LLMJudgeInputFormat::Messages => None,
    };
    UninitializedChatCompletionConfig {
        assistant_template: None,
        extra_body: params.extra_body,
        extra_headers: params.extra_headers,
        frequency_penalty: params.frequency_penalty,
        input_wrappers: None,
        json_mode: Some(params.json_mode),
        max_tokens: params.max_tokens,
        model: params.model,
        presence_penalty: params.presence_penalty,
        reasoning_effort: params.reasoning_effort,
        retries: params.retries,
        seed: params.seed,
        service_tier: params.service_tier,
        stop_sequences: params.stop_sequences,
        system_template: Some(system_template.path),
        temperature: params.temperature,
        templates: Default::default(),
        thinking_budget_tokens: params.thinking_budget_tokens,
        top_p: params.top_p,
        user_template: user_template.map(|t| t.path),
        verbosity: params.verbosity,
        weight: get_weight(params.active),
    }
    .load(
        &SchemaData::load(
            user_schema,
            None,
            None,
            UninitializedSchemas::default(),
            &format!("tensorzero::evaluator::{evaluator_name}"),
        )?,
        &ErrorContext {
            function_name: "tensorzero::evaluator".to_string(),
            variant_name: evaluator_name.to_string(),
        },
    )
}

fn default_timeout() -> f64 {
    300.0
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UninitializedLLMJudgeBestOfNVariantConfig {
    #[serde(default)]
    active: Option<bool>,
    #[serde(default = "default_timeout")]
    timeout_s: f64,
    #[serde(default)]
    candidates: Vec<String>,
    evaluator: UninitializedLLMJudgeChatCompletionVariantConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UninitializedLLMJudgeMixtureOfNVariantConfig {
    #[serde(default)]
    active: Option<bool>,
    #[serde(default = "default_timeout")]
    timeout_s: f64,
    #[serde(default)]
    candidates: Vec<String>,
    fuser: UninitializedLLMJudgeChatCompletionVariantConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UninitializedLLMJudgeDiclVariantConfig {
    #[serde(default)]
    active: Option<bool>,
    embedding_model: String,
    k: u32, // k as in k-nearest neighbors
    model: String,
    system_instructions: Option<ResolvedTomlPathData>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    max_tokens: Option<u32>,
    seed: Option<u32>,
    json_mode: Option<JsonMode>,
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    extra_body: Option<ExtraBodyConfig>,
    #[serde(default)]
    retries: RetryConfig,
    #[serde(default)]
    extra_headers: Option<ExtraHeadersConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UninitializedLLMJudgeChainOfThoughtVariantConfig {
    #[serde(flatten)]
    inner: UninitializedLLMJudgeChatCompletionVariantConfig,
}

fn get_template_path(
    evaluation_name: &str,
    evaluator_name: &str,
    variant_name: &str,
    template_name: &str,
    data: String,
) -> ResolvedTomlPathData {
    ResolvedTomlPathData::new_fake_path(format!(
        "tensorzero::llm_judge::{evaluation_name}::{evaluator_name}::{variant_name}::{template_name}"
    ), data)
}

fn get_weight(active: Option<bool>) -> Option<f64> {
    match active {
        Some(active) => {
            if active {
                Some(1.0)
            } else {
                Some(0.0)
            }
        }
        None => None,
    }
}

impl UninitializedLLMJudgeVariantInfo {
    pub fn load(
        self,
        evaluation_name: &str,
        evaluator_name: &str,
        input_format: &LLMJudgeInputFormat,
        variant_name: &str,
        user_schema: Option<StaticJSONSchema>,
    ) -> Result<VariantInfo, Error> {
        let inner = match self.inner {
            UninitializedLLMJudgeVariantConfig::ChatCompletion(params) => {
                VariantConfig::ChatCompletion(convert_chat_completion_judge_to_variant(
                    evaluation_name,
                    evaluator_name,
                    variant_name,
                    input_format,
                    params,
                    user_schema,
                )?)
            }
            UninitializedLLMJudgeVariantConfig::BestOfNSampling(params) => {
                let evaluator_system_instructions = params.evaluator.system_instructions.data();
                let templated_evaluator_system_instructions = format!(
                    include_str!("llm_judge_system_instructions.txt"),
                    system_instructions = evaluator_system_instructions,
                );
                let evaluator_system_template = PathWithContents::from_path(get_template_path(
                    evaluation_name,
                    evaluator_name,
                    variant_name,
                    "system",
                    templated_evaluator_system_instructions,
                ))?;
                let evaluator_user_template = match input_format {
                    LLMJudgeInputFormat::Serialized => {
                        Some(PathWithContents::from_path(get_template_path(
                            evaluation_name,
                            evaluator_name,
                            variant_name,
                            "user",
                            include_str!("llm_judge_user_template.minijinja").to_string(),
                        ))?)
                    }
                    LLMJudgeInputFormat::Messages => None,
                };
                VariantConfig::BestOfNSampling(
                    UninitializedBestOfNSamplingConfig {
                        weight: get_weight(params.active),
                        timeout_s: params.timeout_s,
                        candidates: params.candidates,
                        evaluator: UninitializedBestOfNEvaluatorConfig {
                            inner: UninitializedChatCompletionConfig {
                                assistant_template: None,
                                extra_body: params.evaluator.extra_body,
                                extra_headers: params.evaluator.extra_headers,
                                frequency_penalty: params.evaluator.frequency_penalty,
                                input_wrappers: None,
                                json_mode: Some(params.evaluator.json_mode),
                                max_tokens: params.evaluator.max_tokens,
                                model: params.evaluator.model,
                                presence_penalty: params.evaluator.presence_penalty,
                                reasoning_effort: params.evaluator.reasoning_effort,
                                retries: params.evaluator.retries,
                                seed: params.evaluator.seed,
                                service_tier: params.evaluator.service_tier,
                                stop_sequences: params.evaluator.stop_sequences,
                                system_template: Some(evaluator_system_template.path),
                                temperature: params.evaluator.temperature,
                                templates: Default::default(),
                                thinking_budget_tokens: params.evaluator.thinking_budget_tokens,
                                top_p: params.evaluator.top_p,
                                user_template: evaluator_user_template.map(|t| t.path),
                                verbosity: params.evaluator.verbosity,
                                weight: None,
                            },
                        },
                    }
                    .load(
                        &SchemaData::load(
                            user_schema,
                            None,
                            None,
                            UninitializedSchemas::default(),
                            &format!("tensorzero::evaluator::{evaluator_name}"),
                        )?,
                        &ErrorContext {
                            function_name: "tensorzero::evaluator".to_string(),
                            variant_name: evaluator_name.to_string(),
                        },
                    )?,
                )
            }
            UninitializedLLMJudgeVariantConfig::MixtureOfNSampling(params) => {
                let fuser_system_instructions = params.fuser.system_instructions.data();
                let templated_fuser_system_instructions = format!(
                    include_str!("llm_judge_system_instructions.txt"),
                    system_instructions = fuser_system_instructions,
                );
                let fuser_system_template = PathWithContents::from_path(get_template_path(
                    evaluation_name,
                    evaluator_name,
                    variant_name,
                    "system",
                    templated_fuser_system_instructions,
                ))?;
                let fuser_user_template = match input_format {
                    LLMJudgeInputFormat::Serialized => {
                        Some(PathWithContents::from_path(get_template_path(
                            evaluation_name,
                            evaluator_name,
                            variant_name,
                            "user",
                            include_str!("llm_judge_user_template.minijinja").to_string(),
                        ))?)
                    }
                    LLMJudgeInputFormat::Messages => None,
                };
                VariantConfig::MixtureOfN(
                    UninitializedMixtureOfNConfig {
                        weight: get_weight(params.active),
                        timeout_s: params.timeout_s,
                        candidates: params.candidates,
                        fuser: UninitializedFuserConfig {
                            inner: UninitializedChatCompletionConfig {
                                assistant_template: None,
                                extra_body: params.fuser.extra_body,
                                extra_headers: params.fuser.extra_headers,
                                frequency_penalty: params.fuser.frequency_penalty,
                                input_wrappers: None,
                                json_mode: Some(params.fuser.json_mode),
                                max_tokens: params.fuser.max_tokens,
                                model: params.fuser.model,
                                presence_penalty: params.fuser.presence_penalty,
                                reasoning_effort: params.fuser.reasoning_effort,
                                retries: params.fuser.retries,
                                seed: params.fuser.seed,
                                service_tier: params.fuser.service_tier,
                                stop_sequences: params.fuser.stop_sequences,
                                system_template: Some(fuser_system_template.path),
                                temperature: params.fuser.temperature,
                                templates: Default::default(),
                                thinking_budget_tokens: params.fuser.thinking_budget_tokens,
                                top_p: params.fuser.top_p,
                                user_template: fuser_user_template.map(|t| t.path),
                                verbosity: params.fuser.verbosity,
                                weight: None,
                            },
                        },
                    }
                    .load(
                        &SchemaData::load(
                            user_schema,
                            None,
                            None,
                            UninitializedSchemas::default(),
                            &format!("tensorzero::evaluator::{evaluator_name}"),
                        )?,
                        &ErrorContext {
                            function_name: "tensorzero::evaluator".to_string(),
                            variant_name: evaluator_name.to_string(),
                        },
                    )?,
                )
            }
            UninitializedLLMJudgeVariantConfig::Dicl(params) => {
                let dicl_system_instructions = params
                    .system_instructions
                    .map(|si| si.data().to_string())
                    .map(|si| {
                        format!(
                            include_str!("llm_judge_system_instructions.txt"),
                            system_instructions = si,
                        )
                    });

                let uninitialized_config = UninitializedDiclConfig {
                    weight: get_weight(params.active),
                    embedding_model: params.embedding_model,
                    k: params.k,
                    model: params.model,
                    system_instructions: dicl_system_instructions.map(|s| {
                        ResolvedTomlPathData::new_fake_path("tensorzero::llm_judge".to_string(), s)
                    }),
                    temperature: params.temperature,
                    top_p: params.top_p,
                    presence_penalty: params.presence_penalty,
                    frequency_penalty: params.frequency_penalty,
                    max_tokens: params.max_tokens,
                    seed: params.seed,
                    json_mode: params.json_mode,
                    extra_body: params.extra_body,
                    extra_headers: params.extra_headers,
                    retries: params.retries,
                    stop_sequences: params.stop_sequences,
                    max_distance: None,
                    ..Default::default()
                };
                VariantConfig::Dicl(uninitialized_config.load()?)
            }
            UninitializedLLMJudgeVariantConfig::ChainOfThought(params) => {
                VariantConfig::ChainOfThought(ChainOfThoughtConfig {
                    inner: convert_chat_completion_judge_to_variant(
                        evaluation_name,
                        evaluator_name,
                        variant_name,
                        input_format,
                        params.inner,
                        user_schema,
                    )?,
                })
            }
        };
        Ok(VariantInfo {
            inner,
            timeouts: self.timeouts.unwrap_or_default(),
        })
    }
}

/// NOTE: this function should not be called.
/// In the code we already have a conversion from UninitializedLLMJudgeVariantConfig to VariantConfig.
/// We want to make sure that there is an UninitializedLLMJudgeVariantConfig for each VariantConfig.
/// This function should complain at compile time if we forget to update it when adding a new variant type.
#[expect(dead_code)]
#[expect(clippy::unnecessary_wraps)]
fn check_convert_variant_to_llm_judge_variant(
    variant: VariantConfig,
) -> Result<UninitializedLLMJudgeVariantConfig, Error> {
    match variant {
        VariantConfig::ChatCompletion(variant) => {
            Ok(UninitializedLLMJudgeVariantConfig::ChatCompletion(
                UninitializedLLMJudgeChatCompletionVariantConfig {
                    active: Some(false),
                    model: variant.model().clone(),
                    system_instructions: ResolvedTomlPathData::new_fake_path(
                        String::new(),
                        String::new(),
                    ),
                    temperature: variant.temperature(),
                    top_p: variant.top_p(),
                    max_tokens: variant.max_tokens(),
                    presence_penalty: variant.presence_penalty(),
                    frequency_penalty: variant.frequency_penalty(),
                    seed: variant.seed(),
                    json_mode: JsonMode::Off,
                    retries: *variant.retries(),
                    stop_sequences: variant.stop_sequences().cloned(),
                    extra_body: variant.extra_body().cloned(),
                    extra_headers: variant.extra_headers().cloned(),
                    reasoning_effort: variant.reasoning_effort().cloned(),
                    service_tier: variant.service_tier().cloned(),
                    thinking_budget_tokens: variant.thinking_budget_tokens(),
                    verbosity: variant.verbosity().cloned(),
                },
            ))
        }
        VariantConfig::BestOfNSampling(variant) => {
            Ok(UninitializedLLMJudgeVariantConfig::BestOfNSampling(
                UninitializedLLMJudgeBestOfNVariantConfig {
                    active: Some(false),
                    timeout_s: variant.timeout_s(),
                    candidates: variant.candidates().clone(),
                    evaluator: UninitializedLLMJudgeChatCompletionVariantConfig {
                        active: Some(false),
                        model: variant.evaluator().inner.model().clone(),
                        system_instructions: ResolvedTomlPathData::new_fake_path(
                            String::new(),
                            String::new(),
                        ),
                        temperature: variant.evaluator().inner.temperature(),
                        top_p: variant.evaluator().inner.top_p(),
                        max_tokens: variant.evaluator().inner.max_tokens(),
                        presence_penalty: variant.evaluator().inner.presence_penalty(),
                        frequency_penalty: variant.evaluator().inner.frequency_penalty(),
                        seed: variant.evaluator().inner.seed(),
                        json_mode: JsonMode::Off,
                        retries: *variant.evaluator().inner.retries(),
                        stop_sequences: variant.evaluator().inner.stop_sequences().cloned(),
                        extra_body: variant.evaluator().inner.extra_body().cloned(),
                        extra_headers: variant.evaluator().inner.extra_headers().cloned(),
                        reasoning_effort: variant.evaluator().inner.reasoning_effort().cloned(),
                        service_tier: variant.evaluator().inner.service_tier().cloned(),
                        thinking_budget_tokens: variant.evaluator().inner.thinking_budget_tokens(),
                        verbosity: variant.evaluator().inner.verbosity().cloned(),
                    },
                },
            ))
        }
        VariantConfig::MixtureOfN(variant) => {
            Ok(UninitializedLLMJudgeVariantConfig::MixtureOfNSampling(
                UninitializedLLMJudgeMixtureOfNVariantConfig {
                    active: Some(false),
                    timeout_s: variant.timeout_s(),
                    candidates: variant.candidates().clone(),
                    fuser: UninitializedLLMJudgeChatCompletionVariantConfig {
                        active: Some(false),
                        model: variant.fuser().inner.model().clone(),
                        system_instructions: ResolvedTomlPathData::new_fake_path(
                            String::new(),
                            String::new(),
                        ),
                        temperature: variant.fuser().inner.temperature(),
                        top_p: variant.fuser().inner.top_p(),
                        max_tokens: variant.fuser().inner.max_tokens(),
                        presence_penalty: variant.fuser().inner.presence_penalty(),
                        frequency_penalty: variant.fuser().inner.frequency_penalty(),
                        seed: variant.fuser().inner.seed(),
                        json_mode: JsonMode::Off,
                        retries: *variant.fuser().inner.retries(),
                        stop_sequences: variant.fuser().inner.stop_sequences().cloned(),
                        extra_body: variant.fuser().inner.extra_body().cloned(),
                        extra_headers: variant.fuser().inner.extra_headers().cloned(),
                        reasoning_effort: variant
                            .fuser()
                            .inner
                            .inference_params_v2
                            .reasoning_effort
                            .clone(),
                        service_tier: variant
                            .fuser()
                            .inner
                            .inference_params_v2
                            .service_tier
                            .clone(),
                        thinking_budget_tokens: variant
                            .fuser()
                            .inner
                            .inference_params_v2
                            .thinking_budget_tokens,
                        verbosity: variant.fuser().inner.inference_params_v2.verbosity.clone(),
                    },
                },
            ))
        }
        VariantConfig::Dicl(variant) => Ok(UninitializedLLMJudgeVariantConfig::Dicl(
            UninitializedLLMJudgeDiclVariantConfig {
                active: Some(false),
                embedding_model: variant.embedding_model().to_string(),
                k: variant.k(),
                model: variant.model().to_string(),
                system_instructions: None,
                temperature: variant.temperature(),
                top_p: variant.top_p(),
                presence_penalty: variant.presence_penalty(),
                frequency_penalty: variant.frequency_penalty(),
                max_tokens: variant.max_tokens(),
                seed: variant.seed(),
                json_mode: variant.json_mode().cloned(),
                extra_body: variant.extra_body().cloned(),
                extra_headers: variant.extra_headers().cloned(),
                retries: *variant.retries(),
                stop_sequences: variant.stop_sequences().cloned(),
            },
        )),
        VariantConfig::ChainOfThought(variant) => {
            Ok(UninitializedLLMJudgeVariantConfig::ChainOfThought(
                UninitializedLLMJudgeChainOfThoughtVariantConfig {
                    inner: UninitializedLLMJudgeChatCompletionVariantConfig {
                        active: Some(false),
                        model: variant.inner.model().to_string().into(),
                        system_instructions: ResolvedTomlPathData::new_fake_path(
                            String::new(),
                            String::new(),
                        ),
                        temperature: variant.inner.temperature(),
                        top_p: variant.inner.top_p(),
                        max_tokens: variant.inner.max_tokens(),
                        presence_penalty: variant.inner.presence_penalty(),
                        frequency_penalty: variant.inner.frequency_penalty(),
                        seed: variant.inner.seed(),
                        json_mode: JsonMode::Off,
                        retries: *variant.inner.retries(),
                        stop_sequences: variant.inner.stop_sequences().cloned(),
                        extra_body: variant.inner.extra_body().cloned(),
                        extra_headers: variant.inner.extra_headers().cloned(),
                        reasoning_effort: variant
                            .inner
                            .inference_params_v2
                            .reasoning_effort
                            .clone(),
                        service_tier: variant.inner.inference_params_v2.service_tier.clone(),
                        thinking_budget_tokens: variant
                            .inner
                            .inference_params_v2
                            .thinking_budget_tokens,
                        verbosity: variant.inner.inference_params_v2.verbosity.clone(),
                    },
                },
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    #[test]
    fn test_uninitialized_evaluation_config_load() {
        // Setup test fixtures
        let evaluation_name = "test_evaluation";

        // Prepare function configs map with a function referenced in the evaluation
        let mut functions = HashMap::new();
        let function_name = "generate_draft";
        let function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            output_schema: create_test_schema(),
            json_mode_tool_call_config: create_json_mode_tool_call_config(create_test_schema()),
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::legacy_from_variants_map(&HashMap::new()),
        });
        functions.insert(function_name.to_string(), Arc::new(function_config));

        // Test case 1: Successful loading with exact match evaluator
        {
            let mut evaluators = HashMap::new();
            evaluators.insert(
                "em_evaluator".to_string(),
                UninitializedEvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: Some(0.4) }),
            );

            let uninitialized_config = UninitializedInferenceEvaluationConfig {
                evaluators,
                function_name: function_name.to_string(),
                description: Some("evaluation description".to_string()),
            };

            let result = uninitialized_config.load(&functions, evaluation_name);
            assert!(result.is_ok());

            let (config, additional_functions, metric_configs) = result.unwrap();
            assert_eq!(config.function_name, function_name);
            assert_eq!(
                config.description.as_deref(),
                Some("evaluation description")
            );
            assert_eq!(config.evaluators.len(), 1);
            match config.evaluators.get("em_evaluator").unwrap() {
                EvaluatorConfig::ExactMatch(params) => assert_eq!(params.cutoff, Some(0.4)),
                EvaluatorConfig::LLMJudge(_) => panic!("Expected ExactMatch evaluator"),
            }
            // No additional function configs for exact match
            assert_eq!(additional_functions.len(), 0);

            // Verify the metrics
            assert_eq!(metric_configs.len(), 1);

            // Check the metric name follows expected format
            let metric_config_name = get_evaluator_metric_name(evaluation_name, "em_evaluator");
            assert_eq!(
                metric_config_name,
                "tensorzero::evaluation_name::test_evaluation::evaluator_name::em_evaluator"
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
                UninitializedLLMJudgeVariantInfo {
                    inner: UninitializedLLMJudgeVariantConfig::ChatCompletion(
                        UninitializedLLMJudgeChatCompletionVariantConfig {
                            active: Some(true),
                            model: Arc::from("gpt-4.1-mini"),
                            system_instructions:
                                "fixtures/config/evaluations/evaluation1/llm_judge_bool/system_instructions.txt"
                                    .into(),
                            temperature: Some(0.7),
                            top_p: None,
                            max_tokens: Some(100),
                            presence_penalty: None,
                            frequency_penalty: None,
                            seed: None,
                            json_mode: JsonMode::Tool,
                            retries: RetryConfig::default(),
                            extra_body: Default::default(),
                            extra_headers: Default::default(),
                            stop_sequences: None,
                            reasoning_effort: None,
                            service_tier: None,
                            thinking_budget_tokens: None,
                            verbosity: None,
                        },
                    ),
                    timeouts: None,
                },
            );

            let llm_judge_config = UninitializedLLMJudgeConfig {
                input_format: LLMJudgeInputFormat::Serialized,
                variants,
                output_type: LLMJudgeOutputType::Boolean,
                optimize: LLMJudgeOptimize::Min,
                include: LLMJudgeIncludeConfig {
                    reference_output: false,
                },
                cutoff: None,
                description: Some("llm judge description".to_string()),
            };

            let mut evaluators = HashMap::new();
            evaluators.insert(
                "llm_judge_evaluation".to_string(),
                UninitializedEvaluatorConfig::LLMJudge(llm_judge_config),
            );

            let uninitialized_config = UninitializedInferenceEvaluationConfig {
                evaluators,
                function_name: function_name.to_string(),
                description: Some("evaluation description llm judge".to_string()),
            };

            let (config, additional_functions, metric_configs) = uninitialized_config
                .load(&functions, evaluation_name)
                .unwrap();
            assert_eq!(config.evaluators.len(), 1);
            assert_eq!(
                config.description.as_deref(),
                Some("evaluation description llm judge")
            );

            // Verify LLM judge evaluator config
            match config.evaluators.get("llm_judge_evaluation").unwrap() {
                EvaluatorConfig::LLMJudge(judge_config) => {
                    assert!(matches!(
                        judge_config.output_type,
                        LLMJudgeOutputType::Boolean
                    ));
                    assert!(matches!(judge_config.optimize, LLMJudgeOptimize::Min));
                    assert!(!judge_config.include.reference_output);
                    assert_eq!(
                        judge_config.description.as_deref(),
                        Some("llm judge description")
                    );
                }
                EvaluatorConfig::ExactMatch(_) => panic!("Expected LLMJudge evaluator config"),
            }

            // Verify additional function config was created
            assert_eq!(additional_functions.len(), 1);
            let function_name =
                get_llm_judge_function_name(evaluation_name, "llm_judge_evaluation");
            assert!(additional_functions.contains_key(&function_name));

            // Verify the function config has the correct type
            match additional_functions[&function_name].as_ref() {
                FunctionConfig::Json(json_config) => {
                    assert_eq!(json_config.variants.len(), 1);
                    assert!(json_config.variants.contains_key("test_variant"));
                    assert!(json_config.schemas.get_implicit_system_schema().is_none());
                    assert!(json_config.schemas.get_implicit_user_schema().is_some());
                    assert!(json_config.output_schema.value.is_object());
                }
                FunctionConfig::Chat(_) => panic!("Expected Json function config"),
            }

            // Verify the metrics
            assert_eq!(metric_configs.len(), 1);

            // Check the metric name follows expected format
            let metric_config_name =
                get_evaluator_metric_name(evaluation_name, "llm_judge_evaluation");
            assert_eq!(
                metric_config_name,
                "tensorzero::evaluation_name::test_evaluation::evaluator_name::llm_judge_evaluation"
            );
            assert!(metric_configs.contains_key(&metric_config_name));

            // Verify all properties of the metric config
            let metric_config = metric_configs.get(&metric_config_name).unwrap();
            assert_eq!(metric_config.r#type, MetricConfigType::Boolean);
            assert_eq!(metric_config.optimize, MetricConfigOptimize::Min);
            assert_eq!(metric_config.level, MetricConfigLevel::Inference);

            // Verify the type conversion from LLMJudgeOutputType to MetricConfigType
            let llm_judge_evaluation = match config.evaluators.get("llm_judge_evaluation").unwrap()
            {
                EvaluatorConfig::LLMJudge(config) => config,
                EvaluatorConfig::ExactMatch(_) => panic!("Expected LLMJudge evaluator"),
            };
            assert_eq!(
                MetricConfigType::from(llm_judge_evaluation.output_type),
                metric_config.r#type
            );

            // Verify the optimize conversion from LLMJudgeOptimize to MetricConfigOptimize
            assert_eq!(
                MetricConfigOptimize::from(llm_judge_evaluation.optimize),
                metric_config.optimize
            );
        }

        // Test case 2.1: Successful loading with LLM judge evaluator with Float output type
        {
            let mut variants = HashMap::new();
            variants.insert(
                "test_variant".to_string(),
                UninitializedLLMJudgeVariantInfo {
                    inner: UninitializedLLMJudgeVariantConfig::ChatCompletion(
                        UninitializedLLMJudgeChatCompletionVariantConfig {
                            active: Some(true),
                            model: Arc::from("gpt-4.1-mini"),
                            system_instructions:
                                "fixtures/config/evaluations/evaluation1/llm_judge_bool/system_instructions.txt"
                                    .into(),
                            temperature: Some(0.7),
                            top_p: None,
                            max_tokens: Some(100),
                            presence_penalty: None,
                            frequency_penalty: None,
                            seed: None,
                            json_mode: JsonMode::Tool,
                            retries: RetryConfig::default(),
                            extra_body: Default::default(),
                            extra_headers: Default::default(),
                            stop_sequences: None,
                            reasoning_effort: None,
                            service_tier: None,
                            thinking_budget_tokens: None,
                            verbosity: None,
                        },
                    ),
                    timeouts: None,
                },
            );

            let llm_judge_config = UninitializedLLMJudgeConfig {
                input_format: LLMJudgeInputFormat::Serialized,
                variants,
                output_type: LLMJudgeOutputType::Float,
                optimize: LLMJudgeOptimize::Max,
                include: LLMJudgeIncludeConfig {
                    reference_output: true,
                },
                cutoff: None,
                description: Some("llm judge description float".to_string()),
            };

            let mut evaluators = HashMap::new();
            evaluators.insert(
                "llm_judge_float".to_string(),
                UninitializedEvaluatorConfig::LLMJudge(llm_judge_config),
            );

            let uninitialized_config = UninitializedInferenceEvaluationConfig {
                evaluators,
                function_name: function_name.to_string(),
                description: Some("evaluation description llm judge float".to_string()),
            };

            let (config, additional_functions, metric_configs) = uninitialized_config
                .load(&functions, evaluation_name)
                .unwrap();
            assert_eq!(config.evaluators.len(), 1);
            assert_eq!(
                config.description.as_deref(),
                Some("evaluation description llm judge float")
            );

            // Verify LLM judge evaluator config
            match config.evaluators.get("llm_judge_float").unwrap() {
                EvaluatorConfig::LLMJudge(judge_config) => {
                    assert!(matches!(
                        judge_config.output_type,
                        LLMJudgeOutputType::Float
                    ));
                    assert!(matches!(judge_config.optimize, LLMJudgeOptimize::Max));
                    assert!(judge_config.include.reference_output);
                    assert_eq!(
                        judge_config.description.as_deref(),
                        Some("llm judge description float")
                    );
                }
                EvaluatorConfig::ExactMatch(_) => panic!("Expected LLMJudge evaluator config"),
            }

            // Verify additional function config was created
            assert_eq!(additional_functions.len(), 1);
            let function_name = get_llm_judge_function_name(evaluation_name, "llm_judge_float");
            assert!(additional_functions.contains_key(&function_name));

            // Verify the metrics
            assert_eq!(metric_configs.len(), 1);

            // Check the metric name follows expected format
            let metric_config_name = get_evaluator_metric_name(evaluation_name, "llm_judge_float");
            assert_eq!(
                metric_config_name,
                "tensorzero::evaluation_name::test_evaluation::evaluator_name::llm_judge_float"
            );
            assert!(metric_configs.contains_key(&metric_config_name));

            // Verify all properties of the metric config
            let metric_config = metric_configs.get(&metric_config_name).unwrap();
            assert_eq!(metric_config.r#type, MetricConfigType::Float);
            assert_eq!(metric_config.optimize, MetricConfigOptimize::Max);
            assert_eq!(metric_config.level, MetricConfigLevel::Inference);

            // Verify the type conversion from LLMJudgeOutputType to MetricConfigType
            let llm_judge_evaluation = match config.evaluators.get("llm_judge_float").unwrap() {
                EvaluatorConfig::LLMJudge(config) => config,
                EvaluatorConfig::ExactMatch(_) => panic!("Expected LLMJudge evaluator"),
            };
            assert_eq!(
                MetricConfigType::from(llm_judge_evaluation.output_type),
                metric_config.r#type
            );

            // Verify the optimize conversion from LLMJudgeOptimize to MetricConfigOptimize
            assert_eq!(
                MetricConfigOptimize::from(llm_judge_evaluation.optimize),
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

            let uninitialized_config = UninitializedInferenceEvaluationConfig {
                evaluators,
                function_name: "nonexistent_function".to_string(),
                description: None,
            };

            let result = uninitialized_config.load(&functions, evaluation_name);
            assert!(result.is_err());
            assert!(matches!(
                *result.unwrap_err().get_details(),
                ErrorDetails::Config { .. }
            ));
        }

        // Test case 4: Error when evaluation name contains "::"
        {
            let mut evaluators = HashMap::new();
            evaluators.insert(
                "em_evaluator".to_string(),
                UninitializedEvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }),
            );

            let uninitialized_config = UninitializedInferenceEvaluationConfig {
                evaluators,
                function_name: function_name.to_string(),
                description: None,
            };

            let result = uninitialized_config.load(&functions, "invalid::evaluation::name");
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
                UninitializedLLMJudgeVariantInfo {
                    inner: UninitializedLLMJudgeVariantConfig::ChatCompletion(
                        UninitializedLLMJudgeChatCompletionVariantConfig {
                            active: Some(true),
                            model: Arc::from("gpt-4.1-mini"),
                            system_instructions:
                                "fixtures/config/evaluations/evaluation1/llm_judge_bool/system_instructions.txt"
                                    .into(),
                            temperature: Some(0.7),
                            top_p: None,
                            max_tokens: Some(100),
                            presence_penalty: None,
                            frequency_penalty: None,
                            seed: None,
                            json_mode: JsonMode::Tool,
                            retries: RetryConfig::default(),
                            extra_body: Default::default(),
                            extra_headers: Default::default(),
                            stop_sequences: None,
                            reasoning_effort: None,
                            service_tier: None,
                            thinking_budget_tokens: None,
                            verbosity: None,
                        },
                    ),
                    timeouts: None,
                },
            );

            let mut test_variant2 = HashMap::new();
            test_variant2.insert(
                "test_variant2".to_string(),
                UninitializedLLMJudgeVariantInfo {
                    inner: UninitializedLLMJudgeVariantConfig::ChatCompletion(
                        UninitializedLLMJudgeChatCompletionVariantConfig {
                            active: Some(true),
                            model: Arc::from("gpt-4"),
                            system_instructions: ResolvedTomlPathData::new_for_tests(PathBuf::from(
                                "fixtures/config/evaluations/evaluation1/llm_judge_bool/system_instructions.txt",
                            ), None),
                            temperature: Some(0.5),
                            top_p: None,
                            max_tokens: Some(200),
                            presence_penalty: None,
                            frequency_penalty: None,
                            seed: None,
                            json_mode: JsonMode::Tool,
                            retries: RetryConfig::default(),
                            extra_body: Default::default(),
                            extra_headers: Default::default(),
                            stop_sequences: None,
                            reasoning_effort: None,
                            service_tier: None,
                            thinking_budget_tokens: None,
                            verbosity: None,
                        },
                    ),
                    timeouts: None,
                },
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
                input_format: LLMJudgeInputFormat::Serialized,
                variants,
                output_type: LLMJudgeOutputType::Boolean,
                optimize: LLMJudgeOptimize::Min,
                include: LLMJudgeIncludeConfig {
                    reference_output: false,
                },
                cutoff: Some(0.3),
                description: None,
            };

            let mut evaluators = HashMap::new();
            evaluators.insert(
                "multiple_active_variants".to_string(),
                UninitializedEvaluatorConfig::LLMJudge(llm_judge_config),
            );

            let uninitialized_config = UninitializedInferenceEvaluationConfig {
                evaluators,
                function_name: function_name.to_string(),
                description: None,
            };

            let result = uninitialized_config.load(&functions, evaluation_name);
            assert!(result.is_err());
            assert_eq!(
                *result.unwrap_err().get_details(),
                ErrorDetails::Config {
                    message: "Evaluator `multiple_active_variants` in `[evaluations.test_evaluation]` must have exactly 1 variant that is active. Found 2 variants with nonzero weights.".to_string(),
                }
            );
        }

        // Test case 6: Error when evaluator name contains "::"
        {
            let evaluation_name = "test_evaluation";
            let function_name = "test_function";

            let mut functions = HashMap::new();
            functions.insert(
                function_name.to_string(),
                Arc::new(FunctionConfig::Json(FunctionConfigJson {
                    variants: HashMap::new(),
                    output_schema: create_test_schema(),
                    schemas: SchemaData::default(),
                    json_mode_tool_call_config: create_json_mode_tool_call_config(
                        create_test_schema(),
                    ),
                    description: None,
                    all_explicit_template_names: HashSet::new(),
                    experimentation: ExperimentationConfig::legacy_from_variants_map(
                        &HashMap::new(),
                    ),
                })),
            );

            let mut evaluators = HashMap::new();
            evaluators.insert(
                "foo::invalid_name".to_string(),
                UninitializedEvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }),
            );

            let uninitialized_config = UninitializedInferenceEvaluationConfig {
                evaluators,
                function_name: function_name.to_string(),
                description: None,
            };

            let result = uninitialized_config.load(&functions, evaluation_name);
            assert!(result.is_err());
            assert_eq!(
                *result.unwrap_err().get_details(),
                ErrorDetails::Config {
                    message:
                        "Evaluator names cannot contain \"::\" (referenced in `[evaluations.test_evaluation.foo::invalid_name]`)"
                            .to_string(),
                }
            );
        }

        // Test case 7: Successful loading with LLM judge evaluator with reference_output = true
        {
            let mut variants = HashMap::new();
            variants.insert(
                "test_variant".to_string(),
                UninitializedLLMJudgeVariantInfo {
                    inner: UninitializedLLMJudgeVariantConfig::ChatCompletion(
                        UninitializedLLMJudgeChatCompletionVariantConfig {
                            active: Some(true),
                            model: Arc::from("gpt-4.1-mini"),
                            system_instructions: ResolvedTomlPathData::new_for_tests(PathBuf::from(
                                "fixtures/config/evaluations/evaluation1/llm_judge_bool/system_instructions.txt",
                            ), None),
                            temperature: Some(0.7),
                            top_p: None,
                            max_tokens: Some(100),
                            presence_penalty: None,
                            frequency_penalty: None,
                            seed: None,
                            json_mode: JsonMode::Tool,
                            retries: RetryConfig::default(),
                            extra_body: Default::default(),
                            extra_headers: Default::default(),
                            stop_sequences: None,
                            reasoning_effort: None,
                            service_tier: None,
                            thinking_budget_tokens: None,
                            verbosity: None,
                        },
                    ),
                    timeouts: None,
                },
            );

            let llm_judge_config = UninitializedLLMJudgeConfig {
                input_format: LLMJudgeInputFormat::Serialized,
                variants,
                output_type: LLMJudgeOutputType::Boolean,
                optimize: LLMJudgeOptimize::Min,
                include: LLMJudgeIncludeConfig {
                    reference_output: true,
                },
                cutoff: None,
                description: None,
            };

            let mut evaluators = HashMap::new();
            evaluators.insert(
                "llm_judge_with_ref".to_string(),
                UninitializedEvaluatorConfig::LLMJudge(llm_judge_config),
            );

            let uninitialized_config = UninitializedInferenceEvaluationConfig {
                evaluators,
                function_name: function_name.to_string(),
                description: None,
            };

            let result = uninitialized_config.load(&functions, evaluation_name);
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
                EvaluatorConfig::ExactMatch(_) => panic!("Expected LLMJudge evaluator config"),
            }
        }

        // Test case 8: Single LLM Judge variant with no 'active' field specified (defaults to active)
        {
            let mut variants = HashMap::new();
            variants.insert(
                "default_active_variant".to_string(),
                UninitializedLLMJudgeVariantInfo {
                    inner: UninitializedLLMJudgeVariantConfig::ChatCompletion(
                        UninitializedLLMJudgeChatCompletionVariantConfig {
                            active: None, // No 'active' field specified
                            model: Arc::from("gpt-4.1-mini"),
                            system_instructions: ResolvedTomlPathData::new_for_tests(PathBuf::from(
                                "fixtures/config/evaluations/evaluation1/llm_judge_bool/system_instructions.txt",
                            ), None),
                            temperature: Some(0.7),
                            top_p: None,
                            max_tokens: Some(100),
                            presence_penalty: None,
                            frequency_penalty: None,
                            seed: None,
                            json_mode: JsonMode::Tool,
                            retries: RetryConfig::default(),
                            extra_body: Default::default(),
                            extra_headers: Default::default(),
                            stop_sequences: None,
                            reasoning_effort: None,
                            service_tier: None,
                            thinking_budget_tokens: None,
                            verbosity: None,
                        },
                    ),
                    timeouts: None,
                },
            );

            let llm_judge_config = UninitializedLLMJudgeConfig {
                input_format: LLMJudgeInputFormat::Serialized,
                variants,
                output_type: LLMJudgeOutputType::Boolean,
                optimize: LLMJudgeOptimize::Max,
                include: LLMJudgeIncludeConfig::default(),
                cutoff: None,
                description: None,
            };

            let mut evaluators = HashMap::new();
            evaluators.insert(
                "llm_judge_default_active".to_string(),
                UninitializedEvaluatorConfig::LLMJudge(llm_judge_config),
            );

            let uninitialized_config = UninitializedInferenceEvaluationConfig {
                evaluators,
                function_name: function_name.to_string(),
                description: None,
            };

            let result = uninitialized_config.load(&functions, evaluation_name);
            assert!(result.is_ok());

            let (_config, additional_functions, _metric_configs) = result.unwrap();
            let function_config_name =
                get_llm_judge_function_name(evaluation_name, "llm_judge_default_active");
            let function_config = additional_functions.get(&function_config_name).unwrap();
            match function_config.as_ref() {
                FunctionConfig::Json(json_config) => {
                    assert_eq!(json_config.variants.len(), 1);
                    let variant = json_config.variants.get("default_active_variant").unwrap();
                    // Check that the weight is Some(1.0) which indicates it defaulted to active
                    match &variant.inner {
                        VariantConfig::ChatCompletion(cc_config) => {
                            assert_eq!(cc_config.weight(), Some(1.0));
                        }
                        _ => panic!("Expected ChatCompletion variant config"),
                    }
                }
                FunctionConfig::Chat(_) => panic!("Expected Json function config"),
            }
        }

        // Test case 9: Single LLM Judge variant explicitly set to inactive (active = false)
        {
            let mut variants = HashMap::new();
            variants.insert(
                "inactive_variant".to_string(),
                UninitializedLLMJudgeVariantInfo {
                    inner: UninitializedLLMJudgeVariantConfig::ChatCompletion(
                        UninitializedLLMJudgeChatCompletionVariantConfig {
                            active: Some(false), // Explicitly inactive
                            model: Arc::from("gpt-4.1-mini"),
                            system_instructions: ResolvedTomlPathData::new_for_tests(PathBuf::from(
                                "fixtures/config/evaluations/evaluation1/llm_judge_bool/system_instructions.txt",
                            ), None),
                            temperature: Some(0.7),
                            top_p: None,
                            max_tokens: Some(100),
                            presence_penalty: None,
                            frequency_penalty: None,
                            seed: None,
                            json_mode: JsonMode::Tool,
                            retries: RetryConfig::default(),
                            extra_body: Default::default(),
                            extra_headers: Default::default(),
                            stop_sequences: None,
                            reasoning_effort: None,
                            service_tier: None,
                            thinking_budget_tokens: None,
                            verbosity: None,
                        },
                    ),
                    timeouts: None,
                },
            );

            let llm_judge_config = UninitializedLLMJudgeConfig {
                input_format: LLMJudgeInputFormat::Serialized,
                variants,
                output_type: LLMJudgeOutputType::Boolean,
                optimize: LLMJudgeOptimize::Max,
                include: LLMJudgeIncludeConfig::default(),
                cutoff: None,
                description: None,
            };

            let mut evaluators = HashMap::new();
            evaluators.insert(
                "llm_judge_inactive".to_string(),
                UninitializedEvaluatorConfig::LLMJudge(llm_judge_config),
            );

            let uninitialized_config = UninitializedInferenceEvaluationConfig {
                evaluators,
                function_name: function_name.to_string(),
                description: None,
            };

            let result = uninitialized_config.load(&functions, evaluation_name);
            assert!(result.is_err());
            assert_eq!(
                *result.unwrap_err().get_details(),
                ErrorDetails::Config {
                    message: format!("Evaluator `llm_judge_inactive` in `[evaluations.{evaluation_name}]` must have exactly 1 variant that is active. You have specified a single inactive variant."),
                }
            );
        }
    }

    /// Test backward compatibility: verify deprecated "static" type still works
    #[test]
    fn test_backward_compatibility_static_type() {
        // Setup: Create a function that the evaluation will reference
        let mut functions = HashMap::new();
        let function_name = "test_function";
        let function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            output_schema: create_test_schema(),
            json_mode_tool_call_config: create_json_mode_tool_call_config(create_test_schema()),
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::legacy_from_variants_map(&HashMap::new()),
        });
        functions.insert(function_name.to_string(), Arc::new(function_config));

        // Test TOML config with deprecated "static" type
        let toml = format!(
            r#"
            type = "static"
            function_name = "{function_name}"

            [evaluators.test_evaluator]
            type = "exact_match"
        "#
        );

        // Deserialize the TOML
        let uninitialized: UninitializedEvaluationConfig = toml::from_str(&toml).unwrap();

        // Verify it loads correctly despite using deprecated "static" type
        let result = uninitialized.load(&functions, "test_backward_compat");
        assert!(
            result.is_ok(),
            "Expected successful load with 'static' type"
        );

        let (config, additional_functions, metric_configs) = result.unwrap();

        // Verify the config was loaded correctly
        assert_eq!(config.function_name, function_name);
        assert_eq!(config.evaluators.len(), 1);
        assert!(config.evaluators.contains_key("test_evaluator"));

        // Verify exact match evaluator was loaded
        match config.evaluators.get("test_evaluator").unwrap() {
            EvaluatorConfig::ExactMatch(_) => {} // Expected
            EvaluatorConfig::LLMJudge(_) => panic!("Expected ExactMatch evaluator"),
        }

        // Verify no additional functions (exact match doesn't need them)
        assert_eq!(additional_functions.len(), 0);

        // Verify metric was created
        assert_eq!(metric_configs.len(), 1);
    }

    // Helper functions for tests
    fn create_test_schema() -> StaticJSONSchema {
        let schema_value = serde_json::json!({
            "type": "object",
            "properties": {
                "result": {
                    "type": "string"
                }
            },
            "required": ["result"]
        });
        StaticJSONSchema::from_value(schema_value).unwrap()
    }
}
