use pyo3::{
    exceptions::PyValueError,
    types::{PyDict, PyDictMethods},
    Py, PyResult, Python,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tensorzero_internal::inference::types::batch::deserialize_json_string;
use tensorzero_internal::{
    clickhouse::ClickHouseConnectionInfo,
    config_parser::{
        Config, MetricConfig, MetricConfigLevel, MetricConfigOptimize, MetricConfigType,
    },
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    inference::types::ResolvedInput,
    variant::VariantConfig,
};
use tensorzero_rust::{input_handling::reresolve_input_for_fine_tuning, Client, TensorZeroError};
use uuid::Uuid;

use crate::convert_error;

pub fn get_template_config(
    py: Python<'_>,
    config: &Config<'_>,
    function_name: &str,
    variant_name: &str,
) -> PyResult<Py<PyDict>> {
    let function_config = config
        .get_function(function_name)
        .map_err(|e| convert_error(py, TensorZeroError::Other { source: e.into() }))?
        .into_owned();

    let variant_config = function_config
        .variants()
        .get(variant_name)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "Variant {variant_name} not found in function {function_name}",
            ))
        })?;
    let VariantConfig::ChatCompletion(config) = variant_config else {
        return Err(PyValueError::new_err(format!(
            "Variant {variant_name} is not a ChatCompletion variant"
        )));
    };
    let template_env = PyDict::new(py);
    if let Some(system) = &config.system_template {
        template_env.set_item("system", &system.contents)?;
    }
    if let Some(assistant) = &config.assistant_template {
        template_env.set_item("assistant", &assistant.contents)?;
    }
    if let Some(user) = &config.user_template {
        template_env.set_item("user", &user.contents)?;
    }
    Ok(template_env.unbind())
}

pub async fn get_curated_inferences(
    config: &Config<'_>,
    clickhouse: &ClickHouseConnectionInfo,
    client: &Client,
    function_name: &str,
    metric_name: Option<&str>,
    threshold: Option<f64>,
    max_samples: Option<u64>,
) -> Result<Vec<ProcessedInferenceData>, TensorZeroError> {
    let function_config = config
        .get_function(function_name)
        .map_err(|e| TensorZeroError::Other { source: e.into() })?;

    let limit_clause = max_samples
        .map(|s| format!("LIMIT {s}"))
        .unwrap_or_default();

    let inference_table_name = match &**function_config {
        FunctionConfig::Chat(_) => "ChatInference",
        FunctionConfig::Json(_) => "JsonInference",
    };

    let metric_name = match metric_name {
        Some(name) => name,
        None => {
            let query = format!(
                r#"
            SELECT
                variant_name,
                input,
                output,
                episode_id
            FROM
                {{inference_table_name:Identifier}}
            WHERE
                function_name = {{function_name:String}}
            {limit_clause}
            FORMAT JSONEachRow
            "#
            );
            let rows = clickhouse
                .run_query_synchronous(
                    query,
                    Some(
                        &[
                            ("function_name", function_name),
                            ("inference_table_name", inference_table_name),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                )
                .await
                .map_err(|e| TensorZeroError::Other { source: e.into() })?;

            let unprocessed_rows: Vec<UnprocessedInferenceData> = rows
                .lines()
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect();

            let processing_futures = unprocessed_rows
                .into_iter()
                .map(|row| row.postprocess(client, function_name));

            let processed_rows = futures::future::try_join_all(processing_futures).await?;

            return Ok(processed_rows);
        }
    };

    if metric_name == "demonstration" {
        return query_demonstration_data(
            clickhouse,
            client,
            function_name,
            inference_table_name,
            max_samples,
        )
        .await;
    }

    let metric_config = config
        .get_metric_or_err(metric_name)
        .map_err(|e| TensorZeroError::Other { source: e.into() })?;

    query_curated_metric_data(
        clickhouse,
        client,
        function_name,
        metric_name,
        inference_table_name,
        metric_config,
        true,
        threshold,
        max_samples,
    )
    .await
}

#[derive(Debug, Deserialize)]
struct UnprocessedInferenceData {
    variant_name: String,
    #[serde(deserialize_with = "deserialize_json_string")]
    input: ResolvedInput,
    #[serde(deserialize_with = "deserialize_json_string")]
    output: Value,
    episode_id: Option<Uuid>,
}

#[derive(Debug, Serialize)]
pub struct ProcessedInferenceData {
    variant_name: String,
    input: ResolvedInput,
    output: Value,
    episode_id: Option<Uuid>,
}

impl UnprocessedInferenceData {
    /// Normally, we would use a new() method for this but since we need to deserialize into the
    /// UnprocessedInferenceData struct and we want to force the caller to call this function,
    /// this is the only way to get a ProcessedInferenceData struct.
    pub async fn postprocess(
        mut self,
        client: &Client,
        function_name: &str,
    ) -> Result<ProcessedInferenceData, TensorZeroError> {
        reresolve_input_for_fine_tuning(&mut self.input, client).await?;
        if function_name.starts_with("tensorzero::llm_judge::") {
            handle_llm_judge_output(&mut self)?;
        }
        Ok(ProcessedInferenceData {
            variant_name: self.variant_name,
            input: self.input,
            output: self.output,
            episode_id: self.episode_id,
        })
    }
}

#[expect(clippy::too_many_arguments)]
async fn query_curated_metric_data(
    clickhouse: &ClickHouseConnectionInfo,
    client: &Client,
    function_name: &str,
    metric_name: &str,
    inference_table_name: &str,
    metric_config: &MetricConfig,
    filter_good: bool,
    threshold: Option<f64>,
    max_samples: Option<u64>,
) -> Result<Vec<ProcessedInferenceData>, TensorZeroError> {
    // Set defaults and prepare limit clause
    let optimize = metric_config.optimize;
    let limit_clause = get_limit_clause(max_samples);

    // Prepare value condition
    let mut value_condition = String::new();
    if filter_good {
        match metric_config.r#type {
            MetricConfigType::Boolean => {
                let value = match optimize {
                    MetricConfigOptimize::Max => 1,
                    MetricConfigOptimize::Min => 0,
                };
                value_condition = format!("AND value = {value}");
            }
            MetricConfigType::Float => {
                if threshold.is_some() {
                    let operator = get_comparison_operator(optimize);
                    value_condition = format!("AND value {operator} {{threshold:Float}}");
                }
            }
        }
    }

    let feedback_table = match metric_config.r#type {
        MetricConfigType::Boolean => "BooleanMetricFeedback",
        MetricConfigType::Float => "FloatMetricFeedback",
    };

    let inference_join_key = match metric_config.level {
        MetricConfigLevel::Episode => "episode_id",
        MetricConfigLevel::Inference => "id",
    };

    // Construct the query
    let query = format!(
        r#"
        SELECT
          i.variant_name,
          i.input,
          i.output,
          i.episode_id
        FROM
          {{inference_table_name:Identifier}} i
        JOIN
          (SELECT
            target_id,
            value,
            ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
          FROM
            {{feedback_table:Identifier}}
          WHERE
            metric_name = {{metric_name:String}}
            {value_condition}
          ) f ON i.{{inference_join_key:Identifier}} = f.target_id and f.rn = 1
        WHERE
          i.function_name = {{function_name:String}}
        {limit_clause}
        FORMAT JSONEachRow
        "#
    );

    // Prepare parameters
    let mut params = vec![
        ("function_name", function_name),
        ("metric_name", metric_name),
        ("inference_table_name", inference_table_name),
        ("inference_join_key", inference_join_key),
        ("feedback_table", feedback_table),
    ];

    // Add threshold if present
    let threshold_str = threshold.map(|t| t.to_string()).unwrap_or_default();
    if threshold.is_some() {
        params.push(("threshold", &threshold_str));
    }

    // Run the query
    let rows = clickhouse
        .run_query_synchronous(query, Some(&params.into_iter().collect()))
        .await
        .map_err(|e| TensorZeroError::Other { source: e.into() })?;

    // Parse the results
    let unprocessed_rows: Result<Vec<UnprocessedInferenceData>, _> =
        rows.lines().map(serde_json::from_str).collect();
    let unprocessed_rows = unprocessed_rows.map_err(|e| TensorZeroError::Other {
        source: Error::new(ErrorDetails::Serialization {
            message: format!("Failed to deserialize inferences: {e:?}"),
        })
        .into(),
    })?;
    let processing_futures = unprocessed_rows
        .into_iter()
        .map(|row| row.postprocess(client, function_name));

    let processed_rows = futures::future::try_join_all(processing_futures).await?;

    Ok(processed_rows)
}

async fn query_demonstration_data(
    clickhouse: &ClickHouseConnectionInfo,
    client: &Client,
    function_name: &str,
    inference_table_name: &str,
    max_samples: Option<u64>,
) -> Result<Vec<ProcessedInferenceData>, TensorZeroError> {
    let limit_clause = get_limit_clause(max_samples);

    let query = format!(
        r#"
        SELECT
            argMax(i.variant_name, d.timestamp) as variant_name,
            argMax(i.input, d.timestamp) as input,
            argMax(d.value, d.timestamp) as output,
            argMax(i.episode_id, d.timestamp) as episode_id
        FROM
          {{inference_table_name:Identifier}} i
        JOIN DemonstrationFeedback d
        ON i.id = d.inference_id
        WHERE i.function_name = {{function_name:String}}
        GROUP BY d.inference_id
        {limit_clause}
        FORMAT JSONEachRow
    "#
    );

    let params = vec![
        ("inference_table_name", inference_table_name),
        ("function_name", function_name),
    ];
    let rows = clickhouse
        .run_query_synchronous(query, Some(&params.into_iter().collect()))
        .await
        .map_err(|e| TensorZeroError::Other { source: e.into() })?;

    let unprocessed_rows: Vec<UnprocessedInferenceData> = rows
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    let processing_futures = unprocessed_rows
        .into_iter()
        .map(|row| row.postprocess(client, function_name));

    let processed_rows = futures::future::try_join_all(processing_futures).await?;

    Ok(processed_rows)
}

fn get_limit_clause(max_samples: Option<u64>) -> String {
    max_samples
        .map(|s| format!("LIMIT {s}"))
        .unwrap_or_default()
}

// Helper function to determine comparison operator based on optimization goal
fn get_comparison_operator(optimize: MetricConfigOptimize) -> &'static str {
    match optimize {
        MetricConfigOptimize::Max => ">=",
        MetricConfigOptimize::Min => "<=",
    }
}

/// When we first introduced LLM Judges, we included the thinking section in the output.
/// We have since removed it, but we need to handle the old data.
/// So, we transform any old LLM Judge outputs to the new format by removing the thinking section from the
/// parsed and raw outputs.
fn handle_llm_judge_output(output: &mut UnprocessedInferenceData) -> Result<(), TensorZeroError> {
    if let Some(parsed) = output.output.get_mut("parsed") {
        if parsed.get("thinking").is_some() {
            if let Some(obj) = parsed.as_object_mut() {
                obj.remove("thinking");
            }
        }

        let raw_json = serde_json::to_string(&parsed).map_err(|e| TensorZeroError::Other {
            source: Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize inferences: {e:?}"),
            })
            .into(),
        })?;

        output.output = json!({
            "parsed": parsed,
            "raw": raw_json
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_handle_llm_judge_output() {
        // Test removing the thinking field from the output
        let mut input = UnprocessedInferenceData {
            variant_name: "test".to_string(),
            input: ResolvedInput {
                system: None,
                messages: vec![],
            },
            output: json!({
                "parsed": {
                    "thinking": "This is a test",
                    "answer": "This is a test"
                },
                "raw": "{\"thinking\": \"This is a test\", \"answer\": \"This is a test\"}"
            }),
            episode_id: None,
        };

        handle_llm_judge_output(&mut input).unwrap();

        let expected = json!({
            "parsed": {
                "answer": "This is a test"
            },
            "raw": "{\"answer\":\"This is a test\"}"
        });

        assert_eq!(input.output, expected);

        // Test the correct output is unmodified
        handle_llm_judge_output(&mut input).unwrap();
        assert_eq!(input.output, expected);

        // Test not modifying the output if the parsed field is not present
        let mut input = UnprocessedInferenceData {
            variant_name: "test".to_string(),
            input: ResolvedInput {
                system: None,
                messages: vec![],
            },
            output: json!({
                "raw": "This is a test"
            }),
            episode_id: None,
        };
        handle_llm_judge_output(&mut input).unwrap();

        let expected = json!({
            "raw": "This is a test"
        });

        assert_eq!(input.output, expected);
    }
}
