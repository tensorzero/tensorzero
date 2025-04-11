use pyo3::{
    exceptions::PyValueError,
    types::{PyDict, PyDictMethods},
    Py, PyResult, Python,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
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
use tensorzero_rust::TensorZeroError;
use uuid::Uuid;

use crate::convert_error;

pub fn get_template_config(
    py: Python<'_>,
    config: &Config<'_>,
    function_name: &str,
    variant_name: &str,
) -> PyResult<Py<PyDict>> {
    let variant_config = config
        .get_function(function_name)
        .map_err(|e| convert_error(py, TensorZeroError::Other { source: e.into() }))?
        .variants()
        .get(variant_name)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "Variant {variant_name} not found in function {function_name}",
            ))
        })?;
    let VariantConfig::ChatCompletion(config) = variant_config else {
        return Err(PyValueError::new_err(format!(
            "Variant {} is not a ChatCompletion variant",
            variant_name
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
    function_name: &str,
    metric_name: Option<&str>,
    threshold: Option<f64>,
    max_samples: Option<u64>,
) -> Result<Vec<Value>, TensorZeroError> {
    let function_config = config
        .get_function(function_name)
        .map_err(|e| TensorZeroError::Other { source: e.into() })?;

    let limit_clause = max_samples
        .map(|s| format!("LIMIT {}", s))
        .unwrap_or_default();

    let inference_table_name = match &**function_config {
        FunctionConfig::Chat(_) => "ChatInference",
        FunctionConfig::Json(_) => "JsonInference",
    };

    let metric_name = match metric_name {
        Some(name) => name,
        None => {
            let rows = clickhouse.run_query_synchronous("SELECT * from {inference_table_name:Identifier} WHERE function_name = {function_name:String} FORMAT JSONEachRow".to_string() + &limit_clause, Some(&[
                ("function_name", function_name),
                ("inference_table_name", inference_table_name),
                ].into_iter().collect())).await.map_err(|e| TensorZeroError::Other { source: e.into() })?;

            let json_rows: Vec<Value> = rows
                .lines()
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect();
            return Ok(json_rows);
        }
    };

    // TODO - handle demonstrations
    let metric_config = config
        .get_metric_or_err(metric_name)
        .map_err(|e| TensorZeroError::Other { source: e.into() })?;

    query_curated_metric_data(
        clickhouse,
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

#[allow(clippy::too_many_arguments)]
async fn query_curated_metric_data(
    clickhouse: &ClickHouseConnectionInfo,
    function_name: &str,
    metric_name: &str,
    inference_table_name: &str,
    metric_config: &MetricConfig,
    filter_good: bool,
    threshold: Option<f64>,
    max_samples: Option<u64>,
) -> Result<Vec<Value>, TensorZeroError> {
    // Set defaults and prepare limit clause
    let optimize = metric_config.optimize;
    let limit_clause = max_samples
        .map(|s| format!("LIMIT {}", s))
        .unwrap_or_default();

    // Prepare value condition
    let mut value_condition = String::new();
    if filter_good {
        match metric_config.r#type {
            MetricConfigType::Boolean => {
                let value = match optimize {
                    MetricConfigOptimize::Max => 1,
                    MetricConfigOptimize::Min => 0,
                };
                value_condition = format!("AND value = {}", value);
            }
            MetricConfigType::Float => {
                if threshold.is_some() {
                    let operator = get_comparison_operator(optimize);
                    value_condition = format!("AND value {} {{threshold:Float}}", operator);
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

    #[derive(Deserialize, Serialize)]
    struct InferenceData {
        variant_name: String,
        #[serde(deserialize_with = "deserialize_json_string")]
        input: ResolvedInput,
        #[serde(deserialize_with = "deserialize_json_string")]
        output: Value,
        episode_id: Option<Uuid>,
    }

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
    let json_rows: Result<Vec<Value>, _> = rows
        .lines()
        .map(|line| {
            let inference: InferenceData = serde_json::from_str(line)?;
            serde_json::to_value(inference)
        })
        .collect();

    json_rows.map_err(|e| TensorZeroError::Other {
        source: Error::new(ErrorDetails::Serialization {
            message: format!("Failed to deserialize inferences: {e:?}"),
        })
        .into(),
    })
}

// Helper function to determine comparison operator based on optimization goal
fn get_comparison_operator(optimize: MetricConfigOptimize) -> &'static str {
    match optimize {
        MetricConfigOptimize::Max => ">=",
        MetricConfigOptimize::Min => "<=",
    }
}
