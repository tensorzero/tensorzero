use uuid::Uuid;

use crate::config::Config;
use crate::db::datasets::{ChatInferenceDatapointInsert, JsonInferenceDatapointInsert};
use crate::endpoints::datasets::v1::types::{
    CreateChatDatapointRequest, CreateJsonDatapointRequest,
};
use crate::endpoints::feedback::{
    validate_parse_demonstration, DemonstrationOutput, DynamicDemonstrationInfo,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::{FetchContext, JsonInferenceOutput};

impl CreateChatDatapointRequest {
    /// Validates and prepares this request for insertion into the database.
    /// Returns the datapoint insert struct and the generated UUID.
    pub async fn into_database_insert(
        self,
        config: &Config,
        fetch_context: &FetchContext<'_>,
        dataset_name: &str,
    ) -> Result<ChatInferenceDatapointInsert, Error> {
        // Validate function exists and is a chat function
        let function_config = config.get_function(&self.function_name)?;
        let FunctionConfig::Chat(_) = &**function_config else {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!(
                    "Function '{}' is not configured as a chat function",
                    self.function_name
                ),
            }));
        };
        function_config.validate_input(&self.input)?;

        let stored_input = self
            .input
            .into_lazy_resolved_input(*fetch_context)?
            .into_stored_input(fetch_context.object_store_info)
            .await?;

        let tool_config =
            function_config.prepare_tool_config(self.dynamic_tool_params, &config.tools)?;
        let dynamic_demonstration_info =
            DynamicDemonstrationInfo::Chat(tool_config.clone().unwrap_or_default());

        // Validate and parse output if provided
        let output = if let Some(output_value) = self.output {
            let validated_output = validate_parse_demonstration(
                &function_config,
                &output_value,
                dynamic_demonstration_info,
            )
            .await?;

            let DemonstrationOutput::Chat(output) = validated_output else {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: "Expected chat output from validate_parse_demonstration".to_string(),
                }));
            };

            Some(output)
        } else {
            None
        };

        let insert = ChatInferenceDatapointInsert {
            dataset_name: dataset_name.to_string(),
            function_name: self.function_name,
            name: self.name,
            id: Uuid::now_v7(),
            episode_id: self.episode_id,
            input: stored_input,
            output,
            tool_params: tool_config.as_ref().map(|x| x.clone().into()),
            tags: self.tags,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        Ok(insert)
    }
}

impl CreateJsonDatapointRequest {
    /// Validates and prepares this request for insertion into the database.
    /// Returns the datapoint insert struct and the generated UUID.
    pub async fn into_database_insert(
        self,
        config: &Config,
        fetch_context: &FetchContext<'_>,
        dataset_name: &str,
    ) -> Result<JsonInferenceDatapointInsert, Error> {
        // Validate function exists and is a JSON function
        let function_config = config.get_function(&self.function_name)?;
        let FunctionConfig::Json(json_function_config) = &**function_config else {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!(
                    "Function '{}' is not configured as a JSON function",
                    self.function_name
                ),
            }));
        };

        // Validate and convert input
        function_config.validate_input(&self.input)?;
        let stored_input = self
            .input
            .into_lazy_resolved_input(crate::inference::types::FetchContext {
                client: fetch_context.client,
                object_store_info: fetch_context.object_store_info,
            })?
            .into_stored_input(fetch_context.object_store_info)
            .await?;

        // Determine the output schema (use provided or default to function's schema)
        let output_schema = self
            .output_schema
            .unwrap_or_else(|| json_function_config.output_schema.value.clone());
        let dynamic_demonstration_info = DynamicDemonstrationInfo::Json(output_schema.clone());

        // Validate and parse output if provided
        let output = if let Some(output_value) = self.output {
            let _validated_output = validate_parse_demonstration(
                &function_config,
                &output_value,
                dynamic_demonstration_info,
            )
            .await?;

            // The provided output is validated, so we write it.
            Some(JsonInferenceOutput {
                raw: Some(serde_json::to_string(&output_value).map_err(|e| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Failed to serialize provided outputto json: {e}"),
                    })
                })?),
                parsed: Some(output_value),
            })
        } else {
            None
        };

        let insert = JsonInferenceDatapointInsert {
            dataset_name: dataset_name.to_string(),
            function_name: self.function_name,
            name: self.name,
            id: Uuid::now_v7(),
            episode_id: self.episode_id,
            input: stored_input,
            output,
            output_schema,
            tags: self.tags,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        Ok(insert)
    }
}
