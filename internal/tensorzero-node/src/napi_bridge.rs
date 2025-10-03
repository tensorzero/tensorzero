/// Macro to reduce boilerplate when calling methods defined in tensorzero-core from NAPI
///
/// All methods like this exposed to Node will return a string that Node client can deserialize into a response object. They may take no parameters, or take a string serialized from the corresponding JSON request object.
///
/// This macro handles:
/// - Deserializing JSON parameters into a Rust struct
/// - Calling the method defined in tensorzero-core
/// - Serializing the result back to JSON
/// - Error mapping
///
/// # Examples
///
/// ```ignore
///     #[napi]
///     pub async fn get_model_usage_timeseries(&self, params: String) -> Result<String, napi::Error> {
///         napi_call!(
///             &self,
///             get_model_usage_timeseries,
///             params,
///             // Rust struct that represents the JSON type exposed to Node.
///             GetModelUsageTimeseriesParams {
///                 time_window,
///                 max_periods
///             }
///         )
///     }
/// ```
#[macro_export]
macro_rules! napi_call {
    ($self:expr, $method:ident, $params:expr, $param_type:ty { $($field:ident),+ }) => {{
        let params_struct: $param_type = serde_json::from_str(&$params)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let result = $self.0.$method($(params_struct.$field),+)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }};

    ($self:expr, $method:ident) => {{
        let result = $self.0.$method()
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }};
}

/// Macro to reduce boilerplate when calling methods defined in tensorzero-core from NAPI
///
/// All methods like this exposed to Node will return a native Node type. They may take no parameters, or take a string serialized from the corresponding JSON request object.
///
/// This macro handles:
/// - Deserializing JSON parameters into a Rust struct
/// - Calling the method defined in tensorzero-core
/// - Error mapping
///
/// # Examples
///
/// ```ignore
///     #[napi]
///     pub async fn count_distinct_models_used(&self) -> Result<u32, napi::Error> {
///         napi_call_no_deserializing!(&self, count_distinct_models_used)
///     }
/// ```
#[macro_export]
macro_rules! napi_call_no_deserializing {
    ($self:expr, $method:ident, $params:expr, $param_type:ty { $($field:ident),+ }) => {{
        let params_struct: $param_type = serde_json::from_str(&$params)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        $self.0.$method($(params_struct.$field),+)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }};

    ($self:expr, $method:ident) => {
        $self.0.$method()
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    };
}
