/// Macro to reduce boilerplate when calling methods defined in tensorzero-core from NAPI
///
/// All methods like this exposed to Node will return a string that Node client can deserialize into a response object. They may take no parameters, or take a string serialized from the corresponding JSON request object.
///
/// This macro has two variants, and handles:
/// - Deserializing JSON parameters into a Rust struct (if the method takes parameters)
/// - Calling the method defined in tensorzero-core
/// - Serializing the result back to JSON
/// - Error mapping
///
/// # Example 1: a method that takes parameters
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
///                 max_periods,
///             }
///         )
///     }
/// ```
///
/// This expands into:
///
/// ```ignore
///     #[napi]
///     pub async fn get_model_usage_timeseries(&self, params: String) -> Result<String, napi::Error> {
///         let params_struct: GetModelUsageTimeseriesParams = serde_json::from_str(&params)
///             .map_err(|e| napi::Error::from_reason(e.to_string()))?;
///         let result = &self.0.get_model_usage_timeseries(time_window, max_periods)
///             .await
///             .map_err(|e| napi::Error::from_reason(e.to_string()))?;
///         serde_json::to_string(&result)
///             .map_err(|e| napi::Error::from_reason(e.to_string()))
///     }
/// ```
///
/// # Example 2: a method that doesn't take parameters
///
/// ```ignore
///     #[napi]
///     pub async fn query_episode_table_bounds(&self) -> Result<String, napi::Error> {
///         napi_call!(&self, query_episode_table_bounds)
///     }
/// ```
///
/// This expands into:
///
/// ```ignore
///     #[napi]
///     pub async fn query_episode_table_bounds(&self) -> Result<String, napi::Error> {
///         let result = &self.0.query_episode_table_bounds()
///             .await
///             .map_err(|e| napi::Error::from_reason(e.to_string()))?;
///         serde_json::to_string(&result)
///             .map_err(|e| napi::Error::from_reason(e.to_string()))
///     }
/// ```
///
/// Note that the order of the struct fields is important: they must match the order of the method's arguments.
#[macro_export]
macro_rules! napi_call {
    // Variant 1: the Rust method takes parameters.
    ($self:expr, $method:ident, $params:expr, $param_type:ty { $($field:ident),+ }) => {{
        // Deserialize the JSON parameters into a Rust struct.
        let params_struct: $param_type = serde_json::from_str(&$params)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        // Call the Rust method with the deserialized parameters. We pass each struct field as an argument to the method in the order they are specified.
        let result = $self.0.$method($(params_struct.$field),+)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        // Serialize the result back to JSON.
        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }};

    // Variant 2: the Rust method takes the whole struct as a parameter, and the struct is re-exported to Node.
    ($self:expr, $method:ident, $params:expr, $param_type:ty) => {{
        // Deserialize the JSON parameters into a Rust struct.
        let params_struct: $param_type = serde_json::from_str(&$params)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        // Call the Rust method with the deserialized parameters. We pass the struct as a single argument to the method.
        let result = $self.0.$method(&params_struct)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        // Serialize the result back to JSON.
        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }};

    // Variant 3: the Rust method doesn't take parameters.
    ($self:expr, $method:ident) => {{
        // Call the Rust method with no parameters.
        let result = $self.0.$method()
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        // Serialize the result back to JSON.
        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }};

    // Variant 4: the Rust method takes a single parameter that doesn't need to be deserialized.
    ($self:expr, $method:ident, $param:expr) => {{
        // Call the Rust method with no parameters.
        let result = $self.0.$method($param)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        // Serialize the result back to JSON.
        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }};
}

/// Macro to reduce boilerplate when calling methods defined in tensorzero-core from NAPI
///
/// All methods like this exposed to Node will return a native Node type. They may take no parameters, or take a string serialized from the corresponding JSON request object.
///
/// This macro has two variants, and handles:
/// - Deserializing JSON parameters into a Rust struct (if the method takes parameters)
/// - Calling the method defined in tensorzero-core
/// - Error mapping
///
/// The main difference bewteen `napi_call_no_deserializing` and `napi_call` is that `napi_call_no_deserializing` does not deserialize the response of the tensorzero-core method back into a JSON string. This is useful for methods that return a native Node type.
///
/// # Example: a method that doesn't take parameters
///
/// ```ignore
///     #[napi]
///     pub async fn count_distinct_models_used(&self) -> Result<u32, napi::Error> {
///         napi_call_no_deserializing!(&self, count_distinct_models_used)
///     }
/// ```
///
/// This expands into:
///
/// ```ignore
///     #[napi]
///     pub async fn count_distinct_models_used(&self) -> Result<u32, napi::Error> {
///         $self.0.count_distinct_models_used()
///             .await
///             .map_err(|e| napi::Error::from_reason(e.to_string()))
///     }
/// ```
///
/// Note that the order of the struct fields is important: they must match the order of the method's arguments.
#[macro_export]
macro_rules! napi_call_no_deserializing {
    // Variant 1: the Rust method takes parameters.
    ($self:expr, $method:ident, $params:expr, $param_type:ty { $($field:ident),+ }) => {{
        // Deserialize the JSON parameters into a Rust struct.
        let params_struct: $param_type = serde_json::from_str(&$params)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        // Call the Rust method with the deserialized parameters. We pass each struct field as an argument to the method in the order they are specified.
        $self.0.$method($(params_struct.$field),+)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))

        // Result is returned directly as a native Node type.
    }};

    // Variant 2: the Rust method takes the whole struct as a parameter, and the struct is re-exported to Node.
    ($self:expr, $method:ident, $params:expr, $param_type:ty) => {{
        // Deserialize the JSON parameters into a Rust struct.
        let params_struct: $param_type = serde_json::from_str(&$params)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        // Call the Rust method with the deserialized parameters. We pass the struct as a single argument to the method.
        $self.0.$method(&params_struct)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))

        // Result is returned directly as a native Node type.
    }};

    // Variant 3: the Rust method doesn't take parameters.
    ($self:expr, $method:ident) => {
        // Call the Rust method with no parameters.
        $self.0.$method()
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))

        // Result is returned directly as a native Node type.
    };

    // Variant 4: the Rust method takes a single parameter that doesn't need to be deserialized.
    ($self:expr, $method:ident, $param:expr) => {{
        // Call the Rust method with the parameter.
        $self.0.$method($param)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?

        // Result is returned directly as a native Node type.
    }};
}
