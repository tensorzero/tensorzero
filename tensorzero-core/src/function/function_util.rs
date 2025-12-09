use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Error;
use crate::error::ErrorDetails;
use crate::function::DEFAULT_FUNCTION_NAME;
use crate::function::FunctionConfig;
use crate::function::FunctionConfigChat;

/// Gets a function by name, handling default function correctly.
pub fn get_function<'a>(
    functions: &'a HashMap<String, Arc<FunctionConfig>>,
    function_name: &str,
) -> Result<Cow<'a, Arc<FunctionConfig>>, Error> {
    if function_name == DEFAULT_FUNCTION_NAME {
        Ok(Cow::Owned(Arc::new(FunctionConfig::Chat(
            FunctionConfigChat::default(),
        ))))
    } else {
        Ok(Cow::Borrowed(functions.get(function_name).ok_or_else(
            || {
                Error::new(ErrorDetails::UnknownFunction {
                    name: function_name.to_string(),
                })
            },
        )?))
    }
}
