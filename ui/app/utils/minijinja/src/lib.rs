use minijinja::{Environment, UndefinedBehavior};

use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct JsExposedEnv {
    env: Environment<'static>,
}

fn annotate_error(err: minijinja::Error) -> JsError {
    JsError::new(&format!("{err:#}"))
}

#[wasm_bindgen]
impl JsExposedEnv {
    pub fn render(&self, template: &str, context: JsValue) -> Result<String, JsError> {
        let tmpl = self.env.get_template(template).map_err(annotate_error)?;
        let context: serde_json::Value = serde_wasm_bindgen::from_value(context)?;
        tmpl.render(context).map_err(annotate_error)
    }

    pub fn has_template(&self, template: &str) -> bool {
        self.env.get_template(template).is_ok()
    }
}

#[wasm_bindgen]
pub fn create_env(templates: JsValue) -> Result<JsExposedEnv, JsError> {
    let templates: HashMap<String, String> = serde_wasm_bindgen::from_value(templates)?;
    let mut env = Environment::new();
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    for (name, template) in templates.into_iter() {
        env.add_template_owned(name, template)
            .map_err(annotate_error)?;
    }
    Ok(JsExposedEnv { env })
}

#[cfg(test)]
mod tests {
    #![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    use super::*;
    use serde_json::json;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_basic_template_rendering() {
        // Create a simple template environment
        let templates = HashMap::from([("hello".to_string(), "Hello, {{ name }}!".to_string())]);
        let js_templates = serde_wasm_bindgen::to_value(&templates).unwrap();

        // Create the environment
        let env = create_env(js_templates).unwrap();

        // Create the context
        let context = serde_wasm_bindgen::to_value(&json!({
            "name": "World"
        }))
        .unwrap();

        // Render the template
        let result = env.render("hello", context).unwrap();
        assert_eq!(result, "Hello, World!");

        assert!(env.has_template("hello"));
        assert!(!env.has_template("goodbye"));
    }

    #[wasm_bindgen_test]
    fn test_template_error_handling() {
        // Test with invalid template syntax
        let templates = HashMap::from([(
            "invalid".to_string(),
            "Hello, {{ invalid syntax }}".to_string(),
        )]);
        let js_templates = serde_wasm_bindgen::to_value(&templates).unwrap();

        // Create should fail
        let env = create_env(js_templates);
        assert!(env.is_err());
    }
}
