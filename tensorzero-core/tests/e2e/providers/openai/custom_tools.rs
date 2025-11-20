#![expect(clippy::print_stdout)]

use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_model_inference_clickhouse,
};

/// Test that OpenAI Responses API accepts and uses a custom tool with text format
#[tokio::test(flavor = "multi_thread")]
async fn test_responses_api_custom_tool_text_format() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "openai::responses::gpt-5-codex",
        "episode_id": episode_id,
        "input": {
            "messages": [{
                "role": "user",
                "content": "Generate Python code to print 'Hello, World!' using the code_generator tool."
            }]
        },
        "additional_tools": [{
            "type": "openai_custom",
            "name": "code_generator",
            "description": "Generates Python code snippets based on the given description",
            "format": {
                "type": "text"
            }
        }],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("response: {response_json:#}");

    // Check that we got tool calls
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content_blocks.is_empty());

    // Find a tool_call block (there should be at least one)
    let tool_call_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "tool_call")
        .collect();
    assert!(
        !tool_call_blocks.is_empty(),
        "Should have at least one tool call block"
    );

    // Check that one of the tool calls is to code_generator
    let code_generator_calls: Vec<&Value> = tool_call_blocks
        .into_iter()
        .filter(|block| block.get("raw_name").unwrap().as_str().unwrap() == "code_generator")
        .collect();
    assert_eq!(code_generator_calls.len(), 1);

    let tool_call = code_generator_calls.first().unwrap();
    let tool_call_id = tool_call.get("id").unwrap().as_str().unwrap();
    let raw_arguments = tool_call.get("raw_arguments").unwrap().as_str().unwrap();
    let name = tool_call.get("name").unwrap().as_str().unwrap();
    assert!(!raw_arguments.is_empty());
    assert!(!tool_call_id.is_empty());
    assert_eq!(name, "code_generator");

    // Check inference_id
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "tensorzero::default");

    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "openai::responses::gpt-5-codex");

    // Check that episode_id is correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    // Check that dynamic_tools contains the custom tool
    let dynamic_tools = result.get("dynamic_tools").unwrap().as_array().unwrap();
    assert_eq!(
        dynamic_tools.len(),
        1,
        "Should have exactly one custom tool"
    );

    // Parse the tool JSON and verify structure
    let tool_json: Value = serde_json::from_str(dynamic_tools[0].as_str().unwrap()).unwrap();
    assert_eq!(
        tool_json.get("type").unwrap().as_str().unwrap(),
        "openai_custom"
    );
    assert_eq!(
        tool_json.get("name").unwrap().as_str().unwrap(),
        "code_generator"
    );
    assert_eq!(
        tool_json.get("description").unwrap().as_str().unwrap(),
        "Generates Python code snippets based on the given description"
    );

    // Verify the format is text
    let format = tool_json.get("format").unwrap();
    assert_eq!(format.get("type").unwrap().as_str().unwrap(), "text");
}

/// Test that OpenAI accepts and uses a custom tool with text format
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_custom_tool_text_format() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "openai::gpt-5-mini",
        "episode_id": episode_id,
        "input": {
            "messages": [{
                "role": "user",
                "content": "Generate Python code to print 'Hello, World!' using the code_generator tool."
            }],
        },
        "additional_tools": [
            {
                "type": "openai_custom",
                "name": "code_generator",
                "description": "Generates Python code snippets based on requirements",
                "format": {
                    "type": "text"
                }
            }
        ],
        "allowed_tools": ["code_generator"],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json
        .get("inference_id")
        .and_then(|v| v.as_str())
        .expect("inference_id should be present in response");

    println!("Response: {response_json:#?}");

    // Check that we got tool calls in the response
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content_blocks.is_empty());

    // Find a tool_call block (there should be at least one)
    let tool_call_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "tool_call")
        .collect();
    assert!(
        !tool_call_blocks.is_empty(),
        "Should have at least one tool call block"
    );

    // Check that one of the tool calls is to code_generator
    let code_generator_calls: Vec<&Value> = tool_call_blocks
        .into_iter()
        .filter(|block| block.get("raw_name").unwrap().as_str().unwrap() == "code_generator")
        .collect();
    assert_eq!(code_generator_calls.len(), 1);

    let tool_call = code_generator_calls.first().unwrap();
    let tool_call_id = tool_call.get("id").unwrap().as_str().unwrap();
    let raw_arguments = tool_call.get("raw_arguments").unwrap().as_str().unwrap();
    let name = tool_call.get("name").unwrap().as_str().unwrap();
    assert!(!raw_arguments.is_empty());
    assert!(!tool_call_id.is_empty());
    assert_eq!(name, "code_generator");

    // Wait for ClickHouse to write the data
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Verify the tool was stored in ClickHouse ModelInference
    let clickhouse = get_clickhouse().await;
    let model_inference =
        select_model_inference_clickhouse(&clickhouse, inference_id.parse().unwrap())
            .await
            .unwrap();

    let raw_request = model_inference
        .get("raw_request")
        .unwrap()
        .as_str()
        .unwrap();

    println!("Raw request: {raw_request}");

    // Verify the custom tool appears in the raw request
    assert!(
        raw_request.contains("code_generator"),
        "Expected custom tool 'code_generator' in raw_request"
    );
    assert!(
        raw_request.contains("\"type\":\"custom\"") || raw_request.contains("\"type\": \"custom\""),
        "Expected custom tool type in raw_request, got: {raw_request}"
    );

    // Query ChatInference table to verify dynamic_tools
    let chat_inference =
        select_chat_inference_clickhouse(&clickhouse, inference_id.parse().unwrap())
            .await
            .unwrap();

    // Check that dynamic_tools contains the custom tool
    let dynamic_tools = chat_inference
        .get("dynamic_tools")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(
        dynamic_tools.len(),
        1,
        "Should have exactly one custom tool"
    );

    // Parse the tool JSON and verify structure
    let tool_json: Value = serde_json::from_str(dynamic_tools[0].as_str().unwrap()).unwrap();
    assert_eq!(
        tool_json.get("type").unwrap().as_str().unwrap(),
        "openai_custom"
    );
    assert_eq!(
        tool_json.get("name").unwrap().as_str().unwrap(),
        "code_generator"
    );
    assert_eq!(
        tool_json.get("description").unwrap().as_str().unwrap(),
        "Generates Python code snippets based on requirements"
    );

    // Verify the format is text
    let format = tool_json.get("format").unwrap();
    assert_eq!(format.get("type").unwrap().as_str().unwrap(), "text");
}

/// Test that OpenAI accepts and uses a custom tool with Lark grammar format
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_custom_tool_grammar_lark() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Simple arithmetic grammar in Lark format
    let lark_grammar = r#"
start: expr

expr: term ((ADD | SUB) term)*
term: factor ((MUL | DIV) factor)*
factor: NUMBER
      | "(" expr ")"

ADD: "+"
SUB: "-"
MUL: "*"
DIV: "/"

NUMBER: /\d+(\.\d+)?/

%import common.WS
%ignore WS
"#;

    let payload = json!({
        "model_name": "openai::gpt-5-mini",
        "episode_id": episode_id,
        "input": {
            "messages": [{
                "role": "user",
                "content": "Use the calculator tool to compute 5 + 3 * 2"
            }],
        },
        "additional_tools": [
            {
                "type": "openai_custom",
                "name": "calculator",
                "description": "Evaluates arithmetic expressions",
                "format": {
                    "type": "grammar",
                    "grammar": {
                    "syntax": "lark",
                    "definition": lark_grammar
                }}
            }
        ],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json
        .get("inference_id")
        .and_then(|v| v.as_str())
        .expect("inference_id should be present in response");

    println!("Response: {response_json:#?}");

    // Check that we got tool calls in the response
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content_blocks.is_empty());

    // Find a tool_call block (there should be at least one)
    let tool_call_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "tool_call")
        .collect();
    assert!(
        !tool_call_blocks.is_empty(),
        "Should have at least one tool call block"
    );

    // Check that one of the tool calls is to calculator
    let calculator_calls: Vec<&Value> = tool_call_blocks
        .into_iter()
        .filter(|block| block.get("raw_name").unwrap().as_str().unwrap() == "calculator")
        .collect();
    assert_eq!(calculator_calls.len(), 1);

    let tool_call = calculator_calls.first().unwrap();
    let tool_call_id = tool_call.get("id").unwrap().as_str().unwrap();
    let raw_arguments = tool_call.get("raw_arguments").unwrap().as_str().unwrap();
    let name = tool_call.get("name").unwrap().as_str().unwrap();
    assert!(!raw_arguments.is_empty());
    assert!(!tool_call_id.is_empty());
    assert_eq!(name, "calculator");

    // Wait for ClickHouse to write the data
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Verify the tool was stored in ClickHouse ModelInference
    let clickhouse = get_clickhouse().await;
    let model_inference =
        select_model_inference_clickhouse(&clickhouse, inference_id.parse().unwrap())
            .await
            .unwrap();

    let raw_request = model_inference
        .get("raw_request")
        .unwrap()
        .as_str()
        .unwrap();

    // Verify the custom tool with grammar appears in the raw request
    assert!(
        raw_request.contains("calculator"),
        "Expected custom tool 'calculator' in raw_request"
    );
    assert!(
        raw_request.contains("\"type\":\"custom\"") || raw_request.contains("\"type\": \"custom\""),
        "Expected custom tool type in raw_request"
    );
    assert!(
        raw_request.contains("lark"),
        "Expected lark grammar syntax in raw_request"
    );

    // Query ChatInference table to verify dynamic_tools
    let chat_inference =
        select_chat_inference_clickhouse(&clickhouse, inference_id.parse().unwrap())
            .await
            .unwrap();

    // Check that dynamic_tools contains the custom tool
    let dynamic_tools = chat_inference
        .get("dynamic_tools")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(
        dynamic_tools.len(),
        1,
        "Should have exactly one custom tool"
    );

    // Parse the tool JSON and verify structure
    let tool_json: Value = serde_json::from_str(dynamic_tools[0].as_str().unwrap()).unwrap();
    assert_eq!(
        tool_json.get("type").unwrap().as_str().unwrap(),
        "openai_custom"
    );
    assert_eq!(
        tool_json.get("name").unwrap().as_str().unwrap(),
        "calculator"
    );
    assert_eq!(
        tool_json.get("description").unwrap().as_str().unwrap(),
        "Evaluates arithmetic expressions"
    );

    // Verify the format is grammar with lark syntax
    let format = tool_json.get("format").unwrap();
    assert_eq!(format.get("type").unwrap().as_str().unwrap(), "grammar");
    let grammar = format.get("grammar").unwrap();
    assert_eq!(grammar.get("syntax").unwrap().as_str().unwrap(), "lark");
    assert!(grammar
        .get("definition")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("start: expr"));
}

/// Test that OpenAI accepts and uses a custom tool with Regex grammar format
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_custom_tool_grammar_regex() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Regex pattern for phone numbers in format XXX-XXX-XXXX
    let regex_pattern = r"^\d{3}-\d{3}-\d{4}$";

    let payload = json!({
        "model_name": "openai::gpt-5-mini",
        "episode_id": episode_id,
        "input": {
            "messages": [{
                "role": "user",
                "content": "Use the phone_formatter tool to format the phone number 415-555-0123"
            }],
        },
        "additional_tools": [
            {
                "type": "openai_custom",
                "name": "phone_formatter",
                "description": "Formats phone numbers in the standard XXX-XXX-XXXX format",
                "format": {
                    "type": "grammar",
                    "grammar": {
                    "syntax": "regex",
                    "definition": regex_pattern
                }}
            }
        ],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json
        .get("inference_id")
        .and_then(|v| v.as_str())
        .expect("inference_id should be present in response");

    println!("Response: {response_json:#?}");

    // Check that we got tool calls in the response
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content_blocks.is_empty());

    // Find a tool_call block (there should be at least one)
    let tool_call_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "tool_call")
        .collect();
    assert!(
        !tool_call_blocks.is_empty(),
        "Should have at least one tool call block"
    );

    // Check that one of the tool calls is to phone_formatter
    let phone_formatter_calls: Vec<&Value> = tool_call_blocks
        .into_iter()
        .filter(|block| block.get("raw_name").unwrap().as_str().unwrap() == "phone_formatter")
        .collect();
    assert_eq!(phone_formatter_calls.len(), 1);

    let tool_call = phone_formatter_calls.first().unwrap();
    let tool_call_id = tool_call.get("id").unwrap().as_str().unwrap();
    let raw_arguments = tool_call.get("raw_arguments").unwrap().as_str().unwrap();
    let name = tool_call.get("name").unwrap().as_str().unwrap();
    assert!(!raw_arguments.is_empty());
    assert!(!tool_call_id.is_empty());
    assert_eq!(name, "phone_formatter");

    // Wait for ClickHouse to write the data
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Verify the tool was stored in ClickHouse ModelInference
    let clickhouse = get_clickhouse().await;
    let model_inference =
        select_model_inference_clickhouse(&clickhouse, inference_id.parse().unwrap())
            .await
            .unwrap();

    let raw_request = model_inference
        .get("raw_request")
        .unwrap()
        .as_str()
        .unwrap();

    // Verify the custom tool with regex grammar appears in the raw request
    assert!(
        raw_request.contains("phone_formatter"),
        "Expected custom tool 'phone_formatter' in raw_request"
    );
    assert!(
        raw_request.contains("\"type\":\"custom\"") || raw_request.contains("\"type\": \"custom\""),
        "Expected custom tool type in raw_request"
    );
    assert!(
        raw_request.contains("regex"),
        "Expected regex grammar syntax in raw_request"
    );

    // Query ChatInference table to verify dynamic_tools
    let chat_inference =
        select_chat_inference_clickhouse(&clickhouse, inference_id.parse().unwrap())
            .await
            .unwrap();

    // Check that dynamic_tools contains the custom tool
    let dynamic_tools = chat_inference
        .get("dynamic_tools")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(
        dynamic_tools.len(),
        1,
        "Should have exactly one custom tool"
    );

    // Parse the tool JSON and verify structure
    let tool_json: Value = serde_json::from_str(dynamic_tools[0].as_str().unwrap()).unwrap();
    assert_eq!(
        tool_json.get("type").unwrap().as_str().unwrap(),
        "openai_custom"
    );
    assert_eq!(
        tool_json.get("name").unwrap().as_str().unwrap(),
        "phone_formatter"
    );
    assert_eq!(
        tool_json.get("description").unwrap().as_str().unwrap(),
        "Formats phone numbers in the standard XXX-XXX-XXXX format"
    );

    // Verify the format is grammar with regex syntax
    let format = tool_json.get("format").unwrap();
    assert_eq!(format.get("type").unwrap().as_str().unwrap(), "grammar");
    let grammar = format.get("grammar").unwrap();
    assert_eq!(grammar.get("syntax").unwrap().as_str().unwrap(), "regex");
    assert_eq!(
        grammar.get("definition").unwrap().as_str().unwrap(),
        regex_pattern
    );
}

/// Test that standard function tools and custom tools can be used together
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_mixed_function_and_custom_tools() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "gpt-5-mini",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "Weather Assistant"},
            "messages": [{
                "role": "user",
                "content": "Get the temperature in Paris, France."
            }],
        },
        "additional_tools": [
            {
                "type": "openai_custom",
                "name": "weather_report",
                "description": "Generates a formatted weather report from temperature data",
                "format": {
                    "type": "text"
                }
            }
        ],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json
        .get("inference_id")
        .and_then(|v| v.as_str())
        .expect("inference_id should be present in response");

    println!("Response: {response_json:#?}");

    // Check that we got content in the response
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content_blocks.is_empty());

    // Check if we got tool calls (LLM behavior can be non-deterministic)
    let tool_call_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "tool_call")
        .collect();

    // If we got tool calls, verify they're to the expected tools
    if tool_call_blocks.is_empty() {
        println!("Note: LLM did not make any tool calls in this run (non-deterministic behavior)");
    } else {
        println!("LLM made {} tool call(s)", tool_call_blocks.len());
        for tool_call in &tool_call_blocks {
            let name = tool_call.get("name").unwrap().as_str().unwrap();
            println!("  - Tool called: {name}");
            assert!(
                name == "get_temperature" || name == "weather_report",
                "Tool call should be to either get_temperature or weather_report, got: {name}"
            );
        }
    }

    // Wait for ClickHouse to write the data
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Verify both tool types were stored in ClickHouse ModelInference
    let clickhouse = get_clickhouse().await;
    let model_inference =
        select_model_inference_clickhouse(&clickhouse, inference_id.parse().unwrap())
            .await
            .unwrap();

    let raw_request = model_inference
        .get("raw_request")
        .unwrap()
        .as_str()
        .unwrap();

    // Verify the custom tool appears in the raw request
    assert!(
        raw_request.contains("weather_report"),
        "Expected custom tool 'weather_report' in raw_request"
    );

    // Verify the function tool (get_temperature) also appears
    assert!(
        raw_request.contains("get_temperature"),
        "Expected function tool 'get_temperature' in raw_request"
    );

    // Both tool types should be present
    assert!(
        raw_request.contains("\"type\":\"custom\"") || raw_request.contains("\"type\": \"custom\""),
        "Expected custom tool type in raw_request"
    );
    assert!(
        raw_request.contains("\"type\":\"function\"")
            || raw_request.contains("\"type\": \"function\""),
        "Expected function tool type in raw_request"
    );

    // Query ChatInference table to verify dynamic_tools
    let chat_inference =
        select_chat_inference_clickhouse(&clickhouse, inference_id.parse().unwrap())
            .await
            .unwrap();

    // Check that dynamic_tools contains both the custom tool and function tool
    let dynamic_tools = chat_inference
        .get("dynamic_tools")
        .unwrap()
        .as_array()
        .unwrap();

    println!("Found {} tool(s) in dynamic_tools:", dynamic_tools.len());
    for (i, tool_str) in dynamic_tools.iter().enumerate() {
        let tool_json: Value = serde_json::from_str(tool_str.as_str().unwrap()).unwrap();
        println!(
            "  Tool {}: type={}, name={}",
            i,
            tool_json.get("type").unwrap().as_str().unwrap(),
            tool_json.get("name").unwrap().as_str().unwrap()
        );
    }

    // Note: dynamic_tools only contains additional_tools passed at inference time,
    // not the function's configured tools (which are stored elsewhere)
    assert_eq!(
        dynamic_tools.len(),
        1,
        "Should have exactly one custom tool (weather_report)"
    );

    // Parse and verify the custom tool
    let tool_json: Value = serde_json::from_str(dynamic_tools[0].as_str().unwrap()).unwrap();
    assert_eq!(
        tool_json.get("type").unwrap().as_str().unwrap(),
        "openai_custom"
    );
    assert_eq!(
        tool_json.get("name").unwrap().as_str().unwrap(),
        "weather_report"
    );
    assert_eq!(
        tool_json.get("description").unwrap().as_str().unwrap(),
        "Generates a formatted weather report from temperature data"
    );
    let format = tool_json.get("format").unwrap();
    assert_eq!(format.get("type").unwrap().as_str().unwrap(), "text");
}

/// Test that OpenAI Responses API accepts and uses a custom tool with Lark grammar format
#[tokio::test(flavor = "multi_thread")]
async fn test_responses_api_custom_tool_grammar_lark() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Simple arithmetic grammar in Lark format
    let lark_grammar = r#"
start: expr

expr: term ((ADD | SUB) term)*
term: factor ((MUL | DIV) factor)*
factor: NUMBER
      | "(" expr ")"

ADD: "+"
SUB: "-"
MUL: "*"
DIV: "/"

NUMBER: /\d+(\.\d+)?/

%import common.WS
%ignore WS
"#;

    let payload = json!({
        "model_name": "openai::responses::gpt-5-codex",
        "episode_id": episode_id,
        "input": {
            "messages": [{
                "role": "user",
                "content": "Use the calculator tool to compute 5 + 3 * 2"
            }]
        },
        "additional_tools": [{
            "type": "openai_custom",
            "name": "calculator",
            "description": "Evaluates arithmetic expressions",
            "format": {
                "type": "grammar",
                "grammar": {
                    "syntax": "lark",
                    "definition": lark_grammar
                }
            }
        }],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("response: {response_json:#}");

    // Check that we got tool calls
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content_blocks.is_empty());

    // Find a tool_call block (there should be at least one)
    let tool_call_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "tool_call")
        .collect();
    assert!(
        !tool_call_blocks.is_empty(),
        "Should have at least one tool call block"
    );

    // Check that one of the tool calls is to calculator
    let calculator_calls: Vec<&Value> = tool_call_blocks
        .into_iter()
        .filter(|block| block.get("raw_name").unwrap().as_str().unwrap() == "calculator")
        .collect();
    assert_eq!(calculator_calls.len(), 1);

    let tool_call = calculator_calls.first().unwrap();
    let tool_call_id = tool_call.get("id").unwrap().as_str().unwrap();
    let raw_arguments = tool_call.get("raw_arguments").unwrap().as_str().unwrap();
    let name = tool_call.get("name").unwrap().as_str().unwrap();
    assert!(!raw_arguments.is_empty());
    assert!(!tool_call_id.is_empty());
    assert_eq!(name, "calculator");

    // Check inference_id
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "tensorzero::default");

    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "openai::responses::gpt-5-codex");

    // Check that episode_id is correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    // Check that dynamic_tools contains the custom tool
    let dynamic_tools = result.get("dynamic_tools").unwrap().as_array().unwrap();
    assert_eq!(
        dynamic_tools.len(),
        1,
        "Should have exactly one custom tool"
    );

    // Parse the tool JSON and verify structure
    let tool_json: Value = serde_json::from_str(dynamic_tools[0].as_str().unwrap()).unwrap();
    assert_eq!(
        tool_json.get("type").unwrap().as_str().unwrap(),
        "openai_custom"
    );
    assert_eq!(
        tool_json.get("name").unwrap().as_str().unwrap(),
        "calculator"
    );
    assert_eq!(
        tool_json.get("description").unwrap().as_str().unwrap(),
        "Evaluates arithmetic expressions"
    );

    // Verify the format is grammar with lark syntax
    let format = tool_json.get("format").unwrap();
    assert_eq!(format.get("type").unwrap().as_str().unwrap(), "grammar");
    let grammar = format.get("grammar").unwrap();
    assert_eq!(grammar.get("syntax").unwrap().as_str().unwrap(), "lark");
    assert!(grammar
        .get("definition")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("start: expr"));
}

/// Test that OpenAI Responses API accepts and uses a custom tool with Regex grammar format
#[tokio::test(flavor = "multi_thread")]
async fn test_responses_api_custom_tool_grammar_regex() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Regex pattern for phone numbers in format XXX-XXX-XXXX
    let regex_pattern = r"^\d{3}-\d{3}-\d{4}$";

    let payload = json!({
        "model_name": "openai::responses::gpt-5-codex",
        "episode_id": episode_id,
        "input": {
            "messages": [{
                "role": "user",
                "content": "Use the phone_formatter tool to format the phone number 415-555-0123"
            }]
        },
        "additional_tools": [{
            "type": "openai_custom",
            "name": "phone_formatter",
            "description": "Formats phone numbers in the standard XXX-XXX-XXXX format",
            "format": {
                "type": "grammar",
                "grammar": {
                    "syntax": "regex",
                    "definition": regex_pattern
                }
            }
        }],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("response: {response_json:#}");

    // Check that we got tool calls
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content_blocks.is_empty());

    // Find a tool_call block (there should be at least one)
    let tool_call_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "tool_call")
        .collect();
    assert!(
        !tool_call_blocks.is_empty(),
        "Should have at least one tool call block"
    );

    // Check that one of the tool calls is to phone_formatter
    let phone_formatter_calls: Vec<&Value> = tool_call_blocks
        .into_iter()
        .filter(|block| block.get("raw_name").unwrap().as_str().unwrap() == "phone_formatter")
        .collect();
    assert_eq!(phone_formatter_calls.len(), 1);

    let tool_call = phone_formatter_calls.first().unwrap();
    let tool_call_id = tool_call.get("id").unwrap().as_str().unwrap();
    let raw_arguments = tool_call.get("raw_arguments").unwrap().as_str().unwrap();
    let name = tool_call.get("name").unwrap().as_str().unwrap();
    assert!(!raw_arguments.is_empty());
    assert!(!tool_call_id.is_empty());
    assert_eq!(name, "phone_formatter");

    // Check inference_id
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "tensorzero::default");

    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "openai::responses::gpt-5-codex");

    // Check that episode_id is correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    // Check that dynamic_tools contains the custom tool
    let dynamic_tools = result.get("dynamic_tools").unwrap().as_array().unwrap();
    assert_eq!(
        dynamic_tools.len(),
        1,
        "Should have exactly one custom tool"
    );

    // Parse the tool JSON and verify structure
    let tool_json: Value = serde_json::from_str(dynamic_tools[0].as_str().unwrap()).unwrap();
    assert_eq!(
        tool_json.get("type").unwrap().as_str().unwrap(),
        "openai_custom"
    );
    assert_eq!(
        tool_json.get("name").unwrap().as_str().unwrap(),
        "phone_formatter"
    );
    assert_eq!(
        tool_json.get("description").unwrap().as_str().unwrap(),
        "Formats phone numbers in the standard XXX-XXX-XXXX format"
    );

    // Verify the format is grammar with regex syntax
    let format = tool_json.get("format").unwrap();
    assert_eq!(format.get("type").unwrap().as_str().unwrap(), "grammar");
    let grammar = format.get("grammar").unwrap();
    assert_eq!(grammar.get("syntax").unwrap().as_str().unwrap(), "regex");
    assert_eq!(
        grammar.get("definition").unwrap().as_str().unwrap(),
        regex_pattern
    );
}

/// Test that standard function tools and custom tools can be used together with Responses API
#[tokio::test(flavor = "multi_thread")]
async fn test_responses_api_mixed_function_and_custom_tools() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "gpt-5-mini",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "Weather Assistant"},
            "messages": [{
                "role": "user",
                "content": "Get the temperature in Paris, France."
            }],
        },
        "additional_tools": [{
            "type": "openai_custom",
            "name": "weather_report",
            "description": "Generates a formatted weather report from temperature data",
            "format": {
                "type": "text"
            }
        }],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("response: {response_json:#}");

    // Check that we got content in the response
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content_blocks.is_empty());

    // Check if we got tool calls (LLM behavior can be non-deterministic)
    let tool_call_blocks: Vec<&Value> = content_blocks
        .iter()
        .filter(|block| block.get("type").unwrap().as_str().unwrap() == "tool_call")
        .collect();

    // If we got tool calls, verify they're to the expected tools
    if tool_call_blocks.is_empty() {
        println!("Note: LLM did not make any tool calls in this run (non-deterministic behavior)");
    } else {
        println!("LLM made {} tool call(s)", tool_call_blocks.len());
        for tool_call in &tool_call_blocks {
            let name = tool_call.get("name").unwrap().as_str().unwrap();
            println!("  - Tool called: {name}");
            assert!(
                name == "get_temperature" || name == "weather_report",
                "Tool call should be to either get_temperature or weather_report, got: {name}"
            );
        }
    }

    // Check inference_id
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "gpt-5-mini");

    // Check that episode_id is correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    // Check that dynamic_tools contains the custom tool
    let dynamic_tools = result.get("dynamic_tools").unwrap().as_array().unwrap();

    println!("Found {} tool(s) in dynamic_tools:", dynamic_tools.len());
    for (i, tool_str) in dynamic_tools.iter().enumerate() {
        let tool_json: Value = serde_json::from_str(tool_str.as_str().unwrap()).unwrap();
        println!(
            "  Tool {}: type={}, name={}",
            i,
            tool_json.get("type").unwrap().as_str().unwrap(),
            tool_json.get("name").unwrap().as_str().unwrap()
        );
    }

    // Note: dynamic_tools only contains additional_tools passed at inference time,
    // not the function's configured tools (which are stored elsewhere)
    assert_eq!(
        dynamic_tools.len(),
        1,
        "Should have exactly one custom tool (weather_report)"
    );

    // Parse and verify the custom tool
    let tool_json: Value = serde_json::from_str(dynamic_tools[0].as_str().unwrap()).unwrap();
    assert_eq!(
        tool_json.get("type").unwrap().as_str().unwrap(),
        "openai_custom"
    );
    assert_eq!(
        tool_json.get("name").unwrap().as_str().unwrap(),
        "weather_report"
    );
    assert_eq!(
        tool_json.get("description").unwrap().as_str().unwrap(),
        "Generates a formatted weather report from temperature data"
    );
    let format = tool_json.get("format").unwrap();
    assert_eq!(format.get("type").unwrap().as_str().unwrap(), "text");
}

/// Test that Anthropic rejects a custom tool with text format (400)
#[tokio::test(flavor = "multi_thread")]
async fn test_non_openai_custom_tool_text_format() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model_name": "anthropic::claude-sonnet-4-5",
        "episode_id": episode_id,
        "input": {
            "messages": [{
                "role": "user",
                "content": "Generate Python code to print 'Hello, World!' using the code_generator tool."
            }],
        },
        "additional_tools": [
            {
                "type": "openai_custom",
                "name": "code_generator",
                "description": "Generates Python code snippets based on requirements",
                "format": {
                    "type": "text"
                }
            }
        ],
        "allowed_tools": ["code_generator"],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let text = response.text().await.unwrap();
    assert!(text.contains("OpenAI custom tools are not supported by this provider"));
}
