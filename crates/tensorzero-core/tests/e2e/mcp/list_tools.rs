use googletest::prelude::*;

use super::common::McpTestClient;

#[gtest]
#[tokio::test]
async fn test_mcp_list_tools() {
    let mcp = McpTestClient::connect().await;
    let tools = mcp.list_tools().await;

    let tool_names: Vec<String> = tools.iter().map(|t| t.name.to_string()).collect();
    expect_that!(
        tool_names,
        unordered_elements_are![eq("list_inferences"), eq("get_inferences")]
    );

    for tool in &tools {
        expect_that!(tool.description.as_deref(), some(not(eq(""))),);
        expect_that!(tool.input_schema.is_empty(), eq(false));
    }

    mcp.cancel().await;
}
