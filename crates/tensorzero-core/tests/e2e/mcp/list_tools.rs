use googletest::prelude::*;

use super::common::McpTestClient;

#[gtest]
#[tokio::test]
async fn test_mcp_list_tools() {
    let mcp = McpTestClient::connect().await;
    let tools = mcp.list_tools().await;

    let tool_names: Vec<String> = tools.iter().map(|t| t.name.to_string()).collect();
    // The MCP server registers all SimpleTool implementations from autopilot-tools.
    // Verify that the core tools are present and that there are more than just two.
    expect_that!(
        tool_names,
        contains_each![eq("list_inferences"), eq("get_inferences")]
    );
    expect_that!(tool_names.len(), gt(2));

    for tool in &tools {
        expect_that!(tool.description.as_deref(), some(not(eq(""))),);
        expect_that!(tool.input_schema.is_empty(), eq(false));
    }

    mcp.cancel().await;
}
