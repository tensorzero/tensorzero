mod common;

use common::start_gateway_on_random_port;
use rmcp::ServiceExt;

#[tokio::test]
async fn test_mcp_list_tools() {
    let child_data = start_gateway_on_random_port("", None).await;

    let mcp_url = format!("http://127.0.0.1:{}/mcp", child_data.addr.port());

    // Connect an MCP client
    let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(mcp_url.as_str());
    let client = ().serve(transport).await.expect("Failed to connect MCP client");

    // Call list_tools
    let result = client
        .list_tools(Default::default())
        .await
        .expect("list_tools failed");

    // Verify expected tools are present
    let tool_names: Vec<&str> = result.tools.iter().map(|t| &*t.name).collect();
    assert!(
        tool_names.contains(&"list_inferences"),
        "Expected `list_inferences` tool, got: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"get_inferences"),
        "Expected `get_inferences` tool, got: {tool_names:?}"
    );

    // Verify tools have descriptions
    for tool in &result.tools {
        assert!(
            tool.description.is_some(),
            "Tool `{}` should have a description",
            tool.name
        );
    }

    // Verify tools have input schemas
    for tool in &result.tools {
        assert!(
            !tool.input_schema.is_empty(),
            "Tool `{}` should have an input schema",
            tool.name
        );
    }

    let _ = client.cancel().await;
}
