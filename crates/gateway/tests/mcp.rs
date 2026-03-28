#![expect(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod common;

use common::start_gateway_on_random_port;
use rmcp::ServiceExt;

/// Parse the MCP server address from gateway startup logs.
///
/// Looks for a line containing "MCP server listening on <addr>" and extracts the address.
fn parse_mcp_address(output: &[String]) -> std::net::SocketAddr {
    for line in output {
        if let Some(rest) = line.split("MCP server listening on ").nth(1) {
            let addr_str = rest.split('"').next().unwrap();
            return addr_str.parse().expect("Failed to parse MCP address");
        }
    }
    panic!("MCP server listening line not found in output: {output:?}");
}

#[tokio::test]
async fn test_mcp_list_tools() {
    let child_data = start_gateway_on_random_port(
        r#"
        [gateway.mcp]
        enabled = true
        bind_address = "0.0.0.0:0"
        "#,
        None,
    )
    .await;

    let mcp_addr = parse_mcp_address(&child_data.output);
    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp_addr.port());

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

#[tokio::test]
async fn test_mcp_disabled_by_default() {
    // No [gateway.mcp] section — MCP server should not start
    let child_data = start_gateway_on_random_port("", None).await;

    let has_mcp_line = child_data
        .output
        .iter()
        .any(|line| line.contains("MCP server listening on"));
    assert!(
        !has_mcp_line,
        "MCP server should not start when not configured"
    );

    // Verify the status line says disabled
    let mcp_disabled = child_data
        .output
        .iter()
        .any(|line| line.contains("MCP Server: disabled"));
    assert!(
        mcp_disabled,
        "Expected 'MCP Server: disabled' in output: {:?}",
        child_data.output
    );
}
