import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tensorzero import AsyncTensorZeroGateway, Message, Text, ToolCall, ToolResult


async def main():
    # Define the MCP server
    server_params = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "--with",
            "mcp-clickhouse",
            "--python",
            "3.13",
            "mcp-clickhouse",
            "2> /dev/null",
        ],
        env={
            "CLICKHOUSE_HOST": "localhost",
            "CLICKHOUSE_PORT": "8123",
            "CLICKHOUSE_USER": "chuser",
            "CLICKHOUSE_DATABASE": "tensorzero",
            "CLICKHOUSE_PASSWORD": "chpassword",
            "CLICKHOUSE_SECURE": "false",  # running locally
        },
    )

    # Initialize the TensorZero Gateway
    async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as t0:
        # Initialize the MCP client
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(
                read,
                write,
            ) as session:
                await session.initialize()

                # Load the tools from the MCP server
                mcp_tools = await session.list_tools()
                mcp_tools = [tool.model_dump() for tool in mcp_tools.tools]

                # Convert MCP tool format to TensorZero tool format
                t0_tools = [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["inputSchema"],
                    }
                    for tool in mcp_tools
                ]

                # Initialize the conversation
                messages = []
                episode_id = None

                while True:
                    # If the last message is not a tool call, prompt the user for input
                    if not messages or messages[-1]["role"] == "assistant":
                        message = input("\n[User]\n")
                        messages.append(Message(role="user", content=message))

                    # Run the inference
                    response = await t0.inference(
                        function_name="clickhouse_copilot",
                        episode_id=episode_id,
                        input={
                            "messages": messages,
                        },
                        additional_tools=t0_tools,
                    )

                    # Update the episode ID
                    episode_id = response.episode_id

                    # Add the assistant message to the conversation
                    assistant_message = Message(role="assistant", content=response.content)

                    messages.append(assistant_message)

                    # Process the tool calls
                    tool_results = []
                    for content_block in response.content:
                        if isinstance(content_block, Text):
                            print("\n[Assistant]")
                            print(content_block.text)
                        elif isinstance(content_block, ToolCall):
                            print(f"\n[Tool Call: {content_block.raw_name}]")
                            print(content_block.raw_arguments)

                            # NB: depending on the use case, you might want to await multiple tool calls in parallel
                            mcp_tool_result = await session.call_tool(content_block.name, content_block.arguments)

                            # MCP servers technically could return multiple blocks, so we concatenate them
                            t0_tool_result_text = "\n".join([block.text for block in mcp_tool_result.content])

                            print("\n[Tool Result]")
                            print(t0_tool_result_text)

                            t0_tool_result = ToolResult(
                                name=content_block.name,
                                result=t0_tool_result_text,
                                id=content_block.id,
                            )
                            tool_results.append(t0_tool_result)

                    # If there are tool results, add them to the conversation
                    if tool_results:
                        user_message = Message(
                            role="user",
                            content=tool_results,
                        )
                        messages.append(user_message)


if __name__ == "__main__":
    asyncio.run(main())
