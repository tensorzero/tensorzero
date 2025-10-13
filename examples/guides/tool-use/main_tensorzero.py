from tensorzero import TensorZeroGateway, ToolCall  # or AsyncTensorZeroGateway

with TensorZeroGateway.build_http(
    gateway_url="http://localhost:3000",
) as t0:
    messages = [{"role": "user", "content": "What is the weather in Tokyo (°F)?"}]

    response = t0.inference(
        function_name="weather_chatbot",
        input={"messages": messages},
    )

    print(response)

    # The model can return multiple content blocks, including tool calls
    # In a real application, you'd be stricter about validating the response
    tool_calls = [content_block for content_block in response.content if isinstance(content_block, ToolCall)]
    assert len(tool_calls) == 1, "Expected the model to return exactly one tool call"

    # Add the tool call to the message history
    messages.append(
        {
            "role": "assistant",
            "content": response.content,
        }
    )

    # Pretend we've called the tool and got a response
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "id": tool_calls[0].id,
                    "name": tool_calls[0].name,
                    "result": "70",  # imagine it's 70°F in Tokyo
                }
            ],
        }
    )

    response = t0.inference(
        function_name="weather_chatbot",
        input={"messages": messages},
    )

    print(response)
