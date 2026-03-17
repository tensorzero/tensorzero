from openai import OpenAI  # or AsyncOpenAI

client = OpenAI(
    base_url="http://localhost:3000/openai/v1",
)

messages = [{"role": "user", "content": "What is the weather in Tokyo (°F)?"}]

response = client.chat.completions.create(
    model="tensorzero::function_name::weather_chatbot",
    messages=messages,
)

print(response)

# The model can return multiple content blocks, including tool calls
# In a real application, you'd be stricter about validating the response
tool_calls = response.choices[0].message.tool_calls
assert len(tool_calls) == 1, "Expected the model to return exactly one tool call"

# Add the tool call to the message history
messages.append(response.choices[0].message)

# Pretend we've called the tool and got a response
messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_calls[0].id,
        "content": "70",  # imagine it's 70°F in Tokyo
    }
)

response = client.chat.completions.create(
    model="tensorzero::function_name::weather_chatbot",
    messages=messages,
)

print(response)
