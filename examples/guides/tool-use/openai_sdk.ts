import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:3000/openai/v1",
});

const messages: any[] = [
  { role: "user", content: "What is the weather in Tokyo (°F)?" },
];

const response = await client.chat.completions.create({
  model: "tensorzero::function_name::weather_chatbot",
  messages,
});

console.log(JSON.stringify(response, null, 2));

// The model can return multiple content blocks, including tool calls
// In a real application, you'd be stricter about validating the response
const toolCalls = response.choices[0].message.tool_calls;
if (!toolCalls || toolCalls.length !== 1) {
  throw new Error("Expected the model to return exactly one tool call");
}

// Add the tool call to the message history
messages.push(response.choices[0].message);

// Pretend we've called the tool and got a response
messages.push({
  role: "tool",
  tool_call_id: toolCalls[0].id,
  content: "70", // imagine it's 70°F in Tokyo
});

const response2 = await client.chat.completions.create({
  model: "tensorzero::function_name::weather_chatbot",
  messages,
});

console.log(JSON.stringify(response2, null, 2));
