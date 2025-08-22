import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:3000/openai/v1",
});

const response = await client.chat.completions.create({
  model: "tensorzero::model_name::openai::gpt-4.1",
  // or: model: "tensorzero::model_name::anthropic::claude-sonnet-4-20250514",
  // or: Google, AWS, Azure, xAI, vLLM, Ollama, and many more
  messages: [
    {
      role: "user",
      content: "Tell me a fun fact.",
    },
  ],
});

console.dir(response, { depth: null });
