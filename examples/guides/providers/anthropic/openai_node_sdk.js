import OpenAI from "openai";

// Run `docker compose up` to start the TensorZero Gateway on localhost:3000...
const client = new OpenAI({
  baseURL: "http://localhost:3000/openai/v1",
});

const response = await client.chat.completions.create({
  model: "tensorzero::model_name::anthropic::claude-3-5-haiku-20241022",
  messages: [
    {
      role: "user",
      content: "What is the capital of Japan?",
    },
  ],
});

console.log(response);
