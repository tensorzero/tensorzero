import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:3000/openai/v1",
});

const response = await client.chat.completions.create({
  model: "tensorzero::function_name::generate_haiku",
  messages: [
    {
      role: "user",
      content: "Write a haiku about TensorZero.",
    },
  ],
});

console.log(JSON.stringify(response, null, 2));
