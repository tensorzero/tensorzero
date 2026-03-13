import OpenAI from "openai";

const client = new OpenAI();

const response = await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [
    {
      role: "user",
      content: "Write a haiku about TensorZero.",
    },
  ],
});

console.log(JSON.stringify(response, null, 2));
