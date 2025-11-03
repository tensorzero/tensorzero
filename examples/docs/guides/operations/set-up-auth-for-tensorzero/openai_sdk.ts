import OpenAI from "openai";

// Good request

const client = new OpenAI({
  apiKey: process.env.TENSORZERO_API_KEY,
  baseURL: "http://localhost:3000/openai/v1",
});

const response = await client.chat.completions.create({
  model: "tensorzero::model_name::openai::gpt-5-mini",
  messages: [
    {
      role: "user",
      content: "Tell me a fun fact.",
    },
  ],
});

console.dir(response, { depth: null });

// Bad request

const badClient = new OpenAI({
  apiKey: "sk-t0-evilevilevil-hackerhackerhackerhackerhackerhackerhackerhacker",
  baseURL: "http://localhost:3000/openai/v1",
});

try {
  const badResponse = await badClient.chat.completions.create({
    model: "tensorzero::model_name::openai::gpt-5-mini",
    messages: [
      {
        role: "user",
        content: "Tell me a fun fact.",
      },
    ],
  });

  console.log(badResponse);
} catch (e) {
  console.log(`Expected error occurred: ${e}`);
}
