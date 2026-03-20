import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:3000/openai/v1",
  apiKey: "not-used",
});

// NB: OpenAI web search can take up to a minute to complete

const response = await client.chat.completions.create({
  // The model is defined in config/tensorzero.toml
  model: "tensorzero::model_name::gpt-5-mini-responses-web-search",
  messages: [
    {
      role: "user",
      content: "What is the current population of Japan?",
    },
  ],
});

console.dir(response, { depth: null });
