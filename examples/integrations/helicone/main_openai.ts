import OpenAI from "openai";
import * as process from "process";

// Retrieve the Helicone API key from the environment variable
const heliconeApiKey = process.env.HELICONE_API_KEY;
if (!heliconeApiKey) {
  throw new Error("HELICONE_API_KEY is not set");
}

// Build our OpenAI client
const client = new OpenAI({
  baseURL: "http://localhost:3000/openai/v1", // our local TensorZero Gateway
});

// Test our `helicone_gpt_4o_mini` model
const response_openai = await client.chat.completions.create({
  model: "tensorzero::model_name::helicone_gpt_4o_mini",
  messages: [{ role: "user", content: "Who is the CEO of OpenAI?" }],
  // @ts-expect-error: custom TensorZero property
  "tensorzero::extra_headers": [
    {
      model_name: "helicone_gpt_4o_mini",
      provider_name: "helicone",
      name: "Helicone-Auth",
      value: `Bearer ${heliconeApiKey}`,
    },
  ],
});

console.log(response_openai);

// Test our `helicone_grok_3` model
const response_xai = await client.chat.completions.create({
  model: "tensorzero::model_name::helicone_grok_3",
  messages: [{ role: "user", content: "Who is the CEO of xAI?" }],
  // @ts-expect-error: custom TensorZero property
  "tensorzero::extra_headers": [
    {
      model_name: "helicone_grok_3",
      provider_name: "helicone",
      name: "Helicone-Auth",
      value: `Bearer ${heliconeApiKey}`,
    },
  ],
});

console.log(response_xai);
