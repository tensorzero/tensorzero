import OpenAI from "openai";
import { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { ResponseFormatJSONSchema } from "openai/resources/shared";

const messages: ChatCompletionMessageParam[] = [
  {
    role: "system",
    content:
      "You are a helpful math tutor. Guide the user through the solution step by step.",
  },
  { role: "user", content: "how can I solve 8x + 7 = -23" },
];

const response_format: ResponseFormatJSONSchema = {
  type: "json_schema",
  json_schema: {
    name: "math_response",
    ...{
      type: "object",
      properties: {
        steps: {
          type: "array",
          items: {
            type: "object",
            properties: {
              explanation: { type: "string" },
              output: { type: "string" },
            },
            required: ["explanation", "output"],
            additionalProperties: false,
          },
        },
        final_answer: { type: "string" },
      },
      required: ["steps", "final_answer"],
      additionalProperties: false,
    },
    strict: true,
  },
};

const openai = new OpenAI({});

const response = await openai.chat.completions.create({
  model: "gpt-4o-2024-08-06",
  messages,
  response_format,
});

console.log(JSON.stringify(response, null, 2));

const openait0 = new OpenAI({
  baseURL: "http://localhost:3000/openai/v1",
});

const response2 = await openait0.chat.completions.create({
  model: "tensorzero::model_name::openai::gpt-4o-2024-08-06",
  messages,
  response_format,
});

console.log(JSON.stringify(response2, null, 2));
