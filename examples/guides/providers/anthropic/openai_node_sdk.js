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

/*
 * By the way, you can...
 *
 * - Set up custom configuration and observability: `build_embedded(config_file="...", clickhouse_url="...")`
 * - Use a standalone HTTP TensorZero Gateway: use `build_http` instead of `build_embedded`
 * - Call custom models and functions: `model_name="my_model"` or `function_name="my_function"`
 */
