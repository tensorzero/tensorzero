import OpenAI from "openai";

const client = new OpenAI();

const response = await client.chat.completions.create({
  model: "tensorzero::function_name::generate_haiku",
  messages: [
    {
      role: "user",
      content: [
        {
          type: "text",
          text: "Write a haiku about artificial intelligence in German.",
        },
        {
          type: "text",
          text: "In german!",
        },
      ],
    },
  ],
});

console.log(JSON.stringify(response, null, 2));
