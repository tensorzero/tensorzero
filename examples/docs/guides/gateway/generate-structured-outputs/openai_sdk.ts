import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:3000/openai/v1",
  apiKey: "unused",
});

const messages = [
  "Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com",
  "My email is contact@company.com",
  "This is John Doe reaching out",
  "I have a question about your product",
];

for (const message of messages) {
  const response = await client.chat.completions.create({
    model: "tensorzero::function_name::extract_data",
    messages: [
      {
        role: "user",
        content: message,
      },
    ],
  });

  console.log(message);
  console.log(JSON.stringify(response));
  console.log();
}
