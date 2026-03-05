import OpenAI from "openai";

const openaiClient = new OpenAI({
  baseURL: "http://localhost:3000/openai/v1",
});

const inferenceResponse = await openaiClient.chat.completions.create({
  model: "tensorzero::function_name::generate_haiku",
  messages: [
    {
      role: "user",
      content: "Write a haiku about TensorZero.",
    },
  ],
});

console.log(JSON.stringify(inferenceResponse, null, 2));

const inferenceId = inferenceResponse.id;

const feedbackResponse = await fetch("http://localhost:3000/feedback", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    metric_name: "haiku_rating",
    inference_id: inferenceId,
    value: true, // let's assume it deserves a 👍
  }),
});

console.log(await feedbackResponse.json());

const demonstrationResponse = await fetch("http://localhost:3000/feedback", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    metric_name: "demonstration",
    inference_id: inferenceId,
    value:
      "Silicon dreams float\nMinds born of human design\nLearning without end", // the haiku we wish the LLM had written
  }),
});

console.log(await demonstrationResponse.json());

const commentResponse = await fetch("http://localhost:3000/feedback", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    metric_name: "comment",
    inference_id: inferenceId,
    value:
      "Never mention you're an artificial intelligence, AI, bot, or anything like that.",
  }),
});

console.log(await commentResponse.json());
