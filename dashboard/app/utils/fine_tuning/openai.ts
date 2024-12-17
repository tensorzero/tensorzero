import { OpenAI } from "openai";

export const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});
