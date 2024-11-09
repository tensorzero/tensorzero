import { JsExposedEnv } from "../minijinja/pkg";
import {
  ParsedInferenceRow,
  ContentBlockOutput,
  JsonInferenceOutput,
} from "utils/clickhouse";
import { render_message } from "./rendering";
import OpenAI from "openai";
import { createReadStream } from "fs";
import * as fs from "fs/promises";
import * as path from "path";

type OpenAIRole = "system" | "user" | "assistant";

type OpenAIMessage = {
  role: OpenAIRole;
  content: string;
};

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export function sample_to_openai_messages(
  sample: ParsedInferenceRow,
  env: JsExposedEnv,
) {
  const system = sample.input.system;
  const messages: OpenAIMessage[] = [];
  if (env.has_template("system")) {
    const rendered_system = env.render("system", system);
    messages.push({
      role: "system",
      content: rendered_system,
    });
  } else if (system) {
    if (typeof system !== "string") {
      throw new Error(
        "System message must be a string when not using templates",
      );
    }
    messages.push({
      role: "system",
      content: system,
    });
  }
  for (const message of sample.input.messages) {
    const rendered_message = render_message(env, message.role, message.content);
    messages.push({
      role: message.role,
      content: rendered_message,
    });
  }
  // Add type check at the beginning
  const isChatInference = Array.isArray(sample.output);
  if (isChatInference) {
    const output = sample.output as ContentBlockOutput[];
    if (output.length !== 1) {
      throw new Error("Chat inference must have exactly one message");
    }
    if (output[0].type !== "text") {
      throw new Error("Chat inference must have a text message as output");
    }
    messages.push({ role: "assistant", content: output[0].text });
  } else {
    const output = sample.output as JsonInferenceOutput;
    messages.push({ role: "assistant", content: output.raw });
  }
  return messages;
}

export async function upload_examples_to_openai(samples: OpenAIMessage[][]) {
  // Convert samples to JSONL format
  const jsonl = samples
    .map((messages) => JSON.stringify({ messages }))
    .join("\n");

  // Write to temporary file
  const tempFile = path.join(process.cwd(), "temp_training_data.jsonl");
  await fs.writeFile(tempFile, jsonl);

  try {
    const file = await client.files.create({
      file: createReadStream(tempFile),
      purpose: "fine-tune",
    });

    return file.id;
  } finally {
    // Clean up temp file
    await fs.unlink(tempFile);
  }
}

export async function create_fine_tuning_job(
  model: string,
  train_file_id: string,
  val_file_id?: string,
) {
  const params: OpenAI.FineTuning.JobCreateParams = {
    model,
    training_file: train_file_id,
  };

  if (val_file_id) {
    params.validation_file = val_file_id;
  }

  const job = await client.fineTuning.jobs.create(params);
  return job.id;
}

export async function poll_fine_tuning_job(job_id: string) {
  const job = await client.fineTuning.jobs.retrieve(job_id);
  return job;
}
