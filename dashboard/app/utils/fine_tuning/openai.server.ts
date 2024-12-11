import * as os from "os";
import { createReadStream } from "fs";
import * as fs from "fs/promises";
import * as path from "path";
import { OpenAIMessage } from "./openai";
import OpenAI from "openai";

export const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function upload_examples_to_openai(samples: OpenAIMessage[][]) {
  // Convert samples to JSONL format
  let tempFile: string | null = null;
  try {
    const jsonl = samples
      .map((messages) => JSON.stringify({ messages }))
      .join("\n");

    // Write to temporary file
    tempFile = path.join(os.tmpdir(), `temp_training_data_${Date.now()}.jsonl`);
    await fs.writeFile(tempFile, jsonl);

    const file = await client.files.create({
      file: createReadStream(tempFile),
      purpose: "fine-tune",
    });

    return file.id;
  } finally {
    // Clean up temp file
    if (tempFile) {
      try {
        await fs.unlink(tempFile);
      } catch (err) {
        console.error(`Error deleting temp file ${tempFile}: ${err}`);
      }
    }
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

  try {
    const job = await client.fineTuning.jobs.create(params);
    return job.id;
  } catch (error) {
    console.error("Error creating fine-tuning job:", error);
    throw error;
  }
}

export async function poll_fine_tuning_job(job_id: string) {
  const job = await client.fineTuning.jobs.retrieve(job_id);
  return job;
}
