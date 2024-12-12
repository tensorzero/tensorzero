/*
This script demonstrates how to fine-tune a model using the OpenAI API.
To run:
`npx tsx --experimental-wasm-modules fine-tune-openai-example.ts`
You will need to have the docker compose for dashboard running.
*/

import { create_env } from "./minijinja/pkg/minijinja_bindings.js";
import { queryGoodBooleanMetricData } from "./clickhouse";
import { tensorzero_inference_to_openai_messages } from "./fine_tuning/openai";
import {
  upload_examples_to_openai,
  create_fine_tuning_job,
  poll_fine_tuning_job,
} from "./fine_tuning/openai.server";
import OpenAI from "openai";

const result = await queryGoodBooleanMetricData(
  "dashboard_fixture_extract_entities",
  "dashboard_fixture_exact_match",
  "JsonInference",
  "id",
  true,
  undefined,
);

const env = await create_env({ system: "You are a helpful assistant." });

const messages = result.map((row) =>
  tensorzero_inference_to_openai_messages(row, env),
);

const file_id = await upload_examples_to_openai(messages);
const job_id = await create_fine_tuning_job("gpt-4o-mini-2024-07-18", file_id);
let finished = false;
let job: OpenAI.FineTuning.FineTuningJob | undefined;
while (!finished) {
  // Sleep for 10 seconds
  await new Promise((resolve) => setTimeout(resolve, 10000));
  job = await poll_fine_tuning_job(job_id);
  finished =
    job.status === "succeeded" ||
    job.status === "failed" ||
    job.status === "cancelled";
}
