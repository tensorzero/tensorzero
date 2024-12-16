/*
This script demonstrates how to fine-tune a model using the OpenAI API.
To run:
`npx tsx --experimental-wasm-modules fine-tune-openai-example.ts`
You will need to have the docker compose for dashboard running.
*/

import { create_env } from "../minijinja/pkg/minijinja_bindings.js";
import { queryGoodBooleanMetricData } from "../clickhouse";
import { start_sft_openai, poll_sft_openai } from "./openai.server";

const result = await queryGoodBooleanMetricData(
  "dashboard_fixture_extract_entities",
  "dashboard_fixture_exact_match",
  "JsonInference",
  "id",
  true,
  undefined,
);

const env = await create_env({ system: "You are a helpful assistant." });
const { job_id } = await start_sft_openai(
  "gpt-4o-mini-2024-07-18",
  result,
  0.1,
  env,
);

let isJobComplete = false;
while (!isJobComplete) {
  const job = await poll_sft_openai({ job_id });
  console.log(job.status);
  console.log(job.fine_tuned_model);
  console.log(job);

  if (
    job.status === "succeeded" ||
    job.status === "failed" ||
    job.status === "cancelled"
  ) {
    isJobComplete = true;
  } else {
    await new Promise((resolve) => setTimeout(resolve, 10000));
  }
}
