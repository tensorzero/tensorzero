/*
This script demonstrates how to fine-tune a model using the OpenAI API.
To run:
`npx tsx --experimental-wasm-modules fine-tune-openai-example.ts`
You will need to have the docker compose for dashboard running.
*/

import { create_env } from "../minijinja/pkg/minijinja_bindings.js";
import { queryGoodBooleanMetricData } from "../clickhouse";
import { start_sft_fireworks, poll_sft_fireworks } from "./fireworks.server.js";

const result = await queryGoodBooleanMetricData(
  "dashboard_fixture_extract_entities",
  "dashboard_fixture_exact_match",
  "JsonInference",
  "id",
  true,
  undefined,
);

const env = await create_env({ system: "You are a helpful assistant." });
const { job_path: initialJobPath } = await start_sft_fireworks(
  "accounts/fireworks/models/llama-v3p1-8b-instruct",
  result,
  0.1,
  env,
);

let isJobComplete = false;
const pollResult = {
  job_path: initialJobPath,
  model_id: undefined as string | undefined,
};

while (!isJobComplete) {
  const response = await poll_sft_fireworks({
    job_path: pollResult.job_path,
    model_id: pollResult.model_id,
  });

  if (!response) {
    console.log("No response from polling");
    continue;
  }

  pollResult.job_path = response.job_path;
  pollResult.model_id = response.model_id;
  console.log("Status:", response.status);
  console.log("Model ID:", response.model_id);

  if (response.status === "DEPLOYED" || response.status === "FAILED") {
    isJobComplete = true;
  } else {
    await new Promise((resolve) => setTimeout(resolve, 10000));
  }
}
