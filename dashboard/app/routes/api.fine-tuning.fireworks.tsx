import { type LoaderFunctionArgs, type ActionFunctionArgs } from "react-router";
import { ErrorWithStatus } from "~/utils/error";
import { ChatCompletionConfig, get_template_env } from "~/utils/config/variant";
import {
  ContentBlockOutput,
  JsonInferenceOutput,
  ParsedInferenceRow,
  getCuratedInferences,
} from "~/utils/clickhouse";
import { getConfig } from "~/utils/config.server";
import {
  FIREWORKS_ACCOUNT_ID,
  FIREWORKS_API_KEY,
  FIREWORKS_API_URL,
} from "~/utils/fine_tuning/fireworks";
import { render_message } from "~/utils/fine_tuning/rendering";
import { JsExposedEnv } from "~/utils/minijinja/pkg/minijinja_bindings";
import { SFTFormValues } from "~/routes/optimization.fine-tuning/types";
import { v7 } from "uuid";
import { FireworksSFTJob } from "~/utils/fine_tuning/fireworks.client";

// Launches a fine-tuning job on Fireworks
// We actually initialize the dataset on Fireworks, upload the dataset, wait
// until it is ready, then create the fine-tuning job
// This usually doesn't take long, so we poll here until the job is running.
export async function action({ request }: ActionFunctionArgs) {
  if (request.method !== "POST") {
    return Response.json({ error: "Method not allowed" }, { status: 405 });
  }

  try {
    const data = (await request.json()) as SFTFormValues;
    const config = await getConfig();
    const current_variant = config.functions[data.function].variants[
      data.variant
    ] as ChatCompletionConfig;
    if (data.model.provider !== "fireworks") {
      return Response.json(
        { error: "Unsupported model provider" },
        { status: 400 },
      );
    }

    // Get curated inferences
    const curatedInferences = await getCuratedInferences(
      data.function,
      config.functions[data.function],
      data.metric,
      config.metrics[data.metric],
    );

    const template_env = await get_template_env(current_variant);
    const validationSplit = data.validationSplitPercent / 100;
    const jobPath = await start_sft_fireworks(
      data.model.name,
      curatedInferences,
      validationSplit,
      template_env,
    );
    return Response.json(
      new FireworksSFTJob(jobPath, "created", undefined, undefined),
    );
  } catch (error) {
    return Response.json(
      { error: (error as Error).message },
      { status: error instanceof ErrorWithStatus ? error.status : 500 },
    );
  }
}

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const jobPath = url.searchParams.get("jobPath");
  const modelId = url.searchParams.get("modelId");

  if (!jobPath) {
    return Response.json({ error: "Job path is required" }, { status: 400 });
  }

  try {
    const job_info = await poll_sft_fireworks(jobPath, modelId || undefined);
    return Response.json(job_info);
  } catch (error) {
    console.log("Error during polling", error);
    return Response.json({ error: (error as Error).message }, { status: 500 });
  }
}

async function start_sft_fireworks(
  modelName: string,
  inferences: ParsedInferenceRow[],
  val_split: number,
  templateEnv: JsExposedEnv,
): Promise<string> {
  const fireworksExamples = inferences.map((inference) =>
    tensorzero_inference_to_fireworks_messages(inference, templateEnv),
  );

  const datasetId = await create_dataset_record(
    FIREWORKS_ACCOUNT_ID,
    fireworksExamples.length,
  );
  await upload_dataset(FIREWORKS_ACCOUNT_ID, datasetId, fireworksExamples);

  // We poll here since this usually does not take long
  while (!dataset_is_ready(FIREWORKS_ACCOUNT_ID, datasetId)) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  const job_path = await create_fine_tuning_job(
    FIREWORKS_ACCOUNT_ID,
    datasetId,
    modelName,
    val_split,
  );

  return job_path;
}

type FireworksMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

type FireworksExample = {
  messages: FireworksMessage[];
};

function tensorzero_inference_to_fireworks_messages(
  sample: ParsedInferenceRow,
  env: JsExposedEnv,
): FireworksExample {
  const messages: FireworksMessage[] = [];

  // Handle system message
  const system = sample.input.system;
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

  // Handle input messages
  for (const message of sample.input.messages) {
    for (const content of message.content) {
      if (content.type === "text") {
        messages.push({
          role: message.role,
          content: render_message(env, message.role, content),
        });
      } else {
        throw new Error(
          "Only text messages are supported for Fireworks fine-tuning",
        );
      }
    }
  }

  // Handle output
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
  } else if ("raw" in sample.output) {
    const output = sample.output as JsonInferenceOutput;
    messages.push({ role: "assistant", content: output.raw });
  } else {
    throw new Error("Invalid inference type");
  }

  return { messages };
}

// Creates a dataset record in Fireworks.
// This is a placeholder for the dataset that gets uploaded in a subsequest call.
// Essentially all this does is make an ID in Fireworks that we reuse.
// We'll use a UUIDv7
async function create_dataset_record(accountId: string, exampleCount: number) {
  const datasetId = v7();
  const url = new URL(
    `v1/accounts/${accountId}/datasets`,
    FIREWORKS_API_URL,
  ).toString();
  const options = {
    method: "POST",
    headers: {
      Authorization: `Bearer ${FIREWORKS_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      datasetId: datasetId,
      dataset: {
        displayName: datasetId,
        exampleCount: exampleCount.toString(),
        userUploaded: {}, // We can use this for e.g. function_name, timestamp, etc. later
        format: "CHAT", // Options here are CHAT, COMPLETION, and FORMAT_UNSPECIFIED
      },
    }),
  };
  const response = await fetch(url, options).then((r) => r.json());
  // TODO(Viraj: check it more robustly)
  console.log("Created dataset record", response);

  return datasetId;
}

// Docs: https://docs.fireworks.ai/api-reference/upload-dataset-files
// Note: if the data is larger than 150MB, we need to do something else
// described here: https://docs.fireworks.ai/api-reference/get-dataset-upload-endpoint
async function upload_dataset(
  accountId: string,
  datasetId: string,
  examples: FireworksExample[],
) {
  const url = new URL(
    `v1/accounts/${accountId}/datasets/${datasetId}:upload`,
    FIREWORKS_API_URL,
  ).toString();

  // Take the data and turn it into JSONL
  const jsonlData = examples
    .map((example) => JSON.stringify(example))
    .join("\n");

  // Create a Blob from the JSONL data
  const blob = new Blob([jsonlData], { type: "application/jsonl" });

  // Create FormData and append the file
  const form = new FormData();
  form.append("file", blob, "dataset.jsonl");

  const options = {
    method: "POST",
    headers: {
      Authorization: `Bearer ${FIREWORKS_API_KEY}`,
    },
    body: form,
  };

  const response = await fetch(url, options).then((r) => r.json());

  return response;
}

// Returns true if the dataset is ready for fine-tuning
// Returns false if the dataset is not ready for fine-tuning
async function dataset_is_ready(accountId: string, datasetId: string) {
  const url = new URL(
    `v1/accounts/${accountId}/datasets/${datasetId}`,
    FIREWORKS_API_URL,
  ).toString();

  const options = {
    method: "GET",
    headers: {
      Authorization: `Bearer ${FIREWORKS_API_KEY}`,
    },
  };

  const response = await fetch(url, options).then((r) => r.json());
  return response.state == "READY";
}

// Docs: https://docs.fireworks.ai/api-reference/create-fine-tuning-job
// IMPORTANT: this function returns a path like "accounts/viraj-ebfe5a/fineTuningJobs/2aecc5ff56364010a143b6b0b0568b5a"
// We need to directly use this path for getting the job status
async function create_fine_tuning_job(
  accountId: string,
  datasetId: string,
  base_model: string,
  val_split: number,
) {
  const url = new URL(
    `v1/accounts/${accountId}/fineTuningJobs`,
    FIREWORKS_API_URL,
  ).toString();

  const body = {
    dataset: `accounts/${accountId}/datasets/${datasetId}`,
    base_model: base_model,
    conversation: {}, // empty due to us using the default conversation template
    evaluationSplit: val_split,
  };

  const options = {
    method: "POST",
    headers: {
      Authorization: `Bearer ${FIREWORKS_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  };

  const response = await fetch(url, options).then((r) => r.json());
  return response.name;
}

export async function poll_sft_fireworks(
  jobPath: string,
  modelId?: string,
): Promise<FireworksSFTJob> {
  if (!modelId) {
    // If we don't have a model ID, training is still running so we need to poll for it
    const status = await get_fine_tuning_job_status(jobPath);

    if (status === "COMPLETED") {
      const modelId = await get_model_id(jobPath);
      await deploy_model(FIREWORKS_ACCOUNT_ID, modelId);
      return new FireworksSFTJob(jobPath, "DEPLOYING", modelId, undefined);
    } else {
      return new FireworksSFTJob(jobPath, "TRAINING", undefined, undefined);
    }
  } else {
    const status = await poll_model_deployment(FIREWORKS_ACCOUNT_ID, modelId);
    if (status === "DEPLOYED") {
      const modelPath = `accounts/${FIREWORKS_ACCOUNT_ID}/models/${modelId}`;
      return new FireworksSFTJob(jobPath, "DEPLOYED", modelId, modelPath);
    } else {
      return new FireworksSFTJob(jobPath, "DEPLOYING", modelId, undefined);
    }
  }
}

type FineTuningJobStatus =
  | "STATE_UNSPECIFIED"
  | "CREATING"
  | "PENDING"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "DELETING";

// Docs: https://docs.fireworks.ai/api-reference/get-fine-tuning-job
async function get_fine_tuning_job_details(job_path: string) {
  const url = new URL(`v1/${job_path}`, FIREWORKS_API_URL).toString();
  const options = {
    method: "GET",
    headers: {
      Authorization: `Bearer ${FIREWORKS_API_KEY}`,
    },
  };

  const info = await fetch(url, options);
  const response = await info.json();

  return response;
}

async function get_fine_tuning_job_status(
  job_path: string,
): Promise<FineTuningJobStatus> {
  const response = await get_fine_tuning_job_details(job_path);
  return response.state;
}

// This is the model ID that we can use to deploy the model
// Note: this should only be called after the job is completed
async function get_model_id(job_path: string) {
  const response = await get_fine_tuning_job_details(job_path);
  return response.modelId;
}

// Docs: https://docs.fireworks.ai/api-reference/create-deployment
// Once a model has been fine-tuned, we should deploy it
// This is a separate step from the fine-tuning job
// NOTE: If unused, the model will be un-deployed after 7 days
async function deploy_model_request(accountId: string, modelId: string) {
  const url = new URL(
    `v1/accounts/${accountId}/deployedModels`,
    FIREWORKS_API_URL,
  ).toString();

  const model_path = `accounts/${accountId}/models/${modelId}`;
  const body = {
    model: model_path,
    displayName: model_path,
    default: true,
    serverless: true,
    public: false,
  };

  const options = {
    method: "POST",
    headers: {
      Authorization: `Bearer ${FIREWORKS_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  };

  const response = await fetch(url, options).then((r) => r.json());

  return response;
}

async function deploy_model(
  accountId: string,
  modelId: string,
): Promise<string> {
  const response = await deploy_model_request(accountId, modelId);
  return response.name;
}

// Returns the status of the deployment
// TODO: this should be a better API call honestly
// We just can't find the right endpoint
async function poll_model_deployment(
  accountId: string,
  modelId: string,
): Promise<string> {
  const response = await deploy_model_request(accountId, modelId);
  const message = response.message;
  if (!message) {
    throw new Error("Failed to get deployment status message");
  }
  const status = message.split(":").pop()?.trim();
  if (!status) {
    throw new Error("Failed to parse deployment status from message");
  }
  return status;
}
