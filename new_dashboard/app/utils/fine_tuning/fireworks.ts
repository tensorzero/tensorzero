import type { SFTFormValues } from "~/routes/optimization/fine-tuning/types";
import type { JsExposedEnv } from "../minijinja/pkg/minijinja_bindings";
import { v7 } from "uuid";
import type {
  ContentBlockOutput,
  JsonInferenceOutput,
  ParsedInferenceRow,
} from "../clickhouse";
import { getCuratedInferences } from "../clickhouse";
import { render_message } from "./rendering";
import { getConfig } from "../config";
import { get_template_env, type ChatCompletionConfig } from "../config/variant";
import { z } from "zod";
import { SFTJob } from "./common";
export const FIREWORKS_API_URL = "https://api.fireworks.ai";
export const FIREWORKS_API_KEY = process.env.FIREWORKS_API_KEY || throwError();
export const FIREWORKS_ACCOUNT_ID =
  process.env.FIREWORKS_ACCOUNT_ID || throwError();

// This is apparently the traditional way to coerce both to strings.
function throwError(): never {
  throw new Error("FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID must be set");
}

export class FireworksSFTJob extends SFTJob {
  constructor(
    public jobPath: string,
    public status: string,
    public jobId: string,
    public modelId?: string,
    public modelPath?: string,
  ) {
    super();
  }

  static async from_form_data(data: SFTFormValues): Promise<FireworksSFTJob> {
    let config = await getConfig();
    // TODO: throw if this isn't a chat completion
    console.log("config", config);
    const currentVariant = config.functions[data.function].variants[
      data.variant
    ] as ChatCompletionConfig;
    const curatedInferences = await getCuratedInferences(
      data.function,
      config.functions[data.function],
      data.metric,
      config.metrics[data.metric],
      data.maxSamples,
    );
    if (!curatedInferences || curatedInferences.length === 0) {
      throw new Error("No curated inferences found");
    }
    const templateEnv = await get_template_env(currentVariant);
    const jobPath = await start_sft_fireworks(
      data.model.name,
      curatedInferences,
      data.validationSplitPercent,
      templateEnv,
    );
    return new FireworksSFTJob(
      jobPath,
      "RUNNING",
      data.jobId,
      undefined,
      undefined,
    );
  }

  display(): string {
    return this.status ?? "";
  }

  result(): string | undefined {
    return this.modelPath;
  }

  provider(): string {
    return "fireworks";
  }

  is_finished(): boolean {
    return this.status === "deployed" || this.status === "failed";
  }

  async poll(): Promise<FireworksSFTJob> {
    if (!this.modelId) {
      // If we don't have a model ID, training is still running so we need to poll for it
      console.log("polling for training status");
      const status = await get_fine_tuning_job_status(this.jobPath);
      console.log("status", status);
      if (status === "COMPLETED") {
        const modelId = await get_model_id(this.jobPath);
        if (!modelId) {
          throw new Error("Model ID not found after job completed");
        }
        console.log("modelId", modelId);
        await deploy_model(FIREWORKS_ACCOUNT_ID, modelId);
        return new FireworksSFTJob(
          this.jobPath,
          "DEPLOYING",
          this.jobId,
          modelId,
          undefined,
        );
      } else {
        return new FireworksSFTJob(
          this.jobPath,
          "TRAINING",
          this.jobId,
          undefined,
          undefined,
        );
      }
    } else {
      // If we do have a model ID, we need to poll for the deployment
      const status = await poll_model_deployment(
        FIREWORKS_ACCOUNT_ID,
        this.modelId,
      );
      console.log("status", status);
      if (status === "DEPLOYED") {
        const modelPath = `accounts/${FIREWORKS_ACCOUNT_ID}/models/${this.modelId}`;
        console.log("modelPath", modelPath);
        return new FireworksSFTJob(
          this.jobPath,
          "DEPLOYED",
          this.jobId,
          this.modelId,
          modelPath,
        );
      } else {
        return new FireworksSFTJob(
          this.jobPath,
          "DEPLOYING",
          this.jobId,
          this.modelId,
          undefined,
        );
      }
    }
  }
}

type DeploymentStatus = "DEPLOYED" | "DEPLOYING";

const FineTuningJobStatusSchema = z.enum([
  "STATE_UNSPECIFIED",
  "CREATING",
  "PENDING",
  "RUNNING",
  "COMPLETED",
  "FAILED",
  "DELETING",
]);

type FineTuningJobStatus = z.infer<typeof FineTuningJobStatusSchema>;

const FineTuningJobResponseSchema = z.object({
  state: FineTuningJobStatusSchema,
  modelId: z.string().optional(),
  // Add more fields as needed
});

type FineTuningJobResponse = z.infer<typeof FineTuningJobResponseSchema>;

// Docs: https://docs.fireworks.ai/api-reference/get-fine-tuning-job
async function get_fine_tuning_job_details(
  job_path: string,
): Promise<FineTuningJobResponse> {
  console.log("getting fine tuning job details");
  const url = new URL(`v1/${job_path}`, FIREWORKS_API_URL).toString();
  const options = {
    method: "GET",
    headers: {
      Authorization: `Bearer ${FIREWORKS_API_KEY}`,
    },
  };

  try {
    const info = await fetch(url, options);
    if (!info.ok) {
      throw new Error(
        `Failed to get fine tuning job details: ${info.status} ${info.statusText}`,
      );
    }
    const response = await info.json();
    // Validate the response with Zod
    const validated = FineTuningJobResponseSchema.parse(response);
    return validated;
  } catch (error) {
    if (error instanceof z.ZodError) {
      throw new Error(`Invalid API response format: ${error.message}`);
    }
    throw new Error(
      `Error getting fine tuning job details: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
    );
  }
}

async function get_fine_tuning_job_status(
  job_path: string,
): Promise<FineTuningJobStatus> {
  const response = await get_fine_tuning_job_details(job_path);
  return response.state;
}

// This is the model ID that we can use to deploy the model
// Note: this should only be called after the job is completed
async function get_model_id(job_path: string): Promise<string | undefined> {
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

export async function start_sft_fireworks(
  modelName: string,
  inferences: ParsedInferenceRow[],
  val_split: number,
  templateEnv: JsExposedEnv,
): Promise<string> {
  const fireworksExamples = inferences.map((inference) =>
    tensorzero_inference_to_fireworks_messages(inference, templateEnv),
  );
  const serializedExamples = JSON.stringify(fireworksExamples, null, 2);

  console.log("fireworksExamples", serializedExamples);

  const datasetId = await create_dataset_record(
    FIREWORKS_ACCOUNT_ID,
    fireworksExamples.length,
  );
  console.log("datasetId", datasetId);
  let uploadResponse = await upload_dataset(
    FIREWORKS_ACCOUNT_ID,
    datasetId,
    fireworksExamples,
  );
  console.log("uploadResponse", uploadResponse);

  // We poll here since this usually does not take long
  while (!(await dataset_is_ready(FIREWORKS_ACCOUNT_ID, datasetId))) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  console.log("dataset is ready");

  const job_path = await create_fine_tuning_job(
    FIREWORKS_ACCOUNT_ID,
    datasetId,
    modelName,
    val_split,
  );

  return job_path;
}

// export async function poll_sft_fireworks(
//   jobPath: string,
//   modelId?: string,
// ): Promise<FireworksSFTJob> {
//   if (!modelId) {
//     // If we don't have a model ID, training is still running so we need to poll for it
//     const status = await get_fine_tuning_job_status(jobPath);

//     if (status === "COMPLETED") {
//       const modelId = await get_model_id(jobPath);
//       await deploy_model(FIREWORKS_ACCOUNT_ID, modelId);
//       return new FireworksSFTJob(jobPath, "DEPLOYING", modelId, undefined);
//     } else {
//       return new FireworksSFTJob(jobPath, "TRAINING", undefined, undefined);
//     }
//   } else {
//     const status = await poll_model_deployment(FIREWORKS_ACCOUNT_ID, modelId);
//     if (status === "DEPLOYED") {
//       const modelPath = `accounts/${FIREWORKS_ACCOUNT_ID}/models/${modelId}`;
//       return new FireworksSFTJob(jobPath, "DEPLOYED", modelId, modelPath);
//     } else {
//       return new FireworksSFTJob(jobPath, "DEPLOYING", modelId, undefined);
//     }
//   }
// }

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

  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(
        `Failed to check dataset status: ${response.status} ${response.statusText}`,
      );
    }
    const data = await response.json();
    if (!data.state) {
      throw new Error("Dataset status response missing state field");
    }
    return data.state === "READY";
  } catch (error) {
    throw new Error(
      `Error checking dataset status: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
    );
  }
}

// Docs: https://docs.fireworks.ai/api-reference/create-fine-tuning-job
// IMPORTANT: this function returns a path like "accounts/viraj-ebfe5a/fineTuningJobs/2aecc5ff56364010a143b6b0b0568b5a"
// We need to directly use this path for getting the job status
async function create_fine_tuning_job(
  accountId: string,
  datasetId: string,
  baseModel: string,
  valSplit: number,
): Promise<string> {
  const url = new URL(
    `v1/accounts/${accountId}/fineTuningJobs`,
    FIREWORKS_API_URL,
  ).toString();

  console.log("creating fine tuning job with url", url);

  const body = {
    dataset: `accounts/${accountId}/datasets/${datasetId}`,
    baseModel: baseModel,
    conversation: {}, // empty due to us using the default conversation template
    evaluationSplit: valSplit / 100,
  };

  console.log("body", body);

  const options = {
    method: "POST",
    headers: {
      Authorization: `Bearer ${FIREWORKS_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  };

  console.log("options", options);

  try {
    const response = await fetch(url, options);
    console.log("response", response);
    if (!response.ok) {
      throw new Error(
        `Failed to create fine tuning job: ${response.status} ${response.statusText}`,
      );
    }
    const data = await response.json();
    console.log("data", data);
    if (!data.name) {
      throw new Error("Fine tuning job response missing name field");
    }
    return data.name;
  } catch (error) {
    throw new Error(
      `Error creating fine tuning job: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
    );
  }
}
