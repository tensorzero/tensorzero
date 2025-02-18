/**
 * This module handles supervised fine-tuning and deployment of models using the Fireworks AI API.
 *
 * The high-level flow is:
 * 1. User submits fine-tuning job configuration via the UI form (SFTFormValues)
 * 2. FireworksSFTJob class is instantiated with the form data
 * 3. Curated training data is retrieved from ClickHouse based on:
 *    - Selected function
 *    - Selected metric
 *    - Max samples limit
 * 4. Training data is formatted according to Fireworks API requirements
 * 5. Fine-tuning job is launched via Fireworks API endpoints (we get a job ID here)
 * 6. Job status is polled periodically to track progress
 * 7. Once complete, the fine-tuned model ID is stored
 * 8. The fine-tuned model is then deployed-- this also needs to be polled
 * 9. Once this is completed we have a path we can use for inference
 *
 * The FireworksSFTJob class extends the base SFTJob class to provide
 * Fireworks-specific implementation of job creation, status polling,
 * result handling, and deployment management.
 */

import type { SFTFormValues } from "~/routes/optimization/supervised-fine-tuning/types";
import type { JsExposedEnv } from "../minijinja/pkg/minijinja_bindings";
import { v7 } from "uuid";
import { render_message } from "./rendering";
import { getConfig } from "../config/index.server";
import { get_template_env, type ChatCompletionConfig } from "../config/variant";
import { z } from "zod";
import { SFTJob, type SFTJobStatus } from "./common";
import { getCuratedInferences } from "../clickhouse/curation.server";
import type {
  ContentBlockOutput,
  JsonInferenceOutput,
} from "../clickhouse/common";
import type { ParsedInferenceExample } from "../clickhouse/curation";

// Base URL for the Fireworks API
export const FIREWORKS_API_URL = "https://api.fireworks.ai";

// Retrieves the API key for the Fireworks API from environment variables
// Logs a warning if the key is not set
export const FIREWORKS_API_KEY = (() => {
  const key = process.env.FIREWORKS_API_KEY;
  if (!key) {
    console.warn("FIREWORKS_API_KEY is not set");
    return "";
  }
  return key;
})();

// Retrieves the account ID for the Fireworks API from environment variables
// Logs a warning if the ID is not set
export const FIREWORKS_ACCOUNT_ID = (() => {
  const id = process.env.FIREWORKS_ACCOUNT_ID;
  if (!id) {
    console.warn("FIREWORKS_ACCOUNT_ID is not set");
    return "";
  }
  return id;
})();

interface FireworksSFTJobParams {
  jobPath: string;
  status: string;
  jobId: string;
  modelId?: string;
  modelPath?: string;
  jobInfo: JobInfo;
  formData: SFTFormValues;
}

type JobInfo =
  | {
      status: "ok";
      info: string | Record<string, unknown>;
    }
  | {
      status: "error";
      message: string;
    };

export class FireworksSFTJob extends SFTJob {
  public jobPath: string;
  public jobStatus: string;
  public jobId: string;
  public modelId?: string;
  public modelPath?: string;
  public jobInfo: JobInfo;
  public formData: SFTFormValues;

  constructor(params: FireworksSFTJobParams) {
    super();
    this.jobPath = params.jobPath;
    this.jobStatus = params.status;
    this.jobId = params.jobId;
    this.modelId = params.modelId;
    this.modelPath = params.modelPath;
    this.jobInfo = params.jobInfo;
    this.formData = params.formData;
  }

  static async from_form_data(data: SFTFormValues): Promise<FireworksSFTJob> {
    const config = await getConfig();
    const currentVariant = config.functions[data.function].variants[
      data.variant
    ] as ChatCompletionConfig;
    if (currentVariant.type != "chat_completion") {
      throw new Error(
        "Supervised fine-tuning is only supported for chat completion variants",
      );
    }
    const curatedInferences = await getCuratedInferences(
      data.function,
      config.functions[data.function],
      data.metric,
      data.metric ? config.metrics[data.metric] : null,
      data.threshold,
      data.maxSamples,
    );
    if (!curatedInferences || curatedInferences.length === 0) {
      throw new Error("No curated inferences found");
    }
    const templateEnv = await get_template_env(currentVariant);
    let jobInfo;
    try {
      jobInfo = await start_sft_fireworks(
        data.model.name,
        curatedInferences,
        data.validationSplitPercent,
        templateEnv,
      );
    } catch (error) {
      throw new Error(
        `Failed to start Fireworks SFT job: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
    const jobPath = jobInfo.name as string;
    return new FireworksSFTJob({
      jobPath: jobPath,
      status: "RUNNING",
      jobId: data.jobId,
      modelId: undefined,
      modelPath: undefined,
      jobInfo: {
        status: "ok",
        info: jobInfo,
      },
      formData: data,
    });
  }

  private get jobUrl(): string {
    const jobId = this.jobPath.split("/").pop();
    if (!jobId) {
      throw new Error("Failed to parse job ID from path");
    }
    return `https://fireworks.ai/dashboard/fine-tuning/v1/${jobId}`;
  }

  status(): SFTJobStatus {
    if (this.jobStatus === "FAILED") {
      const error =
        this.jobInfo.status === "error"
          ? this.jobInfo.message
          : "Unknown error";
      return {
        status: "error",
        modelProvider: "fireworks",
        formData: this.formData,
        jobUrl: this.jobUrl,
        rawData: this.jobInfo,
        error: error,
      };
    }
    if (this.jobStatus === "DEPLOYED") {
      if (!this.modelPath) {
        throw new Error("Model path is undefined for deployed job");
      }
      return {
        status: "completed",
        modelProvider: "fireworks",
        formData: this.formData,
        jobUrl: this.jobUrl,
        result: this.modelPath,
        rawData: this.jobInfo,
      };
    }
    return {
      status: "running",
      modelProvider: "fireworks",
      formData: this.formData,
      jobUrl: this.jobUrl,
      rawData: this.jobInfo,
    };
  }

  async poll(): Promise<FireworksSFTJob> {
    try {
      if (!this.modelId) {
        // If we don't have a model ID, training is still running so we need to poll for it
        const jobInfo = await get_fine_tuning_job_details(this.jobPath);
        const status = jobInfo.state;
        if (status === "COMPLETED") {
          const modelId = jobInfo.modelId;
          if (!modelId) {
            throw new Error("Model ID not found after job completed");
          }
          // Begin deployment process
          await deploy_model_request(FIREWORKS_ACCOUNT_ID, modelId);
          return new FireworksSFTJob({
            jobPath: this.jobPath,
            status: "DEPLOYING",
            jobId: this.jobId,
            modelId: modelId,
            modelPath: undefined,
            jobInfo: {
              status: "ok",
              info: jobInfo,
            },
            formData: this.formData,
          });
        } else {
          return new FireworksSFTJob({
            jobPath: this.jobPath,
            status: "TRAINING",
            jobId: this.jobId,
            modelId: undefined,
            modelPath: undefined,
            jobInfo: {
              status: "ok",
              info: jobInfo,
            },
            formData: this.formData,
          });
        }
      } else {
        // If we do have a model ID, we need to poll for the deployment
        const deployModelResponse = await deploy_model_request(
          FIREWORKS_ACCOUNT_ID,
          this.modelId,
        );
        const status = get_deployment_status(deployModelResponse);
        if (status === "DEPLOYED") {
          const modelPath = `accounts/${FIREWORKS_ACCOUNT_ID}/models/${this.modelId}`;
          return new FireworksSFTJob({
            jobPath: this.jobPath,
            status: "DEPLOYED",
            jobId: this.jobId,
            modelId: this.modelId,
            modelPath: modelPath,
            jobInfo: {
              status: "ok",
              info: deployModelResponse,
            },
            formData: this.formData,
          });
        } else {
          return new FireworksSFTJob({
            jobPath: this.jobPath,
            status: "DEPLOYING",
            jobId: this.jobId,
            modelId: this.modelId,
            modelPath: undefined,
            jobInfo: {
              status: "ok",
              info: deployModelResponse,
            },
            formData: this.formData,
          });
        }
      }
    } catch (error) {
      return new FireworksSFTJob({
        jobPath: this.jobPath,
        status: "error",
        jobId: this.jobId,
        modelId: this.modelId,
        modelPath: undefined,
        jobInfo: {
          status: "error",
          message: error instanceof Error ? error.message : String(error),
        },
        formData: this.formData,
      });
    }
  }
}

const FineTuningJobStatusSchema = z.enum([
  "STATE_UNSPECIFIED",
  "CREATING",
  "PENDING",
  "RUNNING",
  "COMPLETED",
  "FAILED",
  "DELETING",
]);

const FineTuningJobResponseSchema = z.object({
  state: FineTuningJobStatusSchema,
  modelId: z.string().optional(),
  baseModel: z.string().optional(),
  batchSize: z.number().optional(),
  createTime: z.string().optional(),
  createdBy: z.string().optional(),
  dataset: z.string().optional(),
  evaluationSplit: z.number().optional(),
  fineTuningJobId: z.string().optional(),
  fineTuningJobName: z.string().optional(),
  fineTuningJobPath: z.string().optional(),
  evaluation: z.boolean().optional(),
  evaluationDataset: z.string().optional(),
  learningRate: z.number().optional(),
  loraRank: z.number().optional(),
  loraTargetModules: z.array(z.string()).optional(),
  maskToken: z.string().optional(),
  microBatchSize: z.number().optional(),
  name: z.string().optional(),
  padToken: z.string().optional(),
  status: z
    .object({
      code: z.string().optional(),
      message: z.string().optional(),
    })
    .optional(),
});

type FineTuningJobResponse = z.infer<typeof FineTuningJobResponseSchema>;

// Docs: https://docs.fireworks.ai/api-reference/get-fine-tuning-job
async function get_fine_tuning_job_details(
  job_path: string,
): Promise<FineTuningJobResponse> {
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

// Docs: https://docs.fireworks.ai/api-reference/create-deployment
// Once a model has been fine-tuned, we should deploy it
// This is a separate step from the fine-tuning job
// NOTE: If unused, the model will be un-deployed after 7 days
// This function is called both to deploy the model and to poll for the deployment status
// NOTE: If the model has already been requested to be deployed,
// the API actually returns 400 along with the deployment status in the message
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

  try {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!data) {
      throw new Error("Empty response received from deploy model request");
    }
    return data;
  } catch (error) {
    throw new Error(
      `Error deploying model: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
    );
  }
}

// Extracts the deployment status from the deploy model response
function get_deployment_status(
  deployModelResponse: Record<string, unknown>,
): string {
  const message = deployModelResponse.message;
  if (!message || typeof message !== "string") {
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
  inferences: ParsedInferenceExample[],
  validationSplitPercent: number,
  templateEnv: JsExposedEnv,
): Promise<Record<string, unknown>> {
  const fireworksExamples = inferences.map((inference) =>
    tensorzero_inference_to_fireworks_messages(inference, templateEnv),
  );

  const datasetId = await create_dataset_record(
    FIREWORKS_ACCOUNT_ID,
    fireworksExamples.length,
  );
  await upload_dataset(FIREWORKS_ACCOUNT_ID, datasetId, fireworksExamples);

  // We poll here since this usually does not take long
  while (!(await dataset_is_ready(FIREWORKS_ACCOUNT_ID, datasetId))) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  const jobInfo = await create_fine_tuning_job(
    FIREWORKS_ACCOUNT_ID,
    datasetId,
    modelName,
    validationSplitPercent,
  );

  return jobInfo;
}

type FireworksMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

type FireworksExample = {
  messages: FireworksMessage[];
};

export function tensorzero_inference_to_fireworks_messages(
  sample: ParsedInferenceExample,
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
// This is a placeholder for the dataset that gets uploaded in a subsequent call.
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
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(
        `Failed to create dataset record: ${response.status} ${response.statusText}`,
      );
    }
    await response.json();
  } catch (error) {
    throw new Error(
      `Error creating dataset record: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
    );
  }

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

  let response;
  try {
    response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(
        `Failed to upload dataset: ${response.status} ${response.statusText}`,
      );
    }
    const data = await response.json();
    if (!data) {
      throw new Error("Empty response received from upload dataset request");
    }
    return data;
  } catch (error) {
    throw new Error(
      `Error uploading dataset: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
    );
  }
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
): Promise<Record<string, unknown>> {
  const url = new URL(
    `v1/accounts/${accountId}/fineTuningJobs`,
    FIREWORKS_API_URL,
  ).toString();

  const body = {
    dataset: `accounts/${accountId}/datasets/${datasetId}`,
    baseModel: baseModel,
    conversation: {}, // empty due to us using the default conversation template
    evaluationSplit: valSplit / 100,
  };

  const options = {
    method: "POST",
    headers: {
      Authorization: `Bearer ${FIREWORKS_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  };

  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(
        `Failed to create fine tuning job: ${response.status} ${response.statusText}`,
      );
    }
    const data = await response.json();
    if (!data.name) {
      throw new Error("Fine tuning job response missing name field");
    }
    return data;
  } catch (error) {
    throw new Error(
      `Error creating fine tuning job: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
    );
  }
}
