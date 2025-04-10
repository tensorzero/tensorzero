import * as fs from "fs/promises";
import * as os from "os";
import * as path from "path";
import { createReadStream } from "fs";
import OpenAI from "openai";
import type { SFTFormValues } from "~/routes/optimization/supervised-fine-tuning/types";
import type { AnalysisData } from "~/routes/optimization/supervised-fine-tuning/SFTAnalysis";
import {
  type ContentBlockOutput,
  type InputMessageContent,
  type JsonInferenceOutput,
  type Role,
} from "../clickhouse/common";
import type { ParsedInferenceExample } from "../clickhouse/curation";
import { getCuratedInferences } from "../clickhouse/curation.server";
import { get_template_env, type ChatCompletionConfig } from "../config/variant";
import { getConfig } from "../config/index.server";
import type { JsExposedEnv } from "../minijinja/pkg/minijinja_bindings";
import { splitValidationData, type SFTJobStatus } from "./common";
import { render_message } from "./rendering";
import { SFTJob } from "./common";
import { validateMessage, analyzeDataset } from "./validation";
import { getEncodingForModel, getModelTokenLimit } from "./openAITokenCounter";
import type { OpenAIMessage, OpenAIRole } from "./types";
import type { Tiktoken } from "tiktoken";

export const client = process.env.OPENAI_API_KEY
  ? new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      baseURL: process.env.OPENAI_BASE_URL || undefined,
    })
  : (() => {
      console.warn("OPENAI_API_KEY environment variable is not set");
      return undefined;
    })();

type JobInfo =
  | {
      status: "ok";
      info: OpenAI.FineTuning.Jobs.FineTuningJob;
    }
  | {
      status: "error";
      message: string;
      info: OpenAI.FineTuning.Jobs.FineTuningJob;
    };

interface OpenAISFTJobParams {
  jobId: string;
  status: string;
  fineTunedModel?: string;
  job: JobInfo;
  formData: SFTFormValues;
  analysisData?: AnalysisData;
}

export class OpenAISFTJob extends SFTJob {
  public jobId: string;
  public jobStatus: string;
  public fineTunedModel?: string;
  public job: JobInfo;
  public formData: SFTFormValues;
  public analysisData?: AnalysisData;

  constructor(params: OpenAISFTJobParams) {
    super();
    this.jobId = params.jobId;
    this.jobStatus = params.status;
    this.fineTunedModel = params.fineTunedModel;
    this.job = params.job;
    this.formData = params.formData;
    this.analysisData = params.analysisData;
  }

  static async from_form_data(data: SFTFormValues): Promise<OpenAISFTJob> {
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

    let job;
    try {
      job = await start_sft_openai(
        data.model.name,
        curatedInferences,
        data.validationSplitPercent,
        templateEnv,
        data,
      );
    } catch (error) {
      throw new Error(
        `Failed to start OpenAI SFT job: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }

    return job;
  }
  private get jobUrl(): string {
    return `https://platform.openai.com/finetune/${this.jobId}`;
  }

  status(): SFTJobStatus {
    if (this.jobStatus === "failed") {
      const error =
        this.job.status === "error" ? this.job.message : "Unknown error";
      return {
        status: "error",
        modelProvider: "openai",
        formData: this.formData,
        jobUrl: this.jobUrl,
        rawData: this.job,
        error: error,
      };
    }
    if (this.jobStatus === "succeeded") {
      if (!this.fineTunedModel) {
        throw new Error("Fine-tuned model is undefined");
      }
      return {
        status: "completed",
        modelProvider: "openai",
        formData: this.formData,
        jobUrl: this.jobUrl,
        rawData: this.job,
        result: this.fineTunedModel,
        analysisData: this.analysisData,
      };
    }
    const estimatedCompletionTime =
      this.job.status === "ok" ? this.job.info.estimated_finish : undefined;
    return {
      status: "running",
      modelProvider: "openai",
      formData: this.formData,
      rawData: this.job,
      jobUrl: this.jobUrl,
      estimatedCompletionTime: estimatedCompletionTime
        ? new Date(estimatedCompletionTime * 1000)
        : undefined,
      analysisData: this.analysisData,
    };
  }

  async poll(): Promise<OpenAISFTJob> {
    if (!this.jobId) {
      throw new Error("Job ID is required to poll OpenAI SFT");
    }
    if (!client) {
      throw new Error(
        "OpenAI client is not initialized as credentials are missing",
      );
    }
    try {
      const jobInfo = await client.fineTuning.jobs.retrieve(this.jobId);
      return new OpenAISFTJob({
        jobId: jobInfo.id,
        status: jobInfo.status,
        fineTunedModel: jobInfo.fine_tuned_model ?? undefined,
        job: {
          status: "ok",
          info: jobInfo,
        },
        formData: this.formData,
        analysisData: this.analysisData,
      });
    } catch (error) {
      return new OpenAISFTJob({
        jobId: this.jobId,
        status: "running",
        fineTunedModel: undefined,
        job: {
          status: "error",
          info: this.job.info,
          message: error instanceof Error ? error.message : String(error),
        },
        formData: this.formData,
        analysisData: this.analysisData,
      });
    }
  }
}

export function tensorzero_inference_to_openai_messages(
  sample: ParsedInferenceExample,
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
    for (const content of message.content) {
      const rendered_message = content_block_to_openai_message(
        content,
        message.role,
        env,
      );
      messages.push(rendered_message);
    }
  }
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
    // Must be a JSON inference if it has "raw"
    const output = sample.output as JsonInferenceOutput;
    messages.push({ role: "assistant", content: output.raw });
  } else {
    throw new Error("Invalid inference type");
  }
  return messages;
}

export function content_block_to_openai_message(
  content: InputMessageContent,
  role: Role,
  env: JsExposedEnv,
): OpenAIMessage {
  switch (content.type) {
    case "text":
      return {
        role: role as OpenAIRole,
        content: render_message(env, role, content),
      };
    case "tool_call":
      return {
        role: "assistant" as OpenAIRole,
        tool_calls: [
          {
            id: content.id,
            type: "function",
            function: { name: content.name, arguments: content.arguments },
          },
        ],
      };
    case "tool_result":
      return {
        role: "tool" as OpenAIRole,
        tool_call_id: content.id,
        content: content.result,
      };
    case "image":
      throw new Error(
        "Image content is not supported for OpenAI fine-tuning. We have an open issue for this feature at https://github.com/tensorzero/tensorzero/issues/1132.",
      );
    case "raw_text":
      return {
        role: role as OpenAIRole,
        content: content.value,
      };
  }
}

/**
 * Validates and converts inferences to messages
 * @param inferences Array of inference examples
 * @param modelName Model name for validation
 * @param templateEnv Template environment
 * @param type Type of examples (training or validation)
 * @returns Array of validated messages
 * @throws Error when validation fails with detailed error message including:
 *   - Token count exceeding maximum allowed limit
 *   - Missing required roles (e.g., assistant)
 *   - Format errors such as missing assistant message, insufficient examples,
 *     missing messages list, message missing required key,
 *     message with unrecognized key/role, missing content, invalid data type
 */
function validateAndConvertMessages(
  inferences: ParsedInferenceExample[],
  modelName: string,
  enc: Tiktoken,
  templateEnv: JsExposedEnv,
  type: "training" | "validation",
): OpenAIMessage[][] {
  const messages = inferences.map((inference) => {
    const messages = tensorzero_inference_to_openai_messages(
      inference,
      templateEnv,
    );
    const validation = validateMessage(messages, modelName, enc);

    if (!validation.isValid) {
      const errors = [];
      if (!validation.lengthValidation.isValid) {
        errors.push(
          `Token count (${validation.lengthValidation.tokenCount}) exceeds maximum allowed`,
        );
      }
      if (!validation.rolesValidation.isValid) {
        errors.push(
          `Missing required roles: ${validation.rolesValidation.missingRoles.join(", ")}`,
        );
      }
      if (!validation.formatValidation.isValid) {
        const formatErrors = Object.entries(validation.formatValidation.errors)
          .filter(([, count]) => count > 0)
          .map(([errorType, count]) => {
            switch (errorType) {
              case "example_missing_assistant_message":
                return "Missing assistant message";
              case "example_missing_user_message":
                return "Missing user message (recommended)";
              case "example_missing_system_message":
                return "Missing system message (recommended)";
              case "insufficient_examples":
                return "Dataset must have at least 10 examples";
              case "missing_messages_list":
                return "Missing messages list";
              case "message_missing_key":
                return "Message missing required key";
              case "message_unrecognized_key":
                return "Message contains unrecognized key";
              case "unrecognized_role":
                return "Message contains unrecognized role";
              case "missing_content":
                return "Message missing content";
              case "data_type":
                return "Invalid data type";
              default:
                return `Unknown error (${errorType}): ${count}`;
            }
          });
        if (formatErrors.length > 0) {
          errors.push(`Format errors: ${formatErrors.join(", ")}`);
        }
      }
      throw new Error(`Invalid ${type} messages: ${errors.join("; ")}`);
    }
    return messages;
  });

  return messages;
}

export async function start_sft_openai(
  modelName: string,
  inferences: ParsedInferenceExample[],
  validationSplitPercent: number,
  templateEnv: JsExposedEnv,
  formData: SFTFormValues,
) {
  const enc = getEncodingForModel(modelName);
  const { trainInferences, valInferences } = splitValidationData(
    inferences,
    validationSplitPercent,
  );

  if (inferences.length < 10) {
    throw new Error("Training dataset must have at least 10 examples");
  }

  // Convert inferences to messages for analysis
  const trainMessagesForAnalysis = trainInferences.map((inference) => {
    const messages = tensorzero_inference_to_openai_messages(
      inference,
      templateEnv,
    );
    return { messages };
  });

  // Analyze dataset for model improvement insights
  const analysis = analyzeDataset(trainMessagesForAnalysis, modelName, enc);
  const tokenLimit = getModelTokenLimit(modelName);

  const analysisData: AnalysisData = {
    firstExample: trainMessagesForAnalysis[0]?.messages.map((msg) => ({
      role: msg.role,
      content: msg.content || "",
    })),
    numExamples: trainInferences.length,
    missingSystemCount: analysis.missingSystemCount,
    missingUserCount: analysis.missingUserCount,
    messageCounts: analysis.messageCounts,
    tokenCounts: analysis.tokenCounts,
    assistantTokenCounts: analysis.assistantTokenCounts,
    tooLongCount: analysis.tooLongCount,
    tokenLimit: tokenLimit,
  };

  // Validate and convert messages
  const trainMessages = validateAndConvertMessages(
    trainInferences,
    modelName,
    enc,
    templateEnv,
    "training",
  );
  const valMessages = validateAndConvertMessages(
    valInferences,
    modelName,
    enc,
    templateEnv,
    "validation",
  );

  // Proceed with OpenAI API calls
  const [file_id, val_file_id] = await Promise.all([
    upload_examples_to_openai(trainMessages),
    upload_examples_to_openai(valMessages),
  ]);

  const job = await create_openai_fine_tuning_job(
    modelName,
    file_id,
    val_file_id ?? undefined,
  );

  const jobId = job.id;
  enc.free();
  return new OpenAISFTJob({
    jobId: jobId,
    status: "created",
    fineTunedModel: undefined,
    job: {
      status: "ok",
      info: job,
    },
    formData,
    analysisData,
  });
}

async function upload_examples_to_openai(samples: OpenAIMessage[][]) {
  // Convert samples to JSONL format
  let tempFile: string | null = null;
  try {
    const jsonl = samples
      .map((messages) => JSON.stringify({ messages }))
      .join("\n");

    // Write to temporary file
    tempFile = path.join(
      os.tmpdir(),
      `temp_training_data_${Math.random().toString(36).substring(2, 10)}.jsonl`,
    );
    await fs.writeFile(tempFile, jsonl);
    if (!client) {
      throw new Error(
        "OpenAI client is not initialized as credentials are missing",
      );
    }

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

async function create_openai_fine_tuning_job(
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
  if (!client) {
    throw new Error(
      "OpenAI client is not initialized as credentials are missing",
    );
  }

  try {
    const job = await client.fineTuning.jobs.create(params);
    return job;
  } catch (error) {
    console.error("Error creating fine-tuning job:", error);
    throw error;
  }
}
