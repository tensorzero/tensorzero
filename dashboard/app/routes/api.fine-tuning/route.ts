import { json, type ActionFunctionArgs } from "@remix-run/node";
import { getConfig } from "~/utils/config.server";
import { getCuratedInferences } from "~/utils/clickhouse";
import type { FormValues } from "~/routes/optimization.fine-tuning/route";
import { ChatCompletionConfig, get_template_env } from "~/utils/config/variant";
import {
  OpenAIMessage,
  tensorzero_inference_to_openai_messages,
} from "~/utils/fine_tuning/openai";
import {
  upload_examples_to_openai,
  create_fine_tuning_job,
} from "~/utils/fine_tuning/openai.server";

function splitValidationData(
  messages: OpenAIMessage[][],
  validationSplit: number,
) {
  const splitIndex =
    validationSplit > 0
      ? Math.floor(messages.length * (1 - validationSplit / 100))
      : messages.length;

  const trainMessages = messages.slice(0, splitIndex);
  const valMessages = validationSplit > 0 ? messages.slice(splitIndex) : [];

  return {
    trainMessages,
    valMessages,
  };
}

export async function action({ request }: ActionFunctionArgs) {
  if (request.method !== "POST") {
    return json({ error: "Method not allowed" }, { status: 405 });
  }
  console.log("got request?");

  try {
    const data = (await request.json()) as FormValues;
    const config = await getConfig();
    const current_variant = config.functions[data.function].variants[
      data.variant
    ] as ChatCompletionConfig;
    console.log("data", data);
    console.log("data.model", data.model);
    if (data.model.provider !== "openai") {
      return json({ error: "Unsupported model provider" }, { status: 400 });
    }

    // Get curated inferences
    const curatedInferences = await getCuratedInferences(
      data.function,
      config.functions[data.function],
      data.metric,
      config.metrics[data.metric],
    );

    const template_env = await get_template_env(current_variant);
    const messages = curatedInferences?.map((inference) =>
      tensorzero_inference_to_openai_messages(inference, template_env),
    );

    const { trainMessages, valMessages } = splitValidationData(
      messages,
      data.validationSplit,
    );

    const file_id = await upload_examples_to_openai(trainMessages);
    console.log("file_id", file_id);

    let val_file_id: string | null = null;
    if (valMessages.length > 0) {
      val_file_id = await upload_examples_to_openai(valMessages);
      console.log("val_file_id", val_file_id);
    }
    const job_id = await create_fine_tuning_job(
      data.model.name,
      file_id,
      val_file_id ?? undefined,
    );
    console.log("job_id", job_id);

    console.log("started?");
    return json({
      status: "success",
      message: "Fine-tuning job started",
      job_id,
    });
  } catch (error) {
    return json({ error: (error as Error).message }, { status: 500 });
  }
}
