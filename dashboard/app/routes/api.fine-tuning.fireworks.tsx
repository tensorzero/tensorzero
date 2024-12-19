import { type LoaderFunctionArgs, type ActionFunctionArgs } from "react-router";
import { ErrorWithStatus } from "~/utils/error";
import { ChatCompletionConfig, get_template_env } from "~/utils/config/variant";
import { getCuratedInferences } from "~/utils/clickhouse";
import { getConfig } from "~/utils/config.server";
import {
  poll_sft_fireworks,
  start_sft_fireworks,
} from "~/utils/fine_tuning/fireworks";
import { SFTFormValues } from "~/routes/optimization.fine-tuning/types";
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
