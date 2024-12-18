import { type ActionFunctionArgs } from "react-router";
import { getConfig } from "~/utils/config.server";
import { getCuratedInferences } from "~/utils/clickhouse";
import { SFTFormValues } from "~/routes/optimization.fine-tuning/types";
import { ChatCompletionConfig, get_template_env } from "~/utils/config/variant";
import { ErrorWithStatus } from "~/utils/error";
import { start_sft_openai } from "~/utils/fine_tuning/openai";

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
    if (
      data.model.provider !== "openai" &&
      data.model.provider !== "fireworks"
    ) {
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
    const job_status = await start_sft_openai(
      data.model.name,
      curatedInferences,
      validationSplit,
      template_env,
    );

    return Response.json(job_status);
  } catch (error) {
    return Response.json(
      { error: (error as Error).message },
      { status: error instanceof ErrorWithStatus ? error.status : 500 },
    );
  }
}
