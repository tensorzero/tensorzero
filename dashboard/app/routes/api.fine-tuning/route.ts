import { type LoaderFunctionArgs, type ActionFunctionArgs } from "react-router";
import { getConfig } from "~/utils/config.server";
import { getCuratedInferences } from "~/utils/clickhouse";
import type { FormValues } from "~/routes/optimization.fine-tuning/route";
import { ChatCompletionConfig, get_template_env } from "~/utils/config/variant";
import {
  poll_sft_openai,
  start_sft_openai,
} from "~/utils/fine_tuning/openai.server";
import { BadRequestError, ErrorWithStatus, NotFoundError } from "~/utils/error";
import { poll_sft_fireworks } from "~/utils/fine_tuning/fireworks.server";

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const provider_name = url.searchParams.get("provider_name");
  if (!provider_name) {
    return Response.json(
      { error: "Provider name is required" },
      { status: 400 },
    );
  }
  switch (provider_name) {
    case "openai":
      try {
        const searchParamsObj = Object.fromEntries(url.searchParams.entries());
        const job_info = await poll_sft_openai(searchParamsObj);

        if (job_info) {
          return Response.json({
            status: job_info.status,
            fine_tuned_model: job_info.fine_tuned_model,
            job: job_info,
          });
        } else {
          throw new NotFoundError("Job not found");
        }
      } catch (error) {
        return Response.json(
          { error: (error as Error).message },
          { status: error instanceof ErrorWithStatus ? error.status : 500 },
        );
      }

    case "fireworks":
      try {
        const searchParamsObj = Object.fromEntries(url.searchParams.entries());
        const job_info = await poll_sft_fireworks(searchParamsObj);
        return Response.json(job_info);
      } catch (error) {
        return Response.json(
          { error: (error as Error).message },
          { status: 500 },
        );
      }

    default:
      return Response.json({ error: "Provider not found" }, { status: 404 });
  }
}

export async function action({ request }: ActionFunctionArgs) {
  if (request.method !== "POST") {
    return Response.json({ error: "Method not allowed" }, { status: 405 });
  }

  try {
    const data = (await request.json()) as FormValues;
    const config = await getConfig();
    const current_variant = config.functions[data.function].variants[
      data.variant
    ] as ChatCompletionConfig;
    if (data.model.provider !== "openai") {
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
    let params;
    switch (data.model.provider) {
      case "openai": {
        const job_id = await start_sft_openai(
          data.model.name,
          curatedInferences,
          data.validationSplit,
          template_env,
        );
        params = {
          provider: "openai",
          job_id: job_id,
        };
        break;
      }
      // TODO: Add Fireworks here
      default:
        throw new BadRequestError("Unsupported model provider");
    }

    return Response.json({
      status: "success",
      message: "Fine-tuning job started",
      params,
    });
  } catch (error) {
    return Response.json(
      { error: (error as Error).message },
      { status: error instanceof ErrorWithStatus ? error.status : 500 },
    );
  }
}
