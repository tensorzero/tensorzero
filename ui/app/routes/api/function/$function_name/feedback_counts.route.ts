import { data, type LoaderFunctionArgs } from "react-router";
import { resolveFunctionConfig } from "~/utils/config/index.server";
import type { MetricsWithFeedbackResponse } from "~/types/tensorzero";
import { logger } from "~/utils/logger";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader({
  params,
}: LoaderFunctionArgs): Promise<Response> {
  const functionName = params.function_name;

  if (!functionName) {
    const emptyResponse: MetricsWithFeedbackResponse = {
      metrics: [],
    };
    return Response.json(emptyResponse);
  }

  try {
    const result = await resolveFunctionConfig(functionName);
    if (!result) {
      throw data(`Function ${functionName} not found in config`, {
        status: 404,
      });
    }

    const tensorZeroClient = getTensorZeroClient();
    const response =
      await tensorZeroClient.getFunctionMetricsWithFeedback(functionName);
    return Response.json(response);
  } catch (error) {
    logger.error("Error fetching metrics with feedback:", error);
    throw new Response("Error fetching metrics with feedback", {
      status: 500,
      statusText: "Failed to fetch metrics with feedback",
    });
  }
}
