import { data, type LoaderFunctionArgs } from "react-router";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
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
    const config = await getConfig();
    const functionConfig = await getFunctionConfig(functionName, config);
    if (!functionConfig) {
      throw data(`Function ${functionName} not found in config`, {
        status: 404,
      });
    }

    const tensorZeroClient = getTensorZeroClient();
    const result =
      await tensorZeroClient.getFunctionMetricsWithFeedback(functionName);
    return Response.json(result);
  } catch (error) {
    logger.error("Error fetching metrics with feedback:", error);
    throw new Response("Error fetching metrics with feedback", {
      status: 500,
      statusText: "Failed to fetch metrics with feedback",
    });
  }
}
