import { data, type LoaderFunctionArgs } from "react-router";
import { getConfig } from "~/utils/config/index.server";
import { getInferenceTableName } from "~/utils/clickhouse/common";
import {
  queryMetricsWithFeedback,
  type MetricsWithFeedbackData,
} from "~/utils/clickhouse/feedback";
import { logger } from "~/utils/logger";

export async function loader({
  params,
}: LoaderFunctionArgs): Promise<Response> {
  const functionName = params.function_name;

  if (!functionName) {
    return Response.json({ metrics: [] } as MetricsWithFeedbackData);
  }

  try {
    const config = await getConfig();
    const functionConfig = config.functions[functionName];
    if (!functionConfig) {
      throw data(`Function ${functionName} not found in config`, {
        status: 404,
      });
    }
    const inferenceTable = getInferenceTableName(functionConfig);

    const result = await queryMetricsWithFeedback({
      function_name: functionName,
      inference_table: inferenceTable,
    });
    return Response.json(result);
  } catch (error) {
    logger.error("Error fetching metrics with feedback:", error);
    throw new Response("Error fetching metrics with feedback", {
      status: 500,
      statusText: "Failed to fetch metrics with feedback",
    });
  }
}
