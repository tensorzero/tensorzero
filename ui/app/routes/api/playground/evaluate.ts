import type { Route } from "./+types/evaluate";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import { getEnv } from "~/utils/env.server";
import { runNativeEvaluationStreaming } from "~/utils/tensorzero/native_client.server";
import type {
  EvaluationRunEvent,
  EvaluationRunFatalErrorEvent,
  FunctionConfig,
  EvaluationFunctionConfig,
} from "~/types/tensorzero";

function toEvaluationFunctionConfig(
  config: FunctionConfig,
): EvaluationFunctionConfig {
  if (config.type === "chat") {
    return { type: "chat" };
  }
  return { type: "json", output_schema: config.output_schema };
}

export async function action({ request }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const body = await request.json();
  const { evaluationName, variantName, datapointIds } = body as {
    evaluationName: string;
    variantName: string;
    datapointIds: string[];
  };

  if (!evaluationName || !variantName || !datapointIds?.length) {
    return new Response("Missing required fields", { status: 400 });
  }

  const env = getEnv();
  const config = await getConfig();

  const evaluationConfig = config.evaluations[evaluationName];
  if (!evaluationConfig) {
    return new Response(`Evaluation '${evaluationName}' not found`, {
      status: 404,
    });
  }

  const functionConfig = await getFunctionConfig(
    evaluationConfig.function_name,
    config,
  );
  if (!functionConfig) {
    return new Response(
      `Function '${evaluationConfig.function_name}' not found`,
      { status: 404 },
    );
  }

  const evaluationFunctionConfig = toEvaluationFunctionConfig(functionConfig);

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();

      const sendEvent = (event: EvaluationRunEvent) => {
        controller.enqueue(encoder.encode(JSON.stringify(event) + "\n"));
      };

      try {
        await runNativeEvaluationStreaming({
          gatewayUrl: env.TENSORZERO_GATEWAY_URL,
          clickhouseUrl: env.TENSORZERO_CLICKHOUSE_URL,
          evaluationConfig: JSON.stringify(evaluationConfig),
          functionConfig: JSON.stringify(evaluationFunctionConfig),
          evaluationName,
          datapointIds,
          variantName,
          concurrency: 5,
          inferenceCache: "on",
          onEvent: sendEvent,
        });
      } catch (err) {
        const errorEvent: {
          type: "fatal_error";
        } & EvaluationRunFatalErrorEvent = {
          type: "fatal_error",
          evaluation_run_id: null,
          message: err instanceof Error ? err.message : "Unknown error",
        };
        controller.enqueue(encoder.encode(JSON.stringify(errorEvent) + "\n"));
      }

      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
