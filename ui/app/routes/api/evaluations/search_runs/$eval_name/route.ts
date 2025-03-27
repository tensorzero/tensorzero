import type { LoaderFunctionArgs } from "react-router";
import { useEffect } from "react";
import { useFetcher } from "react-router";
import { searchEvalRuns } from "~/utils/clickhouse/evaluations.server";
import type { EvaluationRunInfo } from "~/utils/clickhouse/evaluations";
import { getConfig } from "~/utils/config/index.server";

export async function loader({
  request,
}: LoaderFunctionArgs): Promise<Response> {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const evalName = searchParams.get("eval_name");
  if (!evalName) {
    return new Response("Missing eval_name parameter", { status: 400 });
  }
  const query = searchParams.get("q") || "";
  const config = await getConfig();
  const function_name = config.evals[evalName].function_name;

  if (!evalName) {
    return new Response("Missing eval_name parameter", { status: 400 });
  }

  const runs = await searchEvalRuns(evalName, function_name, query);
  return new Response(JSON.stringify(runs), {
    headers: {
      "Content-Type": "application/json",
    },
  });
}

/**
 * A hook that fetches evaluation runs based on evaluation name and search query.
 * This hook automatically refetches when any of the parameters change.
 *
 * @param params.evalName - The name of the evaluation to search runs for
 * @param params.query - Optional search query to filter evaluation runs
 * @returns An object containing:
 *  - data: The evaluation runs matching the search criteria
 *  - isLoading: Whether the data is currently being fetched
 */
export function useSearchEvalRunsFetcher(params: {
  evalName?: string;
  query?: string;
}): { data: EvaluationRunInfo[]; isLoading: boolean } {
  const runsFetcher = useFetcher();

  useEffect(() => {
    if (params.evalName) {
      const searchParams = new URLSearchParams();
      searchParams.set("eval_name", params.evalName);
      if (params.query) searchParams.set("q", params.query);

      runsFetcher.load(
        `/api/evaluations/search_runs/${params.evalName}?${searchParams}`,
      );
    }
  }, [params.evalName, params.query]);

  return {
    data: runsFetcher.data || [],
    isLoading: runsFetcher.state === "loading",
  };
}
