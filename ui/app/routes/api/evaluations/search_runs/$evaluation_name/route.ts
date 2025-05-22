import type { LoaderFunctionArgs } from "react-router";
import { useEffect } from "react";
import { useFetcher } from "react-router";
import { searchEvaluationRuns } from "~/utils/clickhouse/evaluations.server";
import type { EvaluationRunSearchResult } from "~/utils/clickhouse/evaluations";
import { getConfig } from "~/utils/config/index.server";

export async function loader({
  request,
}: LoaderFunctionArgs): Promise<Response> {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const evaluationName = searchParams.get("evaluation_name");
  if (!evaluationName) {
    return new Response("Missing evaluation_name parameter", { status: 400 });
  }
  const query = searchParams.get("q") || "";
  const config = await getConfig();
  const function_name = config.evaluations[evaluationName].function_name;

  if (!evaluationName) {
    return new Response("Missing evaluation_name parameter", { status: 400 });
  }

  const runs = await searchEvaluationRuns(
    evaluationName,
    function_name,
    query,
    100,
    0,
  );
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
 * @param params.evaluationName - The name of the evaluation to search runs for
 * @param params.query - Optional search query to filter evaluation runs
 * @returns An object containing:
 *  - data: The evaluation runs matching the search criteria
 *  - isLoading: Whether the data is currently being fetched
 */
export function useSearchEvaluationRunsFetcher(params: {
  evaluationName?: string;
  query?: string;
}): { data?: EvaluationRunSearchResult[]; isLoading: boolean } {
  const runsFetcher = useFetcher();

  useEffect(() => {
    if (params.evaluationName) {
      const searchParams = new URLSearchParams();
      searchParams.set("evaluation_name", params.evaluationName);
      if (params.query) searchParams.set("q", params.query);

      runsFetcher.load(
        `/api/evaluations/search_runs/${params.evaluationName}?${searchParams}`,
      );
    }
    // TODO: Fix and stop ignoring lint rule
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.evaluationName, params.query]);

  return {
    data: runsFetcher.data,
    isLoading: runsFetcher.state === "loading",
  };
}
