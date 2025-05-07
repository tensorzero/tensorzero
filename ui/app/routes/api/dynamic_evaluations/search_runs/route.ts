import type { LoaderFunctionArgs } from "react-router";
import { useEffect } from "react";
import { useFetcher } from "react-router";
import { searchDynamicEvaluationRuns } from "~/utils/clickhouse/dynamic_evaluations.server";
import {
  dynamicEvaluationRunSchema,
  type DynamicEvaluationRun,
} from "~/utils/clickhouse/dynamic_evaluations";

export async function loader({
  request,
}: LoaderFunctionArgs): Promise<Response> {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const projectName = searchParams.get("project_name");
  if (!projectName) {
    return new Response("Missing project_name parameter", { status: 400 });
  }
  const query = searchParams.get("q") || "";

  const runs = await searchDynamicEvaluationRuns(100, 0, projectName, query);
  const parsedRuns = runs.map((run) => dynamicEvaluationRunSchema.parse(run));
  return new Response(JSON.stringify(parsedRuns), {
    headers: {
      "Content-Type": "application/json",
    },
  });
}

/**
 * A hook that fetches evaluation runs based on evaluation name and search query.
 * This hook automatically refetches when any of the parameters change.
 *
 * @param params.projectName - The name of the project to search runs for
 * @param params.query - Optional search query to filter evaluation runs
 * @returns An object containing:
 *  - data: The evaluation runs matching the search criteria
 *  - isLoading: Whether the data is currently being fetched
 */
export function useSearchDynamicEvaluationRunsFetcher(params: {
  projectName?: string;
  query?: string;
}): { data?: DynamicEvaluationRun[]; isLoading: boolean } {
  const runsFetcher = useFetcher();

  useEffect(() => {
    if (params.projectName) {
      const searchParams = new URLSearchParams();
      searchParams.set("project_name", params.projectName);
      if (params.query) searchParams.set("q", params.query);

      runsFetcher.load(`/api/dynamic_evaluations/search_runs/?${searchParams}`);
    }
    // TODO: Fix and stop ignoring lint rule
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.projectName, params.query]);

  return {
    data: runsFetcher.data,
    isLoading: runsFetcher.state === "loading",
  };
}
