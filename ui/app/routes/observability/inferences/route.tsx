import {
  listInferencesWithPagination,
  countInferencesByFunction,
} from "~/utils/clickhouse/inference.server";
import type { Route } from "./+types/route";
import InferencesTable from "./InferencesTable";
import { data, isRouteErrorResponse, useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import InferenceSearchBar from "./InferenceSearchBar";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { logger } from "~/utils/logger";
import type { InferenceFilter } from "~/types/tensorzero";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const before = url.searchParams.get("before");
  const after = url.searchParams.get("after");
  const limit = Number(url.searchParams.get("limit")) || 10;
  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }

  // Filter params
  const function_name = url.searchParams.get("function_name") || undefined;
  const variant_name = url.searchParams.get("variant_name") || undefined;
  const episode_id = url.searchParams.get("episode_id") || undefined;
  const search_query = url.searchParams.get("search_query") || undefined;

  // Parse JSON filter if present
  const filterParam = url.searchParams.get("filter");
  let filter: InferenceFilter | undefined;
  if (filterParam) {
    try {
      filter = JSON.parse(filterParam) as InferenceFilter;
    } catch {
      // Invalid JSON - ignore filter
      filter = undefined;
    }
  }

  const [inferenceResult, countsInfo] = await Promise.all([
    listInferencesWithPagination({
      before: before || undefined,
      after: after || undefined,
      limit,
      function_name,
      variant_name,
      episode_id,
      filter,
      search_query,
    }),
    countInferencesByFunction(),
  ]);

  const totalInferences = countsInfo.reduce((acc, curr) => acc + curr.count, 0);

  return {
    inferences: inferenceResult.inferences,
    hasNextPage: inferenceResult.hasNextPage,
    hasPreviousPage: inferenceResult.hasPreviousPage,
    limit,
    totalInferences,
    // Return filter state for UI
    function_name,
    variant_name,
    episode_id,
    search_query,
    filter,
  };
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const {
    inferences,
    hasNextPage,
    hasPreviousPage,
    limit,
    totalInferences,
    function_name,
    variant_name,
    episode_id,
    search_query,
    filter,
  } = loaderData;

  const navigate = useNavigate();

  const topInference = inferences.at(0);
  const bottomInference = inferences.at(inferences.length - 1);

  // Build search params that preserve current filters
  const buildSearchParams = () => {
    const params = new URLSearchParams();
    params.set("limit", String(limit));
    if (function_name) params.set("function_name", function_name);
    if (variant_name) params.set("variant_name", variant_name);
    if (episode_id) params.set("episode_id", episode_id);
    if (search_query) params.set("search_query", search_query);
    if (filter) params.set("filter", JSON.stringify(filter));
    return params;
  };

  const handleNextPage = () => {
    if (bottomInference) {
      const params = buildSearchParams();
      params.set("before", bottomInference.inference_id);
      navigate(`?${params.toString()}`, {
        preventScrollReset: true,
      });
    }
  };

  const handlePreviousPage = () => {
    if (topInference) {
      const params = buildSearchParams();
      params.set("after", topInference.inference_id);
      navigate(`?${params.toString()}`, {
        preventScrollReset: true,
      });
    }
  };

  return (
    <PageLayout>
      <PageHeader heading="Inferences" count={totalInferences} />
      <SectionLayout>
        <InferenceSearchBar />
        <InferencesTable
          inferences={inferences}
          function_name={function_name}
          variant_name={variant_name}
          episode_id={episode_id}
          search_query={search_query}
          filter={filter}
        />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={!hasPreviousPage}
          disableNext={!hasNextPage}
        />
      </SectionLayout>
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  logger.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}
