import {
  listInferencesWithPagination,
  countInferencesByFunction,
} from "~/utils/clickhouse/inference.server";
import type { Route } from "./+types/route";
import InferencesTable, { type InferencesData } from "./InferencesTable";
import { data, isRouteErrorResponse } from "react-router";
import InferenceSearchBar from "./InferenceSearchBar";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { logger } from "~/utils/logger";
import type { InferenceFilter, InferenceMetadata } from "~/types/tensorzero";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { applyPaginationLogic } from "~/utils/pagination";

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

  // Only need the slow path for search queries and advanced filters
  // The fast listInferenceMetadata endpoint now supports function_name, variant_name, and episode_id
  const needsFullInferences = search_query || filter;

  // Create promise for total count - will be streamed to the component
  const totalInferencesPromise = countInferencesByFunction().then(
    (countsInfo) => countsInfo.reduce((acc, curr) => acc + curr.count, 0),
  );

  // Create promise for inferences data - will be streamed to the component
  const inferencesDataPromise: Promise<InferencesData> = (async () => {
    if (!needsFullInferences) {
      // Use faster gateway endpoint - now supports simple filters
      const client = getTensorZeroClient();
      const metadataResponse = await client.listInferenceMetadata({
        before: before || undefined,
        after: after || undefined,
        limit: limit + 1, // Fetch one extra to determine if there's a next page
        function_name,
        variant_name,
        episode_id,
      });

      const {
        items: inferences,
        hasNextPage,
        hasPreviousPage,
      } = applyPaginationLogic(metadataResponse.inference_metadata, limit, {
        before,
        after,
      });

      return { inferences, hasNextPage, hasPreviousPage };
    }

    const inferenceResult = await listInferencesWithPagination({
      before: before || undefined,
      after: after || undefined,
      limit,
      function_name,
      variant_name,
      episode_id,
      filter,
      search_query,
    });

    // Map StoredInference to InferenceMetadata shape for the table
    const inferences: InferenceMetadata[] = inferenceResult.inferences.map(
      (inf) => ({
        id: inf.inference_id,
        episode_id: inf.episode_id,
        function_name: inf.function_name,
        variant_name: inf.variant_name,
        function_type: inf.type,
      }),
    );

    return {
      inferences,
      hasNextPage: inferenceResult.hasNextPage,
      hasPreviousPage: inferenceResult.hasPreviousPage,
    };
  })();

  return {
    inferencesData: inferencesDataPromise,
    totalInferences: totalInferencesPromise,
    limit,
    function_name,
    variant_name,
    episode_id,
    search_query,
    filter,
  };
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const {
    inferencesData,
    totalInferences,
    limit,
    function_name,
    variant_name,
    episode_id,
    search_query,
    filter,
  } = loaderData;

  return (
    <PageLayout>
      <PageHeader heading="Inferences" count={totalInferences} />
      <SectionLayout>
        <InferenceSearchBar />
        <InferencesTable
          data={inferencesData}
          limit={limit}
          function_name={function_name}
          variant_name={variant_name}
          episode_id={episode_id}
          search_query={search_query}
          filter={filter}
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
