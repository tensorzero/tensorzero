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
import type { InferenceFilter, InferenceMetadata } from "~/types/tensorzero";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

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

  const countsInfo = await countInferencesByFunction();
  const totalInferences = countsInfo.reduce((acc, curr) => acc + curr.count, 0);

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

    // Determine if there are more pages based on whether we got more than limit results
    const hasMore = metadataResponse.inference_metadata.length > limit;

    // Pagination direction logic:
    // - When using 'before': we're going to older inferences (next page = older)
    // - When using 'after': we're going to newer inferences (previous page = newer)
    // - When neither: we're on the first page (most recent)
    let hasNextPage: boolean;
    let hasPreviousPage: boolean;
    let inferences: typeof metadataResponse.inference_metadata;

    if (before) {
      // Going backwards in time (older). hasMore means there are older pages.
      hasNextPage = hasMore;
      // We came from a newer page, so there's always a previous (newer) page
      hasPreviousPage = true;
      // Extra item is at the end, so take first 'limit' items
      inferences = metadataResponse.inference_metadata.slice(0, limit);
    } else if (after) {
      // Going forwards in time (newer). hasMore means there are newer pages.
      hasPreviousPage = hasMore;
      // We came from an older page, so there's always a next (older) page
      hasNextPage = true;
      // Extra item is at position 0, so take items from position 1 onwards
      if (hasMore) {
        inferences = metadataResponse.inference_metadata.slice(1, limit + 1);
      } else {
        inferences = metadataResponse.inference_metadata;
      }
    } else {
      // Initial page load - showing most recent
      hasNextPage = hasMore;
      hasPreviousPage = false;
      // Extra item is at the end, so take first 'limit' items
      inferences = metadataResponse.inference_metadata.slice(0, limit);
    }

    return {
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
    };
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
      params.set("before", bottomInference.id);
      navigate(`?${params.toString()}`, {
        preventScrollReset: true,
      });
    }
  };

  const handlePreviousPage = () => {
    if (topInference) {
      const params = buildSearchParams();
      params.set("after", topInference.id);
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
