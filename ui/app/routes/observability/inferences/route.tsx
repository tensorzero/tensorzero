import { listInferencesWithPagination } from "~/utils/clickhouse/inference.server";
import type { Route } from "./+types/route";
import InferencesTable, { type InferencesData } from "./InferencesTable";
import { data } from "react-router";
import InferenceSearchBar from "./InferenceSearchBar";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { LayoutErrorBoundary } from "~/components/ui/error";
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

  // Parse JSON filters if present
  const filtersParam = url.searchParams.get("filters");
  let filters: InferenceFilter | undefined;
  if (filtersParam) {
    try {
      filters = JSON.parse(filtersParam) as InferenceFilter;
    } catch {
      // Invalid JSON - ignore filters
      filters = undefined;
    }
  }

  // Only need the slow path for search queries and advanced filters
  // The fast listInferenceMetadata endpoint now supports function_name, variant_name, and episode_id
  const needsFullInferences = search_query || filters;

  // Create promise for total count - will be streamed to the component
  const client = getTensorZeroClient();
  const totalInferencesPromise = client
    .listFunctionsWithInferenceCount()
    .then((countsInfo) =>
      countsInfo.reduce((acc, curr) => acc + curr.inference_count, 0),
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
      filters,
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
    filters,
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
    filters,
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
          filters={filters}
        />
      </SectionLayout>
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <LayoutErrorBoundary error={error} />;
}
