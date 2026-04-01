import { listInferencesWithPagination } from "~/utils/clickhouse/inference.server";
import type { Route } from "./+types/route";
import InferencesTable, { type InferencesData } from "./InferencesTable";
import { Await, data } from "react-router";
import InferenceSearchBar from "./InferenceSearchBar";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import type { InferenceFilter, InferenceMetadata } from "~/types/tensorzero";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { applyPaginationLogic } from "~/utils/pagination";
import { Suspense } from "react";
import { StatsBar, StatsBarSkeleton } from "~/components/ui/StatsBar";
import { getRelativeTimeString } from "~/utils/date";

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

  // Create promise for total count and summary stats - will be streamed to the component
  const client = getTensorZeroClient();
  const countsInfoPromise = client.listFunctionsWithInferenceCount();
  const totalInferencesPromise = countsInfoPromise.then((countsInfo) =>
    countsInfo.reduce((acc, curr) => acc + curr.inference_count, 0),
  );
  const summaryPromise = countsInfoPromise.then((countsInfo) => {
    const totalInferences = countsInfo.reduce(
      (acc, curr) => acc + curr.inference_count,
      0,
    );
    const activeFunctions = countsInfo.filter(
      (info) => info.inference_count > 0,
    ).length;
    const lastTimestamp = countsInfo.reduce<string | null>((latest, info) => {
      if (!info.last_inference_timestamp || info.inference_count === 0)
        return latest;
      if (!latest) return info.last_inference_timestamp;
      return info.last_inference_timestamp > latest
        ? info.last_inference_timestamp
        : latest;
    }, null);
    const lastInference = lastTimestamp
      ? getRelativeTimeString(new Date(lastTimestamp))
      : "Never";
    return { totalInferences, activeFunctions, lastInference };
  });

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
    summaryData: summaryPromise,
    limit,
    function_name,
    variant_name,
    episode_id,
    search_query,
    filters,
  };
}

function InferencesSummarySkeleton() {
  return <StatsBarSkeleton count={3} />;
}

function InferencesSummary({
  data,
}: {
  data: {
    totalInferences: number;
    activeFunctions: number;
    lastInference: string;
  };
}) {
  return (
    <StatsBar
      items={[
        {
          label: "Total Inferences",
          value: data.totalInferences.toLocaleString(),
        },
        {
          label: "Active Functions",
          value: String(data.activeFunctions),
        },
        {
          label: "Last Inference",
          value: data.lastInference,
        },
      ]}
    />
  );
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const {
    inferencesData,
    totalInferences,
    summaryData,
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
      <Suspense fallback={<InferencesSummarySkeleton />}>
        <Await resolve={summaryData}>
          {(data) => <InferencesSummary data={data} />}
        </Await>
      </Suspense>
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
