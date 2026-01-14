import type { Route } from "./+types/route";
import EpisodesTable from "./EpisodesTable";
import { data, isRouteErrorResponse, useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import EpisodeSearchBar from "./EpisodeSearchBar";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { logger } from "~/utils/logger";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { EpisodeByIdRow, TableBoundsWithCount } from "~/types/tensorzero";
import { Suspense, use } from "react";

export type EpisodesData = {
  episodes: EpisodeByIdRow[];
  bounds: TableBoundsWithCount;
};

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const before = url.searchParams.get("before") || undefined;
  const after = url.searchParams.get("after") || undefined;
  const limitParam = url.searchParams.get("limit");
  const limit = limitParam !== null ? Number(limitParam) : 10;
  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }
  const client = getTensorZeroClient();

  // Don't await - return promises for streaming
  const episodesPromise = client
    .listEpisodes(limit, before, after)
    .then((r) => r.episodes);
  const boundsPromise = client.queryEpisodeTableBounds();

  // Combine into single promise for pagination logic
  const dataPromise: Promise<EpisodesData> = Promise.all([
    episodesPromise,
    boundsPromise,
  ]).then(([episodes, bounds]) => ({ episodes, bounds }));

  // Derive count promise from bounds - convert bigint to number for serialization
  const countPromise = boundsPromise.then((b) => Number(b.count));

  return {
    dataPromise,
    countPromise,
    limit,
  };
}

function PaginationContent({
  data,
  limit,
}: {
  data: Promise<EpisodesData>;
  limit: number;
}) {
  const { episodes, bounds } = use(data);
  const navigate = useNavigate();

  const topEpisode = episodes.at(0);
  const bottomEpisode = episodes.at(-1);

  const handleNextPage = () => {
    if (bottomEpisode) {
      navigate(`?before=${bottomEpisode.episode_id}&limit=${limit}`, {
        preventScrollReset: true,
      });
    }
  };

  const handlePreviousPage = () => {
    if (topEpisode) {
      navigate(`?after=${topEpisode.episode_id}&limit=${limit}`, {
        preventScrollReset: true,
      });
    }
  };

  // These are swapped because the table is sorted in descending order
  const disablePrevious =
    !bounds?.last_id || bounds.last_id === topEpisode?.episode_id;
  const disableNext =
    !bounds?.first_id || bounds.first_id === bottomEpisode?.episode_id;

  return (
    <PageButtons
      onPreviousPage={handlePreviousPage}
      onNextPage={handleNextPage}
      disablePrevious={disablePrevious}
      disableNext={disableNext}
    />
  );
}

export default function EpisodesPage({ loaderData }: Route.ComponentProps) {
  const { dataPromise, countPromise, limit } = loaderData;

  return (
    <PageLayout>
      <PageHeader
        heading="Episodes"
        subheading="Sequences of related inferences sharing a common outcome"
        count={countPromise}
      />
      <SectionLayout>
        <EpisodeSearchBar />
        <EpisodesTable data={dataPromise} />
        <Suspense
          fallback={
            <PageButtons
              onPreviousPage={() => {}}
              onNextPage={() => {}}
              disablePrevious
              disableNext
            />
          }
        >
          <PaginationContent data={dataPromise} limit={limit} />
        </Suspense>
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
