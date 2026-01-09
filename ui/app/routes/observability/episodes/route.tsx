import type { Route } from "./+types/route";
import EpisodesTable from "./EpisodesTable";
import { data, useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import EpisodeSearchBar from "./EpisodeSearchBar";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { EpisodeByIdRow, TableBoundsWithCount } from "~/types/tensorzero";
import { Suspense, use } from "react";
import { isInfraError } from "~/utils/tensorzero/errors";
import { logger } from "~/utils/logger";

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
  // Graceful degradation: return empty data on infra errors
  const episodesPromise = client
    .listEpisodes(limit, before, after)
    .then((r) => r.episodes)
    .catch((error) => {
      if (isInfraError(error)) {
        logger.warn("Infrastructure unavailable, showing degraded episodes");
        return [] as EpisodeByIdRow[];
      }
      throw error;
    });
  const boundsPromise = client.queryEpisodeTableBounds().catch((error) => {
    if (isInfraError(error)) {
      logger.warn("Infrastructure unavailable, showing degraded bounds");
      return { first_id: null, last_id: null, count: BigInt(0) };
    }
    throw error;
  });

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
      <PageHeader heading="Episodes" count={countPromise} />
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
