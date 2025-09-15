import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";
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

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const before = url.searchParams.get("before");
  const after = url.searchParams.get("after");
  const pageSize = Number(url.searchParams.get("pageSize")) || 10;
  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
  }
  const databaseClient = await getNativeDatabaseClient();

  const [episodes, bounds] = await Promise.all([
    databaseClient.queryEpisodeTable(
      pageSize,
      before || undefined,
      after || undefined,
    ),
    databaseClient.queryEpisodeTableBounds(),
  ]);

  return {
    episodes,
    pageSize,
    bounds,
  };
}

export default function EpisodesPage({ loaderData }: Route.ComponentProps) {
  const { episodes, pageSize, bounds } = loaderData;
  const navigate = useNavigate();

  const topEpisode = episodes.at(0);
  const bottomEpisode = episodes.at(-1);

  const handleNextPage = () => {
    if (bottomEpisode) {
      navigate(`?before=${bottomEpisode.episode_id}&pageSize=${pageSize}`, {
        preventScrollReset: true,
      });
    }
  };

  const handlePreviousPage = () => {
    if (topEpisode) {
      navigate(`?after=${topEpisode.episode_id}&pageSize=${pageSize}`, {
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
    <PageLayout>
      <PageHeader heading="Episodes" count={bounds.count} />
      <SectionLayout>
        <EpisodeSearchBar />
        <EpisodesTable episodes={episodes} />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={disablePrevious}
          disableNext={disableNext}
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
