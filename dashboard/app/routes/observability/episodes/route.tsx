import {
  queryEpisodeTable,
  queryEpisodeTableBounds,
} from "~/utils/clickhouse/inference";
import type { Route } from "./+types/route";
import EpisodesTable from "./EpisodesTable";
import { data, isRouteErrorResponse, useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const before = url.searchParams.get("before");
  const after = url.searchParams.get("after");
  const pageSize = Number(url.searchParams.get("pageSize")) || 10;
  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
  }

  const [episodes, bounds] = await Promise.all([
    queryEpisodeTable({
      before: before || undefined,
      after: after || undefined,
      page_size: pageSize,
    }),
    queryEpisodeTableBounds(),
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

  const topEpisode = episodes[0];
  const bottomEpisode = episodes[episodes.length - 1];

  // IMPORTANT: use the last_inference_id to navigate
  const handleNextPage = () => {
    navigate(`?before=${bottomEpisode.last_inference_id}&pageSize=${pageSize}`);
  };

  const handlePreviousPage = () => {
    navigate(`?after=${topEpisode.last_inference_id}&pageSize=${pageSize}`);
  };

  // These are swapped because the table is sorted in descending order
  const disablePrevious = bounds.last_id === topEpisode.last_inference_id;
  const disableNext = bounds.first_id === bottomEpisode.last_inference_id;

  return (
    <div className="container mx-auto px-4 py-8">
      <EpisodesTable episodes={episodes} />
      <PageButtons
        onPreviousPage={handlePreviousPage}
        onNextPage={handleNextPage}
        disablePrevious={disablePrevious}
        disableNext={disableNext}
      />
    </div>
  );
}
export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  console.error(error);

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
