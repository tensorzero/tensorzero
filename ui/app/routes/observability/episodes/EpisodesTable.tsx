import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { TableItemShortUuid } from "~/components/ui/TableItems";
import { toEpisodeUrl } from "~/utils/urls";
import { Suspense } from "react";
import { useLocation, Await, useAsyncError } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import type { EpisodesData } from "./route";

const formatTimeRange = (startTime: Date, endTime: Date, count: number) => {
  const formatOptions: Intl.DateTimeFormatOptions = {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  };
  const start = startTime.toLocaleString("en-US", formatOptions);
  if (count === 1) {
    return start;
  }
  const end = endTime.toLocaleString("en-US", formatOptions);
  return `${start} — ${end}`;
};

function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 10 }).map((_, i) => (
        <TableRow key={i}>
          <TableCell>
            <Skeleton className="h-4 w-24" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-4 w-16" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-4 w-48" />
          </TableCell>
        </TableRow>
      ))}
    </>
  );
}

function TableErrorState() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load episodes";

  return (
    <TableRow>
      <TableCell colSpan={3} className="text-center">
        <div className="flex flex-col items-center gap-2 py-8 text-red-600">
          <span className="font-medium">Error loading data</span>
          <span className="text-muted-foreground text-sm">{message}</span>
        </div>
      </TableCell>
    </TableRow>
  );
}

function TableRows({ data }: { data: EpisodesData }) {
  const { episodes } = data;

  if (episodes.length === 0) {
    return <TableEmptyState message="No episodes found" />;
  }

  return (
    <>
      {episodes.map((episode) => (
        <TableRow key={episode.episode_id} id={episode.episode_id}>
          <TableCell className="max-w-[200px] lg:max-w-none">
            <TableItemShortUuid
              id={episode.episode_id}
              link={toEpisodeUrl(episode.episode_id)}
            />
          </TableCell>
          <TableCell>{episode.count}</TableCell>
          <TableCell className="max-w-[200px] lg:max-w-none">
            <span className="block overflow-hidden text-ellipsis whitespace-nowrap">
              {formatTimeRange(
                new Date(episode.start_time),
                new Date(episode.end_time),
                Number(episode.count),
              )}
            </span>
          </TableCell>
        </TableRow>
      ))}
    </>
  );
}

export default function EpisodesTable({
  data,
}: {
  data: Promise<EpisodesData>;
}) {
  const location = useLocation();

  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Episode ID</TableHead>
            <TableHead>Inference Count</TableHead>
            <TableHead>Time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <Suspense key={location.key} fallback={<SkeletonRows />}>
            <Await resolve={data} errorElement={<TableErrorState />}>
              {(resolvedData) => <TableRows data={resolvedData} />}
            </Await>
          </Suspense>
        </TableBody>
      </Table>
    </div>
  );
}
