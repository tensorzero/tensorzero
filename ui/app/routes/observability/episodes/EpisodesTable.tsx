import {
  Table,
  TableAsyncErrorState,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";
import { toEpisodeUrl, toInferenceUrl } from "~/utils/urls";
import { Suspense } from "react";
import { useLocation, Await } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { Badge } from "~/components/ui/badge";
import type { EpisodesData } from "./route";

function formatDuration(startTime: Date, endTime: Date): string {
  const diffMs = endTime.getTime() - startTime.getTime();
  if (diffMs < 1000) return `${diffMs}ms`;
  if (diffMs < 60_000) return `${(diffMs / 1000).toFixed(1)}s`;
  if (diffMs < 3_600_000)
    return `${Math.floor(diffMs / 60_000)}m ${Math.floor((diffMs % 60_000) / 1000)}s`;
  return `${Math.floor(diffMs / 3_600_000)}h ${Math.floor((diffMs % 3_600_000) / 60_000)}m`;
}

function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 10 }).map((_, i) => (
        <TableRow key={i}>
          <TableCell>
            <Skeleton className="h-4 w-24" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-4 w-12" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-4 w-24" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-4 w-16" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-4 w-32" />
          </TableCell>
        </TableRow>
      ))}
    </>
  );
}

function TableRows({ data }: { data: EpisodesData }) {
  const { episodes } = data;

  if (episodes.length === 0) {
    return <TableEmptyState message="No episodes found" />;
  }

  return (
    <>
      {episodes.map((episode) => {
        const startTime = new Date(episode.start_time);
        const endTime = new Date(episode.end_time);
        const count = Number(episode.count);

        return (
          <TableRow key={episode.episode_id} id={episode.episode_id}>
            <TableCell className="max-w-[200px] lg:max-w-none">
              <TableItemShortUuid
                id={episode.episode_id}
                link={toEpisodeUrl(episode.episode_id)}
              />
            </TableCell>
            <TableCell>
              <Badge
                variant="secondary"
                className="font-mono text-xs font-normal"
              >
                {count}
              </Badge>
            </TableCell>
            <TableCell className="max-w-[120px] lg:max-w-none">
              <TableItemShortUuid
                id={episode.last_inference_id}
                link={toInferenceUrl(episode.last_inference_id)}
              />
            </TableCell>
            <TableCell>
              <span className="text-fg-secondary whitespace-nowrap text-xs">
                {count > 1 ? formatDuration(startTime, endTime) : "—"}
              </span>
            </TableCell>
            <TableCell>
              <TableItemTime timestamp={episode.start_time} />
            </TableCell>
          </TableRow>
        );
      })}
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
            <TableHead>Inferences</TableHead>
            <TableHead>Last Inference</TableHead>
            <TableHead>Duration</TableHead>
            <TableHead>Started</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <Suspense key={location.key} fallback={<SkeletonRows />}>
            <Await
              resolve={data}
              errorElement={
                <TableAsyncErrorState
                  colSpan={5}
                  defaultMessage="Failed to load episodes"
                />
              }
            >
              {(resolvedData) => <TableRows data={resolvedData} />}
            </Await>
          </Suspense>
        </TableBody>
      </Table>
    </div>
  );
}
