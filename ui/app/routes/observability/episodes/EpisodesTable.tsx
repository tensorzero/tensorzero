import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { EpisodeByIdRow } from "~/types/tensorzero";
import { TableItemShortUuid } from "~/components/ui/TableItems";
import { toEpisodeUrl } from "~/utils/urls";

export default function EpisodesTable({
  episodes,
}: {
  episodes: EpisodeByIdRow[];
}) {
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
          {episodes.length === 0 ? (
            <TableEmptyState message="No episodes found" />
          ) : (
            episodes.map((episode) => (
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
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
