import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import type { EpisodeByIdRow } from "~/utils/clickhouse/inference";

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
          {episodes.map((episode) => (
            <TableRow key={episode.episode_id} id={episode.episode_id}>
              <TableCell className="max-w-[200px] lg:max-w-none">
                <a
                  href={`/observability/episode/${episode.episode_id}`}
                  className="block no-underline"
                >
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {episode.episode_id}
                  </code>
                </a>
              </TableCell>
              <TableCell>{episode.count}</TableCell>
              <TableCell className="max-w-[200px] lg:max-w-none">
                <span className="block overflow-hidden text-ellipsis whitespace-nowrap">
                  {formatTimeRange(
                    new Date(episode.start_time),
                    new Date(episode.end_time),
                    episode.count,
                  )}
                </span>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
