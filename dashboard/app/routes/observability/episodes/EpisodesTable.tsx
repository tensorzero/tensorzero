import { useState } from "react";
import type { EpisodeByIdRow, TableBounds } from "~/utils/clickhouse";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useNavigate } from "react-router";

export default function EpisodesTable({
  episodes,
  pageSize,
  bounds,
}: {
  episodes: EpisodeByIdRow[];
  pageSize: number;
  bounds: TableBounds;
}) {
  const [goToId, setGoToId] = useState("");
  const navigate = useNavigate();

  // TODO: wire this to go the the details page for a particular inference, maybe add a popover.
  const handleGoTo = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    // TODO: Implement episode navigation and highlighting
    console.log("Go to episode:", goToId);
    setGoToId("");
  };

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

  const topEpisode = episodes[0];
  const bottomEpisode = episodes[episodes.length - 1];

  // IMPORTANT: use the last_inference_id to navigate
  const handleNextPage = () => {
    navigate(
      `?before=${bottomEpisode.last_inference_id}&page_size=${pageSize}`,
    );
  };

  const handlePreviousPage = () => {
    navigate(`?after=${topEpisode.last_inference_id}&page_size=${pageSize}`);
  };

  // These are swapped because the table is sorted in descending order
  const disablePrevious = bounds.last_id === topEpisode.last_inference_id;
  const disableNext = bounds.first_id === bottomEpisode.last_inference_id;

  return (
    <div>
      <h2 className="mb-4 text-2xl font-semibold">Episodes</h2>
      <div className="mb-6 h-px w-full bg-gray-200"></div>
      <form onSubmit={handleGoTo} className="mb-4">
        <div className="flex gap-2">
          <Input
            type="text"
            placeholder="00000000-0000-0000-0000-000000000000"
            value={goToId}
            onChange={(e) => setGoToId(e.target.value)}
            className="flex-grow"
          />
          <Button type="submit">Go to Episode</Button>
        </div>
      </form>
      <div className="my-6 h-px w-full bg-gray-200"></div>
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
                <a href="#" className="block no-underline">
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
      <div className="mt-4 flex items-center justify-center gap-2">
        <Button
          onClick={handlePreviousPage}
          disabled={disablePrevious}
          className="rounded-md border border-gray-300 bg-white p-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <Button
          onClick={handleNextPage}
          disabled={disableNext}
          className="rounded-md border border-gray-300 bg-white p-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
