import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { formatDate } from "~/utils/date";
import type { DynamicEvaluationRunEpisodeWithFeedback } from "~/utils/clickhouse/dynamic_evaluations";
import { Link } from "react-router";

export default function DynamicEvaluationRunEpisodesTable({
  episodes,
}: {
  episodes: DynamicEvaluationRunEpisodeWithFeedback[];
}) {
  // Extract all unique metric names from all episodes
  const allMetricNames = new Set<string>();
  episodes.forEach((episode) => {
    episode.feedback_metric_names.forEach((metricName) => {
      allMetricNames.add(metricName);
    });
  });

  // Convert to sorted array for consistent column order
  const uniqueMetricNames = Array.from(allMetricNames).sort();

  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Episode ID</TableHead>
            <TableHead>Timestamp</TableHead>
            <TableHead>Tags</TableHead>
            <TableHead>Datapoint Name</TableHead>
            {/* Add dynamic columns for each metric */}
            {uniqueMetricNames.map((metricName) => (
              <TableHead key={metricName}>{metricName}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {episodes.length === 0 ? (
            <TableEmptyState message="No episodes found" />
          ) : (
            episodes.map((episode) => (
              <TableRow key={episode.episode_id}>
                <TableCell className="max-w-[200px]">
                  <Link to={`/observability/episodes/${episode.episode_id}`}>
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {episode.episode_id}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>{formatDate(new Date(episode.timestamp))}</TableCell>
                <TableCell>
                  <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300">
                    {Object.entries(episode.tags)
                      .map(([k, v]) => `${k}: ${v}`)
                      .join(", ")}
                  </code>
                </TableCell>
                <TableCell>
                  <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300">
                    {episode.datapoint_name ?? ""}
                  </code>
                </TableCell>
                {uniqueMetricNames.map((metricName) => {
                  const metricIndex =
                    episode.feedback_metric_names.indexOf(metricName);
                  const metricValue =
                    metricIndex !== -1
                      ? episode.feedback_values[metricIndex]
                      : null;

                  return (
                    <TableCell key={metricName}>
                      <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300">
                        {metricValue !== null ? (
                          metricValue
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </code>
                    </TableCell>
                  );
                })}
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
