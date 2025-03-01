import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import FeedbackValue from "~/components/feedback/FeedbackValue";
import { getMetricName } from "~/utils/clickhouse/helpers";
import type { FeedbackRow } from "~/utils/clickhouse/feedback";
import { formatDate } from "~/utils/date";
import { MetricBadges } from "~/components/metric/MetricBadges";
import { useConfig } from "~/context/config";

export default function FeedbackTable({
  feedback,
}: {
  feedback: FeedbackRow[];
}) {
  const config = useConfig();
  const metrics = config.metrics;
  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Metric</TableHead>
            <TableHead>Value</TableHead>
            <TableHead>Time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {feedback.length === 0 ? (
            <TableRow className="hover:bg-background-primary">
              <TableCell
                colSpan={4}
                className="px-3 py-8 text-center text-foreground-muted"
              >
                No feedback found
              </TableCell>
            </TableRow>
          ) : (
            feedback.map((item) => (
              <TableRow key={item.id}>
                <TableCell className="max-w-[200px]">
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono">
                    {item.id}
                  </code>
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <span>{getMetricName(item)}</span>
                    <MetricBadges metric={metrics[getMetricName(item)]} />
                  </div>
                </TableCell>
                <TableCell>
                  <FeedbackValue feedback={item} />
                </TableCell>
                <TableCell>{formatDate(new Date(item.timestamp))}</TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
