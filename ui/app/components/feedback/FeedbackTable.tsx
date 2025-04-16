import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import FeedbackValue from "~/components/feedback/FeedbackValue";
import { getMetricName } from "~/utils/clickhouse/helpers";
import type { FeedbackRow } from "~/utils/clickhouse/feedback";
import { formatDate } from "~/utils/date";
import MetricBadges from "~/components/metric/MetricBadges";
import { useConfig } from "~/context/config";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";
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
            <TableEmptyState message="No feedback found" />
          ) : (
            feedback.map((item) => (
              <TableRow key={item.id}>
                <TableCell className="max-w-[200px]">
                  <TableItemShortUuid id={item.id} />
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{getMetricName(item)}</span>
                    <MetricBadges
                      metric={metrics[getMetricName(item)]}
                      row={item}
                    />
                  </div>
                </TableCell>
                <TableCell>
                  <FeedbackValue
                    feedback={item}
                    metric={metrics[getMetricName(item)]}
                  />
                </TableCell>
                <TableCell>
                  <TableItemTime timestamp={item.timestamp} />
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
