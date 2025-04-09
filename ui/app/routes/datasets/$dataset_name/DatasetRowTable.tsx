import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableCellTime,
} from "~/components/ui/table";
import { formatDate } from "~/utils/date";
import type { DatasetDetailRow } from "~/utils/clickhouse/datasets";
import { Badge } from "~/components/ui/badge";
import { Link } from "react-router";
import { FunctionLink } from "~/components/function/FunctionLink";

export default function DatasetRowTable({
  rows,
  dataset_name,
}: {
  rows: DatasetDetailRow[];
  dataset_name: string;
}) {
  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Inference Type</TableHead>
            <TableHead>Episode ID</TableHead>
            <TableHead>Function</TableHead>
            <TableHead>Updated</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.length === 0 ? (
            <TableRow className="hover:bg-bg-primary">
              <TableCell
                colSpan={5}
                className="text-fg-muted px-3 py-8 text-center"
              >
                No datapoints found.
              </TableCell>
            </TableRow>
          ) : (
            rows.map((row) => (
              <TableRow key={row.id} id={row.id}>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={`/datasets/${dataset_name}/datapoint/${row.id}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {row.id}
                    </code>
                  </Link>
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <Badge variant="outline">{row.type}</Badge>
                </TableCell>
                <TableCell>
                  <Link
                    to={`/observability/episodes/${row.episode_id}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {row.episode_id}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>
                  <FunctionLink functionName={row.function_name}>
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {row.function_name}
                    </code>
                  </FunctionLink>
                </TableCell>
                <TableCellTime>
                  {formatDate(new Date(row.updated_at))}
                </TableCellTime>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
