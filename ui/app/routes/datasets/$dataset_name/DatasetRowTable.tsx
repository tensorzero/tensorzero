import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { formatDate } from "~/utils/date";
import type { DatasetDetailRow } from "~/utils/clickhouse/datasets";
import { Badge } from "~/components/ui/badge";
import { Link } from "react-router";

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
          {rows.map((row) => (
            <TableRow key={row.id} id={row.id}>
              <TableCell className="max-w-[200px]">
                <Link
                  to={`/datasets/${dataset_name}/datapoint/${row.id}`}
                  className="block no-underline"
                >
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
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
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {row.episode_id}
                  </code>
                </Link>
              </TableCell>
              <TableCell>
                <Link
                  to={`/observability/functions/${row.function_name}`}
                  className="block no-underline"
                >
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {row.function_name}
                  </code>
                </Link>
              </TableCell>
              <TableCell>{formatDate(new Date(row.updated_at))}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
