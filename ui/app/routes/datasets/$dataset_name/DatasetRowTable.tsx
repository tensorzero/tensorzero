import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { DatasetDetailRow } from "~/utils/clickhouse/datasets";
import { Link } from "react-router";
import { TableItemTime, TableItemFunction } from "~/components/ui/TableItems";

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
            <TableHead>Episode ID</TableHead>
            <TableHead>Function</TableHead>
            <TableHead>Updated</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.length === 0 ? (
            <TableEmptyState message="No datapoints found" />
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
                  <TableItemFunction
                    functionName={row.function_name}
                    functionType={row.type}
                    link={`/observability/functions/${row.function_name}`}
                  />
                </TableCell>
                <TableCell>
                  <TableItemTime timestamp={row.updated_at} />
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
