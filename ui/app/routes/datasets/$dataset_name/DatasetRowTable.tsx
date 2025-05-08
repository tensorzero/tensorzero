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
import { Badge } from "~/components/ui/badge";
import { FunctionLink } from "~/components/function/FunctionLink";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";

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
            <TableEmptyState message="No datapoints found" />
          ) : (
            rows.map((row) => (
              <TableRow key={row.id} id={row.id}>
                <TableCell className="max-w-[200px]">
                  <TableItemShortUuid
                    id={row.id}
                    link={`/datasets/${dataset_name}/datapoint/${row.id}`}
                  />
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <Badge variant="outline">{row.type}</Badge>
                </TableCell>
                <TableCell>
                  <TableItemShortUuid
                    id={row.episode_id}
                    link={`/observability/episodes/${row.episode_id}`}
                  />
                </TableCell>
                <TableCell>
                  <FunctionLink functionName={row.function_name}>
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {row.function_name}
                    </code>
                  </FunctionLink>
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
