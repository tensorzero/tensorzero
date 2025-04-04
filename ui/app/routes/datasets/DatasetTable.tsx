import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { formatDate } from "~/utils/date";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import { Link } from "react-router";

export default function DatasetTable({
  counts,
}: {
  counts: DatasetCountInfo[];
}) {
  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Dataset Name</TableHead>
            <TableHead>Datapoint Count</TableHead>
            <TableHead>Last Updated</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {counts.length === 0 ? (
            <TableRow className="hover:bg-bg-primary">
              <TableCell
                colSpan={3}
                className="text-fg-muted px-3 py-8 text-center"
              >
                No datasets found.
              </TableCell>
            </TableRow>
          ) : (
            counts.map((count) => (
              <TableRow key={count.dataset_name} id={count.dataset_name}>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={`/datasets/${count.dataset_name}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {count.dataset_name}
                    </code>
                  </Link>
                </TableCell>
                <TableCell className="max-w-[200px]">{count.count}</TableCell>
                <TableCell>
                  {formatDate(new Date(count.last_updated))}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
