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
import { Badge } from "~/components/ui/badge";
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
            <TableHead>Count</TableHead>
            <TableHead>Last Updated</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {counts.map((count) => (
            <TableRow key={count.dataset_name} id={count.dataset_name}>
              <TableCell className="max-w-[200px]">
                <Link
                  to={`/datasets/${count.dataset_name}`}
                  className="block no-underline"
                >
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {count.dataset_name}
                  </code>
                </Link>
              </TableCell>
              <TableCell className="max-w-[200px]">
                <Badge variant="outline">{count.count}</Badge>
              </TableCell>
              <TableCell>{formatDate(new Date(count.last_updated))}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
