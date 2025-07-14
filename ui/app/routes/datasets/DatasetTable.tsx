import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import { TableItemTime } from "~/components/ui/TableItems";
import { Link } from "~/safe-navigation";

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
            <TableEmptyState message="No datasets found" />
          ) : (
            counts.map((count) => (
              <TableRow key={count.dataset_name} id={count.dataset_name}>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={[
                      "/datasets/:dataset_name",
                      { dataset_name: count.dataset_name },
                    ]}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {count.dataset_name}
                    </code>
                  </Link>
                </TableCell>
                <TableCell className="max-w-[200px]">{count.count}</TableCell>
                <TableCell>
                  <TableItemTime timestamp={count.last_updated} />
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
