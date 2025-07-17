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
import { Link, useFetcher } from "react-router";
import { TableItemTime } from "~/components/ui/TableItems";
import { Button } from "~/components/ui/button";
import { Trash } from "lucide-react";
import { useState } from "react";

export default function DatasetTable({
  counts,
}: {
  counts: DatasetCountInfo[];
}) {
  const fetcher = useFetcher();
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);
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
              <TableRow
                key={count.dataset_name}
                id={count.dataset_name}
                onMouseEnter={() => setHoveredRow(count.dataset_name)}
                onMouseLeave={() => setHoveredRow(null)}
              >
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
                  <TableItemTime timestamp={count.last_updated} />
                </TableCell>
                <TableCell className="text-right">
                  <Button
                    variant="ghost"
                    size="icon"
                    className={
                      hoveredRow === count.dataset_name ? "" : "invisible"
                    }
                    onClick={() => {
                      if (
                        window.confirm(
                          `Are you sure you want to delete the dataset "${count.dataset_name}"?`,
                        )
                      ) {
                        fetcher.submit(
                          { action: "delete", datasetName: count.dataset_name },
                          { method: "post" },
                        );
                      }
                    }}
                  >
                    <Trash />
                  </Button>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
