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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";

export default function DatasetTable({
  counts,
}: {
  counts: DatasetCountInfo[];
}) {
  const fetcher = useFetcher();
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [datasetToDelete, setDatasetToDelete] = useState<string | null>(null);
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
                      setDatasetToDelete(count.dataset_name);
                      setDeleteDialogOpen(true);
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

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              Are you sure you want to delete the dataset{" "}
              <span className="font-mono text-lg font-bold text-red-500">
                {datasetToDelete}
              </span>
              ?
            </DialogTitle>
            <DialogDescription>
              The datapoints will be marked as stale in the database (soft
              deletion). This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (datasetToDelete) {
                  fetcher.submit(
                    { action: "delete", datasetName: datasetToDelete },
                    { method: "post" },
                  );
                }
                setDeleteDialogOpen(false);
                setDatasetToDelete(null);
              }}
            >
              <Trash className="inline h-4 w-4" />
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
