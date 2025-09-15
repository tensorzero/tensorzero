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
import {
  TableItemTime,
  TableItemFunction,
  TableItemShortUuid,
} from "~/components/ui/TableItems";
import { Button } from "~/components/ui/button";
import { Trash } from "lucide-react";
import { useState } from "react";
import { useFetcher } from "react-router";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";

export default function DatasetRowTable({
  rows,
  dataset_name,
  fetcher,
}: {
  rows: DatasetDetailRow[];
  dataset_name: string;
  fetcher?: ReturnType<typeof useFetcher>;
}) {
  const defaultFetcher = useFetcher();
  const activeFetcher = fetcher || defaultFetcher;
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [datapointToDelete, setDatapointToDelete] =
    useState<DatasetDetailRow | null>(null);

  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Episode ID</TableHead>
            <TableHead>Function</TableHead>
            <TableHead>Updated</TableHead>
            <TableHead className="w-[50px]"></TableHead>
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
                <TableCell>
                  <TableItemShortUuid
                    id={row.episode_id}
                    link={`/observability/episodes/${row.episode_id}`}
                  />
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
                <TableCell>
                  <div className="text-right">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="opacity-60 transition-opacity hover:opacity-100"
                      onClick={() => {
                        setDatapointToDelete(row);
                        setDeleteDialogOpen(true);
                      }}
                    >
                      <Trash />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Datapoint</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this datapoint? This action cannot
              be undone.
              {datapointToDelete && (
                <div className="text-muted-foreground mt-2 text-sm">
                  <strong>ID:</strong> {datapointToDelete.id}
                  <br />
                  <strong>Function:</strong> {datapointToDelete.function_name}
                </div>
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex justify-between gap-2">
            <Button
              variant="outline"
              onClick={() => setDeleteDialogOpen(false)}
            >
              Cancel
            </Button>
            <div className="flex-1" />
            <Button
              variant="destructive"
              onClick={() => {
                if (datapointToDelete) {
                  const formData = new FormData();
                  formData.append("action", "delete_datapoint");
                  formData.append("datapoint_id", datapointToDelete.id);
                  formData.append(
                    "function_name",
                    datapointToDelete.function_name,
                  );
                  formData.append("function_type", datapointToDelete.type);
                  activeFetcher.submit(formData, { method: "post" });
                }
                setDeleteDialogOpen(false);
                setDatapointToDelete(null);
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
