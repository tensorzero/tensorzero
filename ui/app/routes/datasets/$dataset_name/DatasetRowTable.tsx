import type { DatasetDetailRow } from "~/types/tensorzero";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import {
  TableItemTime,
  TableItemFunction,
  TableItemShortUuid,
  TableItemText,
} from "~/components/ui/TableItems";
import { Button } from "~/components/ui/button";
import { Trash } from "lucide-react";
import { useState, useEffect } from "react";
import { useFetcher } from "react-router";
import { toFunctionUrl, toDatapointUrl, toEpisodeUrl } from "~/utils/urls";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

export default function DatasetRowTable({
  rows,
  dataset_name,
}: {
  rows: DatasetDetailRow[];
  dataset_name: string;
}) {
  const activeFetcher = useFetcher();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [datapointToDelete, setDatapointToDelete] =
    useState<DatasetDetailRow | null>(null);

  // Handle successful deletion
  useEffect(() => {
    if (activeFetcher.data && activeFetcher.state === "idle") {
      if (activeFetcher.data.success === true) {
        // Close the dialog
        setDeleteDialogOpen(false);
        setDatapointToDelete(null);
      }
    }
  }, [activeFetcher.data, activeFetcher.state]);

  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Episode ID</TableHead>
            <TableHead>Name</TableHead>
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
                    link={toDatapointUrl(dataset_name, row.id)}
                  />
                </TableCell>
                <TableCell>
                  {row.episode_id && (
                    <TableItemShortUuid
                      id={row.episode_id}
                      link={toEpisodeUrl(row.episode_id)}
                    />
                  )}
                </TableCell>
                <TableCell>
                  {/* TODO: switch to using undefined instead of null */}
                  <TableItemText text={row.name ?? null} />
                </TableCell>
                <TableCell>
                  <TableItemFunction
                    functionName={row.function_name}
                    functionType={row.type}
                    link={toFunctionUrl(row.function_name)}
                  />
                </TableCell>
                <TableCell>
                  <TableItemTime timestamp={row.updated_at} />
                </TableCell>
                <TableCell>
                  <div className="text-right">
                    <ReadOnlyGuard
                      asChild
                      onClick={() => {
                        setDatapointToDelete(row);
                        setDeleteDialogOpen(true);
                      }}
                    >
                      <Button
                        variant="ghost"
                        size="icon"
                        className="opacity-60 transition-opacity hover:opacity-100"
                      >
                        <Trash />
                      </Button>
                    </ReadOnlyGuard>
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
              The datapoint will be marked as stale in the database (soft
              deletion). This action cannot be undone.
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
                  activeFetcher.submit(formData, {
                    method: "post",
                    action: ".",
                  });
                }
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
