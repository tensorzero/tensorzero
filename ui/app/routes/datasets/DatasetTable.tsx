import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { DatasetMetadata } from "tensorzero-node";
import { Link, useFetcher } from "react-router";
import { TableItemTime } from "~/components/ui/TableItems";
import { toDatasetUrl } from "~/utils/urls";
import { Button } from "~/components/ui/button";
import { Trash, ChevronUp, ChevronDown, Search } from "lucide-react";
import { useMemo, useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  createColumnHelper,
  flexRender,
  type SortingState,
} from "@tanstack/react-table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

const columnHelper = createColumnHelper<DatasetMetadata>();

export default function DatasetTable({
  counts,
}: {
  counts: DatasetMetadata[];
}) {
  const fetcher = useFetcher();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [datasetToDelete, setDatasetToDelete] = useState<string | null>(null);

  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState("");

  const columns = useMemo(
    () => [
      columnHelper.accessor("dataset_name", {
        header: "Dataset Name",
        cell: (info) => (
          <Link
            to={toDatasetUrl(info.getValue())}
            className="block no-underline"
          >
            <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
              {info.getValue()}
            </code>
          </Link>
        ),
      }),
      columnHelper.accessor("count", {
        header: "Datapoint Count",
        cell: (info) => info.getValue(),
      }),
      columnHelper.accessor("last_updated", {
        header: "Last Updated",
        cell: (info) => <TableItemTime timestamp={info.getValue()} />,
      }),
      columnHelper.display({
        id: "actions",
        header: "",
        cell: (info) => (
          <div className="text-right">
            <ReadOnlyGuard
              asChild
              onClick={() => {
                setDatasetToDelete(info.row.original.dataset_name);
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
        ),
        enableSorting: false,
      }),
    ],
    [],
  );

  const table = useReactTable({
    data: counts,
    columns,
    state: {
      sorting,
      globalFilter,
    },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: "includesString",
  });

  return (
    <div>
      <div className="mb-4">
        <div className="relative">
          <Search className="text-muted-foreground absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search datasets..."
            value={globalFilter ?? ""}
            onChange={(e) => setGlobalFilter(e.target.value)}
            className="border-input bg-background focus:ring-ring w-full rounded-md border py-2 pr-4 pl-10 text-sm focus:border-transparent focus:ring-2 focus:outline-none"
          />
        </div>
      </div>
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <TableHead
                  key={header.id}
                  className={
                    header.column.getCanSort()
                      ? "cursor-pointer select-none"
                      : ""
                  }
                  onClick={header.column.getToggleSortingHandler()}
                >
                  <div className="flex items-center gap-1">
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext(),
                        )}
                    {header.column.getCanSort() && (
                      <div className="flex flex-col">
                        {header.column.getIsSorted() === "asc" ? (
                          <ChevronUp className="h-3 w-3" />
                        ) : header.column.getIsSorted() === "desc" ? (
                          <ChevronDown className="h-3 w-3" />
                        ) : (
                          <div className="flex flex-col">
                            <ChevronUp className="h-2 w-2 opacity-40" />
                            <ChevronDown className="h-2 w-2 opacity-40" />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows.length === 0 ? (
            <TableEmptyState message="No datasets found" />
          ) : (
            table.getRowModel().rows.map((row) => (
              <TableRow key={row.id} id={row.original.dataset_name}>
                {row.getVisibleCells().map((cell, index) => (
                  <TableCell
                    key={cell.id}
                    className={index === 0 ? "max-w-[200px]" : ""}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
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
